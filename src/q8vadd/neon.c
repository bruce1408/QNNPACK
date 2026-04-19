/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/common.h>
#include <qnnpack/q8vadd.h>

/**
 * NEON 优化的 8 位整数向量加法微内核。
 * 执行两个 uint8_t 向量 a 和 b 的逐元素加法：y = clamp((a - za) * ma + (b - zb) * mb >> shift + round + zy)
 * 支持 AArch64 和 ARM32，支持批量处理（32/16/8 元素）和剩余/标量回退。
 *
 * @param n 向量元素数量
 * @param a 输入向量 A (const uint8_t*)
 * @param b 输入向量 B (const uint8_t*)
 * @param y 输出向量 (uint8_t*)
 * @param quantization_params 量化参数：零点 ma/mb、乘子、右移、y_min/max 等
 */
void q8vadd_ukernel__neon(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union qnnp_add_quantization_params quantization_params[restrict static 1])
{
  // 加载 NEON 量化参数到向量寄存器（广播单值）
  const uint8x8_t va_zero_point = vld1_dup_u8(&quantization_params->neon.a_zero_point);  // A 零点 (广播到 8 通道)
  const uint8x8_t vb_zero_point = vld1_dup_u8(&quantization_params->neon.b_zero_point);  // B 零点
  const int16x8_t vy_zero_point = vld1q_dup_s16(&quantization_params->neon.y_zero_point); // 输出零点 (16-bit, 8 通道)
  const int32x4_t va_multiplier = vld1q_dup_s32(&quantization_params->neon.a_multiplier); // A 乘子 (32-bit, 4 通道)
  const int32x4_t vb_multiplier = vld1q_dup_s32(&quantization_params->neon.b_multiplier); // B 乘子
  const int32x4_t vright_shift = vld1q_dup_s32(&quantization_params->neon.right_shift);   // 右移量 (可变移位)
  const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0))); // 零移位掩码 (shift==0 时)
  const uint8x16_t vy_max = vld1q_dup_u8(&quantization_params->neon.y_max);               // 输出上限 (16 通道)
  const uint8x16_t vy_min = vld1q_dup_u8(&quantization_params->neon.y_min);               // 输出下限

  // 主向量化路径：n >= 8 时
  if QNNP_LIKELY(n >= 8) {
#ifdef __aarch64__
    // AArch64 大批量循环：每次 32 元素 (4x8)
    for (; n >= 32; n -= 32) {
      // 加载 32 字节输入 (4x 16-byte 向量)
      const uint8x16_t va01 = vld1q_u8(a); a += 16;
      const uint8x16_t vb01 = vld1q_u8(b); b += 16;
      const uint8x16_t va23 = vld1q_u8(a); a += 16;
      const uint8x16_t vb23 = vld1q_u8(b); b += 16;

      /* 减零点：uint8 -> uint16 (无符号扩展到 int16) */
      const int16x8_t vxa0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
      const int16x8_t vxb0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
      const int16x8_t vxa1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
      const int16x8_t vxb1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));
      const int16x8_t vxa2 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va23), va_zero_point));
      const int16x8_t vxb2 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb23), vb_zero_point));
      const int16x8_t vxa3 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va23), va_zero_point));
      const int16x8_t vxb3 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb23), vb_zero_point));

      /* 乘法累加：int16 -> int32，A * ma + B * mb */
      int32x4_t vacc0_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa0)), va_multiplier);
      int32x4_t vacc1_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa1)), va_multiplier);
      int32x4_t vacc2_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa2)), va_multiplier);
      int32x4_t vacc3_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa3)), va_multiplier);
      int32x4_t vacc0_hi = vmulq_s32(vmovl_high_s16(vxa0), va_multiplier);
      int32x4_t vacc1_hi = vmulq_s32(vmovl_high_s16(vxa1), va_multiplier);
      int32x4_t vacc2_hi = vmulq_s32(vmovl_high_s16(vxa2), va_multiplier);
      int32x4_t vacc3_hi = vmulq_s32(vmovl_high_s16(vxa3), va_multiplier);

      vacc0_lo = vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vxb0)), vb_multiplier);
      vacc1_lo = vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vxb1)), vb_multiplier);
      vacc2_lo = vmlaq_s32(vacc2_lo, vmovl_s16(vget_low_s16(vxb2)), vb_multiplier);
      vacc3_lo = vmlaq_s32(vacc3_lo, vmovl_s16(vget_low_s16(vxb3)), vb_multiplier);
      vacc0_hi = vmlaq_s32(vacc0_hi, vmovl_high_s16(vxb0), vb_multiplier);
      vacc1_hi = vmlaq_s32(vacc1_hi, vmovl_high_s16(vxb1), vb_multiplier);
      vacc2_hi = vmlaq_s32(vacc2_hi, vmovl_high_s16(vxb2), vb_multiplier);
      vacc3_hi = vmlaq_s32(vacc3_hi, vmovl_high_s16(vxb3), vb_multiplier);

      /* 右移并舍入：先加 0x8000... (31-bit)，再变长右移 */
      vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
      vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
      vacc2_lo = vsraq_n_s32(vacc2_lo, vbicq_s32(vacc2_lo, vzero_shift_mask), 31);
      vacc3_lo = vsraq_n_s32(vacc3_lo, vbicq_s32(vacc3_lo, vzero_shift_mask), 31);
      vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
      vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);
      vacc2_hi = vsraq_n_s32(vacc2_hi, vbicq_s32(vacc2_hi, vzero_shift_mask), 31);
      vacc3_hi = vsraq_n_s32(vacc3_hi, vbicq_s32(vacc3_hi, vzero_shift_mask), 31);

      vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
      vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
      vacc2_lo = vrshlq_s32(vacc2_lo, vright_shift);
      vacc3_lo = vrshlq_s32(vacc3_lo, vright_shift);
      vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
      vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);
      vacc2_hi = vrshlq_s32(vacc2_hi, vright_shift);
      vacc3_hi = vrshlq_s32(vacc3_hi, vright_shift);

      /* 打包饱和到 int16，加输出零点 */
      const int16x8_t vacc0 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0_lo), vacc0_hi), vy_zero_point);
      const int16x8_t vacc1 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1_lo), vacc1_hi), vy_zero_point);
      const int16x8_t vacc2 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2_lo), vacc2_hi), vy_zero_point);
      const int16x8_t vacc3 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3_lo), vacc3_hi), vy_zero_point);

      // 无符号饱和打包到 uint8，并 clamp
      uint8x16_t vy01 = vqmovun_high_s16(vqmovun_s16(vacc0), vacc1);
      uint8x16_t vy23 = vqmovun_high_s16(vqmovun_s16(vacc2), vacc3);

      vy01 = vmaxq_u8(vy01, vy_min);
      vy23 = vmaxq_u8(vy23, vy_min);
      vy01 = vminq_u8(vy01, vy_max);
      vy23 = vminq_u8(vy23, vy_max);

      // 存储结果
      vst1q_u8(y, vy01); y += 16;
      vst1q_u8(y, vy23); y += 16;
    }
#else
    // ARM32 中批量循环：每次 16 元素 (2x8)
    for (; n >= 16; n -= 16) {
      const uint8x16_t va01 = vld1q_u8(a); a += 16;
      const uint8x16_t vb01 = vld1q_u8(b); b += 16;

      /* 减零点 */
      const int16x8_t vxa0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
      const int16x8_t vxb0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
      const int16x8_t vxa1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
      const int16x8_t vxb1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));

      /* 乘法累加 */
      int32x4_t vacc0_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa0)), va_multiplier);
      int32x4_t vacc1_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa1)), va_multiplier);
      int32x4_t vacc0_hi = vmulq_s32(vmovl_s16(vget_high_s16(vxa0)), va_multiplier);
      int32x4_t vacc1_hi = vmulq_s32(vmovl_s16(vget_high_s16(vxa1)), va_multiplier);

      __builtin_prefetch(a + 640);  // 预取优化
      __builtin_prefetch(b + 640);

      vacc0_lo = vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vxb0)), vb_multiplier);
      vacc1_lo = vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vxb1)), vb_multiplier);
      vacc0_hi = vmlaq_s32(vacc0_hi, vmovl_s16(vget_high_s16(vxb0)), vb_multiplier);
      vacc1_hi = vmlaq_s32(vacc1_hi, vmovl_s16(vget_high_s16(vxb1)), vb_multiplier);

      /* 右移舍入 */
      vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
      vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
      vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
      vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);

      vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
      vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
      vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
      vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);

      /* 打包加零点 */
      const int16x8_t vacc0 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi)), vy_zero_point);
      const int16x8_t vacc1 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi)), vy_zero_point);

      uint8x16_t vy01 = vcombine_u8(vqmovun_s16(vacc0), vqmovun_s16(vacc1));
      vy01 = vmaxq_u8(vy01, vy_min);
      vy01 = vminq_u8(vy01, vy_max);

      vst1q_u8(y, vy01); y += 16;
    }
#endif
    // 小批量循环：每次 8 元素
    for (; n >= 8; n -= 8) {
      const uint8x8_t va = vld1_u8(a); a += 8;
      const uint8x8_t vb = vld1_u8(b); b += 8;

      /* 减零点 */
      const int16x8_t vxa = vreinterpretq_s16_u16(vsubl_u8(va, va_zero_point));
      const int16x8_t vxb = vreinterpretq_s16_u16(vsubl_u8(vb, vb_zero_point));

      /* 乘法累加 */
      int32x4_t vacc_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa)), va_multiplier);
#ifdef __aarch64__
      int32x4_t vacc_hi = vmulq_s32(vmovl_high_s16(vxa), va_multiplier);
#else
      int32x4_t vacc_hi = vmulq_s32(vmovl_s16(vget_high_s16(vxa)), va_multiplier);
#endif

      vacc_lo = vmlaq_s32(vacc_lo, vmovl_s16(vget_low_s16(vxb)), vb_multiplier);
#ifdef __aarch64__
      vacc_hi = vmlaq_s32(vacc_hi, vmovl_high_s16(vxb), vb_multiplier);
#else
      vacc_hi = vmlaq_s32(vacc_hi, vmovl_s16(vget_high_s16(vxb)), vb_multiplier);
#endif

      /* 右移舍入 */
      vacc_lo = vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
      vacc_hi = vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

      vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
      vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

      /* 打包加零点 */
#ifdef __aarch64__
      const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
#else
      const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), vy_zero_point);
#endif

      uint8x8_t vy = vqmovun_s16(vacc);
      vy = vmax_u8(vy, vget_low_u8(vy_min));
      vy = vmin_u8(vy, vget_low_u8(vy_max));

      vst1_u8(y, vy); y += 8;
    }
    // 剩余 <8 元素：使用左移掩码加载
    if (n != 0) {
      const size_t n_increment = n - 8;
      const int64x1_t vld_shift = vmov_n_s64(8 * n_increment);  // 字节左移量
      const uint8x8_t va = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a + n_increment)), vld_shift));
      const uint8x8_t vb = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(b + n_increment)), vld_shift));

      /* 减零点、乘累加、右移、打包 (同上 8 元素流程) */
      const int16x8_t vxa = vreinterpretq_s16_u16(vsubl_u8(va, va_zero_point));
      const int16x8_t vxb = vreinterpretq_s16_u16(vsubl_u8(vb, vb_zero_point));

      int32x4_t vacc_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa)), va_multiplier);
#ifdef __aarch64__
      int32x4_t vacc_hi = vmulq_s32(vmovl_high_s16(vxa), va_multiplier);
#else
      int32x4_t vacc_hi = vmulq_s32(vmovl_s16(vget_high_s16(vxa)), va_multiplier);
#endif

      vacc_lo = vmlaq_s32(vacc_lo, vmovl_s16(vget_low_s16(vxb)), vb_multiplier);
#ifdef __aarch64__
      vacc_hi = vmlaq_s32(vacc_hi, vmovl_high_s16(vxb), vb_multiplier);
#else
      vacc_hi = vmlaq_s32(vacc_hi, vmovl_s16(vget_high_s16(vxb)), vb_multiplier);
#endif

      vacc_lo = vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
      vacc_hi = vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

      vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
      vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

#ifdef __aarch64__
      const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
#else
      const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), vy_zero_point);
#endif

      uint8x8_t vy = vqmovun_s16(vacc);
      vy = vmax_u8(vy, vget_low_u8(vy_min));
      vy = vmin_u8(vy, vget_low_u8(vy_max));

      // 按剩余数量分 lane 存储 (4/2/1)
      if (n & 4) {
        vst1_lane_u32(__builtin_assume_aligned(y, 1), vreinterpret_u32_u8(vy), 0); y += 4;
        vy = vext_u8(vy, vy, 4);
      }
      if (n & 2) {
        vst1_lane_u16(__builtin_assume_aligned(y, 1), vreinterpret_u16_u8(vy), 0); y += 2;
        vy = vext_u8(vy, vy, 2);
      }
      if (n & 1) {
        vst1_lane_u8(y, vy, 0);
      }
    }
  } else {
    // 标量回退：逐元素处理 (使用 dup 加载单元素)
    for (; n != 0; n--) {
      const uint8x8_t va = vld1_dup_u8(a); a += 1;
      const uint8x8_t vb = vld1_dup_u8(b); b += 1;

      /* 减零点 (低 4 通道) */
      const int16x4_t vxa = vreinterpret_s16_u16(vget_low_u16(vsubl_u8(va, va_zero_point)));
      const int16x4_t vxb = vreinterpret_s16_u16(vget_low_u16(vsubl_u8(vb, vb_zero_point)));

      /* 乘累加 (低 2x32) */
      int32x2_t vacc = vmul_s32(vget_low_s32(vmovl_s16(vxa)), vget_low_s32(va_multiplier));
      vacc = vmla_s32(vacc, vget_low_s32(vmovl_s16(vxb)), vget_low_s32(vb_multiplier));

      /* 右移舍入 */
      vacc = vsra_n_s32(vacc, vbic_s32(vacc, vget_low_s32(vzero_shift_mask)), 31);

      vacc = vrshl_s32(vacc, vget_low_s32(vright_shift));

      // 打包加零点 (重复低通道)
      const int16x4_t vacc16 = vqadd_s16(vqmovn_s32(vcombine_s32(vacc, vacc)), vget_low_s16(vy_zero_point));

      /* 饱和打包 clamp */
      uint8x8_t vy = vqmovun_s16(vcombine_s16(vacc16, vacc16));
      vy = vmin_u8(vy, vget_low_u8(vy_max));
      vy = vmax_u8(vy, vget_low_u8(vy_min));

      vst1_lane_u8(y, vy, 0); y += 1;
    }
  }
}
