/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * 本文件实现了基于 ARM NEON 指令集的 8-bit 无符号整数向量逐元素加法微内核。
 *
 * 量化加法的数学原理：
 *   实际值 a_float = (a - a_zero_point) * a_scale
 *   实际值 b_float = (b - b_zero_point) * b_scale
 *   和 = a_float + b_float
 *   输出量化值 y = clip(round(和 / y_scale) + y_zero_point, y_min, y_max)
 *
 * 为避免浮点运算，使用定点数乘法实现：
 *   acc = (a - a_zero) * a_multiplier + (b - b_zero) * b_multiplier
 *   y   = clip(rounding_shift_right(acc, shift) + y_zero_point, y_min, y_max)
 *
 * 与 SSE2 版本的主要区别：
 *   1. NEON 有直接的减法扩展指令 vsubl_u8（减去零点并扩展到 16-bit）
 *   2. NEON 有硬件四舍五入移位指令 vrshlq_s32（一条指令完成）
 *   3. ARM64 有更多优化指令（如 vmovl_high_s16），可以更高效地处理数据
 */

#include <arm_neon.h>

#include <qnnpack/common.h>
#include <qnnpack/q8vadd.h>


/*
 * q8vadd_ukernel__neon - 使用 ARM NEON 指令集的 uint8 向量加法微内核
 *
 * 参数：
 *   n                    - 待处理的元素个数
 *   a                    - 第一个输入向量的指针（uint8 类型）
 *   b                    - 第二个输入向量的指针（uint8 类型）
 *   y                    - 输出向量的指针（uint8 类型）
 *   quantization_params  - 量化参数结构体，包含零点、乘法因子、移位量等
 *
 * 处理策略（根据 CPU 架构和数据量自适应）：
 *   - ARM64 且 n >= 32：每次处理 32 个元素（最激进的展开）
 *   - ARM32 且 n >= 16：每次处理 16 个元素（中等展开）
 *   - n >= 8：每次处理 8 个元素（基础 SIMD）
 *   - n < 8：标量路径（逐元素处理）
 */
void q8vadd_ukernel__neon(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union qnnp_add_quantization_params quantization_params[restrict static 1])
{
  /* -----------------------------------------------------------------------
   * 加载量化参数到 NEON 寄存器（循环外一次性加载，避免重复）
   * ----------------------------------------------------------------------- */

  /* va_zero_point：输入 a 的零点（uint8x8，广播到 8 个通道） */
  const uint8x8_t va_zero_point = vld1_dup_u8(&quantization_params->neon.a_zero_point);

  /* vb_zero_point：输入 b 的零点（uint8x8，广播到 8 个通道） */
  const uint8x8_t vb_zero_point = vld1_dup_u8(&quantization_params->neon.b_zero_point);

  /* vy_zero_point：输出的零点（int16x8，广播到 8 个通道） */
  const int16x8_t vy_zero_point = vld1q_dup_s16(&quantization_params->neon.y_zero_point);

  /* va_multiplier：输入 a 的定点乘法因子（int32x4，广播到 4 个通道） */
  const int32x4_t va_multiplier = vld1q_dup_s32(&quantization_params->neon.a_multiplier);

  /* vb_multiplier：输入 b 的定点乘法因子（int32x4，广播到 4 个通道） */
  const int32x4_t vb_multiplier = vld1q_dup_s32(&quantization_params->neon.b_multiplier);

  /* vright_shift：右移位数（负数表示右移，int32x4） */
  const int32x4_t vright_shift = vld1q_dup_s32(&quantization_params->neon.right_shift);

  /*
   * vzero_shift_mask：零移位掩码（当 shift=0 时为全 1，否则为全 0）
   * 用于条件性地跳过符号位调整（当不需要移位时，避免不必要的计算）
   */
  const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));

  /* vy_max / vy_min：输出值的截断范围（uint8x16，广播到 16 个通道） */
  const uint8x16_t vy_max = vld1q_dup_u8(&quantization_params->neon.y_max);
  const uint8x16_t vy_min = vld1q_dup_u8(&quantization_params->neon.y_min);

  /* -----------------------------------------------------------------------
   * 快速路径：n >= 8，使用 NEON SIMD 指令批量处理
   * ----------------------------------------------------------------------- */
  if QNNP_LIKELY(n >= 8) {

#ifdef __aarch64__
    /* ===================================================================
     * ARM64 优化路径：每次处理 32 个元素（4 组 8 元素）
     * 利用 ARM64 的额外指令（如 vmovl_high_s16）和更多寄存器
     * =================================================================== */
    for (; n >= 32; n -= 32) {
      /* 加载 32 个 uint8 元素（分为 2 个 128-bit 寄存器） */
      const uint8x16_t va01 = vld1q_u8(a); a += 16;  // 加载 a[0:15]
      const uint8x16_t vb01 = vld1q_u8(b); b += 16;  // 加载 b[0:15]
      const uint8x16_t va23 = vld1q_u8(a); a += 16;  // 加载 a[16:31]
      const uint8x16_t vb23 = vld1q_u8(b); b += 16;  // 加载 b[16:31]

      /* -------------------------------------------------------------------
       * 步骤 1：减去零点并扩展到 int16
       * vsubl_u8：uint8 减法并扩展到 uint16（一条指令完成两个操作）
       * vreinterpretq_s16_u16：将 uint16 重新解释为 int16（无额外开销）
       * ------------------------------------------------------------------- */
      const int16x8_t vxa0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
      const int16x8_t vxb0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
      const int16x8_t vxa1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
      const int16x8_t vxb1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));
      const int16x8_t vxa2 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va23), va_zero_point));
      const int16x8_t vxb2 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb23), vb_zero_point));
      const int16x8_t vxa3 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va23), va_zero_point));
      const int16x8_t vxb3 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb23), vb_zero_point));

      /* -------------------------------------------------------------------
       * 步骤 2：定点乘法并累加
       * vmovl_s16：将 int16 扩展到 int32（低 4 个元素）
       * vmovl_high_s16：将 int16 扩展到 int32（高 4 个元素，ARM64 专用）
       * vmulq_s32：int32 乘法（4 个元素并行）
       * vmlaq_s32：乘加指令（acc = acc + a * b，融合乘法和加法）
       * ------------------------------------------------------------------- */

      /* 计算 a 的贡献：vxa * va_multiplier */
      int32x4_t vacc0_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa0)), va_multiplier);
      int32x4_t vacc1_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa1)), va_multiplier);
      int32x4_t vacc2_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa2)), va_multiplier);
      int32x4_t vacc3_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa3)), va_multiplier);
      int32x4_t vacc0_hi = vmulq_s32(vmovl_high_s16(vxa0), va_multiplier);
      int32x4_t vacc1_hi = vmulq_s32(vmovl_high_s16(vxa1), va_multiplier);
      int32x4_t vacc2_hi = vmulq_s32(vmovl_high_s16(vxa2), va_multiplier);
      int32x4_t vacc3_hi = vmulq_s32(vmovl_high_s16(vxa3), va_multiplier);

      /* 累加 b 的贡献：vacc += vxb * vb_multiplier */
      vacc0_lo = vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vxb0)), vb_multiplier);
      vacc1_lo = vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vxb1)), vb_multiplier);
      vacc2_lo = vmlaq_s32(vacc2_lo, vmovl_s16(vget_low_s16(vxb2)), vb_multiplier);
      vacc3_lo = vmlaq_s32(vacc3_lo, vmovl_s16(vget_low_s16(vxb3)), vb_multiplier);
      vacc0_hi = vmlaq_s32(vacc0_hi, vmovl_high_s16(vxb0), vb_multiplier);
      vacc1_hi = vmlaq_s32(vacc1_hi, vmovl_high_s16(vxb1), vb_multiplier);
      vacc2_hi = vmlaq_s32(vacc2_hi, vmovl_high_s16(vxb2), vb_multiplier);
      vacc3_hi = vmlaq_s32(vacc3_hi, vmovl_high_s16(vxb3), vb_multiplier);

      /* -------------------------------------------------------------------
       * 步骤 3：四舍五入右移（两阶段实现）
       *
       * 阶段 1：符号位调整（仅当 shift != 0 时）
       *   vsraq_n_s32(acc, bic(acc, mask), 31)：
       *   - vbicq_s32(acc, mask)：当 shift=0 时清零，否则保持 acc
       *   - vsraq_n_s32(a, b, 31)：a += (b >> 31)，即加上符号位
       *   - 目的：为负数的四舍五入做准备（负数需要特殊处理）
       *
       * 阶段 2：四舍五入右移
       *   vrshlq_s32(acc, vright_shift)：
       *   - NEON 硬件指令，一条指令完成四舍五入右移
       *   - vright_shift 为负数时表示右移（如 -10 表示右移 10 位）
       *   - 自动处理四舍五入（余数 >= 0.5 时进位）
       * ------------------------------------------------------------------- */
      vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
      vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
      vacc2_lo = vsraq_n_s32(vacc2_lo, vbicq_s32(vacc2_lo, vzero_shift_mask), 31);
      vacc3_lo = vsraq_n_s32(vacc3_lo, vbicq_s32(vacc3_lo, vzero_shift_mask), 31);
      vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
      vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);
      vacc2_hi = vsraq_n_s32(vacc2_hi, vbicq_s32(vacc2_hi, vzero_shift_mask), 31);
      vacc3_hi = vsraq_n_s32(vacc3_hi, vbicq_s32(vacc3_hi, vzero_shift_mask), 31);

      /* 四舍五入右移（NEON 硬件指令，比 SSE2 手动实现快得多） */
      vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
      vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
      vacc2_lo = vrshlq_s32(vacc2_lo, vright_shift);
      vacc3_lo = vrshlq_s32(vacc3_lo, vright_shift);
      vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
      vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);
      vacc2_hi = vrshlq_s32(vacc2_hi, vright_shift);
      vacc3_hi = vrshlq_s32(vacc3_hi, vright_shift);

      /* -------------------------------------------------------------------
       * 步骤 4：打包、饱和、加零点
       * vqmovn_s32：int32 → int16 饱和打包（超出 [-32768, 32767] 会截断）
       * vqmovn_high_s32：将 int32x4 打包到 int16x8 的高 4 个元素（ARM64 专用）
       * vqaddq_s16：int16 饱和加法（加上输出零点）
       * ------------------------------------------------------------------- */
      const int16x8_t vacc0 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0_lo), vacc0_hi), vy_zero_point);
      const int16x8_t vacc1 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1_lo), vacc1_hi), vy_zero_point);
      const int16x8_t vacc2 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2_lo), vacc2_hi), vy_zero_point);
      const int16x8_t vacc3 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3_lo), vacc3_hi), vy_zero_point);

      /* -------------------------------------------------------------------
       * 步骤 5：int16 → uint8 无符号饱和打包
       * vqmovun_s16：int16 → uint8 无符号饱和（负数 → 0，>255 → 255）
       * vqmovun_high_s16：将 int16x8 打包到 uint8x16 的高 8 个元素（ARM64 专用）
       * ------------------------------------------------------------------- */
      uint8x16_t vy01 = vqmovun_high_s16(vqmovun_s16(vacc0), vacc1);
      uint8x16_t vy23 = vqmovun_high_s16(vqmovun_s16(vacc2), vacc3);

      /* 步骤 6：截断到 [y_min, y_max] 范围（激活函数截断） */
      vy01 = vmaxq_u8(vy01, vy_min);
      vy23 = vmaxq_u8(vy23, vy_min);
      vy01 = vminq_u8(vy01, vy_max);
      vy23 = vminq_u8(vy23, vy_max);

      /* 步骤 7：存储 32 个 uint8 结果 */
      vst1q_u8(y, vy01); y += 16;
      vst1q_u8(y, vy23); y += 16;
    }
#else
    /* ===================================================================
     * ARM32 优化路径：每次处理 16 个元素（2 组 8 元素）
     * ARM32 缺少 vmovl_high_s16 等指令，需要用 vget_high_s16 替代
     * =================================================================== */
    for (; n >= 16; n -= 16) {
      /* 加载 16 个 uint8 元素 */
      const uint8x16_t va01 = vld1q_u8(a); a += 16;
      const uint8x16_t vb01 = vld1q_u8(b); b += 16;

      /* 减去零点并扩展到 int16（与 ARM64 相同） */
      const int16x8_t vxa0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
      const int16x8_t vxb0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
      const int16x8_t vxa1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
      const int16x8_t vxb1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));

      /* 定点乘法并累加（ARM32 需要用 vget_high_s16 代替 vmovl_high_s16） */
      int32x4_t vacc0_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa0)), va_multiplier);
      int32x4_t vacc1_lo = vmulq_s32(vmovl_s16(vget_low_s16(vxa1)), va_multiplier);
      int32x4_t vacc0_hi = vmulq_s32(vmovl_s16(vget_high_s16(vxa0)), va_multiplier);
      int32x4_t vacc1_hi = vmulq_s32(vmovl_s16(vget_high_s16(vxa1)), va_multiplier);

      /* 预取下一批数据到缓存（提高内存访问效率） */
      __builtin_prefetch(a + 640);
      __builtin_prefetch(b + 640);

      /* 累加 b 的贡献 */
      vacc0_lo = vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vxb0)), vb_multiplier);
      vacc1_lo = vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vxb1)), vb_multiplier);
      vacc0_hi = vmlaq_s32(vacc0_hi, vmovl_s16(vget_high_s16(vxb0)), vb_multiplier);
      vacc1_hi = vmlaq_s32(vacc1_hi, vmovl_s16(vget_high_s16(vxb1)), vb_multiplier);

      /* 四舍五入右移（与 ARM64 相同） */
      vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
      vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
      vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
      vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);

      vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
      vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
      vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
      vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);

      /* 打包、饱和、加零点（ARM32 需要用 vcombine_s16 代替 vqmovn_high_s32） */
      const int16x8_t vacc0 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi)), vy_zero_point);
      const int16x8_t vacc1 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi)), vy_zero_point);

      /* int16 → uint8 饱和打包 */
      uint8x16_t vy01 = vcombine_u8(vqmovun_s16(vacc0), vqmovun_s16(vacc1));
      vy01 = vmaxq_u8(vy01, vy_min);
      vy01 = vminq_u8(vy01, vy_max);

      /* 存储 16 个 uint8 结果 */
      vst1q_u8(y, vy01); y += 16;
    }
#endif

    /* ===================================================================
     * 基础 SIMD 路径：每次处理 8 个元素（适用于 ARM32 和 ARM64）
     * =================================================================== */
    for (; n >= 8; n -= 8) {
      /* 加载 8 个 uint8 元素 */
      const uint8x8_t va = vld1_u8(a); a += 8;
      const uint8x8_t vb = vld1_u8(b); b += 8;

      /* 减去零点并扩展到 int16 */
      const int16x8_t vxa = vreinterpretq_s16_u16(vsubl_u8(va, va_zero_point));
      const int16x8_t vxb = vreinterpretq_s16_u16(vsubl_u8(vb, vb_zero_point));

      /* 定点乘法并累加 */
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

      /* 四舍五入右移 */
      vacc_lo = vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
      vacc_hi = vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

      vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
      vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

      /* 打包、饱和、加零点 */
#ifdef __aarch64__
      const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
#else
      const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), vy_zero_point);
#endif

      /* int16 → uint8 饱和打包 */
      uint8x8_t vy = vqmovun_s16(vacc);
      vy = vmax_u8(vy, vget_low_u8(vy_min));
      vy = vmin_u8(vy, vget_low_u8(vy_max));

      /* 存储 8 个 uint8 结果 */
      vst1_u8(y, vy); y += 8;
    }

    /* ===================================================================
     * 尾部处理：处理剩余的 1~7 个元素（仍使用 SIMD，但用移位对齐）
     * =================================================================== */
    if (n != 0) {
      /*
       * 计算需要回退的字节数，使得加载的 8 字节中包含剩余的 n 个有效元素。
       * 例如 n=3，则 n_increment=-5，从 (a - 5) 处加载 8 字节，
       * 然后左移 5*8=40 位，使有效数据对齐到高位，再右移回低位。
       */
      const size_t n_increment = n - 8;
      const int64x1_t vld_shift = vmov_n_s64(8 * n_increment);

      /* 加载并左移，将有效的 n 个元素移到低位 */
      const uint8x8_t va = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a + n_increment)), vld_shift));
      const uint8x8_t vb = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(b + n_increment)), vld_shift));

      /* 减去零点并扩展到 int16 */
      const int16x8_t vxa = vreinterpretq_s16_u16(vsubl_u8(va, va_zero_point));
      const int16x8_t vxb = vreinterpretq_s16_u16(vsubl_u8(vb, vb_zero_point));

      /* 定点乘法并累加（与基础 SIMD 路径相同） */
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

      /* 四舍五入右移 */
      vacc_lo = vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
      vacc_hi = vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

      vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
      vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

      /* 打包、饱和、加零点 */
#ifdef __aarch64__
      const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
#else
      const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), vy_zero_point);
#endif

      /* int16 → uint8 饱和打包 */
      uint8x8_t vy = vqmovun_s16(vacc);
      vy = vmax_u8(vy, vget_low_u8(vy_min));
      vy = vmin_u8(vy, vget_low_u8(vy_max));

      /*
       * 按位分段写出剩余的 n 个字节（n 在 1~7 之间）：
       * 利用 n 的二进制位（bit2=4字节，bit1=2字节，bit0=1字节）逐段写出。
       * vext_u8：向量元素提取/旋转指令，用于移动数据到低位
       */
      if (n & 4) {
        /* 写出低 32 位（4 个字节） */
        vst1_lane_u32(__builtin_assume_aligned(y, 1), vreinterpret_u32_u8(vy), 0); y += 4;
        /* 将剩余数据向左旋转 4 字节（将高 4 字节移到低位） */
        vy = vext_u8(vy, vy, 4);
      }
      if (n & 2) {
        /* 写出低 16 位（2 个字节） */
        vst1_lane_u16(__builtin_assume_aligned(y, 1), vreinterpret_u16_u8(vy), 0); y += 2;
        /* 将剩余数据向左旋转 2 字节 */
        vy = vext_u8(vy, vy, 2);
      }
      if (n & 1) {
        /* 写出最低 8 位（1 个字节） */
        vst1_lane_u8(y, vy, 0);
      }
    }
  } else {
    /* ===================================================================
     * 标量路径：n < 8，逐元素处理（避免 SIMD 开销）
     * =================================================================== */
    for (; n != 0; n--) {
      /* 加载单个元素并广播到 8 个通道（为了复用 NEON 指令） */
      const uint8x8_t va = vld1_dup_u8(a); a += 1;
      const uint8x8_t vb = vld1_dup_u8(b); b += 1;

      /* 减去零点并扩展到 int16（只使用低 4 个元素） */
      const int16x4_t vxa = vreinterpret_s16_u16(vget_low_u16(vsubl_u8(va, va_zero_point)));
      const int16x4_t vxb = vreinterpret_s16_u16(vget_low_u16(vsubl_u8(vb, vb_zero_point)));

      /* 定点乘法并累加（使用 64-bit 版本的指令，处理 2 个 int32） */
      int32x2_t vacc = vmul_s32(vget_low_s32(vmovl_s16(vxa)), vget_low_s32(va_multiplier));
      vacc = vmla_s32(vacc, vget_low_s32(vmovl_s16(vxb)), vget_low_s32(vb_multiplier));

      /* 四舍五入右移（64-bit 版本） */
      vacc = vsra_n_s32(vacc, vbic_s32(vacc, vget_low_s32(vzero_shift_mask)), 31);
      vacc = vrshl_s32(vacc, vget_low_s32(vright_shift));

      /* 打包、饱和、加零点 */
      const int16x4_t vacc16 = vqadd_s16(vqmovn_s32(vcombine_s32(vacc, vacc)), vget_low_s16(vy_zero_point));

      /* int16 → uint8 饱和打包 */
      uint8x8_t vy = vqmovun_s16(vcombine_s16(vacc16, vacc16));
      vy = vmin_u8(vy, vget_low_u8(vy_max));
      vy = vmax_u8(vy, vget_low_u8(vy_min));

      /* 写出单个字节 */
      vst1_lane_u8(y, vy, 0); y += 1;
    }
  }
}
