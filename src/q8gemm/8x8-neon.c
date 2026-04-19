/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/q8gemm.h>


 /*
  * QNNPACK库中的8位量化GEMM (General Matrix Multiply) 微内核实现，使用ARM NEON SIMD指令优化。
  * 该函数计算块矩阵乘法 C[mr x nr] += A[mr x k] * B[k x nr]，其中A、B、C均为uint8_t类型。
  * - mr: 输出块行数 (最大8)
  * - nr: 输出块列数 (最大8)
  * - k: 收缩维度 (内循环次数)
  * - a: 输入矩阵A (列优先? row-major, stride指定)
  * - a_stride: A行步长 (字节)
  * - w: 打包权重矩阵B (特殊布局，每nr列打包k通道)
  * - c: 输出矩阵C
  * - c_stride: C行步长
  * - quantization_params: 量化参数，包括kernel_zero_point, multiplier, right_shift等，用于后处理。
  * 算法特点: 展开8x8块，NEON 128位向量处理8个uint8->int16，vmlal_lane_s16进行点积，优化内存访问和SIMD利用。
  */
 
void q8gemm_ukernel_8x8__neon(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    const union qnnp_conv_quantization_params quantization_params[restrict static 1])
{
  /* 初始化累加器 (accumulator)，从权重打包偏置加载。
   * w的前16字节为vacc0x0123 (列0-3)，接下来16字节vacc0x4567 (列4-7)。
   * 其他行vacc1-7复制vacc0，因为初始为零偏置或预加载偏置。
   * int32x4_t用于累积乘积，避免溢出。
   */
  int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
  int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc1x4567 = vacc0x4567;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc2x4567 = vacc0x4567;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc3x4567 = vacc0x4567;
  int32x4_t vacc4x0123 = vacc0x0123;
  int32x4_t vacc4x4567 = vacc0x4567;
  int32x4_t vacc5x0123 = vacc0x0123;
  int32x4_t vacc5x4567 = vacc0x4567;
  int32x4_t vacc6x0123 = vacc0x0123;
  int32x4_t vacc6x4567 = vacc0x4567;
  int32x4_t vacc7x0123 = vacc0x0123;
  int32x4_t vacc7x4567 = vacc0x4567;

  /* 设置输入A的8行指针，根据mr调整小于8行的重复指针 (padding)，确保始终处理8行。
   * a_stride为字节步长，支持非连续内存。
   */
  const uint8_t* a0 = a;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
  if (mr < 4) {
    a3 = a2;
  }
  const uint8_t* a4 = (const uint8_t*) ((uintptr_t) a3 + a_stride);
  if (mr <= 4) {
    a4 = a3;
  }
  const uint8_t* a5 = (const uint8_t*) ((uintptr_t) a4 + a_stride);
  if (mr < 6) {
    a5 = a4;
  }
  const uint8_t* a6 = (const uint8_t*) ((uintptr_t) a5 + a_stride);
  if (mr <= 6) {
    a6 = a5;
  }
  const uint8_t* a7 = (const uint8_t*) ((uintptr_t) a6 + a_stride);
  if (mr != 8) {
    a7 = a6;
  }

  /* 加载内核零点 (kernel zero point)，用于B去量化: vb_centered = vb - zero_point。
   * 广播到uint8x8_t向量。
   */
  const uint8x8_t vb_zero_point = vld1_dup_u8((const uint8_t*) &quantization_params->neon.kernel_zero_point);
  /* 主循环: 处理完整8个k通道块。每次加载8行A的8字节 (uint8x8_t -> int16x8_t)，
   * 加载8列B的8通道 (vb_c0到c7)，使用vmlal_lane_s16更新vacc (8行x2向量: 0123/4567)。
   * vget_low/high_s16分离B向量，lane 0-3对应A low/high s16。
   * 展开内循环以最大化SIMD利用。
   */
  for (; k >= 8; k -= 8) {
    /* 加载8行A的当前8通道数据 (连续8字节)，扩展到int16x8_t用于乘法。
     * vmovl_u8: uint8->uint16, vreinterpretq_s16_u16: uint16->int16 (signed)。
     */
    const uint8x8_t va0 = vld1_u8(a0);
    const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0)); a0 += 8;
    const uint8x8_t va1 = vld1_u8(a1);
    const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1)); a1 += 8;
    const uint8x8_t va2 = vld1_u8(a2);
    const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2)); a2 += 8;
    const uint8x8_t va3 = vld1_u8(a3);
    const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3)); a3 += 8;
    const uint8x8_t va4 = vld1_u8(a4);
    const int16x8_t vxa4 = vreinterpretq_s16_u16(vmovl_u8(va4)); a4 += 8;
    const uint8x8_t va5 = vld1_u8(a5);
    const int16x8_t vxa5 = vreinterpretq_s16_u16(vmovl_u8(va5)); a5 += 8;
    const uint8x8_t va6 = vld1_u8(a6);
    const int16x8_t vxa6 = vreinterpretq_s16_u16(vmovl_u8(va6)); a6 += 8;
    const uint8x8_t va7 = vld1_u8(a7);
    const int16x8_t vxa7 = vreinterpretq_s16_u16(vmovl_u8(va7)); a7 += 8;

    /* 加载B打包权重第c0通道的8列数据 (vb01234567c0)，去零点化为int16x8_t。
     * w布局: 每nr*8字节一组? 实际为k-major, nr块内通道展开。
     */
    const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

    /* 使用vmlal_lane_s16 (multiply-accumulate lane) 更新所有vacc:
     * vacc[row]x0123/4567 += lowB/highB * lane(va[row])
     * lane 0: A low s16[0] * B low s16[0-3]
     * 类似c1 lane1, c2 lane2, c3 lane3 (low A)
     * c4-7使用high A s16[4-7]
     * 高度展开避免分支。
     */
    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
    vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
    vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
    vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
    vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
    vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa7), 0);
    vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa7), 0);

    const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
    vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
    vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
    vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
    vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
    vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa7), 1);
    vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa7), 1);

    const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
    vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
    vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
    vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa6), 2);
    vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa6), 2);
    vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa7), 2);
    vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa7), 2);

    const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
    vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
    vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
    vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
    vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
    vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa7), 3);
    vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa7), 3);

    const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
    vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
    vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
    vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
    vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
    vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa7), 0);
    vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa7), 0);

    const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
    vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
    vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
    vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa6), 1);
    vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa6), 1);
    vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa7), 1);
    vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa7), 1);

    const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
    vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
    vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
    vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
    vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
    vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa7), 2);
    vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa7), 2);

    const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
    vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
    vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
    vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa6), 3);
    vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa6), 3);
    vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa7), 3);
    vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa7), 3);
  }
  /* 剩余k <8通道处理: 使用位移mask (shl负偏移) 将剩余数据左移对齐到高位，低位零填充。
   * 类似主循环，但嵌套if处理k=1~7，避免分支。
   */
  if (k != 0) {
    const size_t a_predecrement = 8 - k;
    const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
    const uint8x8_t va0 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
    const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
    const uint8x8_t va1 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
    const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
    const uint8x8_t va2 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
    const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
    const uint8x8_t va3 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
    const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
    const uint8x8_t va4 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a4 - a_predecrement)), va_shift));
    const int16x8_t vxa4 = vreinterpretq_s16_u16(vmovl_u8(va4));
    const uint8x8_t va5 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a5 - a_predecrement)), va_shift));
    const int16x8_t vxa5 = vreinterpretq_s16_u16(vmovl_u8(va5));
    const uint8x8_t va6 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a6 - a_predecrement)), va_shift));
    const int16x8_t vxa6 = vreinterpretq_s16_u16(vmovl_u8(va6));
    const uint8x8_t va7 = vreinterpret_u8_u64(vshl_u64(vreinterpret_u64_u8(vld1_u8(a7 - a_predecrement)), va_shift));
    const int16x8_t vxa7 = vreinterpretq_s16_u16(vmovl_u8(va7));

    const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
    const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
    vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
    vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
    vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
    vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
    vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
    vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa7), 0);
    vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa7), 0);

    if (k >= 2) {
      const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa7), 1);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa7), 1);

      if (k >= 3) {
        const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
        const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
        vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa6), 2);
        vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa6), 2);
        vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa7), 2);
        vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa7), 2);

        if (k >= 4) {
          const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
          const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
          vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
          vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
          vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
          vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
          vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
          vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
          vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
          vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
          vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
          vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
          vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
          vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
          vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa7), 3);
          vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa7), 3);

          if (k >= 5) {
            const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
            const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
            vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
            vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
            vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa7), 0);
            vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa7), 0);

            if (k >= 6) {
              const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
              const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
              vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
              vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
              vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
              vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
              vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
              vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
              vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
              vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
              vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
              vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
              vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa6), 1);
              vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa6), 1);
              vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa7), 1);
              vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa7), 1);

              if (k >= 7) {
                const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
                const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
                vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
                vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
                vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
                vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
                vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
                vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
                vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
                vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
                vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
                vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
                vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
                vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa7), 2);
                vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa7), 2);
              }
            }
          }
        }
      }
    }
  }

  /* 后处理: 量化步骤。
   * 1. 乘以乘法因子 (fixed-point multiplier)，使用vqrdmulhq_s32 (rounding)。
   * 2. Rounding shift: 如果right_shift!=0，进行31位舍入 (vsraq_n_s32 with mask)。
   * 3. 右移 (vrshlq_s32)，量化到int32。
   */
  const int32x4_t vmultiplier = vld1q_dup_s32(&quantization_params->neon.multiplier);
  vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
  vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
  vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);
  vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier);
  vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier);
  vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier);
  vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier);
  vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier);
  vacc4x0123 = vqrdmulhq_s32(vacc4x0123, vmultiplier);
  vacc4x4567 = vqrdmulhq_s32(vacc4x4567, vmultiplier);
  vacc5x0123 = vqrdmulhq_s32(vacc5x0123, vmultiplier);
  vacc5x4567 = vqrdmulhq_s32(vacc5x4567, vmultiplier);
  vacc6x0123 = vqrdmulhq_s32(vacc6x0123, vmultiplier);
  vacc6x4567 = vqrdmulhq_s32(vacc6x4567, vmultiplier);
  vacc7x0123 = vqrdmulhq_s32(vacc7x0123, vmultiplier);
  vacc7x4567 = vqrdmulhq_s32(vacc7x4567, vmultiplier);

  const int32x4_t vright_shift = vld1q_dup_s32(&quantization_params->neon.right_shift);
  const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
  vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
  vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
  vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), 31);
  vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), 31);
  vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), 31);
  vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), 31);
  vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), 31);
  vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask), 31);
  vacc4x0123 = vsraq_n_s32(vacc4x0123, vbicq_s32(vacc4x0123, vzero_shift_mask), 31);
  vacc4x4567 = vsraq_n_s32(vacc4x4567, vbicq_s32(vacc4x4567, vzero_shift_mask), 31);
  vacc5x0123 = vsraq_n_s32(vacc5x0123, vbicq_s32(vacc5x0123, vzero_shift_mask), 31);
  vacc5x4567 = vsraq_n_s32(vacc5x4567, vbicq_s32(vacc5x4567, vzero_shift_mask), 31);
  vacc6x0123 = vsraq_n_s32(vacc6x0123, vbicq_s32(vacc6x0123, vzero_shift_mask), 31);
  vacc6x4567 = vsraq_n_s32(vacc6x4567, vbicq_s32(vacc6x4567, vzero_shift_mask), 31);
  vacc7x0123 = vsraq_n_s32(vacc7x0123, vbicq_s32(vacc7x0123, vzero_shift_mask), 31);
  vacc7x4567 = vsraq_n_s32(vacc7x4567, vbicq_s32(vacc7x4567, vzero_shift_mask), 31);

  vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
  vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
  vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
  vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);
  vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);
  vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift);
  vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);
  vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift);
  vacc4x0123 = vrshlq_s32(vacc4x0123, vright_shift);
  vacc4x4567 = vrshlq_s32(vacc4x4567, vright_shift);
  vacc5x0123 = vrshlq_s32(vacc5x0123, vright_shift);
  vacc5x4567 = vrshlq_s32(vacc5x4567, vright_shift);
  vacc6x0123 = vrshlq_s32(vacc6x0123, vright_shift);
  vacc6x4567 = vrshlq_s32(vacc6x4567, vright_shift);
  vacc7x0123 = vrshlq_s32(vacc7x0123, vright_shift);
  vacc7x4567 = vrshlq_s32(vacc7x4567, vright_shift);

  /* 输出转换:
   * int32->int16 (vqmovn)，加output zero point (vqaddq_s16)。
   * AArch64优化使用vqmovn_high_s32，非64位用vcombine_s16。
   * int16->uint8 (vqmovun_s16，饱和无符号)，打包成uint8x16_t (两行8列)。
   * clamp到output_min/max。
   */
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
#ifdef __aarch64__
  const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
  const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
  const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
  const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
  const int16x8_t vacc4x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc4x4567), voutput_zero_point);
  const int16x8_t vacc5x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc5x0123), vacc5x4567), voutput_zero_point);
  const int16x8_t vacc6x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc6x0123), vacc6x4567), voutput_zero_point);
  const int16x8_t vacc7x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc7x0123), vacc7x4567), voutput_zero_point);

  uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
  uint8x16_t vout2x01234567_3x01234567 = vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);
  uint8x16_t vout4x01234567_5x01234567 = vqmovun_high_s16(vqmovun_s16(vacc4x01234567), vacc5x01234567);
  uint8x16_t vout6x01234567_7x01234567 = vqmovun_high_s16(vqmovun_s16(vacc6x01234567), vacc7x01234567);
#else
  const int16x8_t vacc0x01234567 =
    vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
  const int16x8_t vacc1x01234567 =
    vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
  const int16x8_t vacc2x01234567 =
    vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
  const int16x8_t vacc3x01234567 =
    vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
  const int16x8_t vacc4x01234567 =
    vqaddq_s16(vcombine_s16(vqmovn_s32(vacc4x0123), vqmovn_s32(vacc4x4567)), voutput_zero_point);
  const int16x8_t vacc5x01234567 =
    vqaddq_s16(vcombine_s16(vqmovn_s32(vacc5x0123), vqmovn_s32(vacc5x4567)), voutput_zero_point);
  const int16x8_t vacc6x01234567 =
    vqaddq_s16(vcombine_s16(vqmovn_s32(vacc6x0123), vqmovn_s32(vacc6x4567)), voutput_zero_point);
  const int16x8_t vacc7x01234567 =
    vqaddq_s16(vcombine_s16(vqmovn_s32(vacc7x0123), vqmovn_s32(vacc7x4567)), voutput_zero_point);

  uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc1x01234567));
  uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc3x01234567));
  uint8x16_t vout4x01234567_5x01234567 = vcombine_u8(vqmovun_s16(vacc4x01234567), vqmovun_s16(vacc5x01234567));
  uint8x16_t vout6x01234567_7x01234567 = vcombine_u8(vqmovun_s16(vacc6x01234567), vqmovun_s16(vacc7x01234567));
#endif
  const uint8x16_t voutput_min = vld1q_dup_u8(&quantization_params->neon.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&quantization_params->neon.output_max);

  vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
  vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
  vout4x01234567_5x01234567 = vmaxq_u8(vout4x01234567_5x01234567, voutput_min);
  vout6x01234567_7x01234567 = vmaxq_u8(vout6x01234567_7x01234567, voutput_min);
  vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
  vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
  vout4x01234567_5x01234567 = vminq_u8(vout4x01234567_5x01234567, voutput_max);
  vout6x01234567_7x01234567 = vminq_u8(vout6x01234567_7x01234567, voutput_max);

  /* 存储输出C: 设置8行C指针，类似A调整padding。
   * 如果nr==8，直接vst1_u8。
   * 否则，按4/2/1字节剥离 (lane store)，使用vextq_u8移位剩余数据。
   */
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + c_stride);
  if (mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + c_stride);
  if (mr < 4) {
    c3 = c2;
  }
  uint8_t* c4 = (uint8_t*) ((uintptr_t) c3 + c_stride);
  if (mr <= 4) {
    c4 = c3;
  }
  uint8_t* c5 = (uint8_t*) ((uintptr_t) c4 + c_stride);
  if (mr < 6) {
    c5 = c4;
  }
  uint8_t* c6 = (uint8_t*) ((uintptr_t) c5 + c_stride);
  if (mr <= 6) {
    c6 = c5;
  }
  uint8_t* c7 = (uint8_t*) ((uintptr_t) c6 + c_stride);
  if (mr != 8) {
    c7 = c6;
  }
  if (nr == 8) {
    vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
    vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
    vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
    vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
    vst1_u8(c4, vget_low_u8(vout4x01234567_5x01234567));
    vst1_u8(c5, vget_high_u8(vout4x01234567_5x01234567));
    vst1_u8(c6, vget_low_u8(vout6x01234567_7x01234567));
    vst1_u8(c7, vget_high_u8(vout6x01234567_7x01234567));
  } else {
    if (nr >= 4) {
      vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 0); c2 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 2); c3 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c4, 1), vreinterpretq_u32_u8(vout4x01234567_5x01234567), 0); c4 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c5, 1), vreinterpretq_u32_u8(vout4x01234567_5x01234567), 2); c5 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c6, 1), vreinterpretq_u32_u8(vout6x01234567_7x01234567), 0); c6 += 4;
      vst1q_lane_u32(__builtin_assume_aligned(c7, 1), vreinterpretq_u32_u8(vout6x01234567_7x01234567), 2); c7 += 4;
      vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      vout4x01234567_5x01234567 = vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 4);
      vout6x01234567_7x01234567 = vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 4);
      nr -= 4;
    }
    if (nr >= 2) {
      vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 0); c2 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 4); c3 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c4, 1), vreinterpretq_u16_u8(vout4x01234567_5x01234567), 0); c4 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c5, 1), vreinterpretq_u16_u8(vout4x01234567_5x01234567), 4); c5 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c6, 1), vreinterpretq_u16_u8(vout6x01234567_7x01234567), 0); c6 += 2;
      vst1q_lane_u16(__builtin_assume_aligned(c7, 1), vreinterpretq_u16_u8(vout6x01234567_7x01234567), 4); c7 += 2;
      vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      vout4x01234567_5x01234567 = vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 2);
      vout6x01234567_7x01234567 = vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 2);
      nr -= 2;
    }
    if (nr != 0) {
      vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
      vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
      vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
      vst1q_lane_u8(c4, vout4x01234567_5x01234567, 0);
      vst1q_lane_u8(c5, vout4x01234567_5x01234567, 8);
      vst1q_lane_u8(c6, vout6x01234567_7x01234567, 0);
      vst1q_lane_u8(c7, vout6x01234567_7x01234567, 8);
    }
  }
}
