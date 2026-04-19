/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* 标准库：断言 */
#include <assert.h>

/* ARM NEON  intrinsics */
#include <arm_neon.h>

/* QNNPACK u8 clamp 接口定义 */
#include <qnnpack/u8clamp.h>


/**
 * NEON 优化的 uint8 Clamp 微内核。
 * 对输入向量 x 执行逐元素 clamp 操作：y[i] = clamp(x[i], output_min, output_max)
 * 支持批量处理 (64/8 元素循环 + 剩余 + 标量回退)，使用 vmin/vmax 饱和 clamp。
 *
 * @param n 向量长度 (元素数)
 * @param x 输入 uint8_t 向量 (const)
 * @param y 输出 uint8_t 向量
 * @param params Clamp 参数 (union: neon.output_min/max，预广播到向量)
 */
void u8clamp_ukernel__neon(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union qnnp_u8_clamping_params params[restrict static 1])
{
  assert(n != 0);  // 前置条件：n > 0

  /* 加载预计算的 clamp 上下限向量 (广播单值到 16 通道) */
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);  // 输出上限向量
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);  // 输出下限向量

  /* 主向量化路径：n >= 8 时，使用 NEON 处理 */
  if QNNP_LIKELY(n >= 8) {
    /* 大批量循环：每次 64 元素 (4x16-byte 向量)，最大吞吐 */
    for (; n >= 64; n -= 64) {
      /* 加载 4 个 16-byte 输入向量 */
      const uint8x16_t vx0 = vld1q_u8(x); x += 16;
      const uint8x16_t vx1 = vld1q_u8(x); x += 16;
      const uint8x16_t vx2 = vld1q_u8(x); x += 16;
      const uint8x16_t vx3 = vld1q_u8(x); x += 16;

      /* 并行 clamp：先 max(输入, min)，再 min(结果, max) */
      const uint8x16_t vy0 = vminq_u8(vmaxq_u8(vx0, voutput_min), voutput_max);
      const uint8x16_t vy1 = vminq_u8(vmaxq_u8(vx1, voutput_min), voutput_max);
      const uint8x16_t vy2 = vminq_u8(vmaxq_u8(vx2, voutput_min), voutput_max);
      const uint8x16_t vy3 = vminq_u8(vmaxq_u8(vx3, voutput_min), voutput_max);

      __builtin_prefetch(x + 640);  // 预取下一批输入，提高缓存命中

      /* 存储 4 个结果向量 */
      vst1q_u8(y, vy0); y += 16;
      vst1q_u8(y, vy1); y += 16;
      vst1q_u8(y, vy2); y += 16;
      vst1q_u8(y, vy3); y += 16;
    }
    /* 中批量循环：每次 8 元素 (8-byte 向量) */
    for (; n >= 8; n -= 8) {
      uint8x8_t vout = vld1_u8(x); x += 8;  // 加载 8-byte 输入
      vout = vmin_u8(vout, vget_low_u8(voutput_max));  // clamp 上限 (低 8 通道)
      vout = vmax_u8(vout, vget_low_u8(voutput_min));  // clamp 下限
      vst1_u8(y, vout); y += 8;  // 存储结果
    }
    /* 剩余 <8 元素：调整指针到有效数据，加载全 8-byte 并 clamp */
    if (n != 0) {
      const size_t n_increment = n - 8;  // 负偏移量
      x = (const uint8_t*) ((uintptr_t) x + n_increment);  // 后移指针到剩余数据起始
      y = (uint8_t*) ((uintptr_t) y + n_increment);

      uint8x8_t vout = vld1_u8(x);  // 加载包含剩余数据的 8-byte (高位垃圾忽略，因 clamp 后只存低 n 字节)
      vout = vmin_u8(vout, vget_low_u8(voutput_max));
      vout = vmax_u8(vout, vget_low_u8(voutput_min));
      vst1_u8(y, vout);  // 存储 (低 n 字节有效，高位覆盖垃圾)
    }
  } else {
    /* 标量回退：逐元素处理 (n < 8)，使用 dup 加载单元素到向量 lane 0 */
    do {
      uint8x8_t vout = vld1_dup_u8(x); x += 1;  // 广播单 uint8 到 8 通道
      vout = vmin_u8(vout, vget_low_u8(voutput_max));
      vout = vmax_u8(vout, vget_low_u8(voutput_min));
      vst1_lane_u8(y, vout, 0); y += 1;  // 只存 lane 0
    } while (--n != 0);
  }
}
