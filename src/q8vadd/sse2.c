/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <qnnpack/common.h>
#include <qnnpack/scalar-utils.h>
#include <qnnpack/q8vadd.h>

/**
 * SSE2 优化的 8 位整数向量加法微内核。
 * 该函数执行两个 uint8_t 向量 a 和 b 的逐元素加法，并应用量化参数进行缩放、偏移和裁剪。
 * 支持批量向量化处理（每次 8 个元素）和剩余元素处理，以及标量回退路径。
 *
 * @param n 向量元素数量
 * @param a 输入向量 A (const uint8_t*)
 * @param b 输入向量 B (const uint8_t*)
 * @param y 输出向量 (uint8_t*)
 * @param quantization_params 量化参数联合体，包含 SSE2 特定乘子、零点、掩码等预计算值
 */
void q8vadd_ukernel__sse2(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union qnnp_add_quantization_params quantization_params[RESTRICT_STATIC 1])
{
  if QNNP_LIKELY(n >= 8) {
    const __m128i vzero_point_product = _mm_load_si128((const __m128i*) &quantization_params->sse2.zero_point_product);
    const __m128i va_multiplier_lo = _mm_load_si128((const __m128i*) &quantization_params->sse2.a_multiplier_lo);
    const __m128i va_multiplier_hi = _mm_load_si128((const __m128i*) &quantization_params->sse2.a_multiplier_hi);
    const __m128i vb_multiplier_lo = _mm_load_si128((const __m128i*) &quantization_params->sse2.b_multiplier_lo);
    const __m128i vb_multiplier_hi = _mm_load_si128((const __m128i*) &quantization_params->sse2.b_multiplier_hi);
    const __m128i vremainder_mask = _mm_load_si128((const __m128i*) quantization_params->sse2.remainder_mask);
    const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) quantization_params->sse2.remainder_threshold);
    const __m128i vshift = _mm_cvtsi32_si128((int) quantization_params->sse2.shift);

    const __m128i vzero = _mm_setzero_si128();

    // 主向量化循环：每次处理 8 个 uint8 元素
    do {
      // 加载 8 字节输入数据（低 64 位）
      const __m128i va = _mm_loadl_epi64((const __m128i*) a);
      a += 8;
      const __m128i vb = _mm_loadl_epi64((const __m128i*) b);
      b += 8;

      // 将 uint8 unpack 到 int16（零扩展）
      const __m128i vxa = _mm_unpacklo_epi8(va, vzero);
      const __m128i vxb = _mm_unpacklo_epi8(vb, vzero);

      /* Multiply by factors */
      const __m128i va_product_lo = _mm_mullo_epi16(vxa, va_multiplier_lo);
      const __m128i va_product_hi =
        _mm_add_epi16(_mm_mulhi_epu16(vxa, va_multiplier_lo), _mm_mullo_epi16(vxa, va_multiplier_hi));

      const __m128i vb_product_lo = _mm_mullo_epi16(vxb, vb_multiplier_lo);
      const __m128i vb_product_hi =
        _mm_add_epi16(_mm_mulhi_epu16(vxb, vb_multiplier_lo), _mm_mullo_epi16(vxb, vb_multiplier_hi));

      /* 累加乘积：unpack 到 32 位并加入零点乘积 */
      __m128i vacc_lo = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(va_product_lo, va_product_hi));
      __m128i vacc_hi = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(va_product_lo, va_product_hi));

      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vb_product_lo, vb_product_hi));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vb_product_lo, vb_product_hi));

      /* 右移定点数并舍入：计算余数并调整 */
      const __m128i vrem_lo =
        _mm_add_epi32(_mm_and_si128(vacc_lo, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo));
      const __m128i vrem_hi =
        _mm_add_epi32(_mm_and_si128(vacc_hi, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi));

      vacc_lo = _mm_sub_epi32(_mm_sra_epi32(vacc_lo, vshift), _mm_cmpgt_epi32(vrem_lo, vremainder_threshold));
      vacc_hi = _mm_sub_epi32(_mm_sra_epi32(vacc_hi, vshift), _mm_cmpgt_epi32(vrem_hi, vremainder_threshold));

      /* 打包、饱和加输出零点、裁剪到 uint8 范围 */
      const __m128i vy_zero_point = _mm_load_si128((const __m128i*) quantization_params->sse2.y_zero_point);
      const __m128i vacc = _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), vy_zero_point);
      __m128i vy = _mm_packus_epi16(vacc, vacc);
      vy = _mm_max_epu8(vy, _mm_load_si128((const __m128i*) quantization_params->sse2.y_min));
      vy = _mm_min_epu8(vy, _mm_load_si128((const __m128i*) quantization_params->sse2.y_max));

      // 存储低 8 字节结果
      _mm_storel_epi64((__m128i*) y, vy);
      y += 8;

      n -= 8;
    } while (n >= 8);

    // 处理剩余少于 8 个元素：使用移位掩码加载
    if (n != 0) {
      const size_t n_decrement = 8 - n;
      const __m128i vload_shift = _mm_cvtsi32_si128(8 * (int32_t) n_decrement);

      // 后移指针加载并掩码高位
      const __m128i va = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) (a - n_decrement)), vload_shift);
      const __m128i vb = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) (b - n_decrement)), vload_shift);

      // 重复相同的计算流程（unpack、乘法、累加、移位、打包）
      const __m128i vxa = _mm_unpacklo_epi8(va, vzero);
      const __m128i vxb = _mm_unpacklo_epi8(vb, vzero);

      /* Multiply by factors */
      const __m128i va_product_lo = _mm_mullo_epi16(vxa, va_multiplier_lo);
      const __m128i va_product_hi =
        _mm_add_epi16(_mm_mulhi_epu16(vxa, va_multiplier_lo), _mm_mullo_epi16(vxa, va_multiplier_hi));

      const __m128i vb_product_lo = _mm_mullo_epi16(vxb, vb_multiplier_lo);
      const __m128i vb_product_hi =
        _mm_add_epi16(_mm_mulhi_epu16(vxb, vb_multiplier_lo), _mm_mullo_epi16(vxb, vb_multiplier_hi));

      /* Accumulate products */
      __m128i vacc_lo = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(va_product_lo, va_product_hi));
      __m128i vacc_hi = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(va_product_lo, va_product_hi));

      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vb_product_lo, vb_product_hi));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vb_product_lo, vb_product_hi));

      /* Shift right and round */
      const __m128i vrem_lo =
        _mm_add_epi32(_mm_and_si128(vacc_lo, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo));
      const __m128i vrem_hi =
        _mm_add_epi32(_mm_and_si128(vacc_hi, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi));

      vacc_lo = _mm_sub_epi32(_mm_sra_epi32(vacc_lo, vshift), _mm_cmpgt_epi32(vrem_lo, vremainder_threshold));
      vacc_hi = _mm_sub_epi32(_mm_sra_epi32(vacc_hi, vshift), _mm_cmpgt_epi32(vrem_hi, vremainder_threshold));

      /* Pack, saturate, and add output zero point */
      const __m128i vy_zero_point = _mm_load_si128((const __m128i*) quantization_params->sse2.y_zero_point);
      const __m128i vacc = _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), vy_zero_point);
      __m128i vy = _mm_packus_epi16(vacc, vacc);
      vy = _mm_max_epu8(vy, _mm_load_si128((const __m128i*) quantization_params->sse2.y_min));
      vy = _mm_min_epu8(vy, _mm_load_si128((const __m128i*) quantization_params->sse2.y_max));

      // 按剩余元素数量分批存储（4/2/1）
      if (n & 4) {
        *((uint32_t*) y) = (uint32_t) _mm_cvtsi128_si32(vy);
        vy = _mm_shuffle_epi32(vy, _MM_SHUFFLE(3, 2, 1, 1));
        y += 4;
      }
      if (n & 2) {
        *((uint16_t*) y) = (uint16_t) _mm_extract_epi16(vy, 0);
        vy = _mm_srli_epi32(vy, 16);
        y += 2;
      }
      if (n & 1) {
        *((uint8_t*) y) = (uint8_t) _mm_cvtsi128_si32(vy);
      }
    }
  } else {
    // 标量回退路径：n < 8 时逐元素计算，避免向量化开销
    const int32_t vzero_point_product = quantization_params->sse2.zero_point_product[0];
    const uint32_t va_multiplier = quantization_params->sse2.a_multiplier;
    const uint32_t vb_multiplier = quantization_params->sse2.b_multiplier;
    const int32_t vremainder_mask = quantization_params->sse2.remainder_mask[0];
    const int32_t vremainder_threshold = quantization_params->sse2.remainder_threshold[0];
    const uint32_t vshift = quantization_params->sse2.shift;
    const int32_t vy_zero_point = (int32_t) quantization_params->sse2.y_zero_point[0];
    const int32_t vy_max = (int32_t) (uint32_t) quantization_params->sse2.y_max[0];
    const int32_t vy_min = (int32_t) (uint32_t) quantization_params->sse2.y_min[0];

    // 逐元素标量循环
    while (n-- != 0) {
      const uint32_t vxa = (uint32_t) *a++;
      const uint32_t vxb = (uint32_t) *b++;

      /* 乘法累加 */
      int32_t vacc = vzero_point_product + (int32_t) (vxa * va_multiplier) + (int32_t) (vxb * vb_multiplier);

      /* 右移并舍入 */
      const int32_t vrem = (vacc & vremainder_mask) - (int32_t) (vacc < 0);
      vacc = asr_s32(vacc, vshift) + (int32_t) (vrem > vremainder_threshold);

      /* 加输出零点并裁剪 */
      int32_t vy = vacc + vy_zero_point;
      vy = vy >= vy_min ? vy : vy_min;
      vy = vy <= vy_max ? vy : vy_max;

      *y++ = (uint8_t) vy;
    }
  }
}
