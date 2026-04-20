/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * 本文件实现了基于 SSE2 指令集的 8-bit 无符号整数向量逐元素加法微内核。
 *
 * 量化加法的数学原理：
 *   输出量化值 y = clip(round((a * a_scale + b * b_scale) / y_scale) + y_zero_point)
 *
 * 为避免浮点运算，使用定点数乘法实现：
 *   acc = zero_point_product + a * a_multiplier + b * b_multiplier
 *   y   = clip(rounding_shift_right(acc, shift) + y_zero_point, y_min, y_max)
 *
 * 其中 zero_point_product 预先吸收了输入零点的贡献，
 * a_multiplier / b_multiplier 是对应的定点缩放因子，
 * shift 是右移位数（用于还原定点数的小数部分）。
 */

#include <immintrin.h>

#include <qnnpack/common.h>
#include <qnnpack/scalar-utils.h>
#include <qnnpack/q8vadd.h>


/*
 * q8vadd_ukernel__sse2 - 使用 SSE2 指令集的 uint8 向量加法微内核
 *
 * 参数：
 *   n                    - 待处理的元素个数
 *   a                    - 第一个输入向量的指针（uint8 类型）
 *   b                    - 第二个输入向量的指针（uint8 类型）
 *   y                    - 输出向量的指针（uint8 类型）
 *   quantization_params  - 量化参数结构体，包含乘法因子、零点、移位量等
 *
 * 处理策略：
 *   - 当 n >= 8 时，使用 SSE2 SIMD 指令每次处理 8 个元素（主循环）
 *   - 主循环结束后若仍有剩余元素（1~7 个），使用尾部处理逻辑
 *   - 当 n < 8 时，直接走标量（逐元素）路径
 */
void q8vadd_ukernel__sse2(
    size_t n,                           /* 待处理的元素个数 */
    const uint8_t* a,                   /* 第一个输入向量指针 */
    const uint8_t* b,                   /* 第二个输入向量指针 */
    uint8_t* y,                         /* 输出向量指针 */
    const union qnnp_add_quantization_params quantization_params[RESTRICT_STATIC 1]) /* 量化参数 */
{
  /* -----------------------------------------------------------------------
   * 快速路径：n >= 8，使用 SSE2 SIMD 指令批量处理
   * ----------------------------------------------------------------------- */
  if QNNP_LIKELY(n >= 8) {
    /* 将量化参数加载到 SSE2 寄存器，避免循环内重复加载 */

    /* zero_point_product：预计算的零点乘积偏置（int32x4，广播到 4 个通道） */
    const __m128i vzero_point_product = _mm_load_si128((const __m128i*) &quantization_params->sse2.zero_point_product);

    /* a_multiplier_lo / a_multiplier_hi：输入 a 的定点乘法因子的低 16 位和高 16 位 */
    const __m128i va_multiplier_lo = _mm_load_si128((const __m128i*) &quantization_params->sse2.a_multiplier_lo);
    const __m128i va_multiplier_hi = _mm_load_si128((const __m128i*) &quantization_params->sse2.a_multiplier_hi);

    /* b_multiplier_lo / b_multiplier_hi：输入 b 的定点乘法因子的低 16 位和高 16 位 */
    const __m128i vb_multiplier_lo = _mm_load_si128((const __m128i*) &quantization_params->sse2.b_multiplier_lo);
    const __m128i vb_multiplier_hi = _mm_load_si128((const __m128i*) &quantization_params->sse2.b_multiplier_hi);

    /* vremainder_mask：用于提取余数低位的掩码，等于 (1 << shift) - 1 */
    const __m128i vremainder_mask = _mm_load_si128((const __m128i*) quantization_params->sse2.remainder_mask);

    /* vremainder_threshold：四舍五入阈值，等于 (1 << shift) / 2 - 1 */
    const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) quantization_params->sse2.remainder_threshold);

    /* vshift：右移位数（标量，加载到 XMM 寄存器低 32 位供 _mm_sra_epi32 使用） */
    const __m128i vshift = _mm_cvtsi32_si128((int) quantization_params->sse2.shift);

    /* vzero：全零寄存器，用于 8-bit → 16-bit 零扩展解包 */
    const __m128i vzero = _mm_setzero_si128();

    /* 主循环：每次处理 8 个 uint8 元素 */
    do {
      /* 从输入向量各加载 8 字节（64 位）到 XMM 寄存器低 64 位 */
      const __m128i va = _mm_loadl_epi64((const __m128i*) a);
      a += 8;
      const __m128i vb = _mm_loadl_epi64((const __m128i*) b);
      b += 8;

      /*
       * 将 8 个 uint8 零扩展为 8 个 uint16，存入 128-bit 寄存器
       * vxa[i] = (uint16_t) a[i]，vxb[i] = (uint16_t) b[i]
       */
      const __m128i vxa = _mm_unpacklo_epi8(va, vzero);
      const __m128i vxb = _mm_unpacklo_epi8(vb, vzero);

      /*
       * 定点乘法：将 uint16 输入与 32-bit 乘法因子相乘，结果为 32-bit。
       * SSE2 没有直接的 16x16→32 无符号乘法，因此拆分为：
       *   product = vx * multiplier
       *           = vx * (multiplier_hi * 2^16 + multiplier_lo)
       *
       * va_product_lo：乘积的低 16 位（_mm_mullo_epi16 取低 16 位）
       * va_product_hi：乘积的高 16 位 = mulhi_u(vxa, lo) + mullo(vxa, hi)
       *   其中 mulhi_u 为无符号高位乘法，mullo 为低位乘法
       */
      const __m128i va_product_lo = _mm_mullo_epi16(vxa, va_multiplier_lo);
      const __m128i va_product_hi =
        _mm_add_epi16(_mm_mulhi_epu16(vxa, va_multiplier_lo), _mm_mullo_epi16(vxa, va_multiplier_hi));

      const __m128i vb_product_lo = _mm_mullo_epi16(vxb, vb_multiplier_lo);
      const __m128i vb_product_hi =
        _mm_add_epi16(_mm_mulhi_epu16(vxb, vb_multiplier_lo), _mm_mullo_epi16(vxb, vb_multiplier_hi));

      /*
       * 将 16-bit 的乘积高低位重新组合为 32-bit 整数，并累加：
       *   vacc = zero_point_product + va_product + vb_product
       *
       * _mm_unpacklo_epi16(lo, hi) 将低 4 个 16-bit 对交织为 4 个 32-bit 整数
       * _mm_unpackhi_epi16(lo, hi) 将高 4 个 16-bit 对交织为 4 个 32-bit 整数
       *
       * vacc_lo 处理元素 0~3，vacc_hi 处理元素 4~7
       */
      __m128i vacc_lo = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(va_product_lo, va_product_hi));
      __m128i vacc_hi = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(va_product_lo, va_product_hi));

      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vb_product_lo, vb_product_hi));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vb_product_lo, vb_product_hi));

      /*
       * 带符号的四舍五入右移（等效于 round(vacc / 2^shift)）：
       *
       * 步骤 1：计算余数 vrem，用于判断是否需要进位
       *   vrem = (vacc & remainder_mask) + (vacc < 0 ? -1 : 0)
       *
       *   说明：
       *   - (vacc & remainder_mask) 提取低 shift 位（即余数的绝对值部分）
       *   - _mm_cmpgt_epi32(zero, vacc) 对负数返回 0xFFFFFFFF（即 -1），非负数返回 0
       *   - 加上 -1 是为了将负数的余数调整为与正数一致的比较基准
       *     （等价于对负数做 floor 除法后的余数修正）
       */
      const __m128i vrem_lo =
        _mm_add_epi32(_mm_and_si128(vacc_lo, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo));
      const __m128i vrem_hi =
        _mm_add_epi32(_mm_and_si128(vacc_hi, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi));

      /*
       * 步骤 2：算术右移并根据余数决定是否加 1（四舍五入）
       *   vacc = (vacc >> shift) - (vrem > remainder_threshold ? 1 : 0)
       *
       *   说明：
       *   - _mm_sra_epi32 执行算术右移（保留符号位）
       *   - _mm_cmpgt_epi32(vrem, threshold) 对余数超过阈值的元素返回 0xFFFFFFFF（即 -1）
       *   - 减去 -1 等价于加 1，实现四舍五入
       *   - remainder_threshold = (1 << shift) / 2 - 1，即 "大于一半则进位"
       */
      vacc_lo = _mm_sub_epi32(_mm_sra_epi32(vacc_lo, vshift), _mm_cmpgt_epi32(vrem_lo, vremainder_threshold));
      vacc_hi = _mm_sub_epi32(_mm_sra_epi32(vacc_hi, vshift), _mm_cmpgt_epi32(vrem_hi, vremainder_threshold));

      /*
       * 将 32-bit 累加结果打包回 uint8，并加上输出零点、做饱和截断：
       *
       * 1. 加载输出零点 vy_zero_point（int16x8）
       * 2. _mm_packs_epi32：将两个 int32x4 饱和打包为 int16x8
       * 3. _mm_adds_epi16：加上输出零点（带饱和，防止溢出）
       * 4. _mm_packus_epi16：将 int16x8 无符号饱和打包为 uint8x16
       *    （负数截断为 0，超过 255 截断为 255）
       */
      const __m128i vy_zero_point = _mm_load_si128((const __m128i*) quantization_params->sse2.y_zero_point);
      const __m128i vacc = _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), vy_zero_point);
      __m128i vy = _mm_packus_epi16(vacc, vacc);

      /* 将输出值截断到 [y_min, y_max] 范围（激活函数截断，如 ReLU6） */
      vy = _mm_max_epu8(vy, _mm_load_si128((const __m128i*) quantization_params->sse2.y_min));
      vy = _mm_min_epu8(vy, _mm_load_si128((const __m128i*) quantization_params->sse2.y_max));

      /* 将低 64 位（8 个 uint8 结果）写入输出缓冲区 */
      _mm_storel_epi64((__m128i*) y, vy);
      y += 8;

      n -= 8;
    } while (n >= 8);

    /* -------------------------------------------------------------------
     * 尾部处理：处理剩余的 1~7 个元素（仍使用 SSE2，但用移位对齐）
     * ------------------------------------------------------------------- */
    if (n != 0) {
      /*
       * 计算需要回退的字节数，使得加载的 8 字节中包含剩余的 n 个有效元素。
       * 例如 n=3，则 n_decrement=5，从 (a - 5) 处加载 8 字节，
       * 然后右移 5*8=40 位，使有效数据对齐到低位。
       */
      const size_t n_decrement = 8 - n;
      const __m128i vload_shift = _mm_cvtsi32_si128(8 * (int32_t) n_decrement);

      /* 加载并右移，将有效的 n 个元素移到低位 */
      const __m128i va = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) (a - n_decrement)), vload_shift);
      const __m128i vb = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) (b - n_decrement)), vload_shift);

      /* 零扩展 8-bit → 16-bit */
      const __m128i vxa = _mm_unpacklo_epi8(va, vzero);
      const __m128i vxb = _mm_unpacklo_epi8(vb, vzero);

      /* 定点乘法（与主循环相同） */
      const __m128i va_product_lo = _mm_mullo_epi16(vxa, va_multiplier_lo);
      const __m128i va_product_hi =
        _mm_add_epi16(_mm_mulhi_epu16(vxa, va_multiplier_lo), _mm_mullo_epi16(vxa, va_multiplier_hi));

      const __m128i vb_product_lo = _mm_mullo_epi16(vxb, vb_multiplier_lo);
      const __m128i vb_product_hi =
        _mm_add_epi16(_mm_mulhi_epu16(vxb, vb_multiplier_lo), _mm_mullo_epi16(vxb, vb_multiplier_hi));

      /* 累加（与主循环相同） */
      __m128i vacc_lo = _mm_add_epi32(vzero_point_product, _mm_unpacklo_epi16(va_product_lo, va_product_hi));
      __m128i vacc_hi = _mm_add_epi32(vzero_point_product, _mm_unpackhi_epi16(va_product_lo, va_product_hi));

      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vb_product_lo, vb_product_hi));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vb_product_lo, vb_product_hi));

      /* 带符号四舍五入右移（与主循环相同） */
      const __m128i vrem_lo =
        _mm_add_epi32(_mm_and_si128(vacc_lo, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo));
      const __m128i vrem_hi =
        _mm_add_epi32(_mm_and_si128(vacc_hi, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi));

      vacc_lo = _mm_sub_epi32(_mm_sra_epi32(vacc_lo, vshift), _mm_cmpgt_epi32(vrem_lo, vremainder_threshold));
      vacc_hi = _mm_sub_epi32(_mm_sra_epi32(vacc_hi, vshift), _mm_cmpgt_epi32(vrem_hi, vremainder_threshold));

      /* 打包、加零点、饱和截断（与主循环相同） */
      const __m128i vy_zero_point = _mm_load_si128((const __m128i*) quantization_params->sse2.y_zero_point);
      const __m128i vacc = _mm_adds_epi16(_mm_packs_epi32(vacc_lo, vacc_hi), vy_zero_point);
      __m128i vy = _mm_packus_epi16(vacc, vacc);
      vy = _mm_max_epu8(vy, _mm_load_si128((const __m128i*) quantization_params->sse2.y_min));
      vy = _mm_min_epu8(vy, _mm_load_si128((const __m128i*) quantization_params->sse2.y_max));

      /*
       * 按位分段写出剩余的 n 个字节（n 在 1~7 之间）：
       * 利用 n 的二进制位（bit2=4字节，bit1=2字节，bit0=1字节）逐段写出。
       */
      if (n & 4) {
        /* 写出低 32 位（4 个字节） */
        *((uint32_t*) y) = (uint32_t) _mm_cvtsi128_si32(vy);
        /* 将寄存器中的第 1 个 32-bit 字移到低位，准备写下一段 */
        vy = _mm_shuffle_epi32(vy, _MM_SHUFFLE(3, 2, 1, 1));
        y += 4;
      }
      if (n & 2) {
        /* 写出低 16 位（2 个字节） */
        *((uint16_t*) y) = (uint16_t) _mm_extract_epi16(vy, 0);
        /* 右移 16 位，准备写最后 1 个字节 */
        vy = _mm_srli_epi32(vy, 16);
        y += 2;
      }
      if (n & 1) {
        /* 写出最低 8 位（1 个字节） */
        *((uint8_t*) y) = (uint8_t) _mm_cvtsi128_si32(vy);
      }
    }
  } else {
    /* -----------------------------------------------------------------------
     * 标量路径：n < 8，逐元素处理
     * ----------------------------------------------------------------------- */

    /* 从量化参数结构体中提取标量值 */
    const int32_t vzero_point_product = quantization_params->sse2.zero_point_product[0];
    const uint32_t va_multiplier = quantization_params->sse2.a_multiplier;
    const uint32_t vb_multiplier = quantization_params->sse2.b_multiplier;
    const int32_t vremainder_mask = quantization_params->sse2.remainder_mask[0];
    const int32_t vremainder_threshold = quantization_params->sse2.remainder_threshold[0];
    const uint32_t vshift = quantization_params->sse2.shift;
    const int32_t vy_zero_point = (int32_t) quantization_params->sse2.y_zero_point[0];
    const int32_t vy_max = (int32_t) (uint32_t) quantization_params->sse2.y_max[0];
    const int32_t vy_min = (int32_t) (uint32_t) quantization_params->sse2.y_min[0];

    while (n-- != 0) {
      /* 零扩展读取输入元素 */
      const uint32_t vxa = (uint32_t) *a++;
      const uint32_t vxb = (uint32_t) *b++;

      /*
       * 定点乘法累加：
       *   acc = zero_point_product + a * a_multiplier + b * b_multiplier
       */
      int32_t vacc = vzero_point_product + (int32_t) (vxa * va_multiplier) + (int32_t) (vxb * vb_multiplier);

      /*
       * 带符号四舍五入右移：
       *   vrem = (vacc & remainder_mask) - (vacc < 0 ? 1 : 0)
       *   vacc = asr(vacc, shift) + (vrem > remainder_threshold ? 1 : 0)
       *
       * asr_s32 为算术右移（保留符号位），定义于 scalar-utils.h
       */
      const int32_t vrem = (vacc & vremainder_mask) - (int32_t) (vacc < 0);
      vacc = asr_s32(vacc, vshift) + (int32_t) (vrem > vremainder_threshold);

      /* 加上输出零点，并截断到 [y_min, y_max] */
      int32_t vy = vacc + vy_zero_point;
      vy = vy >= vy_min ? vy : vy_min;
      vy = vy <= vy_max ? vy : vy_max;

      /* 写出 uint8 结果 */
      *y++ = (uint8_t) vy;
    }
  }
}
