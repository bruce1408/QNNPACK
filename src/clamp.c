/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* 标准库头文件：断言、数学、类型、大小、内存分配 */
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* QNNPACK 核心头文件：状态码、操作符定义、日志 */
#include <qnnpack.h>
#include <qnnpack/operator.h>
#include <qnnpack/log.h>


/**
 * 创建 NC (NCHW 格式，通道优先) uint8 Clamp 操作符。
 * Clamp 操作将输入值限制在 [output_min, output_max] 范围内：y = clamp(x, min, max)。
 * 支持量化 uint8 张量，自动计算 clamping 参数。
 *
 * @param channels 通道数 (必须 >0)
 * @param output_min 输出下限 (uint8_t)
 * @param output_max 输出上限 (uint8_t, 必须 >= output_min)
 * @param flags 标志位 (当前未使用)
 * @param clamp_out 输出操作符指针 (成功时分配)
 * @return qnnp_status_success (成功) 或错误码 (invalid_parameter, out_of_memory 等)
 */
enum qnnp_status qnnp_create_clamp_nc_u8(
    size_t channels,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    qnnp_operator_t* clamp_out)
{
  qnnp_operator_t clamp_op = NULL;  // 操作符结构体指针
  enum qnnp_status status = qnnp_status_uninitialized;  // 默认未初始化状态

  /* 检查 QNNPACK 是否已初始化 (qnnp_initialize() 调用过) */
  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_create_clamp_nc_u8 failed because QNNPACK is not properly initialized");
    goto error;
  }

  status = qnnp_status_invalid_parameter;  // 参数错误状态

  /* 验证通道数 > 0 */
  if (channels == 0) {
    qnnp_log_error(
      "failed to create Clamp operator with %zu channels: number of channels must be non-zero", channels);
    goto error;
  }

  /* 验证 output_min <= output_max */
  if (output_min > output_max) {
    qnnp_log_error(
      "failed to create Clamp operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      output_min, output_max);
    goto error;
  }

  status = qnnp_status_out_of_memory;  // 内存不足状态

  /* 分配操作符结构体内存 */
  clamp_op = calloc(1, sizeof(struct qnnp_operator));
  if (clamp_op == NULL) {
    qnnp_log_error("failed to allocate %zu bytes for qnnp_operator structure", sizeof(struct qnnp_operator));
    goto error;
  }

  /* 设置操作符参数：通道数、clamping 参数 (预计算 min/max 向量) */
  clamp_op->channels = channels;
  clamp_op->u8_clamping_params = qnnp_compute_u8_clamping_params(output_min, output_max);

  /* 设置操作符类型和格式：Clamp 类型，QUInt8 格式 */
  clamp_op->ukernel_type = qnnp_ukernel_type_clamp;
  clamp_op->format = qnnp_format_quint8;

  /* 返回成功，输出操作符指针 */
  *clamp_out = clamp_op;
  return qnnp_status_success;

error:
  /* 错误路径：删除操作符 (释放内存) 并返回错误状态 */
  qnnp_delete_operator(clamp_op);
  return status;
}

/**
 * 设置 NC Clamp 操作符的运行参数 (batch_size, input/output 指针和 stride)。
 * 准备操作符执行前调用，更新输入/输出缓冲区和批次大小。
 *
 * @param clamp 已创建的 Clamp 操作符
 * @param batch_size 批次大小 (元素数，0 表示无操作)
 * @param input 输入数据指针 (uint8_t*, NC 格式)
 * @param input_stride 输入 stride (字节/通道)
 * @param output 输出数据指针 (uint8_t*, NC 格式)
 * @param output_stride 输出 stride (字节/通道)
 * @return qnnp_status_success 或 qnnp_status_uninitialized
 */
enum qnnp_status qnnp_setup_clamp_nc_u8(
    qnnp_operator_t clamp,
    size_t batch_size,
    const uint8_t* input,
    size_t input_stride,
    uint8_t* output,
    size_t output_stride)
{
  /* 检查 QNNPACK 初始化 */
  if (!qnnp_params.initialized) {
    qnnp_log_error("qnnp_setup_clamp_nc_u8 failed because QNNPACK is not properly initialized");
    return qnnp_status_uninitialized;
  }

  /* batch_size == 0：无操作，直接返回成功 */
  if (batch_size == 0) {
    clamp->batch_size = 0;
    return qnnp_status_success;
  }

  /* 更新操作符运行时参数：批次大小、输入/输出指针和 stride */
  clamp->batch_size = batch_size;
  clamp->input = input;
  clamp->input_pixel_stride = input_stride;
  clamp->output = output;
  clamp->output_pixel_stride = output_stride;

  return qnnp_status_success;
}
