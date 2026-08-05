#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr float kMaxFloat = 3.402823466e+38F;

size_t checkedMultiply(size_t left, size_t right, const char* message) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
    throw std::invalid_argument(message);
  }
  return left * right;
}

template <typename T>
__device__ float toFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ float toFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ T fromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ half fromFloat<half>(float value) {
  return __float2half(value);
}

__device__ float warpReduceSum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ float blockReduceSum(float value) {
  __shared__ float warp_sums[kBlockSize / kWarpSize];

  value = warpReduceSum(value);
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < blockDim.x / kWarpSize ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warpReduceSum(value);
  }
  return value;
}

template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t rows, size_t hidden_dim, float eps) {
  __shared__ float inv_rms;

  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    float square_sum = 0.0f;
    const size_t row_offset = row * hidden_dim;
    for (size_t column = threadIdx.x; column < hidden_dim;
         column += blockDim.x) {
      const float value = toFloat(input[row_offset + column]);
      square_sum += value * value;
    }

    square_sum = blockReduceSum(square_sum);
    if (threadIdx.x == 0) {
      inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();

    for (size_t column = threadIdx.x; column < hidden_dim;
         column += blockDim.x) {
      const float normalized = toFloat(input[row_offset + column]) * inv_rms;
      output[row_offset + column] =
          fromFloat<T>(normalized * toFloat(weight[column]));
    }
    __syncthreads();
  }
}

__device__ float blockReduceMax(float value) {
  __shared__ float warp_maxima[kBlockSize / kWarpSize];

  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) {
    warp_maxima[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < blockDim.x / kWarpSize ? warp_maxima[lane] : -kMaxFloat;
  if (warp == 0) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
  }
  return value;
}

template <typename T>
__global__ void attentionOnlineKernel(const T* q, const T* k, const T* v,
                                      T* output, int batch_size,
                                      int target_seq_len, int src_seq_len,
                                      int query_heads, int kv_heads,
                                      int head_dim, bool is_causal) {
  extern __shared__ float shared[];
  float* probabilities = shared;
  float* accumulator = shared + kBlockSize;
  __shared__ float row_max;
  __shared__ float row_sum;
  __shared__ float row_alpha;
  __shared__ float row_new_max;

  const size_t rows = static_cast<size_t>(batch_size) * target_seq_len * query_heads;
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    size_t remaining = row;
    const int query_head = remaining % query_heads;
    remaining /= query_heads;
    const int target_position = remaining % target_seq_len;
    const int batch = remaining / target_seq_len;
    const int group_size = query_heads / kv_heads;
    const int kv_head = query_head / group_size;
    const int effective_src_len =
        is_causal ? min(src_seq_len, target_position + 1) : src_seq_len;
    const size_t q_offset =
        ((static_cast<size_t>(batch) * target_seq_len + target_position) *
             query_heads + query_head) *
        head_dim;

    for (int dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
      accumulator[dimension] = 0.0f;
    }
    if (threadIdx.x == 0) {
      row_max = -kMaxFloat;
      row_sum = 0.0f;
    }
    __syncthreads();

    for (int tile_start = 0; tile_start < effective_src_len;
         tile_start += kBlockSize) {
      const int source_position = tile_start + threadIdx.x;
      float score = -kMaxFloat;
      if (source_position < effective_src_len) {
        const size_t k_offset =
            ((static_cast<size_t>(batch) * src_seq_len + source_position) *
                 kv_heads + kv_head) *
            head_dim;
        float dot = 0.0f;
        for (int dimension = 0; dimension < head_dim; ++dimension) {
          dot += toFloat(q[q_offset + dimension]) *
                 toFloat(k[k_offset + dimension]);
        }
        score = dot / sqrtf(static_cast<float>(head_dim));
      }

      const float tile_max = blockReduceMax(score);
      if (threadIdx.x == 0) {
        row_new_max = fmaxf(row_max, tile_max);
        row_alpha = row_sum == 0.0f ? 0.0f : expf(row_max - row_new_max);
      }
      __syncthreads();

      float probability = 0.0f;
      if (source_position < effective_src_len) {
        probability = expf(score - row_new_max);
      }
      probabilities[threadIdx.x] = probability;
      const float tile_sum = blockReduceSum(probability);
      if (threadIdx.x == 0) {
        row_sum = row_alpha * row_sum + tile_sum;
        row_max = row_new_max;
      }
      __syncthreads();

      for (int dimension = threadIdx.x; dimension < head_dim;
           dimension += blockDim.x) {
        float tile_value = 0.0f;
        for (int offset = 0; offset < kBlockSize; ++offset) {
          const int source = tile_start + offset;
          if (source >= effective_src_len) {
            break;
          }
          const size_t v_offset =
              ((static_cast<size_t>(batch) * src_seq_len + source) * kv_heads +
               kv_head) * head_dim + dimension;
          tile_value += probabilities[offset] * toFloat(v[v_offset]);
        }
        accumulator[dimension] = row_alpha * accumulator[dimension] + tile_value;
      }
      __syncthreads();
    }

    for (int dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
      const size_t output_index =
          ((static_cast<size_t>(batch) * target_seq_len + target_position) *
               query_heads + query_head) *
          head_dim + dimension;
      output[output_index] = fromFloat<T>(accumulator[dimension] / row_sum);
    }
    __syncthreads();
  }
}

}  // namespace

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  if (rows == 0 || hidden_dim == 0) {
    h_output.clear();
    return;
  }
  if (rows > std::numeric_limits<size_t>::max() / hidden_dim) {
    throw std::invalid_argument("RMSNorm tensor size overflow");
  }
  const size_t element_count = rows * hidden_dim;
  if (h_input.size() != element_count || h_weight.size() != hidden_dim) {
    throw std::invalid_argument("RMSNorm input size mismatch");
  }

  h_output.resize(element_count);
  const size_t input_bytes = h_input.size() * sizeof(T);
  const size_t weight_bytes = h_weight.size() * sizeof(T);
  const size_t output_bytes = h_output.size() * sizeof(T);
  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;

  RUNTIME_CHECK(cudaMalloc(&d_input, input_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_weight, weight_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_output, output_bytes));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                           cudaMemcpyHostToDevice));

  const int blocks = static_cast<int>(std::min<size_t>(rows, 65535));
  rmsNormKernel<<<blocks, kBlockSize>>>(d_input, d_weight, d_output, rows,
                                       hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, output_bytes,
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_output));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_input));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      query_heads % kv_heads != 0) {
    throw std::invalid_argument("Invalid attention dimensions");
  }

  const size_t batch = static_cast<size_t>(batch_size);
  const size_t q_elements = checkedMultiply(
      checkedMultiply(checkedMultiply(batch, target_seq_len,
                                      "Attention tensor size overflow"),
                      query_heads, "Attention tensor size overflow"),
      head_dim, "Attention tensor size overflow");
  const size_t kv_elements = checkedMultiply(
      checkedMultiply(checkedMultiply(batch, src_seq_len,
                                      "Attention tensor size overflow"),
                      kv_heads, "Attention tensor size overflow"),
      head_dim, "Attention tensor size overflow");
  if (h_q.size() != q_elements || h_k.size() != kv_elements ||
      h_v.size() != kv_elements) {
    throw std::invalid_argument("Attention input size mismatch");
  }

  h_o.resize(q_elements);
  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;
  const size_t q_bytes = checkedMultiply(q_elements, sizeof(T),
                                         "Attention byte size overflow");
  const size_t kv_bytes = checkedMultiply(kv_elements, sizeof(T),
                                          "Attention byte size overflow");
  const size_t output_bytes = q_bytes;

  RUNTIME_CHECK(cudaMalloc(&d_q, q_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_o, output_bytes));
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice));

  const size_t row_count = checkedMultiply(
      checkedMultiply(batch, target_seq_len, "Attention row size overflow"),
      query_heads, "Attention row size overflow");
  const int blocks = static_cast<int>(std::min<size_t>(row_count, 65535));
  const size_t shared_bytes =
      checkedMultiply(kBlockSize + static_cast<size_t>(head_dim), sizeof(float),
                      "Attention shared memory size overflow");
  int max_shared_memory = 0;
  RUNTIME_CHECK(cudaDeviceGetAttribute(&max_shared_memory,
                                       cudaDevAttrMaxSharedMemoryPerBlock, 0));
  if (shared_bytes > static_cast<size_t>(max_shared_memory)) {
    throw std::invalid_argument("Attention head_dim requires too much shared memory");
  }
  attentionOnlineKernel<<<blocks, kBlockSize, shared_bytes>>>(
      d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, output_bytes,
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_o));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_q));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
