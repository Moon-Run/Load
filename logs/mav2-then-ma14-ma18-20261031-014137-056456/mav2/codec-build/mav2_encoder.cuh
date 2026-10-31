template<typename T, int GROUP_SIZE>
__global__ void __launch_bounds__(GROUP_SIZE) mav2_encode_kernel(
    const void* __restrict__ input,
    void* __restrict__ output,
    float* __restrict__ quant_scales,    // Not strictly needed
    float* __restrict__ ada_scales_out,  // Not strictly needed
    int orgChunkCount,
    size_t in_chunk_stride_bytes,
    size_t out_chunk_stride_bytes,
    size_t scale_chunk_stride_bytes,
    float target_range,
    float lambda,
    float fp8_max_val,
    __nv_saturation_t saturate,
    __nv_fp8_interpretation_t fp8_type, int mav2_mode)
{
    int group_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;

    const char* in_base = (const char*)input + chunk_id * in_chunk_stride_bytes;
    char* out_base = (char*)output + chunk_id * out_chunk_stride_bytes;

    float* chunk_q_scales = (float*)(out_base + orgChunkCount);
    float* chunk_ada_scales = (float*)(out_base + orgChunkCount + scale_chunk_stride_bytes);

    int start_idx = group_id * GROUP_SIZE;
    int idx = start_idx + tid;

    float val = 0.0f;
    if (idx < orgChunkCount) {
        if (std::is_same<T, float>::value) {
            val = reinterpret_cast<const float*>(in_base)[idx];
        } else {
            val = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(in_base)[idx]);
        }
    }

    __shared__ float mav2_scratch[4];
    if (mav2::try_direct(val, reinterpret_cast<unsigned char*>(out_base),
        chunk_q_scales, chunk_ada_scales, idx, group_id, mav2_mode, mav2_scratch)) return;

    float val_sq = val * val;
    val_sq = warpReduceSum(val_sq); 

    static __shared__ float shared_stats[32];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) shared_stats[warp_id] = val_sq;
    __syncthreads();

    float group_sum_sq = (tid < (GROUP_SIZE / WARP_SIZE)) ? shared_stats[lane] : 0.0f;
    if (warp_id == 0) group_sum_sq = warpReduceSum(group_sum_sq);

    __shared__ float ada_scale;
    if (tid == 0) {
        int count = min(GROUP_SIZE, orgChunkCount - start_idx);
        float var = (group_sum_sq / (float)count) + lambda;
        if (var < 1e-12f) var = 1e-12f;
        
        float s = fp8_max_val * rsqrtf(var);
        ada_scale = clampf(s, 1e-3f, 1e3f);

        chunk_ada_scales[group_id] = ada_scale;
    }
    __syncthreads();

    val *= ada_scale;

    #pragma unroll
    for (int s = 1; s < 32; s *= 2) {
        float neighbor = __shfl_xor_sync(0xffffffff, val, s);
        if (tid & s) val = neighbor - val;
        else         val = val + neighbor;
    }

    if (GROUP_SIZE > 32) {
        extern __shared__ float smem_data[];
        smem_data[tid] = val;
        __syncthreads();

        for (int s = 32; s < GROUP_SIZE; s *= 2) {
            int neighbor_idx = tid ^ s;
            float neighbor = smem_data[neighbor_idx];

            float new_val;
            if (tid & s) new_val = neighbor - val;
            else         new_val = val + neighbor;

            __syncthreads();
            val = new_val;
            smem_data[tid] = val;
            __syncthreads();
        }
    }

    val *= rsqrtf((float)GROUP_SIZE);

    float abs_val = fabsf(val);
    abs_val = warpReduceMax(abs_val);

    if (lane == 0) shared_stats[warp_id] = abs_val;
    __syncthreads();

    float group_max = (tid < (GROUP_SIZE / WARP_SIZE)) ? shared_stats[lane] : 0.0f;
    if (warp_id == 0) group_max = warpReduceMax(group_max);

    __shared__ float quant_scale;
    if (tid == 0) {
        float mv = (group_max < 1e-12f) ? 1e-12f : group_max;
        float s = safe_divide(mv, target_range);
        quant_scale = clampf(s, 1e-12f, 1e6f);
        chunk_q_scales[group_id] = quant_scale;
    }
    __syncthreads();

    val = check_nan_inf(val);
    float scaled = clamp_fp8(safe_divide(val, quant_scale));
    unsigned char q8 = (unsigned char)__nv_cvt_float_to_fp8(scaled, saturate, fp8_type);

    unsigned int packed = (unsigned int)q8;
    unsigned int b1 = __shfl_down_sync(0xffffffff, packed, 1);
    unsigned int b2 = __shfl_down_sync(0xffffffff, packed, 2);
    unsigned int b3 = __shfl_down_sync(0xffffffff, packed, 3);

    if ((tid & 3) == 0 && (idx + 3 < orgChunkCount)) {
        unsigned int final_int = packed | (b1 << 8) | (b2 << 16) | (b3 << 24);
        reinterpret_cast<unsigned int*>(out_base)[idx / 4] = final_int;
    }
    else if (idx < orgChunkCount && (tid & 3) != 0 && (idx / 4 != (idx + 3) / 4)) {
        ((__nv_fp8_storage_t*)out_base)[idx] = (__nv_fp8_storage_t)q8;
    }
}

