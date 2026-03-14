#include "utils.h"

typedef sycl::half IT;
typedef sycl::half MT;

#define KV_SHARE_BUFFER
template<uint32_t QK_DIM, uint32_t V_DIM, uint32_t REDUCE_NUM>
inline void sdp_esimd_kernel_with_reduce_fp16I_fp16O(
    int64_t num_heads, 
    int64_t num_heads_kv,
    int64_t batch_idx,
    uint8_t* query,
    uint8_t* key,
    uint8_t* value,
    uint32_t* kv_indptr, 
    uint32_t* kv_indices,
    void* sdp_tmp,
    void* attn_mask,
    uint8_t* output,
    float attn_scale,
    float beta,
    sycl::queue& dpcpp_queue) {

    constexpr size_t q_chunk = 2;
    const size_t GS = num_heads/q_chunk;
    const size_t qk_HD = QK_DIM;
    constexpr size_t v_HD = V_DIM;
    const size_t groups_for_kv_len = REDUCE_NUM; 

    const void * mask = nullptr;
    const size_t query_head_stride = qk_HD;
    const size_t query_seq_stride = qk_HD * num_heads;
    const size_t query_bsz_stride = query_seq_stride;
    const size_t key_bsz_stride = 0;
    const size_t value_head_stride = v_HD;
    const size_t key_head_stride = qk_HD;

    const size_t value_bsz_stride = 0;
    const size_t mask_bsz_stride = 0;
    const size_t mask_head_stride = 0;
    const size_t mask_seq_stride = 0;
    const size_t output_bsz_stride = 0;
    const size_t output_group_stride = v_HD;
    const size_t output_tmp_head_stride = v_HD * groups_for_kv_len + groups_for_kv_len * 2;
    const size_t output_tmp_seq_stride = output_tmp_head_stride* num_heads;
    const size_t output_head_stride = v_HD;
    const size_t output_seq_stride = output_head_stride* num_heads;
    const size_t bsz = 1;
    const size_t num_kv_heads = num_heads_kv;
    const size_t seq_len = 1; //q token num

    const size_t group_num = num_heads / num_kv_heads;
    // const float attn_scale = 1 / std::sqrt((float)qk_HD);

    sycl::range<3> global_size(bsz, seq_len, groups_for_kv_len * GS);
    sycl::range<3> local_size(1, 1, GS);

    // std::cout << "global range " << "(" << num_heads << ", " << seq_len * GS << ")" << std::endl;
    // std::cout << "local range " << "(" << GS << ")" << std::endl;
#if 1
    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class sdp_esimd>(
            sycl::nd_range<3>(global_size, local_size),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {

                const size_t bsz_idx = item.get_group(0);
                const size_t head_idx = item.get_local_id(2) * q_chunk;
                const size_t kv_head_idx = head_idx / group_num;
                const size_t seq_idx = item.get_group(1);
                const size_t group_idx = item.get_group(2); 

                const IT * query_head = (const IT *)query //+ bsz_idx * query_bsz_stride
                                                          + head_idx  * query_head_stride
                                                          + batch_idx * query_seq_stride;
                                                          //+ seq_idx * query_seq_stride;

                const IT * key_head = (const IT *)key //+ bsz_idx * key_bsz_stride
                                                      + kv_head_idx * key_head_stride;
                const IT * value_head = (const IT *)value //+ bsz_idx * value_bsz_stride
                                                          + kv_head_idx * value_head_stride;
                // const MT * mask_head = (const MT *)mask + bsz_idx * mask_bsz_stride
                //                                         + head_idx * mask_head_stride
                //                                         + seq_idx * mask_seq_stride;

                IT * output_head = (IT *)sdp_tmp //+ bsz_idx * output_bsz_stride
                                                + head_idx * output_tmp_head_stride
                                                + group_idx * output_group_stride;
                                                //+ seq_idx * output_tmp_seq_stride;
                IT * max_head = (IT *)sdp_tmp //+ bsz_idx * output_bsz_stride
                                                + head_idx * output_tmp_head_stride
                                                + group_idx + v_HD * groups_for_kv_len;
                                                //+ seq_idx * output_tmp_seq_stride; 
                IT * softmax_head = (IT *)sdp_tmp //+ bsz_idx * output_bsz_stride
                                                + head_idx * output_tmp_head_stride
                                                + group_idx + v_HD * groups_for_kv_len + groups_for_kv_len;
                                                //+ seq_idx * output_tmp_seq_stride; 

                simd<IT, q_chunk*qk_HD> query_row = block_load<IT, q_chunk* qk_HD>(query_head)* attn_scale;

                uint32_t* tokens = (uint32_t*)kv_indices +  kv_indptr[batch_idx];

                const size_t kv_len = kv_indptr[batch_idx+1] - kv_indptr[batch_idx];


                const size_t sub_rows = kv_len / groups_for_kv_len;
                const size_t rem_rows = kv_len % groups_for_kv_len;
                size_t start_row = sub_rows * group_idx + std::min(group_idx, rem_rows);
                size_t end_row = start_row + sub_rows + (group_idx < rem_rows);

                simd<IT, q_chunk*v_HD> accs = 0;   //atten out
                simd<IT, q_chunk> softmax = 0;          //deno
                simd<IT, q_chunk> old_max_attn = -sycl::detail::max_v<IT>();   // -8
                simd<IT, q_chunk> max_attn = -sycl::detail::max_v<IT>();
                simd<IT, q_chunk> attn;

                //simd<uint32_t, sub_rows> real_rows = block_load<uint32_t, sub_rows>((uint32_t*)tokens + sub_rows * group_idx);
                simd<IT, qk_HD> key_row;
                #ifndef KV_SHARE_BUFFER
                simd<IT, v_HD> value_row;
                #endif
                //explicit cache setting only supports fixed size
                //block_load<IT, 128,sycl::ext::intel::esimd::detail::make_L1_L2_properties_t<cache_hint::uncached, cache_hint::cached>

                uint32_t real_row = tokens[start_row];
                for (size_t r = start_row; r < end_row; ++r) {
                    #ifndef KV_SHARE_BUFFER
                    value_row = block_load<IT, v_HD>(value_head + real_row * num_kv_heads * v_HD);
                    #endif
                    key_row = block_load<IT, qk_HD>(key_head + real_row * num_kv_heads * qk_HD); 

                    #pragma unroll
                    for (size_t q = 0; q < q_chunk; ++q) {
                        attn[q] = sycl::ext::intel::esimd::detail::sum<IT, IT, qk_HD>(
                        query_row.template select<qk_HD,1>(q*qk_HD) * key_row);
                    }
                    real_row = tokens[r + 1];

                    // q_chunk  together softmax
                    max_attn  = max<IT, q_chunk, IT>(attn, old_max_attn);
                    simd<IT, q_chunk> attn_exp_1 = sycl::ext::intel::esimd::exp(old_max_attn - max_attn); 
                    simd<IT, q_chunk> attn_exp_2 = sycl::ext::intel::esimd::exp(attn - max_attn); 
                    #pragma unroll
                    for (size_t q = 0; q < q_chunk; ++q) {
                        #ifndef KV_SHARE_BUFFER
                        accs.template select<v_HD, 1>(q*v_HD) = 
                            accs.template select<v_HD, 1>(q*v_HD) * attn_exp_1[q] 
                            + value_row.template select<v_HD,1>(0) * attn_exp_2[q];
                        #else
                        accs.template select<v_HD, 1>(q*v_HD) = 
                            accs.template select<v_HD, 1>(q*v_HD) * attn_exp_1[q] 
                            + key_row.template select<v_HD, 1>(0) * attn_exp_2[q];
                        #endif
                    }
                    softmax = softmax  * attn_exp_1 + attn_exp_2; 
                    old_max_attn = max_attn; 
                }
                #pragma unroll 
                for (size_t q = 0; q < q_chunk; ++q)  {
                    block_store<IT, v_HD>(output_head + q * output_tmp_head_stride, accs.template select<v_HD, 1>(q*v_HD));
                    block_store<IT, 1>(max_head + q * output_tmp_head_stride, max_attn[q]);
                    block_store<IT, 1>(softmax_head + q* output_tmp_head_stride, softmax[q]); 
                }
            });
      });
#endif

#if 1
    constexpr size_t reduce_GS = groups_for_kv_len > 16? 16 : groups_for_kv_len;  //best 16 or 32
    constexpr size_t SUB_GS_NUM = 2;
    constexpr size_t SUB_DIM = v_HD/2;
    constexpr int sub_gs = reduce_GS / SUB_GS_NUM;
    constexpr size_t softmax_offset = reduce_GS * SUB_DIM * sizeof(IT);
    const size_t atten_num = groups_for_kv_len / reduce_GS;
    sycl::range<3> reduce_global_size(1/*bsz*/, num_heads,  (v_HD/SUB_DIM) * reduce_GS);
    sycl::range<3> reduce_local_size(1, 1, reduce_GS);
    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class sdp_reduction_esimd>(
            sycl::nd_range<3>(reduce_global_size, reduce_local_size),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                slm_init<reduce_GS * SUB_DIM * sizeof(IT) + reduce_GS * sizeof(IT) * 2>();
#if 1
                const size_t bsz_idx = item.get_group(0);
                const size_t tid = item.get_local_id(2);
                const size_t head_idx = item.get_group(1);
                const size_t dim_idx = item.get_group(2);

                const IT * atten_head = (const IT *)sdp_tmp //+ bsz_idx * output_bsz_stride
                                                + head_idx * output_tmp_head_stride
                                                + tid * atten_num * output_group_stride
                                                + dim_idx * SUB_DIM;

                IT * max_head = (IT *)sdp_tmp + //bsz_idx * output_bsz_stride
                                                + head_idx * output_tmp_head_stride
                                                + 0  + v_HD * groups_for_kv_len;
                                                //+ seq_idx * output_tmp_seq_stride; 
                IT * softmax_head = (IT *)sdp_tmp //+ bsz_idx * output_bsz_stride
                                                + head_idx * output_tmp_head_stride
                                                + tid * atten_num + v_HD * groups_for_kv_len + groups_for_kv_len;
                                                //+ seq_idx * output_tmp_seq_stride; 


                IT * output_head = (IT *)output //+ bsz_idx * output_bsz_stride
                                                + head_idx * output_head_stride
                                                + dim_idx * SUB_DIM
                                                + batch_idx * output_seq_stride;


                simd<IT, groups_for_kv_len> max_attn = block_load<IT, groups_for_kv_len>(max_head);   
                simd<IT, atten_num> softmax = block_load<IT, atten_num>(softmax_head);
                simd<IT, SUB_DIM> accs{0};
                simd<IT, 1> softmax_merge = 0;

                IT max_attn_max = hmax<IT, IT, groups_for_kv_len>(max_attn);
                //merge current thread
                #pragma unroll
                for (size_t r = 0; r < atten_num; ++r) {
                    simd<IT, SUB_DIM> accs_tmp = block_load<IT, SUB_DIM>(atten_head + r * v_HD);
                    //correction then sum
                    if (max_attn[tid * atten_num + r] < max_attn_max) {
                        IT attn_exp = sycl::ext::intel::esimd::exp(max_attn[tid * atten_num + r] - max_attn_max);
                        softmax_merge += softmax[r] * attn_exp;
                        accs += accs_tmp *attn_exp;
                    }
                    else{
                        softmax_merge += softmax[r]; 
                        accs += accs_tmp;
                    }
                }

                //to merge one group's threads 
                slm_block_store(tid * SUB_DIM * sizeof(IT), accs);  //result after max of max correction
                slm_block_store<IT, 1>(softmax_offset + tid * sizeof(IT), softmax_merge);//deno after max of max correction

                barrier();

                if (tid < SUB_GS_NUM) {  //reduce w/  sub_gs threads
                    simd<IT, SUB_DIM> accs = 0;
                    #pragma unroll
                    for (int i = 0; i < sub_gs; ++i) {
                        accs += slm_block_load<IT, SUB_DIM>((tid * sub_gs + i) * SUB_DIM * sizeof(IT));
                    }
                    slm_block_store<IT, SUB_DIM>(tid * sub_gs * SUB_DIM * sizeof(IT), accs);
                }

                barrier();

                //final reduce
                if (tid == 0) {
                    //reduce to final one deno
                    float softmax_sum = sycl::ext::intel::esimd::detail::sum<IT, IT, reduce_GS>(
                        slm_block_load<IT, reduce_GS>(softmax_offset)
                    );

                    simd<IT, SUB_DIM> accs = 0;
                    #pragma unorll
                    for (int i = 0; i < SUB_GS_NUM; ++i) { // final one result
                        accs += slm_block_load<IT, SUB_DIM>(i * sub_gs * SUB_DIM * sizeof(IT));
                    }

                    simd<IT, SUB_DIM> result = accs / softmax_sum;
                    block_store<IT, SUB_DIM>(output_head, result);
                }
#endif
            });
      });
#endif
}