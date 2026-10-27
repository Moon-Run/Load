"""Real launch-correlated per-rank evidence for the portable TP8/DP2 run."""
from collections import Counter
from ma9_profile import Index, contains, require
import ma11_profile


def analyze(events, arm, rank):
    index = Index(events)
    # A DP=1 run cannot pass this gate. Require actual GPU work belonging to
    # the two-member DP collective, in addition to replica gradient hashes.
    dp = [e for e in index.cpu if e['name'] == 'record_param_comms' and
          e.get('args',{}).get('Group size') == 2 and
          e.get('args',{}).get('Collective name') == 'allreduce' and
          e.get('args',{}).get('In msg nelems',0) > 8192]
    require(dp, 'No DP2 gradient AllReduce metadata in profile')
    require(all(any(e['cat']=='kernel' and 'nccl' in e['name'].lower() and 'AllReduce' in e['name']
                    for e in index.work(scope)[0]) for scope in dp), 'DP collective lacks correlated NCCL kernels')
    result = dict(dp_allreduces=len(dp),rank=rank)
    if arm == 'ma11':
        result.update(ma11_profile.trace_composed_records(events,'F4-cache-zcount-csoftmax',rank%8))
        return result
    require(arm in ('flux-basic','megatron-cuda'), 'Unknown profile arm')
    route = 'flux_tensor' if arm == 'flux-basic' else 'tensor'
    receiver = 'basic_quant_decode_reduce_kernel' if arm == 'flux-basic' else 'fused_ash_decompress_kernel'
    result['roles'] = {}
    claimed = set()
    for role in ('attn','mlp'):
        scopes = [e for e in events if e.get('cat')=='user_annotation' and e.get('ph')=='X' and
                  e['name']==f'scope.{role}.{route}']
        require(len(scopes)==64, 'Expected 32 layers x 2 local microbatches for '+role)
        for scope in scopes:
            work,_ = index.work(scope)
            kernels = [e for e in work if e['cat']=='kernel']
            require(not claimed.intersection(map(id,kernels)), 'Projection kernels attributed twice')
            claimed.update(map(id,kernels))
            names = [e['name'] for e in kernels]
            cpu = [e for e in index.cpu if contains(scope,e)]
            counts = Counter(e['name'] for e in cpu)
            require(sum('fused_ash_compress_kernel' in n for n in names)==1 and
                    sum(receiver in n for n in names)==1 and
                    any('nccl' in n.lower() for n in names), 'Missing standalone encode/exchange/decode kernel')
            require(any('alltoall' in e['name'].lower().replace('_','') for e in cpu), 'NCCL All-to-All missing')
            require(not counts['aten::clone'] and not any('CudaIpcBarrierAllKernel' in n or
                    'taco_decode_ring_kernel' in n or 'reducescatter' in n.lower().replace('_','') for n in names),
                    'Baseline changed to GEMM-RS or clone path')
            require(counts['aten::add'] == (0 if arm=='flux-basic' else 8), 'BF16 receiver fold changed')
            markers = [e['name'] for e in events if e.get('cat')=='user_annotation' and contains(scope,e)]
            for marker in ('megatron_taco.encode','megatron_taco.exchange','megatron_taco.decode_reduce'):
                require(markers.count(marker)==1, 'Missing codec stage: '+marker)
            if arm=='flux-basic':
                require(markers.count('ablation6.flux_gemm_only')==1 and
                        sum(('cutlass' in n.lower() or 'flux_bf16' in n.lower()) and 'sm90' in n.lower() for n in names)==1 and
                        not any('fused_ash_decompress_kernel' in n for n in names), 'Flux GemmOnly/basic receiver changed')
            else:
                # This scope contains the replacement RS only; the original
                # Megatron GEMM remains outside and is checked by source identity.
                require('ablation6.flux_gemm_only' not in markers and
                        not any('basic_quant_decode_reduce_kernel' in n for n in names), 'Native CUDA baseline changed')
        result['roles'][role] = dict(calls=64,route=route,receiver=receiver,encode_kernels=64,
                                    decode_kernels=64,separate_bf16_adds=0 if arm=='flux-basic' else 512)
    return result
