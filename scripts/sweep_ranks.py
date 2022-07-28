import tntorch as tn
import torch
import configargparse
import os
import os.path as osp
import sys
import tqdm
import tntorch as tn
import yaml

sys.path.append(osp.join(os.path.abspath(os.getcwd()), '..',))
from t4dt.t4dt import reduce_tucker, reduce_tt, tensor3d2qtt, tensor3d2oqtt, qtt_stack
from t4dt.t4dt import oqtt2tensor3d, qtt2tensor3d, get_qtt_frame
from t4dt.metrics import compute_metrics

torch.set_default_dtype(torch.float64)

EPS = 1e-4 # NOTE: used for TT

parser = configargparse.ArgumentParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    description='Compress the scene of SDFs with 4D TT decomposition')
parser.add('-c', '--config', is_config_file=True, help='config file path')


parser.add('--data_dir', required=True, type=str, help='Dirrectory with data')
parser.add('--experiment_name', required=True, type=str, help='Experiment name')
parser.add('--output_dir', required=True, type=str, help='Dirrectory for output')

parser.add('--model', required=True, type=str, help='Name of the model')
parser.add('--scene', required=True, type=str, help='Name of the scene')

parser.add('--decomposition', required=True, choices=['TT', 'QTT', 'OQTT','TT-Tucker'], help='TT, TT-Tucker, OQTT or QTT')
parser.add('--num_sample_points', required=True, type=int, help='Number of points to compute IoU')

parser.add_argument('--trunc_values', type=eval, help='SDF limits for a sweep')
parser.add_argument('--max_inner_ranks', type=int, nargs='+', help='Tucker ranks for a sweep')
parser.add_argument('--tt_ranks', type=int, nargs='+', help='TT ranks for a sweep')
parser.add_argument('--rank_multiplier', type=int, help='TT rank multiplier for stacked tensor')

args = parser.parse_args()

frames = []
for frame in sorted(os.listdir(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed'))):
    if frame.startswith('sdf'):
        frames.append(frame)

if len(frames) == 0:
    raise ValueError('SDF must be precomputed first')

result = {}

for min_tsdf, max_tsdf in args.trunc_values:
    print(f'tsdf limits ({min_tsdf}:{max_tsdf})')
    max_rank = max(args.max_inner_ranks)
    local_frames = []
    result[(min_tsdf, max_tsdf)] = {}
    result[(min_tsdf, max_tsdf)][max_rank] = {}

    for frame in tqdm.tqdm(frames):
        sdf = torch.load(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frame))['sdf']
        tsdf = sdf.clamp_min(min_tsdf).clamp_max(max_tsdf).double()
        if args.decomposition == 'TT':
            local_frames.append(tn.Tensor(tsdf, ranks_tt=max_rank))
        elif args.decomposition in ['Tucker', 'TT-Tucker']:
            local_frames.append(tn.Tensor(tsdf, ranks_tucker=max_rank))
        elif args.decomposition == 'QTT':
            local_frames.append(tn.Tensor(tensor3d2qtt(tsdf), ranks_tt=max_rank))
        elif args.decomposition == 'OQTT':
            local_frames.append(tn.Tensor(tensor3d2oqtt(tsdf), ranks_tt=max_rank))

    res = torch.tensor(local_frames[-1].shape)
    T = len(frames)
    largest_dim = max(res.max().item(), T)
    outer_rank = min(args.rank_multiplier * max_rank, largest_dim)

    if args.decomposition == 'TT-Tucker':
        res_decomp = reduce_tucker(
            [t[..., None] for t in local_frames],
            eps=EPS, rank=outer_rank, algorithm='eig')
        preprocessing_fn = lambda x, i: x[..., i].torch()
    elif args.decomposition == 'TT':
        res_decomp = reduce_tt(
            [t[..., None] for t in local_frames],
            eps=EPS, rank=outer_rank, algorithm='eig')
        preprocessing_fn = lambda x, i: x[..., i].torch()
    elif args.decomposition == 'QTT':
        res_decomp = qtt_stack(local_frames, eps=EPS, rank=outer_rank, algorithm='eig')
        preprocessing_fn = lambda x, i: qtt2tensor3d(get_qtt_frame(x, i).torch())
    elif args.decomposition == 'OQTT':
        res_decomp = qtt_stack(local_frames, N=1, eps=EPS, rank=args.rank_multiplier * outer_rank, algorithm='eig')
        preprocessing_fn = lambda x, i: oqtt2tensor3d(get_qtt_frame(x, i, N=1).torch())

    result[(min_tsdf, max_tsdf)][max_rank]['compressed_frames'] = local_frames
    result[(min_tsdf, max_tsdf)][max_rank]['compressed_scene'] = res_decomp
    local_res_decomp = res_decomp.clone()

    for rank in reversed(sorted(args.max_inner_ranks)):
        if rank not in result[(min_tsdf, max_tsdf)]:
            result[(min_tsdf, max_tsdf)][rank] = {}
        local_res_tt = local_res_decomp.clone()

        for tt_rank in reversed(sorted(args.tt_ranks)):
            ranks_tt = local_res_tt.ranks_tt.clone()
            ranks_tucker = local_res_tt.ranks_tucker.clone()
            if args.decomposition == 'TT-Tucker':
                assert len(ranks_tt) == 5 and len(ranks_tucker) == 4
                ranks_tucker[0:3] = min(largest_dim, rank)
                local_res_tt.round_tucker(rmax=ranks_tucker, algorithm='eig')
                ranks_tt[3] = min(largest_dim, tt_rank)
                local_res_tt.round_tt(rmax=ranks_tt[1:-1], algorithm='eig')
            elif args.decomposition == 'TT':
                assert len(ranks_tt) == 5
                ranks_tt[1:3] = min(largest_dim, rank)
                ranks_tt[3] = min(largest_dim, tt_rank)
                local_res_tt.round_tt(rmax=ranks_tt[1:-1], algorithm='eig')
            elif args.decomposition == 'QTT':
                # NOTE: ranks adjacent to time qtt dims
                idxs = torch.arange(len(ranks_tt))
                mask = torch.ones(len(ranks_tt)).bool()
                mask[idxs[3::4]] = False
                mask[idxs[4::4]] = False
                ranks_tt[3::4] = torch.where(ranks_tt[3::4] < tt_rank, ranks_tt[3::4], tt_rank)
                ranks_tt[4::4] = torch.where(ranks_tt[4::4] < tt_rank, ranks_tt[4::4], tt_rank)
                # NOTE: x, y, z qtt ranks
                ranks_tt[mask] = torch.where(ranks_tt[mask] < rank, ranks_tt[mask], rank)
                local_res_tt.round_tt(rmax=ranks_tt[1:-1], algorithm='eig')
            elif args.decomposition == 'OQTT':
                # NOTE: ranks adjacent to time oqtt dims
                idxs = torch.arange(len(ranks_tt))
                mask = torch.ones(len(ranks_tt)).bool()
                mask[idxs[1::2]] = False
                mask[idxs[2::2]] = False
                ranks_tt[1::2] = torch.where(ranks_tt[1::2] < tt_rank, ranks_tt[1::2], tt_rank)
                ranks_tt[2::2] = torch.where(ranks_tt[2::2] < tt_rank, ranks_tt[2::2], tt_rank)
                # NOTE: x, y, z qtt ranks
                ranks_tt[mask] = torch.where(ranks_tt[mask] < rank, ranks_tt[mask], rank)
                local_res_tt.round_tt(rmax=ranks_tt[1:-1], algorithm='eig')

            result[(min_tsdf, max_tsdf)][rank][tt_rank] = {
                'tensor': local_res_tt.clone(),
                'metrics': compute_metrics(
                    [(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frame),
                      osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frame[4:-2] + 'obj'))
                     for frame in frames],
                    local_res_tt,
                    preprocessing_fn,
                    min_tsdf, max_tsdf,
                    args.num_sample_points,
                    [0, len(frames) // 2, len(frames) - 1])} # NOTE: sample first, middle and last frames

os.makedirs(args.output_dir, exist_ok=True)
torch.save(result, osp.join(args.output_dir, f'{args.experiment_name}.pt'))
