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
from t4dt.t4dt import reduce_tucker, tensor3d2qtt, qtt2tensor3d, qtt_stack, get_qtt_frame
from t4dt.metrics import compute_metrics

EPS = 1e-5 # NOTE: used for TT

parser = configargparse.ArgumentParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    description='Compress the scene of SDFs with 4D TT decomposition')
parser.add('-c', '--config', is_config_file=True, help='config file path')


parser.add('--data_dir', required=True, type=str, help='Dirrectory with data')
parser.add('--experiment_name', required=True, type=str, help='Experiment name')
parser.add('--output_dir', required=True, type=str, help='Dirrectory for output')

parser.add('--model', required=True, type=str, help='Name of the model')
parser.add('--scene', required=True, type=str, help='Name of the scene')

parser.add('--decomposition', required=True, choices=['TT', 'QTT', 'TT-Tucker'], help='TT, TT-Tucker or QTT')
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
    result[(min_tsdf, max_tsdf)] = {}

    for max_rank in reversed(sorted(args.max_inner_ranks)):
        result[(min_tsdf, max_tsdf)][max_rank] = {}
        local_res = []

        for frame in tqdm.tqdm(frames):
            sdf = torch.load(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frame))['sdf']
            tsdf = sdf.clamp_min(min_tsdf).clamp_max(max_tsdf)
            if args.decomposition == 'TT':
                local_res.append(tn.Tensor(tsdf, ranks_tt=max_rank))
            elif args.decomposition in ['Tucker', 'TT-Tucker']:
                local_res.append(tn.Tensor(tsdf, ranks_tucker=max_rank))
            elif args.decomposition == 'QTT':
                local_res.append(tn.Tensor(tensor3d2qtt(tsdf), ranks_tt=max_rank))

        if args.decomposition == 'TT-Tucker':
            local_res_decomp = reduce_tucker(
                [t[..., None] for t in local_res],
                eps=EPS, rank=args.rank_multiplier * max_rank, algorithm='svd')
            preprocessing_fn = lambda x, i: x[..., i].torch()
        elif args.decomposition == 'TT':
            raise NotImplementedError()
        elif args.decomposition == 'QTT':
            local_res_decomp = qtt_stack(local_res, eps=EPS, rank=args.rank_multiplier * max_rank)
            preprocessing_fn = lambda x, i: qtt2tensor3d(get_qtt_frame(x, i).torch())

        result[(min_tsdf, max_tsdf)][max_rank]['compressed_frames'] = local_res
        local_res_tt = local_res_decomp.clone()
        ranks_tt = local_res_tt.ranks_tt
        for tt_rank in reversed(sorted(args.tt_ranks)):
            ranks_tt[-2] = tt_rank
            local_res_tt.round_tt(rmax=ranks_tt[1:-1])
            result[(min_tsdf, max_tsdf)][max_rank][tt_rank] = {
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
