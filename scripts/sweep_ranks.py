import tntorch as tn
import torch
import configargparse
import os
import os.path as osp
import sys
import tqdm
import tntorch as tn

sys.path.append(osp.join(os.path.abspath(os.getcwd()), '..',))
from t4dt.t4dt import reduce_tucker
from t4dt.metrics import compute_metrics

EPS = 1e-9 # NOTE: used for TT

parser = configargparse.ArgumentParser(description='Compress the scene of SDFs with 4D TT decomposition')
parser.add('-c', '--config', is_config_file=True, help='config file path')

parser.add('--data_dir', required=True, type=str, help='Dirrectory with data')
parser.add('--experiment_name', required=True, type=str, help='Experiment name')
parser.add('--output_dir', required=True, type=str, help='Dirrectory for output')

parser.add('--model', required=True, type=str, help='Name of the model')
parser.add('--scene', required=True, type=str, help='Name of the scene')

parser.add('--decomposition', required=True, choices=['TT', 'QTT', 'TT-Tucker'], help='TT, TT-Tucker or QTT')
parser.add('--num_sample_points', required=True, type=int, help='Number of points to compute IoU')

args = parser.parse_args()

frames = []
for frame in sorted(os.listdir(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed'))):
    if frame.startswith('sdf'):
        frames.append(frame)

if len(frames) == 0:
    raise ValueError('SDF must be precomputed first')

result = {}

for min_tsdf, max_tsdf in [(-0.5, 0.5), (-0.05, 0.05), (-0.01, 0.01), (-0.005, 0.005)]:
    result[(min_tsdf, max_tsdf)] = {}

    for tucker_rank in [20, 50, 100]:
        local_res = []

        for frame in tqdm.tqdm(frames):
            sdf = torch.load(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frame))['sdf']
            tsdf = sdf.clamp_min(min_tsdf).clamp_max(max_tsdf)
            if args.decomposition == 'TT':
                local_res.append(tn.Tensor(tsdf, ranks_tt=tucker_rank))
            elif args.decomposition in ['Tucker', 'TT-Tucker']:
                local_res.append(tn.Tensor(tsdf, ranks_tucker=tucker_rank))
            elif args.decomposition == 'QTT':
                raise NotImplementedError()

        if args.decomposition == 'TT-Tucker':
            local_res_tucker = reduce_tucker(
                [t[..., None] for t in local_res],
                eps=EPS, rmax=tucker_rank, algorithm='svd')
        elif args.decomposition == 'TT':
            raise NotImplementedError()
        elif args.decomposition == 'QTT':
            raise NotImplementedError()

        local_res_tt = local_res_tucker.clone()
        for tt_rank in [1024, 512, 128]:
            local_res_tt.round_tt(rmax=tt_rank)
            result[(min_tsdf, max_tsdf)][tucker_rank][tt_rank] = {
                'tensor': local_res_tt,
                'metrics': compute_metrics(
                    [(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frame),
                      osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frame[4:-2] + 'obj'))
                     for frame in frames],
                    local_res_tt,
                    min_tsdf, max_tsdf,
                    args.num_sample_points,
                    [0, len(frames) // 2, len(frames) - 1])} # NOTE: sample first, middle and last frames


os.makedirs(args.output_dir, exist_ok=True)
torch.save(result, osp.join(args.output_dir, f'{args.experiment_name}.pt'))
