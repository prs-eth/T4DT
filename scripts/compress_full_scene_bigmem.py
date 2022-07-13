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
parser.add_argument('--tt_ranks', type=int, nargs='+', help='TT ranks for a sweep')

args = parser.parse_args()

frames = []
for frame in sorted(os.listdir(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed'))):
    if frame.startswith('sdf'):
        frames.append(frame)

if len(frames) == 0:
    raise ValueError('SDF must be precomputed first')

result = {}

tsdfs = torch.zeros((512, 512, 512, 284))

for min_tsdf, max_tsdf in args.trunc_values:
    print(f'tsdf limits ({min_tsdf}:{max_tsdf})')
    local_res = []

    for i, frame in tqdm.tqdm(enumerate(frames)):
        sdf = torch.load(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frame))['sdf']
        tsdfs[..., i] = sdf.clamp_min(min_tsdf).clamp_max(max_tsdf)

    for tt_rank in reversed(sorted(args.tt_ranks)):
        local_res_tt = tn.Tensor(tsdfs, ranks_tt=tt_rank)
        local_res[tt_rank] = local_res_tt
    result[(min_tsdf, max_tsdf)] = local_res

os.makedirs(args.output_dir, exist_ok=True)
torch.save(result, osp.join(args.output_dir, f'{args.experiment_name}.pt'))
