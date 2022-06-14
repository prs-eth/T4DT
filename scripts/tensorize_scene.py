import tntorch as tn
import torch
import configargparse
import os
import os.path as osp
import tqdm

MIN_TSDF = -1.
MAX_TSDF = 1.

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Compress the scene of SDFs with 4D TT decomposition')
    parser.add('-c', '--config', is_config_file=True, help='config file path')

    parser.add('--data_dir', required=True, type=str, help='Dirrectory with data')
    parser.add('--experiment_name', required=True, type=str, help='Experiment name')
    parser.add('--output_dir', required=True, type=str, help='Dirrectory for output')

    parser.add('--model', required=True, type=str, help='Name of the model')
    parser.add('--scene', required=True, type=str, help='Name of the scene')

    parser.add('--resolution', required=True, type=int, help='Grid resolution')
    parser.add('--max_rank', required=True, type=int, help='Maximum rank')
    parser.add('--decomposition', required=True, choices=['TT', 'Tucker', 'QTT'], help='TT, Tucker or QTT')

    args = parser.parse_args()

    frames = []
    for frame in sorted(os.listdir(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed'))):
        if frame.startswith('sdf'):
            frames.append(frame)

    if len(frames) == 0:
        raise ValueError('SDF must be precomputed first')

    result = []

    error = 0
    for frame in tqdm.tqdm(frames):
        sdf = torch.load(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frame))['sdf']
        tsdf = sdf.clamp_min(MIN_TSDF)
        tsdf = sdf.clamp_max(MAX_TSDF)
        if args.decomposition == 'TT':
            result.append(tn.Tensor(tsdf, ranks_tt=args.max_rank))
        elif args.decomposition == 'Tucker':
            result.append(tn.Tensor(tsdf, ranks_tucker=args.max_rank))
        # error += torch.norm(tsdf - result[-1].torch())
        # tqdm.tqdm.write(f'Error: {error.item() / (i + 1)}')

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(result, osp.join(args.output_dir, args.experiment_name + '.pt'))
