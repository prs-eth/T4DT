import os
import sys
import os.path as osp

sys.path.append('../')

DATASET_DIR = '/scratch2/data/cape_release/'
MESH_LIB = 'trimesh'

from cape_utils import dataset_utils as du

cape = du.CAPE_utils(MESH_LIB, DATASET_DIR)
models = {}

for fname in os.listdir(osp.join(DATASET_DIR, 'seq_lists')):
    if fname.startswith('seq_list'):
        model = fname.split('_')[-1][:-4] # Note: removed .txt
        models[model] = set()

        with open(osp.join(DATASET_DIR, 'seq_lists', fname), 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            while f:
                seq_name = f.readline().strip().split()[0]
                models[model].add(seq_name)

                cape.extract_mesh_seq(model, seq_name, option='posed')
