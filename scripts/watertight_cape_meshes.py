import os
import os.path as osp
import subprocess
import json
import tqdm


DATASET_DIR = '/scratch2/data/cape_release/'
MANIFOLD_PLUS_PATH = '/scratch2/libs/ManifoldPlus/build/manifold'

processes = set()
max_processes = 10

models = {}

for model in tqdm.tqdm(os.listdir(osp.join(DATASET_DIR, 'meshes'))):
    print(model)
    scenes = {}
    for scene in tqdm.tqdm(os.listdir(osp.join(DATASET_DIR, 'meshes', model))):
        scenes[scene] = []
        for frame in tqdm.tqdm(os.listdir(osp.join(DATASET_DIR, 'meshes', model, scene, 'posed'))):
            input_name = osp.join(DATASET_DIR, 'meshes', model, scene, 'posed', frame)
            output_name = osp.join(DATASET_DIR, 'meshes', model, scene, 'posed', 'watertight_' + frame)
            if frame.startswith('watertight_') or osp.exists(output_name):
                scenes[scene].append(output_name)
            else:
                processes.add(subprocess.Popen([MANIFOLD_PLUS_PATH, '--input', input_name,  '--output', output_name,'--depth', '8'], stdout=subprocess.DEVNULL))
                if len(processes) >= max_processes:
                    os.wait()
                    processes.difference_update([
                        p for p in processes if p.poll() is not None])
                scenes[scene].append(frame)

        if scenes[scene]:
            models[model] = scenes

with open('../info/watertight_meshes.json', 'w') as f:
    json.dump(models, f)
