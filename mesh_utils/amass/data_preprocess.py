import glob
import os

import subprocess

import torch
from os import path as osp
from torch.utils.data import Dataset, DataLoader
from amass.data.prepare_data import prepare_amass
from scipy.spatial.transform import Rotation

from mesh_utils.utility.constants import Defaults, DataPaths, Splits
from mesh_utils.utility.utils import init_logger

data_logger = init_logger('data_logger', log_path=r'/mnt/DS_SHARED/users/talb/projects/3d_avatar_generation')


def _unzip_files(npz_root):
    os.chdir(npz_root)
    for file in [f for f in os.listdir('uploaded') if f.endswith('bz2')]:
        name, _ = osp.splitext(file)
        subprocess.check_call(['tar', '-xvjf', f'uploaded/{file}'])

def _prepare_amass(splits,
                   db_out,
                   npz_path=DataPaths.AMASS_NPZ):
    if splits is None:
        splits = Splits.COMPACT
    splits['train'] = list(set(splits['train']).difference(set(splits['test'] + splits['vald'])))
    prepare_amass(splits, npz_path, db_out)


class AmassParametricDataset(Dataset):
    def __init__(self, dataset_dir, num_betas=Defaults.BETAS):
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt', '')
            self.ds[k] = torch.load(data_fname)
        self.num_betas = num_betas

    def __len__(self):
        return len(self.ds['trans'])

    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        data['pose'] = torch.Tensor(Rotation.from_rotvec(data['pose'].reshape(-1, 3)).as_rotvec().reshape(-1))
        data['root_orient'] = data['pose'][:3]
        data['pose_body'] = data['pose'][3:66]
        data['pose_hand'] = data['pose'][66:]
        data['betas'] = data['betas'][:self.num_betas]

        return data


def get_data_loaders(db_dir, batch_size=Defaults.BATCH_SIZE, num_betas=Defaults.BETAS, splits=None):
    _prepare_amass(splits, db_dir)
    data_logger.info(f'AMASS prepared successfully.')
    def to_dataset(s):
        data_logger.info(f'Dataset creation for {s}')
        data_dir = osp.join(db_dir, 'stage_III', s)
        ds = AmassParametricDataset(dataset_dir=data_dir, num_betas=num_betas)
        return ds

    def to_data_loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)

    datasets = [to_dataset(s) for s in ['train', 'vald', 'test']]
    dims = {k: v.shape for k, v in datasets[0][0].items()}
    data_loaders = [to_data_loader(ds) for ds in datasets]
    return dims, data_loaders
