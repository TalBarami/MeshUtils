import torch
from os import path as osp
import h5py
from pathlib import Path

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from mesh_utils.amass.data_preprocess import get_data_loaders, data_logger
from mesh_utils.utility.constants import Defaults, DataPaths, Splits


class PointCloudGenerator:
    def __init__(self, body_model_path=DataPaths.BODY_MODEL, num_betas=Defaults.BETAS, features=('root_orient', 'betas', 'pose_body', 'pose_hand'), smpl_like=True, num_dmpls=None, dmpl_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bm = BodyModel(bm_fname=body_model_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_path).to(self.device)
        self.faces = c2c(self.bm.f)
        self.num_verts = self.bm.init_v_template.shape[1]
        self.smpl_like = smpl_like
        self.features = features

    def forward(self, data_batch):
        if self.smpl_like and 'pose_hand' in self.features:
            data_batch['pose_hand'][3:15 * 3] = 0
            data_batch['pose_hand'][16 * 3:] = 0
        out = self.bm.forward(**{k: v.to(self.device) for k, v in data_batch.items() if k in self.features})
        return out.v, out.Jtr

    def __call__(self, data_batch):
        return self.forward(data_batch)


def generate_data(db_dir, out_path):
    data_logger.info(f'Generating H5 data file from {db_dir} to {out_path}')
    dims, (train, val, test) = get_data_loaders(db_dir, splits=Splits.SUGGESTED)
    keys = list(dims.keys()) + ['point_cloud', 'skeletons', 'skeleton_h']
    generator = PointCloudGenerator()
    Path(out_path).mkdir(parents=True, exist_ok=True)
    for name, dl in [('train', train), ('val', val), ('test', test)]:
        data_logger.info(f'Predicting for {name}')
        db = {k: torch.Tensor() for k in keys}
        for batch in dl:
            pcloud, Jtr = generator(batch)

            for k, v in batch.items():
                db[k] = torch.cat([db[k], v], dim=0)
            db['point_cloud'] = torch.cat([db['point_cloud'], pcloud.detach().cpu()], dim=0)
            skel_h = Jtr.detach().cpu()
            db['skeleton_h'] = torch.cat([db['skeleton_h'], skel_h], dim=0)
        db['skeletons'] = torch.cat([db['skeleton_h'][:, :23], db['skeleton_h'][:, 37:38]], dim=1)
        for k, v in db.items():
            db[k] = db[k].numpy()
        db['faces'] = generator.faces
        db['num_verts'] = generator.num_verts
        with h5py.File(osp.join(out_path, f'{name}.h5'), 'w') as f:
            data_logger.info(f'Saving {name}')
            for k, v in db.items():
                data_logger.info(f'Dataset {k}: {v.shape}')
                f.create_dataset(k, data=v)


if __name__ == '__main__':
    suggested_db = osp.join(DataPaths.AMASS_PROCESSED, 'amass_full')
    suggested_out = osp.join(DataPaths.POINT_CLOUDS_DB, 'smpl_like')
    generate_data(suggested_db, suggested_out)
