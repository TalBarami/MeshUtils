from os import path as osp

_support_data_root = '/mnt/DS_SHARED/users/talb/projects/amass/support_data'
_db_homedir = '/mnt/DS_SHARED/users/talb/data/AMASS/'


class DataPaths:
    AMASS_NPZ: str = osp.join(_support_data_root, 'amass_npz')
    AMASS_PROCESSED: str = osp.join(_support_data_root, 'prepared_data')
    BODY_MODEL: str = osp.join(_support_data_root, 'body_models/smplh/male/model.npz')
    POINT_CLOUDS_DB: str = osp.join(_db_homedir, 'h5')


class Defaults:
    BATCH_SIZE: int = 128
    BETAS: int = 16
    DMPLS: int = 8


class Splits:
    COMPACT: dict = {'vald': ['SFU', ], 'test': ['SSM_synced'], 'train': ['MPI_Limits']}
    SUGGESTED: dict = {
        'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT',
                  'BML', 'EKUT', 'TCD_handMocap', 'ACCAD']
    }
