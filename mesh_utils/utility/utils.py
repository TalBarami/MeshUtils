import logging
import pickle
from os import path as osp
from pathlib import Path

import numpy as np


def init_logger(log_name, log_path=None):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_path is not None:
        fh = logging.FileHandler(osp.join(log_path, f'{log_name}.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info(f'Initialization Success: {log_name}')
    return logger

def init_directories(*dirs):
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)

def write_pkl(p, dst):
    with open(dst, 'wb') as f:
        pickle.dump(p, f)

def read_pkl(file):
    try:
        with open(file, 'rb') as p:
            return pickle.load(p)
    except Exception as e:
        print(f'Error while reading {file}: {e}')
        raise e

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: points- points and data, [N, D]
        npoint: number of points to samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point, centroids.astype(np.int32)