import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from utility.skeleton_layout import SkeletonLayout


def get_cmap_string(palette, domain):
    domain_unique = np.unique(domain)
    hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
    mpl_cmap = cm.get_cmap(palette, lut=len(domain_unique))

    def cmap_out(X, **kwargs):
        return mpl_cmap(hash_table[X], **kwargs)

    return cmap_out

def plot_reconstruction_compare(pc, org_segments, predicted_segments, skeleton, reconstructed_skeleton):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(221, projection='3d')
    plot_3d_point_cloud(pc.T[0], pc.T[1], pc.T[2], org_segments, axis=ax, show=False, title='Manual Segments')
    ax = fig.add_subplot(222, projection='3d')
    plot_3d_point_cloud(pc.T[0], pc.T[1], pc.T[2], predicted_segments, axis=ax, show=False, title='Predicted Segments')
    ax = fig.add_subplot(223, projection='3d')
    plot_3d_point_cloud(skeleton.T[0], skeleton.T[1], skeleton.T[2], axis=ax, show=False, title='Ground Truth Skeleton')
    ax = fig.add_subplot(224, projection='3d')
    plot_3d_point_cloud(reconstructed_skeleton.T[0], reconstructed_skeleton.T[1], reconstructed_skeleton.T[2], axis=ax, show=True, title='Skeleton Reconstruction')


def plot_3d_point_cloud(x, y, z, segments=None, show=True, show_axis=True, in_u_sphere=False, marker='.', s=8, alpha=.8,
                        figsize=(12, 12), elev=10, azim=240, axis=None, title=None, *args, **kwargs):
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    if segments is None:
        c = None
    else:
        cmap = get_cmap_string(palette='tab20', domain=segments)
        c = [cmap(s) for s in segments]
    sc = ax.scatter(x, y, z, marker=marker, s=s, c=c, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        #         miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
        #         mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        miv, mav = -0.8, 0.8
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if len(x) < 60:
        for i in range(len(x)):
            ax.text(x[i], y[i], z[i], i)
        for s, t in SkeletonLayout().pairs():
            if np.sum([x[s], y[s], z[s]]) != 0 and np.sum([x[t], y[t], z[t]]) != 0:
                ax.plot([x[s], x[t]], [y[s], y[t]], zs=[z[s], z[t]])

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig