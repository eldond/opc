"""Provides various utility functions"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import copy

from op_specs import display

__all__ = ['make_unfold_fig', 'set_axes_equal', 'mirror', 'copy_point', 'unfold_figs', 'make_main_fig']

unfold_figs = []


def make_main_fig(which_figures):
    if 'assembled' in which_figures:
        # fig, axs = plt.subplots(2, 2)
        fig = plt.figure()
        axs = np.array(
            [
                [fig.add_subplot(221, projection='3d'), fig.add_subplot(222)],
                [fig.add_subplot(223), fig.add_subplot(224)],
            ]
        )
        axs[1, 1].set_xlabel('x')
        axs[1, 1].set_ylabel('y')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('z')
        axs[0, 1].set_xlabel('y')
        axs[0, 1].set_ylabel('z')
        for ax in axs.flatten():
            ax.set_aspect('equal', adjustable='box')
    else:
        fig = axs = None
    return fig, axs


def make_unfold_fig():
    if display['debug_unfold']:
        figf_ = plt.figure()
        axf = np.array(
            [
                [figf_.add_subplot(221, projection='3d'), figf_.add_subplot(222)],
                [figf_.add_subplot(223), figf_.add_subplot(224)],
            ]
        )
        axf[1, 1].set_xlabel('x')
        axf[1, 1].set_ylabel('y')
        axf[1, 0].set_xlabel('x')
        axf[1, 0].set_ylabel('z')
        axf[0, 1].set_xlabel('y')
        axf[0, 1].set_ylabel('z')
    else:
        figf_, axf = plt.subplots(1, squeeze=False)
        axf[0, 0].set_xlabel('x')
        axf[0, 0].set_ylabel('z')
    for ax_ in axf.flatten():
        ax_.set_aspect('equal', adjustable='box')

    unfold_figs.append(figf_)

    return axf


def set_axes_equal(ax3):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax3.set_aspect('equal') and ax3.axis('equal') not working for 3D.

    https://stackoverflow.com/a/31364297/6605826

    :param ax3: a Matplotlib axes instance with 3D projection
    """

    x_limits = ax3.get_xlim3d()
    y_limits = ax3.get_ylim3d()
    z_limits = ax3.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax3.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax3.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax3.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return ax3


def mirror(part, as_new=False):
    """
    Mirrors a part, either as a new part, or by completing half or a part.
    Vertices to be mirrored should be named right_* or left_*
    :param part: dict
    :param as_new: bool
    :return: dict
    """
    ind = [vv[3] for k, vv in part.items() if (not k.startswith('fold')) and (isinstance(vv, tuple))]
    ii = int(np.nanmax(ind)) + 1
    new_part = {}
    for k, vv in part.items():
        if k.startswith('fold'):
            if as_new:
                new_part[k] = vv
        elif k == 'unfold':
            if as_new:
                newv = copy.copy(vv)
                newv['xt'] = -newv.get('xt', 0)
                newv['xo'] = -newv.get('xo', 0)
                newv['zr'] = -newv.get('zr', 0)
                newv['yr'] = -newv.get('yr', 0)
                new_part[k] = newv
        elif ((not np.isnan(vv[3])) and (('left' in k) or ('right' in k))) or as_new:
            newk = k.replace('right', 'left')
            newv = (-vv[0], vv[1], vv[2], (vv[3] if as_new else 2 * ii - vv[3]))
            new_part[newk] = newv
    if as_new:
        return new_part
    else:
        part.update(new_part)
        return part


def copy_point(point, new_index):
    return point[0], point[1], point[2], new_index
