#!/bin/python3

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import interpolate
import copy

specs = dict(
    shoulder_width=40.0,
    chest_depth_at_mid=30.0,
    arm_hole_width=15.0,
    behind_arm_margin=5.0,
    above_arm_margin=5.0,
    prow_angle=10*np.pi/180.0,
    center_rib_cutout_width=25.0,
    center_rib_cutout_depth=5.0,
    center_rib_cutout_angle=45.0*np.pi/180.0,
    center_rib_cutout_bevel=5.0,
    front_slant=10.0*np.pi/180.0,
    chest_height=25.0,
    head_cutout_width=20.0,
    behind_head_margin=2.0,
    head_cutout_depth=20.0,
    head_round_depth=5.0,
    head_cutout_angle=10.0*np.pi/180.0,
    grill_half_width=10.0,
    grill_height=15.0,
    arm_top_cut_width=8.0,
    arm_top_cut_depth=15.0,
    arm_back_cover_margin=3.0,
    hand_hole_margins=dict(top=1.5, bottom=1.5, front=1.5, back=2.0),
    hand_hole_depth=12.0,
)

display = dict(
    arm_angle=10.0*np.pi/180.0,
    arm_spacex=3.0,
    arm_spacez=-3.0,
)

reference_images = dict(
    front_image='20190826_202031.jpg',
    front_image_height=115.0 + 22.0,
    front_image_center=(-8, -34+specs['chest_height']),
)

available_figures = ['assembled', 'unfolded_torso_back', 'unfolded_torso_extra', 'unfolded_right_arm']
which_figures = available_figures[3]

if 'assembled' in which_figures:
    # fig, axs = plt.subplots(2, 2)
    fig = plt.figure()
    axs = np.array([[
        fig.add_subplot(221, projection='3d'), fig.add_subplot(222)],
        [fig.add_subplot(223), fig.add_subplot(224)],
    ])

    for ax in axs.flatten():
        ax.set_aspect('equal', adjustable='box')
else:
    fig = axs = None


def make_unfold_fig():
    figf_ = plt.figure()
    axf = np.array([[
        figf_.add_subplot(221, projection='3d'), figf_.add_subplot(222)],
        [figf_.add_subplot(223), figf_.add_subplot(224)],
    ])
    for ax_ in axf.flatten():
        ax_.set_aspect('equal', adjustable='box')
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
    plot_radius = 0.5*max([x_range, y_range, z_range])

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
                new_part[k] = newv
        elif ((not np.isnan(vv[3])) and (('left' in k) or ('right' in k))) or as_new:
            newk = k.replace('right', 'left')
            newv = (-vv[0], vv[1], vv[2], (vv[3] if as_new else 2*ii-vv[3]))
            new_part[newk] = newv
    if as_new:
        return new_part
    else:
        part.update(new_part)
        return part


def copy_point(point, new_index):
    return point[0], point[1], point[2], new_index


# General shortcuts
rt = specs['shoulder_width']/2.0
bk = -specs['arm_hole_width']-specs['behind_arm_margin']

# Define front rib
front_margin = specs['chest_depth_at_mid']+bk
prow_depth = np.tan(specs['prow_angle'])*rt
cutout_bevel_x = specs['center_rib_cutout_bevel'] * np.cos(specs['center_rib_cutout_angle'])
cutout_bevel_y = specs['center_rib_cutout_bevel'] * np.sin(specs['center_rib_cutout_angle'])
cut_b1y = specs['center_rib_cutout_depth'] - cutout_bevel_y
cut_b1x = specs['center_rib_cutout_width']/2.0 - cutout_bevel_x
rib = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(xr=90, zt=-(front_margin+prow_depth+2)),
    front_center=(0, front_margin+prow_depth, 0, 0),
    front_right_corner=(rt, front_margin, 0, 1),
    outer_right_corner=(rt, 0, 0, 2),
    inner_right_corner=(specs['center_rib_cutout_width']/2.0, 0, 0, 3),
    cutout_right=(specs['center_rib_cutout_width']/2.0, cut_b1y, 0, 4),
    cutout_right_front=(cut_b1x, specs['center_rib_cutout_depth'], 0, 5),
)
mirror(rib)

# Define side panel
armhole_height = specs['chest_height'] - specs['above_arm_margin']
slant_dy = np.tan(specs['front_slant'])*specs['chest_height']
armx = rt + (specs['chest_height'] - specs['above_arm_margin'])*np.sin(display['arm_angle'])
armz = (specs['chest_height'] - specs['above_arm_margin'])*(1-np.cos(display['arm_angle']))
right_side = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(zr=90, xo=rt, yo=bk),
    bottom_front=copy_point(rib['front_right_corner'], 0),
    bottom_armpit_front=copy_point(rib['outer_right_corner'], 1),
    top_armpit_front=(rt, 0, armhole_height, 2),
    arm_end_front=(armx, 0, armz, 3),
    arm_end_back=(armx, -specs['arm_hole_width'], armz, 4),
    top_armpit_back=(rt, -specs['arm_hole_width'], armhole_height, 5),
    bottom_armpit_back=(rt, -specs['arm_hole_width'], 0, 6),
    back_bottom_corner=(rt, bk, 0, 7),
    back_top_corner=(rt, bk, specs['chest_height'], 8),
    front_top_corner=(rt, front_margin-slant_dy, specs['chest_height'], 9),
    fold_shoulder=(2, 5),
)
left_side = mirror(right_side, as_new=True)

# Define cab front / windshield
front_right = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(
        xo=0, yo=front_margin-slant_dy+prow_depth, zo=specs['chest_height'],
        zr=-specs['prow_angle']*180.0/np.pi,
        xr=-specs['front_slant']*180.0/np.pi,
        yt=-front_margin-slant_dy+prow_depth,
    ),
    center_top=(0, front_margin-slant_dy+prow_depth, specs['chest_height'], 0),
    right_top=copy_point(right_side['front_top_corner'], 1),
    right_bottom=copy_point(right_side['bottom_front'], 2),
    center_bottom=copy_point(rib['front_center'], 3),
    # fold1=(0, 3),
)
front_left = mirror(front_right, as_new=True)

# Define back
back = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(xr=0, yr=0, xt=0, yt=0, zt=0),
    right_inner_3=(specs['grill_half_width'], bk, -specs['grill_height'], -3),
    right_inner_2=(specs['grill_half_width'], bk, -specs['grill_height']+specs['arm_back_cover_margin'], -2),
    right_inner_1=(specs['grill_half_width']+specs['arm_top_cut_width'], bk, 0, -1),
    right_bottom=copy_point(right_side['back_bottom_corner'], 0),
    right_top=copy_point(right_side['back_top_corner'], 1),
)
mirror(back)
# back['fold1'] = (1, 7)
back['fold2'] = (back['right_inner_1'][-1], back['left_inner_1'][-1])

head_front = bk+specs['behind_head_margin']+specs['head_cutout_depth']
flapy = bk+specs['behind_head_margin'] + \
        (specs['head_cutout_depth']-specs['head_round_depth'])*np.cos(specs['head_cutout_angle'])
flapy2 = bk+specs['behind_head_margin'] + specs['head_cutout_depth']*np.cos(specs['head_cutout_angle'])
flapz = specs['chest_height']+(specs['head_cutout_depth']-specs['head_round_depth'])*np.sin(specs['head_cutout_angle'])
flapz2 = specs['chest_height']+specs['head_cutout_depth']*np.sin(specs['head_cutout_angle'])
head_cutout = dict(
    origin=(0, 0, 0, np.NaN),
    right_front_corner=(
        specs['head_cutout_width']/2.0-specs['head_round_depth'], head_front, specs['chest_height'], -1),
    right_corner=(specs['head_cutout_width']/2.0, head_front-specs['head_round_depth'], specs['chest_height'], 0),
    back_right_corner=(specs['head_cutout_width']/2.0, bk+specs['behind_head_margin'], specs['chest_height'], 1),
    flap_right_corner=(specs['head_cutout_width']/2.0, flapy, flapz, 2),
    flap_right_front_corner=(specs['head_cutout_width']/2.0-specs['head_round_depth'], flapy2, flapz2, 3),
)
mirror(head_cutout)
head_cutout['fold_close'] = (1, np.nanmax([v[3] if len(v) == 4 else 0 for v in head_cutout.values()])-2)

# Top
top = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(xr=90, yo=back['right_top'][1], zo=back['right_top'][2]),
    right_back=copy_point(right_side['back_top_corner'], 1),
    right_front=copy_point(right_side['front_top_corner'], 2),
    center_front=copy_point(front_right['center_top'], 3),
)
mirror(top)
head_cutout['unfold'] = top['unfold']

# Front grill thing
rib_back = [rib['cutout_right'], rib['cutout_right_front'], rib['cutout_left_front']]
rib_back_x = [rb[0] for rb in rib_back]
rib_back_y = [rb[1] for rb in rib_back]
if specs['grill_half_width'] > specs['center_rib_cutout_width']/2.0:
    grill_back = rib['inner_right_corner'][1]
else:
    grill_back = scipy.interpolate.interp1d(rib_back_x, rib_back_y, bounds_error=False, fill_value='extrapolate')(
        specs['grill_half_width'])
gy = front_right['center_bottom'][1] - specs['grill_half_width'] * np.tan(specs['prow_angle'])
grill = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(
        auto='z', cx=0, cy=grill_back, cz=-specs['grill_height']/2.0, zt=-specs['grill_height']+rib['unfold']['zt']),
    top_center=copy_point(front_right['center_bottom'], 0),
    top_right=(specs['grill_half_width'], gy, 0, 1),
    top_back_right=(specs['grill_half_width'], grill_back, 0, 2),
    bottom_back_right=(specs['grill_half_width'], grill_back, -specs['grill_height'], 3),
    bottom_right=(specs['grill_half_width'], gy, -specs['grill_height'], 4),
    bottom_center=(front_right['center_bottom'][0], front_right['center_bottom'][1], -specs['grill_height'], 5),
)
mirror(grill)
grill['fold_center'] = (grill['top_center'][-1], grill['bottom_center'][-1])
grill['fold_right'] = (grill['top_right'][-1], grill['bottom_right'][-1])
grill['fold_left'] = (grill['top_left'][-1], grill['bottom_left'][-1])

# Bottom of front grill
grill_bottom = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(xr=90, zo=-specs['grill_height'], zt=rib['unfold']['zt']),
)
grill_bottom.update({k: v for k, v in grill.items() if k.startswith('bottom_')})

# Arms
right_arm_origin = (display['arm_spacex'], 0, display['arm_spacez'], np.NaN)
ra_pivot_x = rib['front_right_corner'][0] + right_arm_origin[0]
ra_pivot_y = rib['front_right_corner'][1] + right_arm_origin[1]
ra_pivot_zt = rib['front_right_corner'][2] + right_arm_origin[2]
ra_pivot_zb = -specs['grill_height'] + right_arm_origin[2]

right_arm_top = dict(
    origin=right_arm_origin,
    front_right_corner=(copy_point(grill['top_right'], 0)),
    outer_right_corner=(copy_point(rib['front_right_corner'], 1)),
    outer_back_corner=(copy_point(back['right_bottom'], 2)),
    inner_back_corner=(grill['top_right'][0]+specs['arm_top_cut_width'], back['right_bottom'][1], 0, 3),
    inner_cutout_corner=(
        grill['top_right'][0]+specs['arm_top_cut_width'], back['right_bottom'][1]+specs['arm_top_cut_depth'], 0, 4,
    ),
    inner_front_corner=(grill['top_right'][0], rib['inner_right_corner'][1], 0, 5),
)

right_arm_bottom = dict(
    origin=right_arm_origin,
    front_right_corner=(grill['top_right'][0], grill['top_right'][1], -specs['grill_height'], 0),
    outer_right_corner=(rib['front_right_corner'][0], rib['front_right_corner'][1], -specs['grill_height'], 1),
    outer_back_corner=(back['right_bottom'][0], back['right_bottom'][1], -specs['grill_height'], 2),
    inner_back_corner=(grill['top_right'][0], back['right_bottom'][1], -specs['grill_height'], 3),
)

right_arm_outer = dict(
    origin=right_arm_origin,
    # front_right_top_corner=copy_point(right_arm_top['front_right_corner'], 0),
    outer_right_top_corner=copy_point(right_arm_top['outer_right_corner'], 1),
    outer_back_top_corner=copy_point(right_arm_top['outer_back_corner'], 2),
    outer_back_bottom_corner=(
        right_arm_top['outer_back_corner'][0], right_arm_top['outer_back_corner'][1], -specs['grill_height'], 3),
    outer_right_bottom_corner=(
        right_arm_top['outer_right_corner'][0], right_arm_top['outer_right_corner'][1], -specs['grill_height'], 4),
    # front_right_bottom_corner=(
    #     right_arm_top['front_right_corner'][0], right_arm_top['front_right_corner'][1], -specs['grill_height'], 5),
)
# right_arm_outer['fold1'] = (
#     right_arm_outer['outer_right_top_corner'][-1], right_arm_outer['outer_right_bottom_corner'][-1])

right_arm_front = dict(
    origin=right_arm_origin,
    unfold=dict(zr=90-specs['prow_angle']*180.0/np.pi, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zt),
    front_right_top_corner=copy_point(right_arm_top['front_right_corner'], 0),
    outer_right_top_corner=copy_point(right_arm_top['outer_right_corner'], 1),
    outer_right_bottom_corner=(
        right_arm_top['outer_right_corner'][0], right_arm_top['outer_right_corner'][1], -specs['grill_height'], 2),
    front_right_bottom_corner=(
        right_arm_top['front_right_corner'][0], right_arm_top['front_right_corner'][1], -specs['grill_height'], 3),
)

right_arm_back = dict(
    origin=right_arm_origin,
    top_inner=copy_point(right_arm_top['inner_back_corner'], 0),
    top_outer=copy_point(right_arm_top['outer_back_corner'], 1),
    bottom_outer=copy_point(right_arm_bottom['outer_back_corner'], 2),
    bottom_inner=copy_point(right_arm_bottom['inner_back_corner'], 3),
    inner=(
        right_arm_bottom['inner_back_corner'][0], right_arm_bottom['inner_back_corner'][1],
        specs['arm_back_cover_margin']-specs['grill_height'], 4,
    ),
)

right_arm_inner = dict(
    origin=right_arm_origin,
    inner_top=copy_point(right_arm_top['inner_front_corner'], 0),
    front_top=copy_point(right_arm_top['front_right_corner'], 1),
    front_bottom=copy_point(right_arm_bottom['front_right_corner'], 2),
    back_bottom=copy_point(right_arm_bottom['inner_back_corner'], 3),
    back_top=copy_point(right_arm_back['inner'], 4),
    inner=(right_arm_top['inner_front_corner'][0],
           right_arm_top['inner_cutout_corner'][1], right_arm_back['inner'][2], 5),
)

fr_y = right_arm_inner['front_top'][1]-specs['hand_hole_margins']['front']
bk_y = right_arm_inner['front_top'][1]-specs['hand_hole_margins']['front']-specs['hand_hole_depth']
tp_z = right_arm_inner['front_top'][2]-specs['hand_hole_margins']['top']
bt_z = right_arm_inner['front_bottom'][2]+specs['hand_hole_margins']['bottom']
tp_y = right_arm_inner['inner_top'][1]+specs['hand_hole_margins']['back']
raiit = right_arm_inner['inner_top']
raii = right_arm_inner['inner']
bk_z = tp_z - (raiit[2]-raii[2]) / (raiit[1]-raii[1]) * (tp_y-bk_y)

hhrx = right_arm_inner['front_top'][0]
right_arm_hand_cutout = dict(
    origin=right_arm_origin,
    top_front=(hhrx, fr_y, tp_z, 0),
    bottom_front=(hhrx, fr_y, bt_z, 1),
    bottom_back=(hhrx, bk_y, bt_z, 2),
    back_top=(hhrx, bk_y, bk_z, 3),
    top_back=(hhrx, tp_y, tp_z, 4),
)

left_arm_top = mirror(right_arm_top, as_new=True)
left_arm_bottom = mirror(right_arm_bottom, as_new=True)
left_arm_outer = mirror(right_arm_outer, as_new=True)
left_arm_front = mirror(right_arm_front, as_new=True)
left_arm_back = mirror(right_arm_back, as_new=True)
left_arm_inner = mirror(right_arm_inner, as_new=True)
left_arm_hand_cutout = mirror(right_arm_hand_cutout, as_new=True)


def plot_path(part, close=True, unfold=False, uidx=0):
    """
    Forms a path from the vertices defined for a part and plots it
    :param part: dict
    :param close: bool
    :param unfold: bool
    :param uidx: int
    """
    x, y, z, i = np.array([np.array(point) for k, point in part.items() if 'fold' not in k[0:6]]).T

    origin = x[np.isnan(i)], y[np.isnan(i)], z[np.isnan(i)]
    x = x[~np.isnan(i)] + origin[0]
    y = y[~np.isnan(i)] + origin[1]
    z = z[~np.isnan(i)] + origin[2]
    i = i[~np.isnan(i)]
    j = i.argsort()
    x = x[j]
    y = y[j]
    z = z[j]
    i = i[j]
    if close:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])
        i = np.append(i, -1000)

    if unfold:
        while (uidx+1) > len(axsf):
            axsf.append(None)
        if axsf[uidx] is None:
            axsf[uidx] = make_unfold_fig()
        x0 = copy.copy(x)
        y0 = copy.copy(y)
        z0 = copy.copy(z)
        auto_unfold = part.get('unfold', {}).get('auto', None)
        # Translation along all axes
        xt = part.get('unfold', {}).get('xt', 0)
        yt = part.get('unfold', {}).get('yt', 0)
        zt = part.get('unfold', {}).get('zt', 0)
        # Center of part for auto-unfolds
        cx = part.get('unfold', {}).get('cx', np.mean(x))
        cy = part.get('unfold', {}).get('cy', np.mean(y))
        cz = part.get('unfold', {}).get('cz', np.mean(z))
        if auto_unfold is None:
            # Rotation about X and Z axes
            xr = part.get('unfold', {}).get('xr', 0) * np.pi/180.0
            zr = part.get('unfold', {}).get('zr', 0) * np.pi/180.0
            # Origin of rotation
            xo = part.get('unfold', {}).get('xo', 0)
            yo = part.get('unfold', {}).get('yo', 0)
            zo = part.get('unfold', {}).get('zo', 0)
            # Rotate about x axis
            y = (y0-yo)*np.cos(xr)+yo - (z0-zo)*np.sin(xr)
            z = (z0-zo)*np.cos(xr)+zo + (y0-yo)*np.sin(xr)
            # Rotate about z axis
            y00 = copy.copy(y)
            x = (x0-xo)*np.cos(zr)+xo + (y00-yo)*np.sin(zr)
            y = (y00-yo)*np.cos(zr)+yo - (x0-xo)*np.sin(zr)
        if auto_unfold == 'z':
            # Automatic unfold while leaving z alone (flatten x-y)
            theta = np.arctan2(y-cy, x-cx) - np.pi
            dth = np.diff(theta)
            sdth = np.sign(dth)
            dx = np.diff(x)
            dy = np.diff(y)
            ds = np.sqrt(dx**2+dy**2) * sdth

            x = np.cumsum(np.append(0, ds))
            y = x*0

        # Translate
        x += xt
        y += yt
        z += zt
        axu = axsf[uidx]
    else:
        axu = axs

    p11 = axu[1, 1].plot(x, y)
    p10 = axu[1, 0].plot(x, z)
    p01 = axu[0, 1].plot(y, z)
    p00 = axu[0, 0].plot(x, y, z)

    folds = [k for k in part if k.startswith('fold')]
    for fold in folds:
        f = part[fold]
        xf = np.append(x[i == f[0]], x[i == f[1]])
        yf = np.append(y[i == f[0]], y[i == f[1]])
        zf = np.append(
            z[i == f[0]], z[i == f[1]])

        axu[1, 1].plot(xf, yf, linestyle='--', color=p11[0].get_color())
        axu[1, 0].plot(xf, zf, linestyle='--', color=p10[0].get_color())
        axu[0, 1].plot(yf, zf, linestyle='--', color=p01[0].get_color())
        axu[0, 0].plot(xf, yf, zf, linestyle='--', color=p00[0].get_color())
    return


def plot_images():
    """Plots reference images behind the wireframe"""
    # Get the pictures
    front_pic = mpl.image.imread(reference_images['front_image'])
    front_pic = np.swapaxes(front_pic, 0, 1)

    # Get dimensions
    fh = reference_images['front_image_height']
    fw = fh * np.shape(front_pic)[1] / np.shape(front_pic)[0]
    fcx = float(reference_images['front_image_center'][0])
    fcy = float(reference_images['front_image_center'][1])
    axs[1, 0].imshow(front_pic, extent=(
        fcx - fw / 2.0, fcx + fw / 2.0,
        fcy - fh / 2.0, fcy + fh / 2.0,
    ))
    return


def plot_unfolded(part, uidx=0):
    plot_path(part, unfold=True, uidx=uidx)
    return


if 'assembled' in which_figures:
    plot_images()

    # Torso
    plot_path(rib)
    plot_path(right_side)
    plot_path(left_side)
    plot_path(top)
    plot_path(front_right)
    plot_path(front_left)
    plot_path(back)
    plot_path(head_cutout, close=True)
    plot_path(grill)
    plot_path(grill_bottom)

    # Arms
    plot_path(right_arm_top)
    plot_path(right_arm_bottom)
    plot_path(right_arm_outer)
    plot_path(right_arm_front)
    plot_path(right_arm_back)
    plot_path(right_arm_inner)
    plot_path(right_arm_hand_cutout)

    plot_path(left_arm_top)
    plot_path(left_arm_bottom)
    plot_path(left_arm_outer)
    plot_path(left_arm_front)
    plot_path(left_arm_back)
    plot_path(left_arm_inner)
    plot_path(left_arm_hand_cutout)

    set_axes_equal(axs[0, 0])

# Unfolded
axsf = []
# Torso
if 'unfolded_torso_back' in which_figures:
    plot_unfolded(back)
    plot_unfolded(top)
    plot_unfolded(head_cutout)
    plot_unfolded(right_side)
    plot_unfolded(left_side)
if 'unfolded_torso_extra' in which_figures:
    plot_unfolded(front_right, 1)
    plot_unfolded(front_left, 1)
    plot_unfolded(rib, 1)
    plot_unfolded(grill_bottom, 1)
    plot_unfolded(grill, 1)

# Arms
if 'right_arm' in which_figures:
    plot_unfolded(right_arm_outer, 2)
    plot_unfolded(right_arm_front, 2)


for axs_ in axsf:
    if axs_ is not None:
        set_axes_equal(axs_[0, 0])
plt.show()
