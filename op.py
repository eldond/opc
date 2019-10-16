#!/bin/python3

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import interpolate
import copy

np.seterr(all='raise')

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
    grill_half_width=10.0,
    grill_height=15.0,
    arm_top_cut_width=8.0,
    arm_top_cut_depth=15.0,
    arm_back_cover_margin=3.0,
    hand_hole_margins=dict(top=2.0, bottom=2.0, front=2.0, back=2.0),
    hand_hole_depth=12.0,
    window_margins=dict(top=3.0, bottom=3.0, outer=3.0, inner=2.0),
    internal_support_back_depth=7.0,
    internal_support_top_height=2.0,
    internal_support_front_margin=1.0,
    internal_support_back_vert_extent=15.0,
    internal_support_back_taper_extent=10.0,
    internal_support_curve_height=10.0,
    internal_support_top_cut_angle=150.0*np.pi/180.0,
    neck_guard_height=4.0,
)

display = dict(
    head_cutout_angle=0.0,  # 10.0*np.pi/180.0,
    arm_angle=0.0,  # 10.0*np.pi/180.0,
    arm_spacex=3.0,
    arm_spacez=-3.0,
    all_unfold_dx=[0, 75.0, 55.0, 125.0],
    all_unfold_dy=[0, 0.0, 80.0, 80.0],
    debug_unfold=False,
    mark_points=True,
)

reference_images = dict(
    front_image='20190826_202031.jpg',
    front_image_height=115.0 + 22.0,
    front_image_center=(-8, -34+specs['chest_height']),
)

available_figures = [
    'assembled', 'unfolded_torso_back', 'unfolded_torso_extra', 'unfolded_right_arm', 'unfolded_left_arm',
]
which_figures = available_figures[0:3]
mpl.rcParams['figure.figsize'] = [10.75, 8.25]

if 'assembled' in which_figures:
    # fig, axs = plt.subplots(2, 2)
    fig = plt.figure()
    axs = np.array([[
        fig.add_subplot(221, projection='3d'), fig.add_subplot(222)],
        [fig.add_subplot(223), fig.add_subplot(224)],
    ])
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

if [wf for wf in which_figures if 'unfolded' in wf]:
    figaf, axaf = plt.subplots(1)
    axaf.set_aspect('equal', adjustable='box')
else:
    figaf = axaf = None

unfold_figs = []


def make_unfold_fig():
    if display['debug_unfold']:
        figf_ = plt.figure()
        axf = np.array([[
            figf_.add_subplot(221, projection='3d'), figf_.add_subplot(222)],
            [figf_.add_subplot(223), figf_.add_subplot(224)],
        ])
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
                newv['yr'] = -newv.get('yr', 0)
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

wdz = np.cos(specs['front_slant'])
wds = np.sin(specs['front_slant'])
wdx = np.cos(specs['prow_angle'])
window_top = front_right['center_top'][2] - specs['window_margins']['top']*wdz
window_top2 = window_top - 7*wdz
window_bottom = front_right['center_bottom'][2] + specs['window_margins']['bottom']*wdz
window_inner_x = 0 + specs['window_margins']['inner']*wdx
window_inner_x2 = window_inner_x + 3*wdx
window_inner_x3 = window_inner_x + 2*wdx
window_outer_x = front_right['right_top'][0] - specs['window_margins']['outer']*wdx


def rwy(wx, wz):
    """Gives y coordinate for (x, z) on the front right window"""
    xct, yct, zct = front_right['center_top'][0:3]
    dx = wx-xct
    dz = wz-zct
    dy = -dz * np.sin(specs['front_slant']) - dx * np.sin(specs['prow_angle'])
    return yct + dy


right_window = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=front_right['unfold'],
    logo_bottom_inner=(window_inner_x, 0, window_top2, -2),
    logo_bottom_outer=(window_inner_x3, 0, window_top2, -1),
    inner_top=(window_inner_x2, 0, window_top, 0),
    outer_top=(window_outer_x, 0, window_top, 1),
    outer_bottom=(window_outer_x, 0, window_bottom, 2),
    inner_bottom=(window_inner_x+wdx, 0, window_bottom, 3),
)
for pt in right_window:
    if pt not in ['origin', 'unfold']:
        v = right_window[pt]
        newy = rwy(v[0], v[2])
        right_window[pt] = (v[0], newy, v[2], v[3])

left_window = mirror(right_window, as_new=True)

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
        (specs['head_cutout_depth']-specs['head_round_depth'])*np.cos(display['head_cutout_angle'])
flapy2 = bk+specs['behind_head_margin'] + specs['head_cutout_depth']*np.cos(display['head_cutout_angle'])
flapz = specs['chest_height']+(specs['head_cutout_depth'] -
                               specs['head_round_depth'])*np.sin(display['head_cutout_angle'])
flapz2 = specs['chest_height']+specs['head_cutout_depth']*np.sin(display['head_cutout_angle'])
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
def y_along_rib_back(xval):
    rib_back = [rib['cutout_right'], rib['cutout_right_front'], rib['cutout_left_front']]
    rib_back_x = [rb[0] for rb in rib_back]
    rib_back_y = [rb[1] for rb in rib_back]
    if xval > specs['center_rib_cutout_width'] / 2.0:
        yval = rib['inner_right_corner'][1]
    else:
        yval = scipy.interpolate.interp1d(rib_back_x, rib_back_y, bounds_error=False, fill_value='extrapolate')(xval)
    return yval


# if specs['grill_half_width'] > specs['center_rib_cutout_width']/2.0:
#     grill_back = rib['inner_right_corner'][1]
# else:
#     grill_back = scipy.interpolate.interp1d(rib_back_x, rib_back_y, bounds_error=False, fill_value='extrapolate')(
#         specs['grill_half_width'])
grill_back = y_along_rib_back(specs['grill_half_width'])
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

# Internal supports
isrx = specs['head_cutout_width']/2.0
isr_curve_front = head_front-specs['internal_support_front_margin']
isr_curve_dy = (isr_curve_front - (bk+specs['internal_support_back_depth']))/2.0
isr_curve_dz = specs['internal_support_curve_height']
isr_curve_bottom = specs['chest_height'] - specs['internal_support_top_height'] - isr_curve_dz
nseg = 16
th = np.arange(nseg+1)/float(nseg) * np.pi
isr_curve_y = isr_curve_front - isr_curve_dy * (1-np.cos(th))
isr_curve_z = isr_curve_bottom + isr_curve_dz * np.sin(th)

internal_support_right = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(zr=90, xt=-35, zt=-40),
    back_inner_bottom=(
        isrx, bk+specs['internal_support_back_depth'],
        specs['chest_height']-specs['internal_support_back_vert_extent'], 0,
    ),
    back_taper_bottom=(
        isrx, bk,
        specs['chest_height']-specs['internal_support_back_vert_extent']-specs['internal_support_back_taper_extent'],
        1,
    ),
    back_top=(isrx, bk, specs['chest_height'], 2),
    front_top=(isrx, rwy(isrx, specs['chest_height']), specs['chest_height'], 3),
    front_bottom=(isrx, rwy(isrx, 0), 0, 4),
    front_inner_bottom=(isrx, y_along_rib_back(isrx), 0, 5),
    front_inner=(isrx, isr_curve_front, specs['chest_height']-specs['internal_support_back_vert_extent'], 6),
)
for k in range(nseg+1):
    internal_support_right['curve_{}'.format(k)] = (isrx, isr_curve_y[k], isr_curve_z[k], 7+k)

internal_support_left = mirror(internal_support_right, as_new=True)

internal_support_top = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(auto='x', zt=60, cz=isr_curve_bottom, cy=isr_curve_front-isr_curve_dy, cx=0),
)
path = ['front_inner'] + [k for k in internal_support_right if k.startswith('curve_')] + ['back_inner_bottom']
for k, pt in enumerate(path[::-1]):
    internal_support_top['curve_outer_right_{}'.format(nseg+1+1-k)] = (
        rt, internal_support_right[pt][1], internal_support_right[pt][2], k)
cut_back = (-1+np.cos(specs['internal_support_top_cut_angle']))*isr_curve_dy + isr_curve_front
for k, pt in enumerate(path):

    if internal_support_right[pt][1] >= cut_back:
        internal_support_top['curve_inner_right_{}'.format(k)] = (
            isrx, internal_support_right[pt][1], internal_support_right[pt][2], k+nseg+2)

mirror(internal_support_top)
# Mark folds on internal support top
mark_fold_every = 2
for k in range(len(path)):
    if k % mark_fold_every == 0:
        if 'curve_inner_right_{}'.format(k) in internal_support_top:
            new_idx = internal_support_top['curve_inner_right_{}'.format(k)][3]
            internal_support_top['fold_{}_L'.format(k)] = (
                internal_support_top['curve_outer_left_{}'.format(k)][3],
                internal_support_top['curve_inner_left_{}'.format(k)][3],
            )
        else:
            new_idx = internal_support_top['curve_outer_left_{}'.format(k)][3]
        internal_support_top['fold_{}'.format(k)] = (
            internal_support_top['curve_outer_right_{}'.format(k)][3], new_idx)

# Neck guard
neck_guard = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(auto='zchain', xt=50),
    right_top_front_corner=copy_point(head_cutout['right_front_corner'], 0),
    right_top_corner=copy_point(head_cutout['right_corner'], 1),
    right_top_outer_support=(specs['head_cutout_width']/2.0, head_front, specs['chest_height'], 2),
)
ngt_pts = [pt for pt in neck_guard if '_top_' in pt]
for pt in ngt_pts:
    a = neck_guard[pt]
    neck_guard[pt.replace('_top_', '_bottom_')] = (a[0], a[1], a[2]-specs['neck_guard_height'], 2*len(ngt_pts)-1-a[3])
mirror(neck_guard)
ngt_pts = [pt for pt in neck_guard if '_top_' in pt]
for pt in ngt_pts:
    neck_guard['fold_' + pt.replace('_top_', '_')] = (neck_guard[pt][3], neck_guard[pt.replace('_top_', '_bottom_')][3])

neck_guard_front_tab = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(xt=50-head_cutout['right_front_corner'][0], xr=90, zo=specs['chest_height'], yo=head_front, xo=0),
    right_back_corner=(specs['head_cutout_width']/2.0-specs['head_round_depth'], head_front, specs['chest_height'], 0),
    right_front_corner=(
        specs['head_cutout_width']/2.0-specs['head_round_depth'], head_front+3, specs['chest_height'], 1),
)
mirror(neck_guard_front_tab)

ngrt_dx = 3*np.sqrt(2)/2.0
ngrt_dy = 3*np.sqrt(2)/2.0
neck_guard_right_tab = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=copy.deepcopy(neck_guard_front_tab['unfold']),
    inner=(specs['head_cutout_width']/2.0-specs['head_round_depth'], head_front, specs['chest_height'], 0),
    back=(specs['head_cutout_width']/2.0, head_front-specs['head_round_depth'], specs['chest_height'], 1),
    outer=(specs['head_cutout_width']/2.0+ngrt_dx, head_front-specs['head_round_depth']+ngrt_dy, specs['chest_height'], 2),
    front=(specs['head_cutout_width']/2.0-specs['head_round_depth']+ngrt_dx, head_front+ngrt_dy, specs['chest_height'], 3),
)
neck_guard_right_tab['unfold']['yr'] = 45
neck_guard_right_tab['unfold']['xo'] = specs['head_cutout_width']/2.0-specs['head_round_depth']
neck_guard_left_tab = mirror(neck_guard_right_tab, as_new=True)
neck_guard_left_tab['unfold']['xt'] = neck_guard_right_tab['unfold']['xt']

neck_back_guard = dict(
    origin=(0, 0, 0, np.NaN),
    unfold=dict(auto='x', xt=45, zt=+10),
    top_right_tab=(
        specs['head_cutout_width']/2.0, bk+np.max([specs['behind_head_margin']-3, 0]), specs['chest_height'], 0
    ),
    top_right=(specs['head_cutout_width']/2.0, bk+specs['behind_head_margin'], specs['chest_height'], 1),
    bottom_right=(
        specs['head_cutout_width']/2.0, bk+specs['behind_head_margin'],
        specs['chest_height']-specs['neck_guard_height'], 2),
)
mirror(neck_back_guard)
neck_back_guard['fold'] = (neck_back_guard['top_right'][-1], neck_back_guard['top_left'][-1])

# Arms
right_arm_origin = (display['arm_spacex'], 0, display['arm_spacez'], np.NaN)
ra_pivot_x = rib['front_right_corner'][0] + right_arm_origin[0]
ra_pivot_y = rib['front_right_corner'][1] + right_arm_origin[1]
ra_pivot_zt = rib['front_right_corner'][2] + right_arm_origin[2]
ra_pivot_zb = -specs['grill_height'] + right_arm_origin[2]

right_arm_top = dict(
    origin=right_arm_origin,
    unfold=dict(zr=90, yr=-90, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zt),
    front_right_corner=(copy_point(grill['top_right'], 0)),
    outer_right_corner=(copy_point(rib['front_right_corner'], 1)),
    outer_back_corner=(copy_point(back['right_bottom'], 2)),
    inner_back_corner=(grill['top_right'][0]+specs['arm_top_cut_width'], back['right_bottom'][1], 0, 3),
    inner_cutout_corner=(
        grill['top_right'][0]+specs['arm_top_cut_width'], back['right_bottom'][1]+specs['arm_top_cut_depth'], 0, 4,
    ),
    inner_front_corner=(grill['top_right'][0], rib['inner_right_corner'][1], 0, 5),
)
# xyzo = right_arm_top['outer_right_corner']
# right_arm_top['unfold'] = dict(zr=90, yr=-90, xo=xyzo[0], yo=xyzo[1], zo=xyzo[2], yt=-xyzo[1]-right_arm_origin[0])

right_arm_bottom = dict(
    origin=right_arm_origin,
    unfold=dict(zr=90, yr=90, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zb),
    front_right_corner=(grill['top_right'][0], grill['top_right'][1], -specs['grill_height'], 0),
    outer_right_corner=(rib['front_right_corner'][0], rib['front_right_corner'][1], -specs['grill_height'], 1),
    outer_back_corner=(back['right_bottom'][0], back['right_bottom'][1], -specs['grill_height'], 2),
    inner_back_corner=(grill['top_right'][0], back['right_bottom'][1], -specs['grill_height'], 3),
)

right_arm_outer = dict(
    origin=right_arm_origin,
    unfold=dict(zr=90, xr=0, yr=0, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zt),
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
    unfold=dict(zr=180-specs['prow_angle']*180.0/np.pi, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zt),
    front_right_top_corner=copy_point(right_arm_top['front_right_corner'], 0),
    outer_right_top_corner=copy_point(right_arm_top['outer_right_corner'], 1),
    outer_right_bottom_corner=(
        right_arm_top['outer_right_corner'][0], right_arm_top['outer_right_corner'][1], -specs['grill_height'], 2),
    front_right_bottom_corner=(
        right_arm_top['front_right_corner'][0], right_arm_top['front_right_corner'][1], -specs['grill_height'], 3),
)

arm_outer_len = right_arm_top['outer_back_corner'][1]-right_arm_top['outer_right_corner'][1]
right_arm_back = dict(
    origin=right_arm_origin,
    unfold=dict(xt=arm_outer_len, yt=-arm_outer_len),
    top_inner=copy_point(right_arm_top['inner_back_corner'], 0),
    top_outer=copy_point(right_arm_top['outer_back_corner'], 1),
    bottom_outer=copy_point(right_arm_bottom['outer_back_corner'], 2),
    bottom_inner=copy_point(right_arm_bottom['inner_back_corner'], 3),
    inner=(
        right_arm_bottom['inner_back_corner'][0], right_arm_bottom['inner_back_corner'][1],
        specs['arm_back_cover_margin']-specs['grill_height'], 4,
    ),
)

arm_width = right_arm_bottom['outer_back_corner'][0]-right_arm_bottom['inner_back_corner'][0]
right_arm_inner = dict(
    origin=right_arm_origin,
    unfold=dict(yr=180, zr=90, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zb, zt=-arm_width, yt=arm_width),
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
    unfold=right_arm_inner['unfold'],
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


def plot_path(part, close=True, unfold=False, uidx=0, mark_points=display['mark_points']):
    """
    Forms a path from the vertices defined for a part and plots it
    :param part: dict
    :param close: bool
    :param unfold: bool
    :param uidx: int
    :param mark_points: bool
    """
    x, y, z, i = np.array([np.array(point) for k, point in part.items() if 'fold' not in k[0:6]]).T
    names = np.array([k for k in part if 'fold' not in k[0:6]])
    origin = x[np.isnan(i)], y[np.isnan(i)], z[np.isnan(i)]
    names = names[~np.isnan(i)]
    x = x[~np.isnan(i)] + origin[0]
    y = y[~np.isnan(i)] + origin[1]
    z = z[~np.isnan(i)] + origin[2]
    i = i[~np.isnan(i)]
    j = i.argsort()
    x = x[j]
    y = y[j]
    z = z[j]
    i = i[j]
    names = names[j]
    if close:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])
        i = np.append(i, -1000)
        names = np.append(names, 'automatic_closer____')

    def draw_path(axx, xx, yy, **kw):
        pp = axx.plot(xx, yy, **kw)
        pcx = np.nanmean(xx)
        pcy = np.nanmean(yy)
        co = pp[0].get_color()

        def marky_mark(x0_, x1_, y0_, y1_):
            ccx = (x1_ + x0_) / 2.0
            ccy = (y1_ + y0_) / 2.0
            outx = ccx - pcx
            outy = ccy - pcy
            ll = np.sqrt((outx ** 2 + outy ** 2))
            outs = 3.0
            outx *= outs / ll
            outy *= outs / ll
            if (outy > 0) and (outx / outy > 4):
                outx = outs
                outy = 0
            if (outx > 0) and (outy / outx > 4):
                outy = outs
                outx = 0
            m1x = x1_ * 0.75 + x0_ * 0.25
            m2x = x1_ * 0.25 + x0_ * 0.75
            m1y = y1_ * 0.75 + y0_ * 0.25
            m2y = y1_ * 0.25 + y0_ * 0.75
            dxx = x1_ - x0_
            dyy = y1_ - y0_
            # direction = np.sign(np.arctan2(dyy, dxx))
            # outx *= direction
            # outy *= direction
            axx.plot([m1x + outx, x1_ + outx], [m1y + outy, y1_ + outy], color=co, lw=0.5, alpha=0.2)
            axx.plot([m2x + outx, x0_ + outx], [m2y + outy, y0_ + outy], color=co, lw=0.5, alpha=0.2)
            axx.text(
                ccx + outx, ccy + outy, '{:0.2f}\n{:0.2f}'.format(abs(dxx), abs(dyy)),
                color=co, ha='center', va='center', size=5,
            )

        if mark_points and unfold:
            lastx = np.NaN
            lasty = np.NaN
            last_curve = False
            lastnn = ''
            lastx_curve = np.NaN
            lasty_curve = np.NaN
            for xx_, yy_, nn_ in zip(xx, yy, names):
                is_curve = nn_.startswith('curve')
                diff_curve = '_'.join(lastnn.split('_')[:-1]) != '_'.join(nn_.split('_')[:-1])
                if (not (last_curve and is_curve)) or diff_curve:
                    if xx_ != lastx:
                        axx.axvline(xx_, color='gray', lw=0.5, alpha=0.2)
                    if yy_ != lasty:
                        axx.axhline(yy_, color='gray', lw=0.5, alpha=0.2)
                if last_curve and ((not is_curve) or diff_curve):
                    axx.axvline(lastx, color='gray', lw=0.5, alpha=0.2)
                    axx.axhline(lasty, color='gray', lw=0.5, alpha=0.2)
                if ((xx_ != lastx) or (yy_ != lasty)) and (not is_curve):
                    marky_mark(lastx, xx_, lasty, yy_)
                if last_curve and ((not is_curve) or diff_curve):
                    marky_mark(lastx_curve, lastx, lasty_curve, lasty)

                if ((not last_curve) or diff_curve) and is_curve:
                    lastx_curve = xx_
                    lasty_curve = yy_
                lastx = xx_
                lasty = yy_
                last_curve = is_curve
                lastnn = nn_
        return pp

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
        cx = part.get('unfold', {}).get('cx', np.nanmean(x))
        cy = part.get('unfold', {}).get('cy', np.nanmean(y))
        cz = part.get('unfold', {}).get('cz', np.nanmean(z))
        if auto_unfold is None:
            # Rotation about X and Z axes
            xr = part.get('unfold', {}).get('xr', 0) * np.pi/180.0
            yr = part.get('unfold', {}).get('yr', 0) * np.pi/180.0
            zr = part.get('unfold', {}).get('zr', 0) * np.pi/180.0
            # Origin of rotation
            xo = part.get('unfold', {}).get('xo', 0)
            yo = part.get('unfold', {}).get('yo', 0)
            zo = part.get('unfold', {}).get('zo', 0)
            # Rotate about x axis
            y = (y0-yo)*np.cos(xr)+yo - (z0-zo)*np.sin(xr)
            z = (z0-zo)*np.cos(xr)+zo + (y0-yo)*np.sin(xr)
            # Rotate about y axis
            z00 = copy.copy(z)
            x = (x0 - xo) * np.cos(yr) + xo - (z00 - zo) * np.sin(yr)
            z = (z00 - zo) * np.cos(yr) + zo + (x0 - xo) * np.sin(yr)
            # Rotate about z axis
            x00 = copy.copy(x)
            y00 = copy.copy(y)
            x = (x00-xo)*np.cos(zr)+xo + (y00-yo)*np.sin(zr)
            y = (y00-yo)*np.cos(zr)+yo - (x00-xo)*np.sin(zr)
        elif auto_unfold == 'z':
            # Automatic unfold while leaving z alone (flatten x-y)
            theta = np.arctan2(y-cy, x-cx) - np.pi
            dth = np.diff(theta)
            dth[dth > (1.5 * np.pi)] -= 2.0 * np.pi
            dth[dth < -(1.5 * np.pi)] += 2.0 * np.pi
            sdth = np.sign(dth)
            dx = np.diff(x)
            dy = np.diff(y)
            ds = np.sqrt(dx**2+dy**2) * sdth

            x = np.cumsum(np.append(0, ds))
            y = x*0
        elif auto_unfold == 'x':
            # Automatic unfold while leaving x alone (flatten y-z)
            theta = np.arctan2(z-cz, y-cy) - np.pi
            dth = np.diff(theta)
            dth[dth > (1.5 * np.pi)] -= 2.0 * np.pi
            dth[dth < -(1.5 * np.pi)] += 2.0 * np.pi
            sdth = np.sign(dth)
            dy = np.diff(y)
            dz = np.diff(z)
            ds = np.sqrt(dz**2+dy**2) * sdth

            z = np.cumsum(np.append(0, ds))
            y = x*0
        elif auto_unfold == 'zchain':
            dx = np.diff(x)
            dy = np.diff(y)
            ds = np.sqrt(dy**2 + dx**2)
            theta_d = np.arctan2(dy, dx)
            d_theta = np.empty(len(dx))
            d_theta[0] = 0.0
            last_theta = d_theta[0]
            sign_change = np.ones(len(dx))
            for ii in range(1, len(dx)):
                if ds[ii] > 0:
                    d_theta[ii] = theta_d[ii] - last_theta
                    if d_theta[ii] > np.pi:
                        d_theta[ii] -= 2*np.pi
                    if d_theta[ii] < -np.pi:
                        d_theta[ii] += 2*np.pi
                    last_theta = theta_d[ii]
                    if ds[ii-1] > 0:
                        sign_change[ii] = 1 - 2 * abs(d_theta[ii]) > np.pi/2.0
                else:
                    d_theta[ii] = 0
                # print(ii, 'ds', ds[ii], 'th', theta_d[ii]/np.pi, 'last', last_theta/np.pi, 'd', d_theta[ii]/np.pi)

            sign_change = 1 - 2 * (abs(d_theta) >= np.pi * 0.99)
            the_sign = np.cumproduct(sign_change)

            x = np.cumsum(np.append(0, ds*the_sign))
            y = x * 0

        # Translate
        x += xt
        y += yt
        z += zt
        axu = axsf[uidx]

        paf = axaf.plot(x+display['all_unfold_dx'][uidx], z+display['all_unfold_dy'][uidx])
    else:
        axu = axs
        paf = None

    if unfold and not display['debug_unfold']:
        p10 = draw_path(axu[0, 0], x, z)
        p00 = p01 = p11 = None
    else:
        p11 = draw_path(axu[1, 1], x, y)
        p10 = draw_path(axu[1, 0], x, z)
        p01 = draw_path(axu[0, 1], y, z)
        p00 = axu[0, 0].plot(x, y, z)

    folds = [k for k in part if k.startswith('fold')]
    for fold in folds:
        f = part[fold]
        xf = np.append(x[i == f[0]], x[i == f[1]])
        yf = np.append(y[i == f[0]], y[i == f[1]])
        zf = np.append(
            z[i == f[0]], z[i == f[1]])
        if unfold:
            axaf.plot(
                xf+display['all_unfold_dx'][uidx], zf+display['all_unfold_dy'][uidx],
                linestyle='--', color=paf[0].get_color(),
            )

        if unfold and not display['debug_unfold']:
            axu[0, 0].plot(xf, zf, linestyle='--', color=p10[0].get_color())
        else:
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


def plot_unfolded(part, uidx=0, mark_points=display['mark_points']):
    plot_path(part, unfold=True, uidx=uidx, mark_points=mark_points)
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
    plot_path(right_window)
    plot_path(left_window)
    plot_path(back)
    plot_path(head_cutout, close=True)
    plot_path(grill)
    plot_path(grill_bottom)
    plot_path(internal_support_right)
    plot_path(internal_support_left)
    plot_path(internal_support_top)
    plot_path(neck_guard)
    plot_path(neck_guard_front_tab)
    plot_path(neck_guard_right_tab)
    plot_path(neck_guard_left_tab)
    plot_path(neck_back_guard)

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
    fig.savefig('costume_assembled.pdf')

# Unfolded
axsf = []
# Torso
if 'unfolded_torso_back' in which_figures:
    plot_unfolded(back)
    plot_unfolded(top)
    plot_unfolded(head_cutout)
    plot_unfolded(right_side, mark_points=False)
    plot_unfolded(left_side)
if 'unfolded_torso_extra' in which_figures:
    plot_unfolded(front_right, 1)
    plot_unfolded(front_left, 1)
    plot_unfolded(right_window, 1)
    plot_unfolded(left_window, 1, mark_points=False)
    plot_unfolded(rib, 1)
    plot_unfolded(grill_bottom, 1)
    plot_unfolded(grill, 1)
    plot_unfolded(internal_support_right, 1)
    plot_unfolded(internal_support_left, 1)
    plot_unfolded(internal_support_top, 1, mark_points='macro')
    plot_unfolded(neck_guard, 1)
    plot_unfolded(neck_guard_right_tab, 1)
    plot_unfolded(neck_guard_left_tab, 1)
    plot_unfolded(neck_guard_front_tab, 1)
    plot_unfolded(neck_back_guard, 1)

# Arms
if 'unfolded_right_arm' in which_figures:
    plot_unfolded(right_arm_outer, 2)
    plot_unfolded(right_arm_front, 2)
    plot_unfolded(right_arm_top, 2)
    plot_unfolded(right_arm_bottom, 2)
    plot_unfolded(right_arm_back, 2)
    plot_unfolded(right_arm_inner, 2)
    plot_unfolded(right_arm_hand_cutout, 2)

if 'unfolded_left_arm' in which_figures:
    plot_unfolded(left_arm_outer, 3)
    plot_unfolded(left_arm_front, 3)
    plot_unfolded(left_arm_top, 3)
    plot_unfolded(left_arm_bottom, 3)
    plot_unfolded(left_arm_back, 3)
    plot_unfolded(left_arm_inner, 3)
    plot_unfolded(left_arm_hand_cutout, 3)

for axs_ in axsf:
    if (axs_ is not None) and display['debug_unfold']:
        set_axes_equal(axs_[0, 0])

for iuf, uf in enumerate(unfold_figs):
    uf.savefig('costume_folded_{}.pdf'.format(iuf))

if figaf is not None:
    figaf.savefig('costume_all_folded.pdf')

plt.show()
