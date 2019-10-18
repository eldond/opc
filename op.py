#!/bin/python3

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import interpolate
import copy

from op_specs import specs, display, reference_images
from op_util import set_axes_equal, mirror, copy_point, make_main_fig, unfold_figs
from op_plot import plot_path, plot_images, plot_unfolded, plot_info

from op_arm import *

np.seterr(all='raise')

available_figures = [
    'assembled', 'unfolded_torso_back', 'unfolded_torso_extra', 'unfolded_right_arm', 'unfolded_left_arm',
]
which_figures = available_figures[0:3]
mpl.rcParams['figure.figsize'] = [10.75, 8.25]

fig, axs = make_main_fig(which_figures)

if [wf for wf in which_figures if 'unfolded' in wf]:
    figaf, axaf = plt.subplots(1)
    axaf.set_aspect('equal', adjustable='box')
else:
    figaf = axaf = None

plot_info.update({
    'fig': fig,
    'axs': axs,
    'figaf': figaf,
    'axaf': axaf,
})

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
    outer=(
        specs['head_cutout_width']/2.0+ngrt_dx, head_front-specs['head_round_depth']+ngrt_dy, specs['chest_height'], 2
    ),
    front=(
        specs['head_cutout_width']/2.0-specs['head_round_depth']+ngrt_dx, head_front+ngrt_dy, specs['chest_height'], 3
    ),
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
plot_info['axsf'] = axsf

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
