"""Defines arm geometry"""

import matplotlib as mpl
from matplotlib import pyplot as plt
import copy
import numpy as np

from op_specs import specs, display, reference_images
from op_util import mirror, copy_point

__all__ = [
    'right_arm_top',
    'right_arm_bottom',
    'right_arm_outer',
    'right_arm_front',
    'right_arm_back',
    'right_arm_inner',
    'right_arm_hand_cutout',
]

__all__ += [a.replace('right', 'left') for a in __all__ if 'right' in a]

# General shortcuts
rt = specs['shoulder_width'] / 2.0
bk = -specs['arm_hole_width'] - specs['behind_arm_margin']
front_margin = specs['chest_depth_at_mid'] + bk
prow_depth = np.tan(specs['prow_angle']) * rt
gy = front_margin + prow_depth - specs['grill_half_width'] * np.tan(specs['prow_angle'])


right_arm_origin = (display['arm_spacex'], 0, display['arm_spacez'], np.NaN)
ra_pivot_x = rt + right_arm_origin[0]
ra_pivot_y = front_margin + right_arm_origin[1]
ra_pivot_zt = 0 + right_arm_origin[2]
ra_pivot_zb = -specs['grill_height'] + right_arm_origin[2]

right_arm_top = dict(
    origin=right_arm_origin,
    unfold=dict(zr=90, yr=-90, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zt),
    front_right_corner=(specs['grill_half_width'], gy, 0, 0),
    outer_right_corner=(rt, front_margin, 0, 1),
    outer_back_corner=(rt, bk, 0, 2),
    inner_back_corner=(specs['grill_half_width'] + specs['arm_top_cut_width'], bk, 0, 3),
    inner_cutout_corner=(specs['grill_half_width'] + specs['arm_top_cut_width'], bk + specs['arm_top_cut_depth'], 0, 4),
    inner_front_corner=(specs['grill_half_width'], 0, 0, 5),
)

# xyzo = right_arm_top['outer_right_corner']
# right_arm_top['unfold'] = dict(zr=90, yr=-90, xo=xyzo[0], yo=xyzo[1], zo=xyzo[2], yt=-xyzo[1]-right_arm_origin[0])

right_arm_bottom = dict(
    origin=right_arm_origin,
    unfold=dict(zr=90, yr=90, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zb),
    front_right_corner=(specs['grill_half_width'], gy, -specs['grill_height'], 0),
    outer_right_corner=(rt, front_margin, -specs['grill_height'], 1),
    outer_back_corner=(rt, bk, -specs['grill_height'], 2),
    inner_back_corner=(specs['grill_half_width'], bk, -specs['grill_height'], 3),
)

right_arm_outer = dict(
    origin=right_arm_origin,
    unfold=dict(zr=90, xr=0, yr=0, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zt),
    # front_right_top_corner=copy_point(right_arm_top['front_right_corner'], 0),
    outer_right_top_corner=copy_point(right_arm_top['outer_right_corner'], 1),
    outer_back_top_corner=copy_point(right_arm_top['outer_back_corner'], 2),
    outer_back_bottom_corner=(
        right_arm_top['outer_back_corner'][0],
        right_arm_top['outer_back_corner'][1],
        -specs['grill_height'],
        3,
    ),
    outer_right_bottom_corner=(
        right_arm_top['outer_right_corner'][0],
        right_arm_top['outer_right_corner'][1],
        -specs['grill_height'],
        4,
    ),
    # front_right_bottom_corner=(
    #     right_arm_top['front_right_corner'][0], right_arm_top['front_right_corner'][1], -specs['grill_height'], 5),
)
# right_arm_outer['fold1'] = (
#     right_arm_outer['outer_right_top_corner'][-1], right_arm_outer['outer_right_bottom_corner'][-1])

right_arm_front = dict(
    origin=right_arm_origin,
    unfold=dict(zr=180 - specs['prow_angle'] * 180.0 / np.pi, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zt),
    front_right_top_corner=copy_point(right_arm_top['front_right_corner'], 0),
    outer_right_top_corner=copy_point(right_arm_top['outer_right_corner'], 1),
    outer_right_bottom_corner=(
        right_arm_top['outer_right_corner'][0],
        right_arm_top['outer_right_corner'][1],
        -specs['grill_height'],
        2,
    ),
    front_right_bottom_corner=(
        right_arm_top['front_right_corner'][0],
        right_arm_top['front_right_corner'][1],
        -specs['grill_height'],
        3,
    ),
)

arm_outer_len = right_arm_top['outer_back_corner'][1] - right_arm_top['outer_right_corner'][1]
right_arm_back = dict(
    origin=right_arm_origin,
    unfold=dict(xt=arm_outer_len, yt=-arm_outer_len),
    top_inner=copy_point(right_arm_top['inner_back_corner'], 0),
    top_outer=copy_point(right_arm_top['outer_back_corner'], 1),
    bottom_outer=copy_point(right_arm_bottom['outer_back_corner'], 2),
    bottom_inner=copy_point(right_arm_bottom['inner_back_corner'], 3),
    inner=(
        right_arm_bottom['inner_back_corner'][0],
        right_arm_bottom['inner_back_corner'][1],
        specs['arm_back_cover_margin'] - specs['grill_height'],
        4,
    ),
)

arm_width = right_arm_bottom['outer_back_corner'][0] - right_arm_bottom['inner_back_corner'][0]
right_arm_inner = dict(
    origin=right_arm_origin,
    unfold=dict(yr=180, zr=90, xo=ra_pivot_x, yo=ra_pivot_y, zo=ra_pivot_zb, zt=-arm_width, yt=arm_width),
    inner_top=copy_point(right_arm_top['inner_front_corner'], 0),
    front_top=copy_point(right_arm_top['front_right_corner'], 1),
    front_bottom=copy_point(right_arm_bottom['front_right_corner'], 2),
    back_bottom=copy_point(right_arm_bottom['inner_back_corner'], 3),
    back_top=copy_point(right_arm_back['inner'], 4),
    inner=(
        right_arm_top['inner_front_corner'][0],
        right_arm_top['inner_cutout_corner'][1],
        right_arm_back['inner'][2],
        5,
    ),
)

fr_y = right_arm_inner['front_top'][1] - specs['hand_hole_margins']['front']
bk_y = right_arm_inner['front_top'][1] - specs['hand_hole_margins']['front'] - specs['hand_hole_depth']
tp_z = right_arm_inner['front_top'][2] - specs['hand_hole_margins']['top']
bt_z = right_arm_inner['front_bottom'][2] + specs['hand_hole_margins']['bottom']
tp_y = right_arm_inner['inner_top'][1] + specs['hand_hole_margins']['back']
raiit = right_arm_inner['inner_top']
raii = right_arm_inner['inner']
bk_z = tp_z - (raiit[2] - raii[2]) / (raiit[1] - raii[1]) * (tp_y - bk_y)

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
