"""Defines leg geometry"""

import copy
import numpy as np
from op_specs import specs, display, reference_images
from op_util import mirror, copy_point

__all__ = [
    'right_leg_back',
    'right_leg_inner',
    'right_leg_outer',
    'right_leg_front_upper',
    'right_leg_under_pad',
    'right_leg_front_lower',
]
__all__ += [a.replace('right', 'left') for a in __all__ if 'right' in a]


right_leg_origin = (2.0, -14.0, -61.0, np.NaN)

fr_y = specs['leg_depth'] + specs['leg_front_pad']
fr_z = specs['leg_height'] - fr_y * np.sin(specs['leg_top_angle'])
pad_bottom_dz = specs['leg_front_pad'] * np.sin(specs['leg_top_angle'])

right_leg_back = {
    'origin': right_leg_origin,
    'unfold': {},
    'inner_bottom': (0, 0, 0, 0),
    'outer_bottom': (specs['leg_width'], 0, 0, 1),
    'outer_crumple_zone': (specs['leg_width'], 0, specs['foot_cutout_height'], 2),
    'outer_top': (specs['leg_width'], 0, specs['leg_height'], 3),
    'inner_top': (0, 0, specs['leg_height'], 4),
    'inner_crumple_zone': (0, 0, specs['foot_cutout_height'], 5),
}
# right_leg_back['fold'] = (right_leg_back['outer_crumple_zone'][-1], right_leg_back['inner_crumple_zone'][-1])

ll_dz = fr_z - specs['leg_front_pad_height'] - pad_bottom_dz
ll_dy = specs['leg_front_pad']
foot_y = specs['leg_depth'] + ll_dy - specs['foot_cutout_height'] * ll_dy / ll_dz

right_leg_inner = {
    'origin': right_leg_origin,
    'unfold': dict(zr=-90, xo=right_leg_origin[0], yo=right_leg_origin[1], zo=right_leg_origin[2]),
    'back_bottom': (0, 0, 0, 1),
    'back_top': (0, 0, specs['leg_height'], 2),
    'front_top': (0, fr_y, fr_z, 3),
    'pad_bottom_front': (0, fr_y, fr_z - specs['leg_front_pad_height'], 4),
    'pad_bottom_back': (0, specs['leg_depth'], fr_z - specs['leg_front_pad_height'] - pad_bottom_dz, 5),
    'foot_cutout_top': (0, foot_y, specs['foot_cutout_height'], 6),
    'front_bottom': (0, foot_y - specs['foot_cutout_height'] / 2.0, 0, 8),
}

right_leg_outer = copy.deepcopy(right_leg_inner)
right_leg_outer['unfold']['xo'] += specs['leg_width']
right_leg_outer['unfold']['zr'] *= -1
for k in right_leg_outer:
    rlok = right_leg_outer[k]
    if isinstance(rlok, tuple) and (len(rlok) == 4) and ~np.isnan(rlok[3]):
        right_leg_outer[k] = (specs['leg_width'], rlok[1], rlok[2], rlok[3])

right_leg_front_upper = {
    'origin': right_leg_origin,
    'unfold': dict(
        zr=-180, xo=right_leg_origin[0], yo=right_leg_origin[1] + fr_y, zo=right_leg_origin[2] + fr_z, xt=-fr_y
    ),
    'top_inner': (0, fr_y, fr_z, 0),
    'top_outer': (specs['leg_width'], fr_y, fr_z, 1),
    'bottom_outer': (specs['leg_width'], fr_y, fr_z - specs['leg_front_pad_height'], 2),
    'bottom_inner': (0, fr_y, fr_z - specs['leg_front_pad_height'], 3),
}

right_leg_under_pad = {
    'origin': right_leg_origin,
    'unfold': dict(
        zr=-180,
        xr=90 - specs['leg_top_angle'] * 180.0 / np.pi,
        xo=right_leg_origin[0],
        yo=right_leg_origin[1] + fr_y,
        zo=right_leg_origin[2] + fr_z - specs['leg_front_pad_height'],
        xt=-fr_y,
    ),
    'outer_front': (specs['leg_width'], fr_y, fr_z - specs['leg_front_pad_height'], 0),
    'inner_front': (0, fr_y, fr_z - specs['leg_front_pad_height'], 1),
    'inner_back': (0, specs['leg_depth'], fr_z - specs['leg_front_pad_height'] - pad_bottom_dz, 2),
    'outer_back': (specs['leg_width'], specs['leg_depth'], fr_z - specs['leg_front_pad_height'] - pad_bottom_dz, 3),
}

right_leg_front_lower = {
    'origin': right_leg_origin,
    'unfold': dict(
        xo=right_leg_origin[0],
        yo=right_leg_origin[1] + specs['leg_depth'],
        zo=right_leg_origin[2] + fr_z - specs['leg_front_pad_height'] - pad_bottom_dz,
        zr=-180,
        xr=-90 + np.arctan2(ll_dz, ll_dy) * 180.0 / np.pi,
        xt=-fr_y,
        zt=pad_bottom_dz - np.sqrt((fr_y - specs['leg_depth']) ** 2 + pad_bottom_dz ** 2),
    ),
    'top_inner': (0, specs['leg_depth'], fr_z - specs['leg_front_pad_height'] - pad_bottom_dz, 0),
    'top_outer': (specs['leg_width'], specs['leg_depth'], fr_z - specs['leg_front_pad_height'] - pad_bottom_dz, 1),
    'bottom_outer': (specs['leg_width'], foot_y, specs['foot_cutout_height'], 2),
    'bottom_inner': (0, foot_y, specs['foot_cutout_height'], 3),
}

left_leg_back = mirror(right_leg_back, as_new=True)
left_leg_inner = mirror(right_leg_inner, as_new=True)
left_leg_outer = mirror(right_leg_outer, as_new=True)
left_leg_front_upper = mirror(right_leg_front_upper, as_new=True)
left_leg_under_pad = mirror(right_leg_under_pad, as_new=True)
left_leg_front_lower = mirror(right_leg_front_lower, as_new=True)

for part in [
    left_leg_back,
    left_leg_inner,
    left_leg_outer,
    left_leg_front_upper,
    left_leg_under_pad,
    left_leg_front_lower,
]:
    part['unfold']['zt'] = part['unfold'].get('zt', 0) - specs['leg_height'] - 5.0
