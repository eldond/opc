#!/bin/python3

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# fig, axs = plt.subplots(2, 2)
fig = plt.figure()
axs = np.array([[
    fig.add_subplot(221, projection='3d'), fig.add_subplot(222)],
    [fig.add_subplot(223), fig.add_subplot(224)],
])
for ax in axs.flatten():
    ax.set_aspect('equal', adjustable='box')

specs = dict(
    shoulder_width=40.0,
    chest_depth_at_mid=30.0,
    arm_hole_width=15.0,
    behind_arm_margin=5.0,
    above_arm_margin=2.0,
    prow_angle=10*np.pi/180.0,
    center_rib_cutout_width=25.0,
    center_rib_cutout_depth=5.0,
    center_rib_cutout_angle=45.0*np.pi/180.0,
    center_rib_cutout_bevel=5.0,
    front_slant=10.0*np.pi/180.0,
    chest_height=40.0,
    head_cutout_width=20.0,
    behind_head_margin=2.0,
    head_cutout_depth=20.0,
    head_round_depth=5.0,
    head_cutout_angle=10.0*np.pi/180.0,
    grill_half_width=10.0,
    grill_height=20.0,
)

display = dict(
    arm_angle=10.0*np.pi/180.0,
)


def mirror(part, as_new=False):
    ind = [v[3] for k, v in part.items() if not k.startswith('fold')]
    ii = int(np.nanmax(ind)) + 1
    new_part = {}
    for k, v in part.items():
        if k.startswith('fold'):
            if as_new:
                new_part[k] = v
        elif ((not np.isnan(v[3])) and (('left' in k) or ('right' in k))) or as_new:
                newk = k.replace('right', 'left')
                newv = (-v[0], v[1], v[2], (v[3] if as_new else 2*ii-v[3]))
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
front = dict(
    origin=(0, 0, 0, np.NaN),
    center_top=(0, front_margin-slant_dy+prow_depth, specs['chest_height'], 0),
    right_top=copy_point(right_side['front_top_corner'], 1),
    right_bottom=copy_point(right_side['bottom_front'], 2),
    center_bottom=copy_point(rib['front_center'], 3),
    fold1=(0, 3),
)
mirror(front)

# Define back
back = dict(
    origin=(0, 0, 0, np.NaN),
    right_bottom=copy_point(right_side['back_bottom_corner'], 0),
    right_top=copy_point(right_side['back_top_corner'], 1),
    right_front=copy_point(right_side['front_top_corner'], 2),
    center_front=copy_point(front['center_top'], 3),
)
mirror(back)
back['fold1'] = (1, 7)
head_front = bk+specs['behind_head_margin']+specs['head_cutout_depth']
flapy = bk+specs['behind_head_margin'] + \
        (specs['head_cutout_depth']-specs['head_round_depth'])*np.cos(specs['head_cutout_angle'])
flapy2 = bk+specs['behind_head_margin'] + specs['head_cutout_depth']*np.cos(specs['head_cutout_angle'])
flapz = specs['chest_height']+(specs['head_cutout_depth']-specs['head_round_depth'])*np.sin(specs['head_cutout_angle'])
flapz2 = specs['chest_height']+specs['head_cutout_depth']*np.sin(specs['head_cutout_angle'])
head_cutout = dict(
    origin=(0, 0, 0, np.NaN),
    right_front_corner=(specs['head_cutout_width']/2.0-specs['head_round_depth'], head_front, specs['chest_height'], -1),
    right_corner=(specs['head_cutout_width']/2.0, head_front-specs['head_round_depth'], specs['chest_height'], 0),
    back_right_corner=(specs['head_cutout_width']/2.0, bk+specs['behind_head_margin'], specs['chest_height'], 1),
    flap_right_corner=(specs['head_cutout_width']/2.0, flapy, flapz, 2),
    flap_right_front_corner=(specs['head_cutout_width']/2.0-specs['head_round_depth'], flapy2, flapz2, 3),
)
mirror(head_cutout)
head_cutout['fold_close'] = (1, np.nanmax([v[3] if len(v) == 4 else 0 for v in head_cutout.values()])-2)

# Front grill thing
gy = front['center_bottom'][1] - specs['grill_half_width'] * np.tan(specs['prow_angle'])
grill = dict(
    origin=(0, 0, 0, np.NaN),
    top_center=copy_point(front['center_bottom'], 0),
    top_right=(specs['grill_half_width'], gy, 0, 1),
    bottom_right=(specs['grill_half_width'], gy, -specs['grill_height'], 2),
    bottom_center=(front['center_bottom'][0], front['center_bottom'][1], -specs['grill_height'], 3),
)
mirror(grill)
grill['fold_center'] = (0, 3)


def plot_path(part, close=True):
    x, y, z, i = np.array([np.array(point) for k, point in part.items() if not k.startswith('fold')]).T

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

    p11 = axs[1, 1].plot(x, y)
    p10 = axs[1, 0].plot(x, z)
    p01 = axs[0, 1].plot(y, z)
    p00 = axs[0, 0].plot(x, y, z)

    folds = [k for k in part if k.startswith('fold')]
    for fold in folds:
        f = part[fold]
        xf = np.append(x[i == f[0]], x[i == f[1]])
        yf = np.append(y[i == f[0]], y[i == f[1]])
        zf = np.append(
            z[i == f[0]], z[i == f[1]])
        axs[1, 1].plot(xf, yf, linestyle='--', color=p11[0].get_color())
        axs[1, 0].plot(xf, zf, linestyle='--', color=p10[0].get_color())
        axs[0, 1].plot(yf, zf, linestyle='--', color=p01[0].get_color())
        axs[0, 0].plot(xf, yf, zf, linestyle='--', color=p00[0].get_color())
    return


def plot_images():
    front_image = '20190826_202031.jpg'
    front_image_height = 115.0 + 22.0
    front_image_center = (-10, 30)
    front_pic = mpl.image.imread(front_image)
    front_pic = np.swapaxes(front_pic, 0, 1)
    front_image_width = front_image_height * np.shape(front_pic)[1] / np.shape(front_pic)[0]
    axs[1, 0].imshow(front_pic, extent=(
        front_image_center[0] - front_image_width / 2.0, front_image_center[0] + front_image_width / 2.0,
        front_image_center[1] - front_image_height / 2.0, front_image_center[1] + front_image_width / 2.0,
    ))
    return


plot_images()

plot_path(rib)
plot_path(right_side)
plot_path(left_side)
plot_path(front)
plot_path(back)
plot_path(head_cutout, close=True)
plot_path(grill)

plt.show()
