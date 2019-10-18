"""
Defines specifications for primary dimensions and display setup
"""

import numpy as np

__all__ = ['specs', 'display', 'reference_images']

specs = dict(
    shoulder_width=40.0,
    chest_depth_at_mid=30.0,
    arm_hole_width=15.0,
    behind_arm_margin=5.0,
    above_arm_margin=5.0,
    prow_angle=10 * np.pi / 180.0,
    center_rib_cutout_width=25.0,
    center_rib_cutout_depth=5.0,
    center_rib_cutout_angle=45.0 * np.pi / 180.0,
    center_rib_cutout_bevel=5.0,
    front_slant=10.0 * np.pi / 180.0,
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
    internal_support_top_cut_angle=150.0 * np.pi / 180.0,
    neck_guard_height=4.0,
    leg_height=25.0,
    leg_width=11.0,
    leg_depth=14.0,
    leg_front_pad=2.0,
    leg_front_pad_height=8.0,
    leg_top_angle=10.0*np.pi/180.0,
    foot_cutout_height=4.0,
)

display = dict(
    head_cutout_angle=0.0,  # 10.0*np.pi/180.0,
    arm_angle=0.0,  # 10.0*np.pi/180.0,
    arm_spacex=3.0,
    arm_spacez=-3.0,
    all_unfold_dx=[0, 75.0, 55.0, 125.0, 125.0, 125.0],
    all_unfold_dy=[0, 0.0, 80.0, 80.0, 80.0, 80.0],
    debug_unfold=False,
    mark_points=True,
)

reference_images = dict(
    front_image='20190826_202031.jpg',
    front_image_height=115.0 + 22.0,
    front_image_center=(-8, -34 + specs['chest_height']),
)
