"""Provides key plotting functions"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy

from op_specs import specs, display, reference_images
from op_util import make_unfold_fig, unfold_figs

__all__ = ['plot_path', 'plot_images', 'plot_unfolded']

plot_info = {}


def plot_path(part, close=True, unfold=False, uidx=0, mark_points=display['mark_points']):
    """
    Forms a path from the vertices defined for a part and plots it
    :param part: dict
    :param close: bool
    :param unfold: bool
    :param uidx: int
    :param mark_points: bool
    """
    axsf = plot_info.get('axsf', None)

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
                ccx + outx,
                ccy + outy,
                '{:0.2f}\n{:0.2f}'.format(abs(dxx), abs(dyy)),
                color=co,
                ha='center',
                va='center',
                size=5,
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
        while (uidx + 1) > len(axsf):
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
            xr = part.get('unfold', {}).get('xr', 0) * np.pi / 180.0
            yr = part.get('unfold', {}).get('yr', 0) * np.pi / 180.0
            zr = part.get('unfold', {}).get('zr', 0) * np.pi / 180.0
            # Origin of rotation
            xo = part.get('unfold', {}).get('xo', 0)
            yo = part.get('unfold', {}).get('yo', 0)
            zo = part.get('unfold', {}).get('zo', 0)
            # Rotate about x axis
            y = (y0 - yo) * np.cos(xr) + yo - (z0 - zo) * np.sin(xr)
            z = (z0 - zo) * np.cos(xr) + zo + (y0 - yo) * np.sin(xr)
            # Rotate about y axis
            z00 = copy.copy(z)
            x = (x0 - xo) * np.cos(yr) + xo - (z00 - zo) * np.sin(yr)
            z = (z00 - zo) * np.cos(yr) + zo + (x0 - xo) * np.sin(yr)
            # Rotate about z axis
            x00 = copy.copy(x)
            y00 = copy.copy(y)
            x = (x00 - xo) * np.cos(zr) + xo + (y00 - yo) * np.sin(zr)
            y = (y00 - yo) * np.cos(zr) + yo - (x00 - xo) * np.sin(zr)
        elif auto_unfold == 'z':
            # Automatic unfold while leaving z alone (flatten x-y)
            theta = np.arctan2(y - cy, x - cx) - np.pi
            dth = np.diff(theta)
            dth[dth > (1.5 * np.pi)] -= 2.0 * np.pi
            dth[dth < -(1.5 * np.pi)] += 2.0 * np.pi
            sdth = np.sign(dth)
            dx = np.diff(x)
            dy = np.diff(y)
            ds = np.sqrt(dx ** 2 + dy ** 2) * sdth

            x = np.cumsum(np.append(0, ds))
            y = x * 0
        elif auto_unfold == 'x':
            # Automatic unfold while leaving x alone (flatten y-z)
            theta = np.arctan2(z - cz, y - cy) - np.pi
            dth = np.diff(theta)
            dth[dth > (1.5 * np.pi)] -= 2.0 * np.pi
            dth[dth < -(1.5 * np.pi)] += 2.0 * np.pi
            sdth = np.sign(dth)
            dy = np.diff(y)
            dz = np.diff(z)
            ds = np.sqrt(dz ** 2 + dy ** 2) * sdth

            z = np.cumsum(np.append(0, ds))
            y = x * 0
        elif auto_unfold == 'zchain':
            dx = np.diff(x)
            dy = np.diff(y)
            ds = np.sqrt(dy ** 2 + dx ** 2)
            theta_d = np.arctan2(dy, dx)
            d_theta = np.empty(len(dx))
            d_theta[0] = 0.0
            last_theta = d_theta[0]
            sign_change = np.ones(len(dx))
            for ii in range(1, len(dx)):
                if ds[ii] > 0:
                    d_theta[ii] = theta_d[ii] - last_theta
                    if d_theta[ii] > np.pi:
                        d_theta[ii] -= 2 * np.pi
                    if d_theta[ii] < -np.pi:
                        d_theta[ii] += 2 * np.pi
                    last_theta = theta_d[ii]
                    if ds[ii - 1] > 0:
                        sign_change[ii] = 1 - 2 * abs(d_theta[ii]) > np.pi / 2.0
                else:
                    d_theta[ii] = 0
                # print(ii, 'ds', ds[ii], 'th', theta_d[ii]/np.pi, 'last', last_theta/np.pi, 'd', d_theta[ii]/np.pi)

            sign_change = 1 - 2 * (abs(d_theta) >= np.pi * 0.99)
            the_sign = np.cumproduct(sign_change)

            x = np.cumsum(np.append(0, ds * the_sign))
            y = x * 0

        # Translate
        x += xt
        y += yt
        z += zt
        axu = axsf[uidx]

        paf = plot_info['axaf'].plot(x + display['all_unfold_dx'][uidx], z + display['all_unfold_dy'][uidx])
    else:
        axu = plot_info['axs']
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
        zf = np.append(z[i == f[0]], z[i == f[1]])
        if unfold:
            plot_info['axaf'].plot(
                xf + display['all_unfold_dx'][uidx],
                zf + display['all_unfold_dy'][uidx],
                linestyle='--',
                color=paf[0].get_color(),
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
    plot_info['axs'][1, 0].imshow(front_pic, extent=(fcx - fw / 2.0, fcx + fw / 2.0, fcy - fh / 2.0, fcy + fh / 2.0))
    return


def plot_unfolded(part, uidx=0, mark_points=display['mark_points']):
    plot_path(part, unfold=True, uidx=uidx, mark_points=mark_points)
    return
