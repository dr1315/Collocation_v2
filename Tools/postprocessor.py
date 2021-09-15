import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.colors as colors
sys.path.append("/g/data/k10/dr1709/code/Personal/Tools")
import him8analysis as h8a
import collocation as col


def put_predicitions_back(predictions, nan_mask, data_shape):
    prediction_arr = np.full(nan_mask.shape, np.nan)
    prediction_arr[~nan_mask] = predictions
    prediction_arr = prediction_arr.reshape(data_shape)
    return prediction_arr

def plot_predictions(arr, scn, save_plot=False, apply_mask=False, class_name='Cloud', img_name='prediction',
                     img_dir='/mnt/c/Users/drob0013/PhD/Diagnostic_Images/NN_Images'):
    if apply_mask:
        mask = np.full(arr.shape, False)
        min_y, max_y = int(6500 / 11000 * arr.shape[0]), int(9750 / 11000 * arr.shape[0])
        min_x, max_x = int(2900 / 11000 * arr.shape[0]), int(8250 / 11000 * arr.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        plottable_arr = arr.copy()[mask].reshape(new_shape)
        fsize = (15, 10)
        xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
        ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
    else:
        plottable_arr = arr.copy()
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor = 'yellow')
    ax.set_facecolor("w")
    ax.set_title(
        r'$\bf{NN \: Prediction}$',
        loc='left',
        fontsize=12
    )
    ax.set_title(
        '{}'.format(scn.start_time.strftime('%d %B %Y %H:%M UTC')),
        loc='right',
        fontsize=12
    )
    ax.text(
        0.5,
        -0.05,
        r'$\bf{Class: }$' + str(class_name),
        size=12,
        ha='center',
        transform=ax.transAxes
    )
    cmap = plt.get_cmap('bwr')
    # cmap.set_bad(color='black', alpha=1.)
    im = plt.imshow(
        plottable_arr,
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap
    )
    cbaxes = fig.add_axes([0.975, 0.1, 0.05, 0.8])
    plt.colorbar(im, cax=cbaxes, orientation='vertical', label='NN Prediction')
    plt.clim(vmin=0., vmax=1.)
    if save_plot:
        if img_name[-4:] != '.png':
            img_name += '.png'
        fname = os.path.join(img_dir, img_name)
        plt.savefig(fname, bbox_inches='tight', dpi=1200)
    else:
        plt.show()

def plot_regressions(arr, scn, apply_mask=False, regression_name='Cloud Top Height', unit='km',
                     min_lim=0., max_lim=1., save_plot=False, img_name='regression',
                     img_dir='/mnt/c/Users/drob0013/PhD/Diagnostic_Images/NN_Images'):
    if apply_mask:
        mask = np.full(arr.shape, False)
        min_y, max_y = int(6500 / 11000 * arr.shape[0]), int(9750 / 11000 * arr.shape[0])
        min_x, max_x = int(2900 / 11000 * arr.shape[0]), int(8250 / 11000 * arr.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        plottable_arr = arr.copy()[mask].reshape(new_shape)
        fsize = (15, 10)
        xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
        ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
    else:
        plottable_arr = arr.copy()
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    ax.set_facecolor("w")
    ax.set_title(
        r'$\bf{NN \: Prediction}$',
        loc='left',
        fontsize=12
    )
    ax.set_title(
        '{}'.format(scn.start_time.strftime('%d %B %Y %H:%M UTC')),
        loc='right',
        fontsize=12
    )
    ax.text(
        0.5,
        -0.05,
        str(regression_name),
        size=12,
        ha='center',
        transform=ax.transAxes
    )
    cmap = plt.get_cmap('bwr')
    # cmap.set_bad(color='black', alpha=1.)
    im = plt.imshow(
        plottable_arr,
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap
    )
    cbaxes = fig.add_axes([0.975, 0.1, 0.05, 0.8])
    plt.colorbar(im, cax=cbaxes, orientation='vertical', label='%s [%s]' % (regression_name, unit))
    plt.clim(vmin=min_lim, vmax=max_lim)
    if save_plot:
        if img_name[-4:] != '.png':
            img_name += '.png'
        fname = os.path.join(img_dir, img_name)
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

def plot_classes(arr, scn, apply_mask=False, class_names=['Cloud'], list_of_colours=['white'],
                 save_plot=False, img_name='class_prediction',
                 img_dir='/mnt/c/Users/drob0013/PhD/Diagnostic_Images/NN_Images'):
    if apply_mask:
        mask = np.full(arr.shape, False)
        min_y, max_y = int(6500 / 11000 * arr.shape[0]), int(9750 / 11000 * arr.shape[0])
        min_x, max_x = int(2900 / 11000 * arr.shape[0]), int(8250 / 11000 * arr.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        plottable_arr = arr.copy()[mask].reshape(new_shape)
        fsize = (15, 10)
        xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
        ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
    else:
        plottable_arr = arr.copy()
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    ax.set_facecolor("w")
    ax.set_title(
        r'$\bf{NN \: Prediction}$',
        loc='left',
        fontsize=12
    )
    ax.set_title(
        '{}'.format(scn.start_time.strftime('%d %B %Y %H:%M UTC')),
        loc='right',
        fontsize=12
    )
    ax.text(
        0.5,
        -0.05,
        'Multiclass Prediction',
        size=12,
        ha='center',
        transform=ax.transAxes
    )
    cmap = colors.ListedColormap(list_of_colours)
    boundaries = np.arange(
        0.5,
        len(class_names) + 1.5,
        1.
    )
    # print(boundaries)
    class_nums = boundaries[:-1] + 0.5
    # print(class_nums)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    im = plt.imshow(
        plottable_arr,
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        norm=norm
    )
    cbaxes = fig.add_axes([0.975, 0.1, 0.05, 0.8])
    cb = plt.colorbar(im, cax=cbaxes, orientation='vertical', ticks=class_nums)
    cb.ax.set_yticks(boundaries[:-1] + 0.5)
    cb.ax.set_yticklabels(class_names)
    if save_plot:
        if img_name[-4:] != '.png':
            img_name += '.png'
        fname = os.path.join(img_dir, img_name)
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

def plot_jma_mask(arr, scn, apply_mask=False, class_names=['Cloud'], list_of_colours=['white'],
                  save_plot=False, img_name='class_prediction',
                  img_dir='/mnt/c/Users/drob0013/PhD/Diagnostic_Images/NN_Images'):
    if apply_mask:
        mask = np.full(arr.shape, False)
        min_y, max_y = int(6500 / 11000 * arr.shape[0]), int(9750 / 11000 * arr.shape[0])
        min_x, max_x = int(2900 / 11000 * arr.shape[0]), int(8250 / 11000 * arr.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        plottable_arr = arr.copy()[mask].reshape(new_shape)
        fsize = (15, 10)
        xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
        ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
    else:
        plottable_arr = arr.copy()
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    ax.set_facecolor("w")
    ax.set_title(
        r'$\bf{JMA \: Cloud \: Mask}$',
        loc='left',
        fontsize=12
    )
    ax.set_title(
        '{}'.format(scn.start_time.strftime('%d %B %Y %H:%M UTC')),
        loc='right',
        fontsize=12
    )
    ax.text(
        0.5,
        -0.05,
        'Multiclass Prediction',
        size=12,
        ha='center',
        transform=ax.transAxes
    )
    cmap = colors.ListedColormap(list_of_colours)
    boundaries = np.arange(
        0.5,
        len(class_names) + 1.5,
        1.
    )
    # print(boundaries)
    class_nums = boundaries[:-1] + 0.5
    # print(class_nums)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    im = plt.imshow(
        plottable_arr,
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        norm=norm
    )
    cbaxes = fig.add_axes([0.975, 0.1, 0.05, 0.8])
    cb = plt.colorbar(im, cax=cbaxes, orientation='vertical', ticks=class_nums)
    cb.ax.set_yticks(boundaries[:-1] + 0.5)
    cb.ax.set_yticklabels(class_names)
    if save_plot:
        if img_name[-4:] != '.png':
            img_name += '.png'
        fname = os.path.join(img_dir, img_name)
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

def plot_binary_continuous_difference(bin_arr, cont_arr, scn, apply_mask=False,
                                      save_plot=False, img_name='difference',
                                      img_dir='/mnt/c/Users/drob0013/PhD/Diagnostic_Images/NN_Images'):
    if apply_mask:
        mask = np.full(bin_arr.shape, False)
        min_y, max_y = int(6500 / 11000 * bin_arr.shape[0]), int(9750 / 11000 * bin_arr.shape[0])
        min_x, max_x = int(2900 / 11000 * bin_arr.shape[0]), int(8250 / 11000 * bin_arr.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        plottable_bin_arr = bin_arr.copy()[mask].reshape(new_shape)
        plottable_cont_arr = cont_arr.copy()[mask].reshape(new_shape)
        fsize = (15, 10)
        xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
        ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
    else:
        plottable_bin_arr = bin_arr.copy()
        plottable_cont_arr = cont_arr.copy()
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    ax.set_facecolor("w")
    ax.set_title(
        r'$\bf{NN \: Prediction}$',
        loc='left',
        fontsize=12
    )
    ax.set_title(
        '{}'.format(scn.start_time.strftime('%d %B %Y %H:%M UTC')),
        loc='right',
        fontsize=12
    )
    ax.text(
        0.5,
        -0.05,
        'Difference in Mask Predictions',
        size=12,
        ha='center',
        transform=ax.transAxes
    )
    where_one = plottable_bin_arr == 1
    arr_deltas = np.abs(plottable_bin_arr - plottable_cont_arr)
    cloud_arr = arr_deltas.copy()
    cloud_arr[~where_one] = np.nan
    non_cloud_arr = arr_deltas.copy()
    non_cloud_arr[where_one] = np.nan
    im1 = plt.imshow(
        cloud_arr,
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap='autumn',
        vmin=0.,
        vmax=1.
    )
    im2 = plt.imshow(
        non_cloud_arr,
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax),
        cmap='winter',
        vmin=0.,
        vmax=1.
    )
    cbaxes1 = fig.add_axes([0.975, 0.1, 0.05, 0.8])
    plt.colorbar(im1, cax=cbaxes1, orientation='vertical', label='Binary Cloud Prediction Delta')
    cbaxes2 = fig.add_axes([0.025, 0.1, 0.05, 0.8])
    plt.colorbar(im2, cax=cbaxes2, orientation='vertical', label='Binary Non-Cloud Prediction Delta')
    if save_plot:
        if img_name[-4:] != '.png':
            img_name += '.png'
        fname = os.path.join(img_dir, img_name)
        plt.savefig(fname, bbox_inches='tight', dpi=1200)
    else:
        plt.show()
