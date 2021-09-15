from satpy import Scene, find_files_and_readers
import sys
import os
import numpy as np
from time import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
from datetime import datetime
from PIL import Image


def read_h8_folder(fn, verbose=False):
    """
        Converts a folder of raw .hsd data into a satpy Scene, as well as
        load in available bands.

        :param fn: str type. Path to the folder containing
                   the .hsd files to be read.
        :param verbose: boolean type. If True, will print out
                        list of available dataset names.
        :return: Satpy Scene of the input data.
    """
    # Find all the .hsd files to be read
    files = find_files_and_readers(reader='ahi_hsd',
                                   base_dir=fn)
    # Put all the .hsd files into a satpy Scene
    scn = Scene(reader='ahi_hsd',
                filenames=files)
    # Find available bands
    bands = scn.available_dataset_names()
    # Load each band
    for band in bands:
        scn.load([band])
    # Print out loaded bands
    if verbose:
        print("Available Band: %s" % bands)
    return scn

def load_JMA_cloud_mask(fn):
    """
        Loads the JMA operational binary cloud mask as a numpy array.

        :param fn: str type. Full/path/to/filename of cloud mask file.
        :return: numpy array of binary cloud mask.
    """
    from netCDF4 import Dataset
    # Load netcdf file
    dst = Dataset(fn)
    cloud_mask = np.array(dst['CloudMaskBinary'])
    cloud_mask = cloud_mask.astype('float')
    cloud_mask[cloud_mask == -128] = np.nan
    return cloud_mask

def band_to_array(scn, band):
    """
        Converts a loaded band from the given satpy Scene into a numpy array.

        :param scn: satpy Scene of Himawari-8 data.
        :param band: str type. Band name from Scene's available dataset names.
        :return: numpy array of band data
    """
    # Define loaded band data
    dst = scn[band]
    # Convert data into numpy array
    return np.asarray(dst)

def halve_res_quick(array):
    """
        Halves the resolution of the input 2D array, e.g. from 0.5km to 1km,
        by averaging every 2x2 grid of pixels into 1 pixel.

        :param array: 2D numpy array of data.
        :return: 2D numpy array with half the resolution of the input array.
    """
    # Define new shape of lower res array
    shape = (int(array.shape[0] / 2),
             array.shape[0] // int(array.shape[0] / 2),
             int(array.shape[1] / 2),
             array.shape[1] // int(array.shape[1] / 2))
    # Return array with lower res
    return  array.reshape(shape).mean(-1).mean(1)

def halve_res(array):
    """
        Halves the resolution of the input 2D array, e.g. from 1km to 2km,
        by averaging every 2x2 grid of pixels into 1 pixel.

        :param array: 2D numpy array of data.
        :return: 2D numpy array with half the resolution of the input array.
    """
    ### Generate the array of mean values ###
    mean_arr = np.zeros((int(array.shape[0]/2), int(array.shape[-1]/2)))
    count_arr = np.zeros((int(array.shape[0]/2), int(array.shape[-1]/2)))
    for i in range(2):
        for j in range(2):
            arr = array[i::2, j::2].copy()
            arr[np.isnan(arr)] = 0.
            mean_arr += arr
            count_arr += ~np.isnan(array[i::2, j::2])
    mean_arr = mean_arr / count_arr
    ### Generate the array of standard deviation values ###
    std_arr = np.zeros((int(array.shape[0]/2), int(array.shape[-1]/2)))
    for i in range(2):
        for j in range(2):
            arr = array[i::2, j::2].copy()
            arr = np.square(arr - mean_arr)
            arr[np.isnan(arr)] = 0.
            std_arr += arr
    std_arr = std_arr / count_arr # Variance
    std_arr = np.sqrt(std_arr)
    ### Return the mean and standard deviation arrays ###
    return  mean_arr, std_arr

def quarter_res(array):
    """
        Quarters the resolution of the input 2D array, e.g. from 0.5km to 2km,
        by averaging every 4x4 grid of pixels into 1 pixel.

        :param array: 2D numpy array of data.
        :return: 2D numpy array with half the resolution of the input array.
    """
    ### Generate the array of mean values ###
    mean_arr = np.zeros((int(array.shape[0]/4), int(array.shape[-1]/4)))
    count_arr = np.zeros((int(array.shape[0]/4), int(array.shape[-1]/4)))
    for i in range(4):
        for j in range(4):
            arr = array[i::4, j::4].copy()
            arr[np.isnan(arr)] = 0.
            mean_arr += arr
            count_arr += ~np.isnan(array[i::4, j::4])
    mean_arr = mean_arr / count_arr
    ### Generate the array of standard deviation values ###
    std_arr = np.zeros((int(array.shape[0]/4), int(array.shape[-1]/4)))
    for i in range(4):
        for j in range(4):
            arr = array[i::4, j::4].copy()
            arr = np.square(arr - mean_arr)
            arr[np.isnan(arr)] = 0.
            std_arr += arr
    std_arr = std_arr / count_arr # Variance
    std_arr = np.sqrt(std_arr)
    ### Return the mean and standard deviation arrays ###
    return  mean_arr, std_arr

def double_res(array):
    """
        Regrids a 2D input array to an equivalent 2D array with double the res,
        e.g. 1km to 0.5km.
        NB// This does NOT give extra information. This simply allows lower
             res data to be used w/ higher res data. e.g. for RGB false colours.

        :param array: 2D numpy array of data.
        :return: 2D numpy array regridded to double the resolution of
                 the input array.
    """
    # Define original no. rows
    row_init = array.shape[0]
    # Define final no. rows
    row_fin = 2 * row_init
    # Define initial no. columns
    col_init = array.shape[1]
    # Define final np. columns
    col_fin = 2 * col_init
    # Repeat the columns
    array = np.repeat(array, 2)
    # Reshape the flattened output
    array = np.reshape(array, [row_init, col_fin])
    # Repeat the rows
    array = np.tile(array, (1,2))
    # Return array in its final shape
    return np.reshape(array, [row_fin, col_fin])

def fifth_res(array):
    """
        Changes the resolution of the input 2D array to a fifth
        of the original resolution, e.g. from 1km to 5km,
        by averaging every 5x5 grid of pixels into 1 pixel.

        :param array: 2D numpy array of data.
        :return: 2D numpy array with half the resolution of the input array.
    """
    ### Generate the array of mean values ###
    mean_arr = np.zeros((int(array.shape[0]/5), int(array.shape[-1]/5)))
    count_arr = np.zeros((int(array.shape[0]/5), int(array.shape[-1]/5)))
    for i in range(5):
        for j in range(5):
            arr = array[i::5, j::5].copy()
            arr[np.isnan(arr)] = 0.
            mean_arr += arr
            count_arr += ~np.isnan(array[i::5, j::5])
    mean_arr = mean_arr / count_arr
    ### Generate the array of standard deviation values ###
    std_arr = np.zeros((int(array.shape[0]/5), int(array.shape[-1]/5)))
    for i in range(5):
        for j in range(5):
            arr = array[i::5, j::5].copy()
            arr = np.square(arr - mean_arr)
            arr[np.isnan(arr)] = 0.
            std_arr += arr
    std_arr = std_arr / count_arr # Variance
    std_arr = np.sqrt(std_arr)
    ### Return the mean and standard deviation arrays ###
    return  mean_arr, std_arr

def tenth_res(array):
    """
        Changes the resolution of the input 2D array to a tenth
        of the original resolution, e.g. from 0.5km to 5km,
        by averaging every 10x10 grid of pixels into 1 pixel.

        :param array: 2D numpy array of data.
        :return: 2D numpy array with half the resolution of the input array.
    """
    ### Generate the array of mean values ###
    mean_arr = np.zeros((int(array.shape[0]/10), int(array.shape[-1]/10)))
    count_arr = np.zeros((int(array.shape[0]/10), int(array.shape[-1]/10)))
    for i in range(10):
        for j in range(10):
            arr = array[i::10, j::10].copy()
            arr[np.isnan(arr)] = 0.
            mean_arr += arr
            count_arr += ~np.isnan(array[i::10, j::10])
    mean_arr = mean_arr / count_arr
    ### Generate the array of standard deviation values ###
    std_arr = np.zeros((int(array.shape[0]/10), int(array.shape[-1]/10)))
    for i in range(10):
        for j in range(10):
            arr = array[i::10, j::10].copy()
            arr = np.square(arr - mean_arr)
            arr[np.isnan(arr)] = 0.
            std_arr += arr
    std_arr = std_arr / count_arr # Variance
    std_arr = np.sqrt(std_arr)
    ### Return the mean and standard deviation arrays ###
    return  mean_arr, std_arr

def downsample_array(array, divide_by):
    """
        Decreases the resolution of the input 2D array to the
        resolution specified by divide_by, e.g. divide_by=2 will
        half the resolution of the array by averaging every 2x2
        grid of pixels into 1 pixel.

        :param array: 2D numpy array of data.
        :param divide_by: int type. The number by which the resolution will be divided by.
        :return: 2D numpy array with 1/(divide_by) the resolution of the input array.
    """
    ### Generate the array of mean values ###
    mean_arr = np.zeros((int(array.shape[0]/divide_by), int(array.shape[-1]/divide_by)))
    count_arr = np.zeros((int(array.shape[0]/divide_by), int(array.shape[-1]/divide_by)))
    for i in range(divide_by):
        for j in range(divide_by):
            arr = array[i::divide_by, j::divide_by].copy()
            arr[np.isnan(arr)] = 0.
            mean_arr += arr
            count_arr += ~np.isnan(array[i::divide_by, j::divide_by])
    mean_arr = mean_arr / count_arr
    ### Generate the array of standard deviation values ###
    std_arr = np.zeros((int(array.shape[0]/divide_by), int(array.shape[-1]/divide_by)))
    for i in range(divide_by):
        for j in range(divide_by):
            arr = array[i::divide_by, j::divide_by].copy()
            arr = np.square(arr - mean_arr)
            arr[np.isnan(arr)] = 0.
            std_arr += arr
    std_arr = std_arr / count_arr # Variance
    std_arr = np.sqrt(std_arr)
    ### Return the mean and standard deviation arrays ###
    return  mean_arr, std_arr

def downsample_him_lons(array, divide_by, central_longitude):
    """
        Decreases the resolution of the input 2D array of Himawari-8
        longitude values to the resolution specified by divide_by,
        e.g. divide_by=2 will half the resolution of the array by
        averaging every 2x2 grid of pixels into 1 pixel.

        :param array: 2D numpy array of data.
        :param divide_by: int type. The number by which the resolution will be divided by.
        :return: 2D numpy array with 1/(divide_by) the resolution of the input array.
    """
    ### Shift meridian to be defined by geostationary satellite ###
    shifted_longitude = array - central_longitude  # For geostationary satellite coordinates
    shifted_longitude[shifted_longitude < -180.] += 360.
    shifted_longitude[shifted_longitude > 180.] -= 360.
    ### Downsample the shifted coordinates ###
    shifted_longitude, shifted_longitude_std = downsample_array(shifted_longitude, divide_by)
    ### Shift meridian back to correct position ###
    shifted_longitude = shifted_longitude + central_longitude  # For geostationary satellite coordinates
    shifted_longitude[shifted_longitude < -180.] += 360.
    shifted_longitude[shifted_longitude > 180.] -= 360.
    return shifted_longitude, shifted_longitude_std

def upsample_array(array, times_by):
    """
        Increases the resolution of the 2D input array to an
        equivalent 2D array with the resolution specified by
        times_by, e.g. if times_by=2, each value is doubled
        into a 2x2 grid.
        NB// This does NOT give extra information. This simply allows lower
             res data to be used w/ higher res data. e.g. for RGB false colours.

        :param array: 2D numpy array of data.
        :return: 2D numpy array regridded to (times_by)x the resolution of
                 the input array.
    """
    # Define original no. rows
    row_init = array.shape[0]
    # Define final no. rows
    row_fin = times_by * row_init
    # Define initial no. columns
    col_init = array.shape[1]
    # Define final np. columns
    col_fin = times_by * col_init
    # Repeat the columns
    array = np.repeat(array, times_by)
    # Reshape the flattened output
    array = np.reshape(array, [row_init, col_fin])
    # Repeat the rows
    array = np.tile(array, (1,times_by))
    # Return array in its final shape
    return np.reshape(array, [row_fin, col_fin])

def normalise_array(array):
    arr = (array - np.nanmin(array))
    arr /= np.nanmax(arr)
    return arr

def generate_band_arrays(scn, mask = False):
    """
    Creates a multi-dimensional array of satpy Scene band data.

    :param scn: satpy Scene of Himawari-8 data.
    :param mask: numpy boolean array to mask Himawari data
    :return: dictionary of masked arrays
    """
    start = time()
    print('Generating array of all 16 Himawari bands')
    if len(scn.available_dataset_names()) != 16:
        raise Exception('Himawari scene is missing data.\nBands in scene:\n%s'
                        % scn.available_dataset_names)
    data = True
    idx=0
    for band in scn.available_dataset_names():
        print('Adding %s' % band)
        band_data = np.asarray(scn[band])
        if band == 'B03':
            idx+=2
            band_mean_data, band_std_data = quarter_res(band_data)
            if mask is not False:
                band_mean_data = band_mean_data[mask]
                band_std_data = band_std_data[mask]
            band_mean_data = band_mean_data.reshape(len(band_mean_data), 1)
            band_std_data = band_std_data.reshape(len(band_std_data), 1)
            if type(data) == type(band_data):
                data = np.concatenate((data, band_mean_data, band_std_data), axis=1).reshape(len(band_mean_data),
                                                                                             idx)
            else:
                data = np.concatenate((band_mean_data, band_std_data), axis=1).reshape(len(band_mean_data), 2)
        elif band == 'B01' or band == 'B02' or band == 'B04':
            idx += 2
            band_mean_data, band_std_data = halve_res(band_data)
            if mask is not False:
                band_mean_data = band_mean_data[mask]
                band_std_data = band_std_data[mask]
            band_mean_data = band_mean_data.reshape(len(band_mean_data), 1)
            band_std_data = band_std_data.reshape(len(band_std_data), 1)
            if type(data) == type(band_data):
                data = np.concatenate((data, band_mean_data, band_std_data), axis=1).reshape(len(band_mean_data),
                                                                                             idx)
            else:
                data = np.concatenate((band_mean_data, band_std_data), axis=1).reshape(len(band_mean_data), 2)
        else:
            idx += 1
            if mask is not False:
                band_data = band_data[mask]
            band_data = band_data.reshape(len(band_data), 1)
            if type(data) == type(band_data):
                data = np.concatenate((data, band_data), axis=1).reshape(len(band_data),
                                                                         idx)
            else:
                data = band_data
    print('Took %0.2fs' % (time()-start))
    return data

def generate_band_arrays_v2(scn, mask = False):
    """
    Creates a multi-dimensional array of satpy Scene band data.

    :param scn: satpy Scene of Himawari-8 data.
    :param mask: numpy boolean array to mask Himawari data
    :return: dictionary of masked arrays
    """
    start = time()
    print('Generating array of all 16 Himawari bands')
    if len(scn.available_dataset_names()) != 16:
        raise Exception('Himawari scene is missing data.\nBands in scene:\n%s'
                        % scn.available_dataset_names)
    data = True
    idx=0
    for band in scn.available_dataset_names():
        print('Adding %s' % band)
        band_data = np.asarray(scn[band])
        if band == 'B03':
            band_mean_data, band_std_data = tenth_res(band_data)
        elif band == 'B01' or band == 'B02' or band == 'B04':
            band_mean_data, band_std_data = fifth_res(band_data)
        else:
            band_mean_data, band_std_data = fifth_res(double_res(band_data))
        idx += 2
        if mask is not False:
            band_mean_data = band_mean_data[mask]
            band_std_data = band_std_data[mask]
        band_mean_data = band_mean_data.reshape(len(band_mean_data), 1)
        band_std_data = band_std_data.reshape(len(band_std_data), 1)
        if type(data) == type(band_data):
            data = np.concatenate((data, band_mean_data, band_std_data), axis=1).reshape(len(band_mean_data),
                                                                                         idx)
        else:
            data = np.concatenate((band_mean_data, band_std_data), axis=1).reshape(len(band_mean_data), 2)
    print('Took %0.2fs' % (time()-start))
    return data

# def solar_zenith_angles(scn):


def gamma_stretch(array, gamma=1.):
    """
    Stretches the values within the given array by 1/gamma.

    :param array: numpy array of raw data to be converted.
    :param array: float type. Gamma factor to be applied to data.
    :return: RGB compatible numpy array of 8-bit integers.
    """
    # Return array of stretched values
    return array ** (1./gamma)

def true_colour_RGB(scn, res='1km', apply_mask=True, array_only=False):
    """
    Generate a true colour RGB image of the given satpy Scene at the given resolution.
    NB// res may be: 0.5km --> highest resolution; slow, but makes full use of
                               0.64micron channel resolution.
                     1km  -->  default resolution; balance between speed and detail.
                     2km  -->  lowest resolution; fast and compatible with RGBs
                               made using IR bands, but loses detail.

    :param scn: satpy Scene from which the RGB image is to be made.
    :param res: str type. Resolution of the RGB image. Compatible values are: 0.5km, 1km, 2km.
    :param apply_mask: boolean type. If True, will crop the data so only Australasia is shown.
    :param array_only: boolean type. If True, will return the formatted array instead of the
                       matplotlib.pylot figure
    :return: matplotlib.pyplot figure or np.ndarray of the scene.
    """
    # Define compatible resolutions
    comp_ress = ['0.5km', '1km', '2km', '5km']
    # Define the loaded bands to be used
    bnd_1, bnd_2, bnd_3 = (band_to_array(scn, 'B01'),
                           band_to_array(scn, 'B02'),
                           band_to_array(scn, 'B03'))
    # Check and change resolutions depending on input res
    if res not in comp_ress:
        raise ValueError('res = %s is not a compatible value \nCompatible values for res: %s'
                         % (res, comp_ress))

    if res == '0.5km':
        # Regrid 1km res bands to 0.5km equivalent
        bnd_1 = upsample_array(bnd_1, times_by=2)
        bnd_2 = upsample_array(bnd_2, times_by=2)
    elif res == '1km':
        # Halve resolution of 0.64micron channel to 1km res
        bnd_3 = downsample_array(bnd_3, divide_by=2)[0]
    elif res == '2km':
        # Reduce resolution of all bands to 2km res
        bnd_1 = downsample_array(bnd_1, divide_by=2)[0]
        bnd_2 = downsample_array(bnd_2, divide_by=2)[0]
        bnd_3 = downsample_array(bnd_3, divide_by=4)[0]
    elif res == '5km':
        # Reduce resolution of all bands to 2km res
        bnd_1 = downsample_array(bnd_1, divide_by=5)[0]
        bnd_2 = downsample_array(bnd_2, divide_by=5)[0]
        bnd_3 = downsample_array(bnd_3, divide_by=10)[0]
    # Ensure values outside 0-100% are corrected
    bnd_1[bnd_1 > 100.] = 100.
    bnd_1[bnd_1 < 0.] = 0.
    bnd_2[bnd_2 > 100.] = 100.
    bnd_2[bnd_2 < 0.] = 0.
    bnd_3[bnd_3 > 100.] = 100.
    bnd_3[bnd_3 < 0.] = 0.
    # Apply mask
    if apply_mask:
        mask = np.full(bnd_1.shape, False)
        min_y, max_y = int(6500 / 11000 * bnd_3.shape[0]), int(9750 / 11000 * bnd_3.shape[0])
        min_x, max_x = int(2900 / 11000 * bnd_3.shape[0]), int(8250 / 11000 * bnd_3.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        bnd_1 = bnd_1[mask].reshape(new_shape)
        bnd_2 = bnd_2[mask].reshape(new_shape)
        bnd_3 = bnd_3[mask].reshape(new_shape)
    # Convert raw band data to 8-bit integer arrays
    r = gamma_stretch(bnd_3/100., gamma=2.)
    g = gamma_stretch(bnd_2/100., gamma=2.)
    b = gamma_stretch(bnd_1/100., gamma=2.)
    # Define RGB array to be converted into an image
    rgb_arr = np.dstack((r, g, b))
    if not array_only:
        # Define figure and attributes
        if apply_mask:
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
        else:
            fsize = (15, 15)
            xmin, xmax = -5500000, 5500000
            ymin, ymax = -5500000, 5500000
        rgb_fig = plt.figure(figsize=fsize)
        rgb_ax = rgb_fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
        rgb_ax.add_feature(feature.COASTLINE, edgecolor='yellow')
        rgb_ax.imshow(rgb_arr, origin='upper', transform=ccrs.Geostationary(140.735785863),
                      extent=(xmin, xmax, ymin, ymax))
        # Extract scene time
        start = scn.start_time
        rgb_ax.set_title('Himawari-8 True Colour RGB', fontweight='bold',
                         loc='left', fontsize=12)
        rgb_ax.set_title('{}'.format(start.strftime('%d %B %Y %H:%M UTC')),
                         loc='right', fontsize=12)
        rgb_ax.text(0.5, -0.05, '$R$: 0.64$\mu m$   $G$: 0.51$\mu m$   $B$: 0.47$\mu m$',
                    size=12, ha='center', transform=rgb_ax.transAxes)
        # Return figure generated from array
        return rgb_fig
    else:
        return rgb_arr

def plot_single_band(scn, band_number='13', res='1km', apply_mask=True, array_only=False, cmap='bwr'):
    """
    Generate an image of the given band data at the given resolution.
    NB// res may be: 0.5km --> highest resolution; slow, but makes full use of
                               0.64micron channel resolution.
                     1km  -->  default resolution; balance between speed and detail.
                     2km  -->  lowest resolution; fast and compatible with RGBs
                               made using IR bands, but loses detail.

    :param scn: satpy Scene from which the RGB image is to be made.
    :param res: str type. Resolution of the RGB image. Compatible values are: 0.5km, 1km, 2km.
    :param apply_mask: boolean type. If True, will crop the data so only Australasia is shown.
    :param array_only: boolean type. If True, will return the formatted array instead of the
                       matplotlib.pylot figure
    :param cmap: str type. The colour map used when plotting the band data.
    :return: matplotlib.pyplot figure or np.ndarray of the scene.
    """
    all_bands = ['B0' + str(n) for n in range(1, 10)] + ['B' + str(n) for n in range(10, 17)]
    halfkm_bands = [all_bands[2]]
    onekm_bands = all_bands[:2] + [all_bands[3]]
    twokm_bands = all_bands[4:]
    band = str(band_number)
    if len(band) == 1:
        band = '0' + band
    band = 'B' + band
    if band not in all_bands:
        raise ValueError('%s not an available band' % band_number)
    if band in all_bands[:6]:
        unit = '%'
    else:
        unit = 'K'
    print(band)
    # Define compatible resolutions
    comp_ress = ['0.5km', '1km', '2km', '5km']
    # Define the loaded bands to be used
    print('Getting band data')
    band_data = band_to_array(scn, band)
    # Check and change resolutions depending on input res
    if res not in comp_ress:
        raise ValueError('res = %s is not a compatible value \nCompatible values for res: %s'
                         % (res, comp_ress))
    print('Changing to specified res')
    if res == '0.5km':
        if band in onekm_bands: # Regrid 1km res bands to 0.5km equivalent
            band_data = upsample_array(band_data, times_by=2)
        elif band in twokm_bands: # Regrid 2km res bands to 0.5km equivalent
            band_data = upsample_array(band_data, times_by=4)
    elif res == '1km':
        if band in halfkm_bands: # Halve resolution of 0.64micron channel to 1km res
            band_data = downsample_array(band_data, divide_by=2)[0]
        elif band in twokm_bands: # Regrid 2km res bands to 1km equivalent
            band_data = upsample_array(band_data, times_by=2)
    elif res == '2km':
        if band in halfkm_bands: # Quarter resolution of 0.64micron channel to 2km res
            band_data = downsample_array(band_data, divide_by=4)[0]
        elif band in onekm_bands: # Halve resolution of 1km channels to 2km res
            band_data = downsample_array(band_data, divide_by=2)[0]
    elif res == '5km':
        if band in halfkm_bands: # Reduce resolution of 0.64micron channel to 5km res
            band_data = downsample_array(band_data, divide_by=10)[0]
        elif band in onekm_bands: # Halve resolution of 1km channels to 5km res
            band_data = downsample_array(band_data, divide_by=5)[0]
        elif band in twokm_bands: # Halve resolution of 2km channels to 5km res
            band_data = upsample_array(band_data, times_by=2)
            band_data = downsample_array(band_data, divide_by=5)[0]
    # Apply mask
    if apply_mask:
        print('Applying mask')
        mask = np.full(band_data.shape, False)
        min_y, max_y = int(6500 / 11000 * band_data.shape[0]), int(9750 / 11000 * band_data.shape[0])
        min_x, max_x = int(2900 / 11000 * band_data.shape[0]), int(8250 / 11000 * band_data.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        band_data = band_data[mask].reshape(new_shape)
    if not array_only:
        # Define figure and attributes
        if apply_mask:
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
        else:
            fsize = (15, 15)
            xmin, xmax = -5500000, 5500000
            ymin, ymax = -5500000, 5500000
        print('Plotting data')
        arr_fig = plt.figure(figsize=fsize)
        arr_ax = arr_fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
        arr_ax.add_feature(feature.COASTLINE, edgecolor='yellow')
        heatmap_im = arr_ax.imshow(
            band_data,
            origin='upper',
            transform=ccrs.Geostationary(140.735785863),
            extent=(xmin, xmax, ymin, ymax),
            cmap=cmap
        )
        cbaxes = arr_fig.add_axes([1., 0.1, 0.005, 0.8])
        plt.colorbar(heatmap_im, cax=cbaxes, orientation='vertical', label=r'[%s]' % unit)
        # Extract scene time
        start = scn.start_time
        arr_ax.set_title('Himawari-8 Channel %d Data' % band_number, fontweight='bold',
                         loc='left', fontsize=12)
        arr_ax.set_title('{}'.format(start.strftime('%d %B %Y %H:%M UTC')),
                         loc='right', fontsize=12)
        # Return figure generated from array
        return arr_fig
    else:
        return arr_arr

def smoke_index(scn, res='1km', apply_mask=True, array_only=False):
    """
    Generate a "smoke index" heatmap of the given satpy Scene at the given resolution.
    NB// res may be: 0.5km --> highest resolution; slow, but makes full use of
                               0.64micron channel resolution.
                     1km  -->  default resolution; balance between speed and detail.
                     2km  -->  lowest resolution; fast and compatible with RGBs
                               made using IR bands, but loses detail.

    :param scn: satpy Scene from which the heatmap is to be made.
    :param res: str type. Resolution of the heatmap. Compatible values are: 0.5km, 1km, 2km.
    :param apply_mask: boolean type. If True, will crop the data so only Australasia is shown.
    :param array_only: boolean type. If True, will return the formatted array instead of the
                       matplotlib.pylot figure
    :return: matplotlib.pyplot figure or np.ndarray of the scene.
    """
    # Define compatible resolutions
    comp_ress = ['0.5km', '1km', '2km']
    # Define the loaded bands to be used
    bnd_1, bnd_2, bnd_3 = (band_to_array(scn, 'B01'),
                           band_to_array(scn, 'B02'),
                           band_to_array(scn, 'B03'))
    bnd_4, bnd_5, bnd_6 = (band_to_array(scn, 'B04'),
                           band_to_array(scn, 'B05'),
                           band_to_array(scn, 'B06'))
    bnd_7 = band_to_array(scn, 'B07')
    # Check and change resolutions depending on input res
    if res not in comp_ress:
        raise ValueError('res = %s is not a compatible value \nCompatible values for res: %s'
                         % (res, comp_ress))
    if res != '1km':
        if res == '0.5km':
            # Regrid 1km res bands to 0.5km equivalent
            bnd_1 = double_res(bnd_1)
            bnd_2 = double_res(bnd_2)
            bnd_4 = double_res(bnd_4)
            bnd_5 = double_res(double_res(bnd_5))
            bnd_6 = double_res(double_res(bnd_6))
            bnd_7 = double_res(double_res(bnd_7))
        else:
            # Reduce resolution of all bands to 2km res
            bnd_1 = halve_res_quick(bnd_1)
            bnd_2 = halve_res_quick(bnd_2)
            bnd_3 = halve_res_quick(halve_res_quick(bnd_3))
            bnd_4 = halve_res_quick(bnd_4)
    else:
        # Halve resolution of 0.64micron channel to 1km res
        bnd_3 = halve_res_quick(bnd_3)
        bnd_5 = double_res(bnd_5)
        bnd_6 = double_res(bnd_6)
        bnd_7 = double_res(bnd_7)
    # Ensure values outside 0-100% are corrected
    bnd_1[bnd_1 > 100.] = 100.
    bnd_1[bnd_1 < 0.] = 0.
    bnd_2[bnd_2 > 100.] = 100.
    bnd_2[bnd_2 < 0.] = 0.
    bnd_3[bnd_3 > 100.] = 100.
    bnd_3[bnd_3 < 0.] = 0.
    bnd_4[bnd_4 > 100.] = 100.
    bnd_4[bnd_4 < 5.] = np.nan
    bnd_5[bnd_5 > 100.] = 100.
    bnd_5[bnd_5 < 0.] = 0.
    bnd_6[bnd_6 > 100.] = 100.
    bnd_6[bnd_6 < 0.] = 0.
    # Apply mask
    if apply_mask:
        mask = np.full(bnd_1.shape, False)
        min_y, max_y = int(5500/11000 * bnd_1.shape[0]), int(10500/11000 * bnd_1.shape[0])
        min_x, max_x = int(2500/11000 * bnd_1.shape[0]), int(10000/11000 * bnd_1.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        bnd_1 = bnd_1[mask].reshape(new_shape)
        bnd_2 = bnd_2[mask].reshape(new_shape)
        bnd_3 = bnd_3[mask].reshape(new_shape)
        bnd_4 = bnd_4[mask].reshape(new_shape)
        bnd_5 = bnd_5[mask].reshape(new_shape)
        bnd_6 = bnd_6[mask].reshape(new_shape)
        bnd_7 = bnd_7[mask].reshape(new_shape)
    # Define RGB array to be converted into an image
    vis = (bnd_1 + bnd_2) / bnd_4
    sw_ir = (bnd_5 + bnd_6) / 2.
    # sw_ir = normalise_array(sw_ir)
    # smoke_index_final = (vis - sw_ir) / (vis + sw_ir)
    smoke_index_final = vis
    smoke_index_final[smoke_index_final > 4.] = np.nan
    # smoke_index_final[smoke_index_final < 10.] = np.nan
    # smoke_index_a = normalise_array((vis - bnd_3) / vis)
    # smoke_index_b = normalise_array(bnd_5 + bnd_6)
    # smoke_index = normalise_array(smoke_index)
    # smoke_index = gamma_stretch(smoke_index, gamma=0.33)
    # smoke_index[smoke_index < 0.5] = np.nan
    # smoke_index[smoke_index > 0.8] = np.nan
    # smoke_index /= bnd_6
    # smoke_index_final = np.dstack((normalise_array(bnd_6), normalise_array(bnd_5), normalise_array(bnd_4)))
    if not array_only:
        # Define figure and attributes
        if apply_mask:
            fsize = (15, 10)
            xmin, xmax = 2500*1000 - 5500000, 10000*1000 - 5500000
            ymin, ymax = 5500000 - 10500*1000, 5500000 - 5500*1000
        else:
            fsize = (15, 15)
            xmin, xmax = -5500000, 5500000
            ymin, ymax = -5500000, 5500000
        rgb_fig = plt.figure(figsize=fsize)
        rgb_ax = rgb_fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
        rgb_ax.add_feature(feature.COASTLINE, edgecolor='yellow')
        # tRGB = true_colour_RGB(scn=scn, res=res, apply_mask=apply_mask, array_only=True)
        # rgb_ax.imshow(tRGB, origin='upper', transform=ccrs.Geostationary(140.735785863),
        #               extent=(xmin, xmax, ymin, ymax))
        im = rgb_ax.imshow(smoke_index_final, origin='upper', transform=ccrs.Geostationary(140.735785863),
                      extent=(xmin, xmax, ymin, ymax), cmap='Spectral')
        # Extract scene time
        start = scn.start_time
        rgb_ax.set_title('Himawari-8 "Smoke Index" Heatmap', fontweight='bold',
                         loc='left', fontsize=12)
        rgb_ax.set_title('{}'.format(start.strftime('%d %B %Y %H:%M UTC')),
                         loc='right', fontsize=12)
        cbaxes = rgb_fig.add_axes([0.1, 0.05, 0.8, 0.05])
        rgb_fig.colorbar(im, cax=cbaxes, orientation='horizontal', label='Smoke Index Value')
        # rgb_fig.clim(vmin=-1., vmax=1.)
        # Return figure generated from array
        return rgb_fig
    else:
        return rgb_arr

def satpy_true_RGB(scn, array_only=False):
    """
    Generate a rayleigh corrected true-colour rgb using satpy's in-built method.

    :param scn:
    :param array_only:
    :return:
    """
    scn.load(['true_color'])
    new_scn = scn.resample(scn.min_area(), resampler='native')

def natural_colour_RGB(scn, res='1km', apply_mask=True, array_only=False):
    """
    Generate a natural colour (see Himawari-8 false colour training) RGB image of
    the given satpy Scene at the given resolution.
    NB// res may be: 1km  -->  default resolution; balance between speed and detail.
                     2km  -->  lowest resolution; fast and compatible with RGBs
                               made using IR bands, but loses detail.

    :param scn: satpy Scene from which the RGB image is to be made.
    :param res: str type. Resolution of the RGB image. Compatible values are: 0.5km, 1km, 2km.
    :param apply_mask: boolean type. If True, will crop the data so only Australasia is shown.
    :param array_only: boolean type. If True, will return the formatted array instead of the
                       matplotlib.pylot figure
    :return: matplotlib.pyplot figure or np.ndarray of the scene.
    """
    # Define compatible resolutions
    comp_ress = ['1km', '2km', '5km']
    # Define the loaded bands to be used
    bnd_3, bnd_4, bnd_5 = (band_to_array(scn, 'B03'),
                           band_to_array(scn, 'B04'),
                           band_to_array(scn, 'B05'))
    # Check and change resolutions depending on input res
    if res not in comp_ress:
        raise ValueError('res = %s is not a compatible value \nCompatible values for res: %s'
                         % (res, comp_ress))
    if res == '1km':
        # Halve resolution of 0.64micron channel to 1km res
        bnd_3 = halve_res_quick(bnd_3)
        bnd_5 = double_res(bnd_5)
    elif res == '2km':
        # Reduce resolution of all bands to 2km res
        bnd_3 = halve_res_quick(halve_res_quick(bnd_3))
        bnd_4 = halve_res_quick(bnd_4)
    elif res == '5km':
        # Reduce resolution of all bands to 5km res
        bnd_3 = tenth_res(bnd_3)[0]
        bnd_4 = fifth_res(bnd_4)[0]
        bnd_5 = fifth_res(double_res(bnd_5))[0]
    # Ensure values greater than 100% are removed???
    bnd_3[bnd_3 > 100.] = 100.
    bnd_3[bnd_3 < 0.] = 0.
    bnd_4[bnd_4 > 100.] = 100.
    bnd_4[bnd_4 < 0.] = 0.
    bnd_5[bnd_5 > 100.] = 100.
    bnd_5[bnd_5 < 0.] = 0.
    # Apply mask
    if apply_mask:
        mask = np.full(bnd_3.shape, False)
        min_y, max_y = int(6500/11000 * bnd_3.shape[0]), int(9750/11000 * bnd_3.shape[0])
        min_x, max_x = int(2900/11000 * bnd_3.shape[0]), int(8250/11000 * bnd_3.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        bnd_3 = bnd_3[mask].reshape(new_shape)
        bnd_4 = bnd_4[mask].reshape(new_shape)
        bnd_5 = bnd_5[mask].reshape(new_shape)
    # Convert raw band data to 8-bit integer arrays
    r = gamma_stretch(bnd_5/100., gamma=2.)
    g = gamma_stretch(bnd_4/100., gamma=2.)
    b = gamma_stretch(bnd_3/100., gamma=2.)
    # Define RGB array to be converted into an image
    rgb_arr = np.dstack((r, g, b))
    if not array_only:
        # Define figure and attributes
        if apply_mask:
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
        else:
            fsize = (15, 15)
            xmin, xmax = -5500000, 5500000
            ymin, ymax = -5500000, 5500000
        rgb_fig = plt.figure(figsize=fsize)
        rgb_ax = rgb_fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
        # rgb_ax.add_feature(feature.COASTLINE, edgecolor='yellow')
        rgb_ax.coastlines(resolution='110m', color='yellow')
        rgb_ax.imshow(rgb_arr, origin='upper', transform=ccrs.Geostationary(140.735785863),
                      extent=(xmin, xmax, ymin, ymax))
        # Extract scene time
        start = scn.start_time
        rgb_ax.set_title('Himawari-8 Natural Colour RGB', fontweight='bold',
                         loc='left', fontsize=12)
        rgb_ax.set_title('{}'.format(start.strftime('%d %B %Y %H:%M UTC')),
                         loc='right', fontsize=12)
        rgb_ax.text(0.5, -0.05, '$R$: 1.6$\mu m$   $G$: 0.86$\mu m$   $B$: 0.64$\mu m$',
                    size=12, ha='center', transform=rgb_ax.transAxes)
        # Return figure generated from array
        return rgb_fig
    else:
        return rgb_arr

def EUMETSAT_dust_RGB(scn, res='2km', apply_mask=True, array_only=False):
    """
    Generate a natural colour (see Himawari-8 false colour training) RGB image of
    the given satpy Scene at the given resolution.
    NB// res may be: 1km  -->  higher resolution; balance between speed and detail,
                               as well as compatibility with visible and near-IR bands.
                     2km  -->  default resolution; standard resolution of IR bands

    :param scn: satpy Scene from which the RGB image is to be made.
    :param res: str type. Resolution of the RGB image. Compatible values are: 0.5km, 1km, 2km.
    :param apply_mask: boolean type. If True, will crop the data so only Australasia is shown.
    :param array_only: boolean type. If True, will return the formatted array instead of the
                       matplotlib.pylot figure
    :return: matplotlib.pyplot figure or np.ndarray of the scene.
    """
    # Define compatible resolutions
    comp_ress = ['1km', '2km', '5km']
    # Define the loaded bands to be used
    bnd_11, bnd_13, bnd_15 = (band_to_array(scn, 'B11'),
                              band_to_array(scn, 'B13'),
                              band_to_array(scn, 'B15'))
    # Check and change resolutions depending on input res
    if res not in comp_ress:
        raise ValueError('res = %s is not a compatible value \nCompatible values for res: %s'
                         % (res, comp_ress))
    if res == '1km':
        # Regrid all bands from 2km res to 1km res
        bnd_11 = double_res(bnd_11)
        bnd_13 = double_res(bnd_13)
        bnd_15 = double_res(bnd_15)
    elif res == '5km':
        bnd_11 = fifth_res(double_res(bnd_11))[0]
        bnd_13 = fifth_res(double_res(bnd_13))[0]
        bnd_15 = fifth_res(double_res(bnd_15))[0]
    # Define dust-specific bands
    psuedo_r = bnd_15 - bnd_13
    psuedo_g = bnd_13 - bnd_11
    # Clean up data according to ranges
    psuedo_r[psuedo_r > 2.] = 2.
    psuedo_r[psuedo_r < -4.] = -4.
    psuedo_g[psuedo_g > 15.] = 15.
    psuedo_g[psuedo_g < 0.] = 0.
    bnd_13[bnd_13 > 289.] = 289.
    bnd_13[bnd_13 < 261.] = 261.
    # Apply mask
    if apply_mask:
        mask = np.full(bnd_11.shape, False)
        min_y, max_y = int(6500 / 11000 * bnd_11.shape[0]), int(9750 / 11000 * bnd_11.shape[0])
        min_x, max_x = int(2900 / 11000 * bnd_11.shape[0]), int(8250 / 11000 * bnd_11.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        psuedo_r = psuedo_r[mask].reshape(new_shape)
        psuedo_g = psuedo_g[mask].reshape(new_shape)
        bnd_13 = bnd_13[mask].reshape(new_shape)
    # Convert raw band data to 8-bit integer arrays
    r = (psuedo_r + 4.)/6
    g = gamma_stretch(psuedo_g/15., gamma=2.5)
    b = (bnd_13 - 261)/(289. - 261.)
    # Define RGB array to be converted into an image
    rgb_arr = np.dstack((r, g, b))
    if not array_only:
        # Define figure and attributes
        if apply_mask:
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
        else:
            fsize = (15, 15)
            xmin, xmax = -5500000, 5500000
            ymin, ymax = -5500000, 5500000
        rgb_fig = plt.figure(figsize=fsize)
        rgb_ax = rgb_fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
        rgb_ax.add_feature(feature.COASTLINE, edgecolor='yellow')
        rgb_ax.imshow(rgb_arr, origin='upper', transform=ccrs.Geostationary(140.735785863),
                      extent=(xmin, xmax, ymin, ymax))
        # Extract scene time
        start = scn.start_time
        rgb_ax.set_title('Himawari-8 EUMETSAT-Style Dust RGB', fontweight='bold',
                         loc='left', fontsize=12)
        rgb_ax.set_title('{}'.format(start.strftime('%d %B %Y %H:%M UTC')),
                         loc='right', fontsize=12)
        rgb_ax.text(0.5, -0.05, '$R$: 12.4$\mu m$ - 10.4$\mu m$   $G$: 10.4$\mu m$ - 8.6$\mu m$   $B$: 10.4$\mu m$',
                    size=12, ha='center', transform=rgb_ax.transAxes)
        # Return figure generated from array
        return rgb_fig
    else:
        return rgb_arr

def natural_fire_RGB(scn, res='1km', apply_mask=True, array_only=False):
    """
    Generate a modified version of a natural colour RGB image of the given
    satpy Scene at the given resolution. This RGB is more sensitive to fires,
    highlighting them in red.
    From https://weather.msfc.nasa.gov/sport/training/quickGuides/rgb/QuickGuide_NatColorFire_NASA_SPoRT.pdf
    NB// res may be: 1km  -->  default resolution; balance between speed and detail.
                     2km  -->  lowest resolution; fast and compatible with RGBs
                               made using IR bands, but loses detail.

    :param scn: satpy Scene from which the RGB image is to be made.
    :param res: str type. Resolution of the RGB image. Compatible values are: 0.5km, 1km, 2km.
    :param apply_mask: boolean type. If True, will crop the data so only Australasia is shown.
    :param array_only: boolean type. If True, will return the formatted array instead of the
                       matplotlib.pylot figure
    :return: matplotlib.pyplot figure or np.ndarray of the scene.
    """
    # Define compatible resolutions
    comp_ress = ['1km', '2km']
    # Define the loaded bands to be used
    bnd_3, bnd_4, bnd_7 = (band_to_array(scn, 'B03'),
                           band_to_array(scn, 'B04'),
                           band_to_array(scn, 'B06'))
    # Check and change resolutions depending on input res
    if res not in comp_ress:
        raise ValueError('res = %s is not a compatible value \nCompatible values for res: %s'
                         % (res, comp_ress))
    if res != '1km':
            # Reduce resolution of all bands to 2km res
            bnd_3 = halve_res_quick(halve_res_quick(bnd_3))
            bnd_4 = halve_res_quick(bnd_4)
    else:
        # Halve resolution of 0.64micron channel to 1km res
        bnd_3 = halve_res_quick(bnd_3)
        # Regrid IR band from 2km res to 1km res
        bnd_7 = double_res(bnd_7)
    # Ensure values greater than 100% are removed???
    bnd_3[bnd_3 > 100.] = 100.
    bnd_3[bnd_3 < 0.] = 0.
    bnd_4[bnd_4 > 100.] = 100.
    bnd_4[bnd_4 < 0.] = 0.
    bnd_7[bnd_7 > 401.] = 401.
    bnd_7[bnd_7 < 0.] = 0.
    # Apply mask
    if apply_mask:
        mask = np.full(bnd_3.shape, False)
        min_y, max_y = int(6500 / 11000 * bnd_3.shape[0]), int(9750 / 11000 * bnd_3.shape[0])
        min_x, max_x = int(2900 / 11000 * bnd_3.shape[0]), int(8250 / 11000 * bnd_3.shape[0])
        mask[min_y:max_y, min_x:max_x] = True
        new_shape = (int(max_y - min_y), int(max_x - min_x))
        bnd_3 = bnd_3[mask].reshape(new_shape)
        bnd_4 = bnd_4[mask].reshape(new_shape)
        bnd_7 = bnd_7[mask].reshape(new_shape)
    # Convert raw band data to 8-bit integer arrays
    r = gamma_stretch(bnd_7/401., 2.5)
    g = gamma_stretch(bnd_4/100., 1.2)
    b = bnd_3/100.
    # Define RGB array to be converted into an image
    rgb_arr = np.dstack((r, g, b))
    if not array_only:
        # Define figure and attributes
        if apply_mask:
            fsize = (15, 10)
            xmin, xmax = 2900 * 1000 - 5500000, 8250 * 1000 - 5500000
            ymin, ymax = 5500000 - 9750 * 1000, 5500000 - 6500 * 1000
        else:
            fsize = (15, 15)
            xmin, xmax = -5500000, 5500000
            ymin, ymax = -5500000, 5500000
        rgb_fig = plt.figure(figsize=fsize)
        rgb_ax = rgb_fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
        rgb_ax.add_feature(feature.COASTLINE, edgecolor='yellow')
        rgb_ax.imshow(rgb_arr, origin='upper', transform=ccrs.Geostationary(140.735785863),
                      extent=(xmin, xmax, ymin, ymax))
        # Extract scene time
        start = scn.start_time
        rgb_ax.set_title('Himawari-8 Fire-Sensitive Natural Colour RGB', fontweight='bold',
                         loc='left', fontsize=12)
        rgb_ax.set_title('{}'.format(start.strftime('%d %B %Y %H:%M UTC')),
                         loc='right', fontsize=12)
        rgb_ax.text(0.5, -0.05, '$R$: 2.3$\mu m$   $G$: 0.86$\mu m$   $B$: 0.64$\mu m$',
                    size=12, ha='center', transform=rgb_ax.transAxes)
        # Return figure generated from array
        return rgb_fig
    else:
        return rgb_arr

def generate_images(scn, img_dir, scn_name, apply_mask=False):
    # Create and save a natural colour image at 2km res
    natural_colour_img = natural_colour_RGB(scn, res='2km', apply_mask=apply_mask)
    if apply_mask:
        tail = '_Aus_only.png'
    else:
        tail = '.png'
    img_name = os.path.join(img_dir, 'natural_colour_RGB_' + scn_name + tail)
    natural_colour_img.savefig(img_name, bbox_inches='tight', dpi=1200)
    # Create and save a natural-fire image at 2km res
    natural_fire_img = natural_fire_RGB(scn, res='2km', apply_mask=apply_mask)
    img_name = os.path.join(img_dir, 'natural_fire_RGB_' + scn_name + tail)
    natural_fire_img.savefig(img_name, bbox_inches='tight', dpi=1200)
    # Create and save a EUMETSAT dust image at 2km res
    dust_img = EUMETSAT_dust_RGB(scn, res='2km', apply_mask=apply_mask)
    img_name = os.path.join(img_dir, 'EUMETSAT_dust_RGB_' + scn_name + tail)
    dust_img.savefig(img_name, bbox_inches='tight', dpi=1200)
    # Create and save a true colour satellite image at 2km res
    true_RGB_fig = true_colour_RGB(scn, res='2km', apply_mask=apply_mask)
    img_name = os.path.join(img_dir, 'true_colour_RGB_' + scn_name + tail)
    true_RGB_fig.savefig(img_name, bbox_inches='tight', dpi=1200)

def pretty_img(scn, apply_mask=False):
    true_arr = true_colour_RGB(scn, res='2km', apply_mask=apply_mask, array_only=True)
    nat_arr = natural_colour_RGB(scn, res='2km', apply_mask=apply_mask, array_only=True)
    comp_arr = true_arr + nat_arr
    for i in range(3):
        comp_arr[:,:,i] /= np.nanmax(comp_arr[:,:,i])
    if apply_mask:
        fsize = (15, 10)
        xmin, xmax = 2500 * 1000 - 5500000, 10000 * 1000 - 5500000
        ymin, ymax = 5500000 - 10500 * 1000, 5500000 - 5500 * 1000
    else:
        fsize = (15, 15)
        xmin, xmax = -5500000, 5500000
        ymin, ymax = -5500000, 5500000
    comp_fig = plt.figure(figsize=fsize)
    comp_ax = comp_fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    comp_ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    comp_ax.imshow(comp_arr, origin='upper', transform=ccrs.Geostationary(140.735785863),
                  extent=(xmin, xmax, ymin, ymax))
    # Extract scene time
    start = scn.start_time
    comp_ax.set_title('Himawari-8 Custom RGB', fontweight='bold',
                     loc='left', fontsize=12)
    comp_ax.set_title('{}'.format(start.strftime('%d %B %Y %H:%M UTC')),
                     loc='right', fontsize=12)
    cbaxes = comp_fig.add_axes([0.1, 0.05, 0.8, 0.05])
    cbaxes.text(0.5, 0.5, 'Combination of True Colour and Natural Colour RGBs',
                size=16, ha='center', transform=cbaxes.transAxes)
    # Return figure generated from array
    return comp_fig

def plot_heatmap_over_rgb(scn, heatmap_arr, rgb_type='true_colour', res='2km',
                          colourmap='bwr', unit='km', limits=None, subtitle='Heatmap'):
    """
    Plots a heatmap of an input array over the chosen RGB image.

    :param scn:
    :param heatmap_arr:
    :param rgb_type:
    :param res:
    :param colourmap:
    :return:
    """
    compatible_rgb_types = [
        'true_colour',
        'dust',
        'natural_colour',
        'natural_fire'
    ]
    if rgb_type in compatible_rgb_types:
        if rgb_type == 'true_colour':
            rgb_arr = true_colour_RGB(scn=scn, res=res, apply_mask=False, array_only=True)
            rgb_title = 'Himawari-8 True Colour RGB'
            rgb_label = '$R$: 0.64$\mu m$   $G$: 0.51$\mu m$   $B$: 0.47$\mu m$'
        elif rgb_type == 'dust':
            rgb_arr = EUMETSAT_dust_RGB(scn=scn, res=res, apply_mask=False, array_only=True)
            rgb_title = 'Himawari-8 EUMETSAT-Style Dust RGB'
            rgb_label = '$R$: 12.4$\mu m$ - 10.4$\mu m$   $G$: 10.4$\mu m$ - 8.6$\mu m$   $B$: 10.4$\mu m$'
        elif rgb_type == 'natural_colour':
            rgb_arr = natural_colour_RGB(scn=scn, res=res, apply_mask=False, array_only=True)
            rgb_title = 'Himawari-8 Natural Colour RGB'
            rgb_label = '$R$: 1.6$\mu m$   $G$: 0.86$\mu m$   $B$: 0.64$\mu m$'
        elif rgb_type == 'natural_fire':
            rgb_arr = natural_fire_RGB(scn=scn, res=res, apply_mask=False, array_only=True)
            rgb_title = 'Himawari-8 Fire-Sensitive Natural Colour RGB'
            rgb_label = '$R$: 2.3$\mu m$   $G$: 0.86$\mu m$   $B$: 0.64$\mu m$'
    else:
        raise Exception('rgb_type must be of compatible type:\n', compatible_rgb_types)
    start = scn.start_time
    fsize = (15, 15)
    xmin, xmax = -5500000, 5500000
    ymin, ymax = -5500000, 5500000
    comp_fig = plt.figure(figsize=fsize)
    comp_ax = comp_fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    comp_ax.add_feature(feature.COASTLINE, edgecolor='yellow')
    comp_ax.imshow(
        rgb_arr,
        origin='upper',
        transform=ccrs.Geostationary(140.735785863),
        extent=(xmin, xmax, ymin, ymax)
    )
    comp_ax.set_title(
        rgb_title+'\n'+subtitle,
        fontweight='bold',
        loc='left',
        fontsize=12
    )
    comp_ax.set_title(
        '{}'.format(start.strftime('%d %B %Y %H:%M UTC')),
         loc='right',
        fontsize=12
    )
    comp_ax.text(
        0.5,
        -0.05,
        rgb_label,
        size=12,
        ha='center',
        transform=comp_ax.transAxes
    )
    if type(limits) != type(None):
        heatmap_im = comp_ax.imshow(
            heatmap_arr,
            origin='upper',
            transform=ccrs.Geostationary(140.735785863),
            extent=(xmin, xmax, ymin, ymax),
            cmap=colourmap,
            vmin=limits[0],
            vmax=limits[1]
        )
    else:
        heatmap_im = comp_ax.imshow(
            heatmap_arr,
            origin='upper',
            transform=ccrs.Geostationary(140.735785863),
            extent=(xmin, xmax, ymin, ymax),
            cmap=colourmap
        )
    cbaxes = comp_fig.add_axes([1., 0.1, 0.005, 0.8])
    plt.colorbar(heatmap_im, cax=cbaxes, orientation='vertical', label=r'[%s]' % unit)
    return comp_fig

def main(him_folder):
    print('Loading data')
    scn = read_h8_folder(him_folder)
    for n in range(1, 16+1, 1):
        print('Loading channel %s' % n)
        plot_single_band(
            scn=scn,
            band_number=n,
        )
        plt.show()


if __name__ == '__main__':
    full_path_to_him_folder = sys.argv[-1]
    main(full_path_to_him_folder)
    # Load in folder name
    # NB// Must include full path to folder, including folder name
    # dst = sys.argv[-1]
    # img_locs = os.path.join('/mnt/c/Users/drob0013/PhD/Data/Images', dst[-13:])
    # if not os.path.isdir(img_locs):
    #     os.mkdir(img_locs)
    # Load in satpy Scene
    # scn = read_h8_folder(dst)
    # Create and save a natural colour image at 1km res
    # natural_colour_fig = natural_colour_RGB(scn, res='2km', apply_mask=False)
    # plt.show()
    # natural_colour_fig.savefig('%s/masked_natural_colour_RGB_%s_Aus_only_low-res.png'
    #                            % (img_locs, dst[-13:]),
    #                            bbox_inches='tight')
    # Create and save a natural-fire image at 1km res
    # natural_fire_fig = natural_fire_RGB(scn, res='2km', apply_mask=True)
    # natural_fire_fig.savefig('%s/natural_fire_RGB_%s_Aus_only.png'
    #                          % (img_locs, dst[-13:]),
    #                          bbox_inches='tight')
    # Create and save a EUMETSAT dust image at 1km res
    # dust_fig = EUMETSAT_dust_RGB(scn, res='5km', apply_mask=True)
    # dust_fig.savefig('%s/masked_EUMETSAT_dust_RGB_%s_Aus_only_low-res.png'
    #                  % (img_locs, dst[-13:]),
    #                  bbox_inches='tight')
    # Create and save a true colour satellite image at 1km res
    # true_RGB_fig = true_colour_RGB(scn, res='0.5km', apply_mask=True)
    # true_RGB_fig.savefig('%s/true_colour_RGB_%s_Aus_only_high-res.png'
    #                      % (img_locs, dst[-13:]),
    #                      bbox_inches='tight', dpi=1200)
    # Create and save a "pretty RGB" composite image
    # pretty_fig = pretty_img(scn, apply_mask=False)
    # pretty_fig.savefig('%s/pretty_composite_RGB_%s.png'
    #                      % (img_locs, dst[-13:]),
    #                      bbox_inches='tight')
    # smoke_fig = smoke_index(scn, res='2km', apply_mask=True)
    # smoke_fig.savefig('%s/smoke_index_heatmap_%s.png'
    #                   % (img_locs, dst[-13:]),
    #                   bbox_inches='tight')
    # plt.show()
    ### General Debugging Area ###
    # lons, lats = scn['B16'].area.get_lonlats()
    # plt.imshow(lons, interpolation='nearest', cmap='plasma')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(lats, interpolation='nearest', cmap='plasma')
    # plt.colorbar()
    # plt.show()
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.Geostationary(140.735785863))
    # ax.add_feature(feature.COASTLINE, edgecolor = 'yellow')
    # ax.set_global()
    # band = nat_fire_array(scn)
    # plt.imshow(band, transform=ccrs.Geostationary(140.735785863),
    #            extent=(-5500000, 5500000, -5500000, 5500000), cmap='hot')
    # plt.colorbar(orientation='horizontal', label='Band 16 BT [K]')
    # plt.show()
    # plt.savefig('nat_fire_array.png', bbox_inches='tight')
