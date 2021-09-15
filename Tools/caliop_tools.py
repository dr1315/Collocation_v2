from pyhdf.SD import SD, SDC
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime

def number_to_bit(arr):
    '''

    :param arr:
    :return:
    '''
    return np.binary_repr(arr, width=16)
number_to_bit = np.vectorize(number_to_bit)

def height_to_y(arr1, arr2):
    '''

    :param arr:
    :return:
    '''
    return np.where(arr2 == arr1)[0]
height_to_y = np.vectorize(height_to_y)

def simple_bit_conversion(arr):
    '''

    :param arr:
    :return:
    '''
    return int(arr[-3:], 2)
simple_bit_conversion = np.vectorize(simple_bit_conversion)

def cloud_bit_conversion(arr):
    '''

    :param arr:
    :return:
    '''
    if int(arr[-3:], 2) == 2:
        return int(arr[-12:-9], 2)
    else:
        return 0
cloud_bit_conversion = np.vectorize(cloud_bit_conversion)

def custom_feature_conversion(arr):
    '''
    Converts a bit array of CALIOP vertical feature mask to a custom
    array where features are marked by integers corresponding to:
    NaN - Invalid
    0 - Clear Air
    1 - Low Overcast (Transparent)
    2 - Low Overcast (Opaque)
    3 - Transition Stratocumulus
    4 - Low Broken Cumulus
    5 - Altocumulus (Transparent)
    6 - Altostratus (Opaque)
    7 - Cirrus (Transparent)
    8 - Deep Convective (Opaque)
    9 - Clean Marine
    10 - Dust
    11 - Polluted Continental/Smoke
    12 - Clean Continental
    13 - Polluted Dust
    14 - Elevated Smoke (Tropospheric)
    15 - Dusty Marine
    16 - PSC Aerosol
    17 - Volcanic Ash
    18 - Sulfate/Other
    19 - Elevated Smoke (Stratospheric)

    :param arr: input 16-bit binary representation of data.
    :return: integer from given values.
    '''
    # Check bits 1-3 for feature type
    # Check if clear air
    if int(arr[-3:], 2) == 1:
        output = 0
    # Check if cloud
    elif int(arr[-3:], 2) == 2:
        output = int(arr[-12:-9], 2) + 1
    # Check if tropospheric aerosol
    elif int(arr[-3:], 2) == 3:
        if int(arr[-12:-9], 2) == 0:
            output = np.nan
        else:
            output = int(arr[-12:-9], 2) + 8
    # Check if stratospheric aerosol
    elif int(arr[-3:], 2) == 4:
        if int(arr[-12:-9], 2) == 0:
            output = np.nan
        else:
            output = int(arr[-12:-9], 2) + 15
    # Set all other values to NaN
    else:
        output = np.nan
    return output
custom_feature_conversion = np.vectorize(custom_feature_conversion)

def calipso_to_datetime(cal_utc_time):
    """
    Adapted from pixel_collocation used in MSci project. Will convert CALIOP
    profile UTC time to a datetime object.

    :param cal_utc_time: str type. CALIOP UTC profile time as strings
    :return datetime object. datetime version of input UTC time
    """
    utc_time = str(cal_utc_time)
    # print('UTC time: ', utc_time)
    if len(utc_time) != 0:
        # year = '20' + utc_time[:2]
        # month = utc_time[2:4]
        # day = utc_time[4:6]
        fraction_of_day = float('0' + utc_time[6:])
        seconds_through_day = fraction_of_day * 24 * 3600
        # print('Fraction of day: %0.4f\n Seconds through day: %0.4f' % (fraction_of_day, seconds_through_day))
        hh_mm_ss = str(datetime.timedelta(seconds=(seconds_through_day)))
        if len(hh_mm_ss) >= 12:
            hh_mm_ss = hh_mm_ss[:12]
        new_time = '20' + utc_time[:6] + 'T' + hh_mm_ss
        if new_time[:8] == '20150631':
            print(new_time)
        if '.' not in hh_mm_ss:
            try:
                new_time = datetime.datetime.strptime(new_time, '%Y%m%dT%H:%M:%S')
                return new_time
            except:
                return datetime.datetime.strptime('20000101T00:00:00', '%Y%m%dT%H:%M:%S')
        else:
            try:
                new_time = datetime.datetime.strptime(new_time, '%Y%m%dT%H:%M:%S.%f')
                return new_time
            except:
                return datetime.datetime.strptime('20000101T00:00:00.000', '%Y%m%dT%H:%M:%S.%f')
calipso_to_datetime = np.vectorize(calipso_to_datetime)

def map_caliop_run(caliop_profile):
    '''
    Create a map showing the input CALIOP run
    :param caliop_profile:
    :return:
    '''
    ax = plt.axes(projection=ccrs.Geostationary(central_longitude=140.735785863))
    lats = caliop_profile.select('Latitude').get().flatten()
    longs = caliop_profile.select('Longitude').get().flatten()
    ax.add_feature(feature.LAND)
    ax.add_feature(feature.OCEAN)
    ax.add_feature(feature.COASTLINE)
    ax.set_global()
    ax.gridlines(crs=ccrs.Geostationary(central_longitude=140.735785863),
                 draw_labels=True, linewidth=2, color='gray',
                 alpha=0.5, linestyle='-')
    plt.plot(longs, lats, 'r',
            transform = ccrs.Geodetic())
    return ax

def curtain_plot(caliop_filename, fig, ax, option='VFM', apply_mask=False,
                 mask_lims=(-55.05, -9.139722, 112.933333, 159.25)):
    """
    Creates a curtain plot from the given CALIOP file.

    :param caliop_filename:
    :param apply_mask:
    :return: matplotlib.pyplot figure of curatin plot.

    :param caliop_filename: str type. Name of the CALIOP file.
    :param fig: matplotlib.pyplot figure on which to draw the
                plotted data.
    :param ax: matplotlib.pyplot figure axis on which to draw
               the plotted data.
    :param option: str type. Name of the data to be plotted.
                   can choose from:
                   - VFM
                   - Backscatter
                   - OD
                   - QA
                   - CAD
    :param apply_mask: boolean type. If True, will only produce
                       a curtain plot within the limits specified
                       by mask_lims.
    :param mask_lims: tuple type. Tuple of minimum latitude,
                      maximum latitude, minimum longitude,
                      maximum longitude within which the data
                      will be plotted.
    :return: matplotlib.pyplot figure with curatin plot drawn
             onto specified axis.
    """
    options = {'metadata': ['Feature CALIOP Name',
                            'Clear Value',
                            'Fill Value',
                            'Heatmap Limits',
                            'Title'],
               'VFM': ['Feature_Classification_Flags',
                       0.5,
                       0.5,
                       [0., 20.],
                       'Vertical Feature Mask'],
               'Backscatter': ['Integrated_Attenuated_Backscatter_532',
                               0.,
                               -9999.,
                               [0., 2.],
                               'Backscatter'],
               'OD': ['Feature_Optical_Depth_532',
                      0.,
                      -9999.,
                      [0., 5.],
                      'Optical Depth'],
               'QA': ['Layer_IAB_QA_Factor',
                      0.,
                      -9999.,
                      [0., 1.],
                      'Quality Assurance'],
               'CAD': ['CAD_Score',
                       0.,
                       -9999.,
                       [-101., 106.],
                       'CAD Score']}
    data_to_plot = options[option]
    custom_type_arr = np.asarray(['Clear', # Clear Air
                                  'LO (Tr)', # Low Overcast (Transparent)
                                  'LO (Op)', # Low Overcast (Opaque)
                                  'TSc', # Transition Stratocumulus
                                  'LBC', # Low Broken Cumulus
                                  'Ac (Tr)', # Altocumulus (Transparent)
                                  'As (Op)', # Altostratus (Opaque)
                                  'Ci (Tr)', # Cirrus (Transparent)
                                  'DC (Op)', # Deep Convective (Opaque)
                                  'CM', # Clean Marine
                                  'D', # Dust
                                  'PC/S', # Polluted Continental/Smoke
                                  'CC', # Clean Continental
                                  'PD', # Polluted Dust
                                  'ES (Ts)', # Elevated Smoke (Tropospheric)
                                  'DM', # Dusty Marine
                                  'PSCA', # PSC Aerosol
                                  'VA', # Volcanic Ash
                                  'S/O', # Sulfate/Other
                                  'ES (Ss)']) # Elevated Smoke (Stratospheric)

    file = SD(caliop_filename, SDC.READ)
    lats = file.select('Latitude').get()
    # lats = np.average(lats, axis=1)
    lats = lats.flatten()
    lats = lats.reshape(lats.shape[0], 1)
    longs = file.select('Longitude').get()
    big_bar_shape = longs.shape
    # longs = np.average(longs, axis=1)
    longs = longs.flatten()
    longs = longs.reshape(longs.shape[0], 1)
    trop_height = np.around(file.select('Tropopause_Height').get(), 2)
    trop_height = np.repeat(trop_height, np.full((trop_height.shape[0]), 3), axis=0)
    base_alts = file.select('Layer_Base_Altitude').get()
    base_alts = np.repeat(base_alts, np.full((base_alts.shape[0]), 3), axis=0)
    top_alts = file.select('Layer_Top_Altitude').get()
    top_alts = np.repeat(top_alts, np.full((top_alts.shape[0]), 3), axis=0)
    vflags = file.select(data_to_plot[0]).get()
    vflags = np.repeat(vflags, np.full((vflags.shape[0]), 3), axis=0)
    if option == 'VFM':
        vflags = custom_feature_conversion(number_to_bit(vflags)).astype('float') + 0.5
    ### Create big bar array ###
    # big_bars = np.full(big_bar_shape, np.nan)
    # if option == 'VFM':
    #     big_bars[::2] = np.full((3,), 6.5)
    # else:
    #     big_bars[::2] = np.full((3,), np.nanmax(vflags) / 4.)
    # big_bars = big_bars.flatten()
    ### Apply Mask (Default is Australia Only) ###
    if apply_mask:
        min_lat, max_lat = mask_lims[0], mask_lims[1]
        min_lon, max_lon = mask_lims[2], mask_lims[3]
        mask = np.where((lats > min_lat) & (lats < max_lat) & (longs > min_lon) & (longs < max_lon))[0]
        lats = lats[mask]
        longs = longs[mask]
        trop_height = trop_height[mask]
        base_alts = base_alts[mask]
        top_alts = top_alts[mask]
        vflags = vflags[mask]
    #     big_bars = big_bars[mask]
    # big_bars = np.dstack(tuple((big_bars,) for i in range(100)))
    alts_res = 0.01
    alts = np.arange(start=-0.5, # - (100*alts_res)
                     stop=30. + alts_res,
                     step=alts_res)
    plottables = np.full((vflags.shape[0], alts.shape[0]), data_to_plot[1])
    for idx in np.ndindex(vflags.shape[0]):
        fill = vflags[idx]
        top_col = top_alts[idx]
        base_col = base_alts[idx]
        non_clear_locs = np.where(fill != data_to_plot[2])[0]
        for fill_loc in non_clear_locs:
            fill_top = top_col[fill_loc]
            fill_base = base_col[fill_loc]
            plottables[idx, np.where(np.logical_and(alts <= fill_top, alts >= fill_base))] = fill[fill_loc]
    # if option == 'VFM':
    #     plottables[::2, 50:100] = 14.5
    #     plottables[1::2, 50:100] = 16.5
    # else:
    #     plottables[::2, 50:100] = np.nanmax(vflags) / 2.
    #     plottables[1::2, 50:100] = np.nanmax(vflags)
    # plottables[:, :100] = big_bars
    if option == 'VFM':
        cmap = cm.tab20
    else:
        cmap = cm.plasma
    if option == 'Backscatter':
        data_to_plot[3][-1] = np.nanmax(vflags)
    cmap.set_bad('black', 1.)
    cax = ax.imshow(np.transpose(plottables, (1, 0)),
                    interpolation='nearest',
                    origin='lower',
                    aspect='auto',
                    vmin=data_to_plot[3][0],
                    vmax=data_to_plot[3][-1],
                    cmap=cmap)
    ax.set_title(data_to_plot[4])
    x_ticks = np.linspace(0, len(lats), 15).astype('int')
    x_ticks_mod = x_ticks
    x_ticks_mod[0] = 0
    x_ticks_mod[-1] = x_ticks_mod[-1] - 1
    lats = np.around(lats[x_ticks_mod], 2)
    longs = np.around(longs[x_ticks_mod], 2)
    x_labs = np.asarray([str(lats[idx]) + '\n' + str(longs[idx]) for idx in np.ndindex(lats.shape)])
    # for idx in np.ndindex(ns.shape):
    #     label = str(lats[idx]) + ns[idx] + str(longs[idx])
    #     x_labs.append(label)
    # x_labs = np.asarray(x_labs)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labs)
    y_ticks = np.linspace(50, len(alts), 16).astype('int')
    y_ticks_mod = y_ticks
    y_ticks_mod[-1] = y_ticks_mod[-1] - 1
    y_labs = np.around(alts[y_ticks_mod], 2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labs)
    # if option == 'VFM':
    #     cbar = fig.colorbar(cax, ticks=np.arange(0.5, 19.5+1, 1),
    #                         orientation='vertical', ax=ax, pad=0.01)
    #     cbar.ax.set_yticklabels(custom_type_arr,
    #                             fontsize=8)
    #     ax.plot(trop_height * (1 / alts_res), 'k-')
    # else:
    #     fig.colorbar(cax, cmap=cmap, ax=ax, orientation='vertical', pad=0.01)
    #     ax.plot(trop_height * (1 / alts_res), 'w-')
    return cax, cmap

def mass_plot(filename, apply_mask=False, mask_lims=(-55.05, -9.139722, 112.933333, 159.25)):
    """
    Generates a single figure containing multiple curtain plots.

    :param filename:
    :param apply_mask:
    :param mask_lims:
    :return:
    """
    fig, axes = plt.subplots(5, 1, figsize=(20, 16), sharex=True, sharey=True)
    fig.suptitle('Curtain Plots for %s' % filename[-60:-4], fontsize=14)
    curtain_plot(filename, fig, axes[0], 'VFM', apply_mask, mask_lims)
    curtain_plot(filename, fig, axes[1], 'Backscatter', apply_mask, mask_lims)
    curtain_plot(filename, fig, axes[2], 'OD', apply_mask, mask_lims)
    curtain_plot(filename, fig, axes[3], 'QA', apply_mask, mask_lims)
    curtain_plot(filename, fig, axes[-1], 'CAD', apply_mask, mask_lims)
    fig.text(0.015, 0.5, 'Altitude [$km$]', fontsize=14, va='center', rotation='vertical')  # y label
    fig.tight_layout(rect=[0.03, 0.04, 0.96, 0.95])
    return fig

def pdf_plots(path_to_save, pdfname, datafname):
    """


    :param path_to_save:
    :param pdfname:
    :param datafname:
    :return:
    """
    from matplotlib.backends.backend_pdf import PdfPages
    fullname = os.path.join(path_to_save, pdfname+'.pdf')
    pdf_pages = PdfPages(fullname)
    plottable_list = ['VFM', 'Backscatter', 'OD', 'QA', 'CAD']
    i=1
    ### 1st Page ###
    fig, axes = plt.subplots(3, 1,
                             figsize=(16., 9.),
                             dpi=300,
                             sharex=True,
                             sharey=True)
    fig.suptitle('Curtain Plots for %s' % datafname[-60:-4], fontsize=14)
    for i, n in enumerate(plottable_list[:3]):
        curtain_plot(datafname, fig, axes[i], n,
                     apply_mask=True)
    fig.text(0.015, 0.5, 'Altitude [$km$]', fontsize=14, va='center', rotation='vertical')  # y label
    fig.tight_layout(rect=[0.03, 0.04, 0.99, 0.95])
    pdf_pages.savefig(fig, bbox_inches='tight')
    ### 2nd Page ###
    fig, axes = plt.subplots(2, 1,
                             figsize=(16., 9.),
                             dpi=300,
                             sharex=True,
                             sharey=True)
    fig.suptitle('Curtain Plots for %s' % datafname[-60:-4], fontsize=14)
    for i, n in enumerate(plottable_list[3:]):
        curtain_plot(datafname, fig, axes[i], n,
                     apply_mask=True)
    fig.text(0.015, 0.5, 'Altitude [$km$]', fontsize=14, va='center', rotation='vertical')  # y label
    fig.tight_layout(rect=[0.03, 0.04, 0.99, 0.95])
    pdf_pages.savefig(fig, bbox_inches='tight')
    pdf_pages.close()


# if __name__ == '__main__':
#     fname = '/mnt/c/Users/drob0013/PhD/Data/CALIOP/CAL_LID_L2_05kmMLay-Standard-V4-20.2018-03-19T03-51-52ZD.hdf'
    # fig = mass_plot(fname, True)
    # fig.savefig(os.path.join('/mnt/c/Users/drob0013/PhD/Diagnostic_Images',
    #                          '%s_curtain_plots.png' % fname[-60:-4]))
    # plt.show(block=True)
    # pdf_plots('/mnt/c/Users/drob0013/PhD/Diagnostic_Images',
    #           'mass_plots',
    #           fname)
