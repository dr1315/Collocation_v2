import os
import sys
import numpy as np
import pandas as pd
from pysolar.solar import get_altitude_fast
from pyorbital.orbital import get_observer_look
from pyorbital.astronomy import get_alt_az
import datetime as dt
from datetime import timezone
sys.path.append("/g/data/k10/dr1709/code/Personal/Tools")
import him8analysis as h8a
import collocation as col

### For NN Training and Validation ###

def normalise_data(dataframe):
    norm_vals = {'1': [50., 100.], # 120.,
                 '2': [50., 100.], # 135.,
                 '3': [50., 100.], # 165.,
                 '4': [50., 100.], # 205.,
                 '5': [50., 100.], # 121.,
                 '6': [50., 100.], # 132.,
                 '7': [273.15, 70.], # 401.,
                 '8': [273.15, 70.], # 317.,
                 '9': [273.15, 70.], # 327.,
                 '10': [273.15, 70.], # 327.,
                 '11': [273.15, 70.], # 344.,
                 '12': [273.15, 70.], # 328.,
                 '13': [273.15, 70.], # 371.,
                 '14': [273.15, 70.], # 348.,
                 '15': [273.15, 70.], # 403.,
                 '16': [273.15, 70.], # 410.,
                 'LAT': [90., 180.],
                 'LON': [180., 360.],
                 'DATE': 366.,
                 'TIME': (24. * 3600 + 1.),
                 'ANGLES': 360.,
                 'SZA': 90.,
                 'OZA': 45.}
    normalised_data = {}
    ### Normalise Band Inputs ###
    for band_number in range(1, 16+1):
        if band_number <= 4:
            for value_type in ['Mean', 'Sigma']:
                key = 'Himawari Band %s %s at 2km Resolution' % (str(band_number), value_type)
                norm_data = (dataframe[key] - norm_vals[str(band_number)][0]) / norm_vals[str(band_number)][1]
                normalised_data[key] = np.array(list(norm_data))
        else:
            key = 'Himawari Band %s %s at 2km Resolution' % (str(band_number), 'Value')
            norm_data = (dataframe[key] - norm_vals[str(band_number)][0]) / norm_vals[str(band_number)][1]
            normalised_data[key] = np.array(list(norm_data))
    ### Normalise Latitudes ###
    norm_lats = (dataframe['Himawari Latitude'] + norm_vals['LAT'][0]) / norm_vals['LAT'][-1]
    norm_lats = np.array(list(norm_lats))
    normalised_data['Latitude'] = norm_lats
    ### Normalise Longitudes ###
    norm_lons = (dataframe['Himawari Longitude'] + norm_vals['LON'][0]) / norm_vals['LON'][-1]
    norm_lons = np.array(list(norm_lons))
    normalised_data['Longitude'] = norm_lons
    ### Normalise Date and Time Inputs ###
    # Date #
    dtime_start = dataframe['Himawari Scene Start Time']
    dtime_delta = dataframe['Himawari Scene End Time'] - dtime_start
    dtime_avg = dtime_start + dtime_delta / 2
    ydays = []
    for d in dtime_avg:
        yday = d.timetuple().tm_yday
        ydays.append([yday])
    ydays = np.array(ydays)
    ydays = ydays / norm_vals['DATE']
    normalised_data['Date'] = ydays
    # Time #
    dsecs = []
    for t in dtime_avg:
        dsec = (t.hour * 3600) + (t.minute * 60) + (t.second)
        dsecs.append([dsec])
    dsecs = np.array(dsecs)
    dsecs = dsecs / norm_vals['TIME']
    normalised_data['Time'] = dsecs
    angles = ['Himawari Solar Zenith Angle', # -90 /90
              # 'Himawari Solar Azimuth Angle',
              # 'Himawari Observer Elevation Angle',
              'Himawari Observer Zenith Angle', # -45 /45
              # 'Himawari Observer Azimuth Angle'
              ]
    angle = np.array(list(dataframe['Himawari Solar Zenith Angle'])) - norm_vals['SZA']
    normalised_data['Himawari Solar Zenith Angle'] = angle / norm_vals['SZA']
    angle = np.array(list(dataframe['Himawari Observer Zenith Angle'])) - norm_vals['OZA']
    normalised_data['Himawari Observer Zenith Angle'] = angle / norm_vals['OZA']
    # for angle in angles:
    #     normalised_data[angle] = np.array(list(dataframe[angle])) / norm_vals['ANGLES']
        # if angle == 'Himawari Solar Azimuth Angle':
        #     normalised_data[angle] = (np.array(list(dataframe[angle])) + 180.) / norm_vals['ANGLES']
        # else:
        #     normalised_data[angle] = np.array(list(dataframe[angle])) / norm_vals['ANGLES']
    # print(normalised_data)
    return normalised_data

def format_inputs(normalised_data):
    inputs = [
              'Himawari Band 1 Mean at 2km Resolution',
              'Himawari Band 1 Sigma at 2km Resolution',
              'Himawari Band 2 Mean at 2km Resolution',
              'Himawari Band 2 Sigma at 2km Resolution',
              'Himawari Band 3 Mean at 2km Resolution',
              'Himawari Band 3 Sigma at 2km Resolution',
              'Himawari Band 4 Mean at 2km Resolution',
              'Himawari Band 4 Sigma at 2km Resolution',
              'Himawari Band 5 Value at 2km Resolution',
              'Himawari Band 6 Value at 2km Resolution',
              'Himawari Band 7 Value at 2km Resolution',
              'Himawari Band 8 Value at 2km Resolution',
              'Himawari Band 9 Value at 2km Resolution',
              'Himawari Band 10 Value at 2km Resolution',
              'Himawari Band 11 Value at 2km Resolution',
              'Himawari Band 12 Value at 2km Resolution',
              'Himawari Band 13 Value at 2km Resolution',
              'Himawari Band 14 Value at 2km Resolution',
              'Himawari Band 15 Value at 2km Resolution',
              'Himawari Band 16 Value at 2km Resolution',
              # 'Latitude',
              # 'Longitude',
              # 'Date',
              # 'Time',
              'Himawari Solar Zenith Angle',
              # 'Himawari Solar Azimuth Angle',
              # 'Himawari Observer Elevation Angle',
              'Himawari Observer Zenith Angle',
              # 'Himawari Observer Azimuth Angle'
              ]
    for input in inputs:
        data = normalised_data[input]
        normalised_data[input] = data.reshape(len(data), 1)
    arr_list = [normalised_data[input] for input in inputs]
    data_inputs = np.hstack(tuple(arr_list))
    return data_inputs

def process_auxiliaries(dataframe):
    print('Adding Auxiliary Information')
    ODs = list(dataframe['CALIOP ODs for 532nm'])
    sum_ODs = []
    for i in ODs:
        i[i == -9999.] = 0.
        sum_ODs.append(sum(i))
    sum_ODs = np.array(sum_ODs)
    surface_types = list(dataframe['CALIOP IGBP Surface Types'])
    surface_types = [i[-1] for i in surface_types]
    surface_types = np.array(surface_types)
    top_heights = (basic_height_regression_classifiers(dataframe) * 30.6) - 0.5
    top_heights = top_heights.flatten()
    lats = np.array(list(dataframe['Himawari Latitude']))
    SZAs = np.array(list(dataframe['Himawari Solar Zenith Angle']))
    OZAs = np.array(list(dataframe['Himawari Observer Zenith Angle']))
    spatial_diffs = get_spatial_diff(dataframe)
    print('Done')
    return sum_ODs, surface_types, top_heights, lats, SZAs, OZAs, spatial_diffs

def get_sza(dataframe):
    ODs = list(dataframe['CALIOP ODs for 532nm'])
    dts = list(dataframe['CALIOP Pixel Scan Times'])
    lats = list(dataframe['CALIOP Latitudes'])
    lons = list(dataframe['CALIOP Longitudes'])
    SZAs = []
    print('Calculating Solar Zenith Angles')
    n = 1
    for i, j, k, l in zip(ODs, dts, lats, lons):
        print('Calculating Angle %d/%d       ' % (n, len(ODs)), end='\r')
        thickest = np.argmax(i)
        thick_obj_dt = j[thickest].replace(tzinfo=dt.timezone.utc)
        thick_obj_lat = k[thickest]
        thick_obj_lon = l[thickest]
        SZAs += [90. - get_altitude_fast(thick_obj_lat, thick_obj_lon, thick_obj_dt)]
        n+=1
    SZAs = np.array(SZAs)
    print('All %d Solar Zenith Angles Calculated          ' % len(ODs))
    return SZAs

def get_spatial_diff(dataframe):
    him_lats = np.array(list(dataframe['Himawari Latitude']))
    him_lons = np.array(list(dataframe['Himawari Longitude']))
    him_lons = him_lons - 140.7
    him_lons[him_lons <= -180.] += 360.
    him_lons[him_lons > 180.] -= 360.
    cal_lats = np.array([i[-1] for i in list(dataframe['CALIOP Latitudes'])])
    cal_lons = np.array([i[-1] for i in list(dataframe['CALIOP Longitudes'])])
    cal_lons = cal_lons - 140.7
    cal_lons[cal_lons <= -180.] += 360.
    cal_lons[cal_lons > 180.] -= 360.
    him_lats = np.deg2rad(him_lats)
    him_lons = np.deg2rad(him_lons)
    cal_lats = np.deg2rad(cal_lats)
    cal_lons = np.deg2rad(cal_lons)
    dlat = (him_lats - cal_lats)
    dlon = (him_lons - cal_lons)
    a = (np.sin(dlat / 2) ** 2) + (np.cos(him_lats) * np.cos(cal_lats) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return c*6372.8

def basic_binary_classifiers(dataframe):
    features = list(dataframe['CALIOP Vertical Feature Mask'])
    bin_features = []
    for item in features:
        if 0 in item:
            bin_features.append(np.array([1., 0.]))
        else:
            bin_features.append(np.array([0., 1.]))
    bin_features = np.array(bin_features)
    print("# of 0's: %d" % np.sum(np.all(bin_features == np.array([1., 0.]), axis=1)))
    print("# of 1's: %d" % np.sum(np.all(bin_features == np.array([0., 1.]), axis=1)))
    return np.array(bin_features)

def basic_binary_cloud_classifiers(dataframe):
    features = list(dataframe['CALIOP Vertical Feature Mask'])
    bin_features = []
    for item in features:
        cloud_object = [i in range(1,8+1) for i in item]
        cloud_in_pixel = sum(cloud_object) > 0
        if cloud_in_pixel:
            bin_features.append(np.array([0., 1.]))
        else:
            bin_features.append(np.array([1., 0.]))
    bin_features = np.array(bin_features)
    print("# of 0's: %d" % np.sum(np.all(bin_features == np.array([1., 0.]), axis=1)))
    print("# of 1's: %d" % np.sum(np.all(bin_features == np.array([0., 1.]), axis=1)))
    return np.array(bin_features)

def high_OD_binary_classifiers(dataframe):
    features = list(dataframe['CALIOP ODs for 532nm'])
    bin_features = []
    for item in features:
        if sum(item) > 0.3:
            bin_features.append(np.array([1., 0.]))
        else:
            bin_features.append(np.array([0., 1.]))
    bin_features = np.array(bin_features)
    print(bin_features.shape, bin_features[:10])
    print("# of 0's: %d" % np.sum(np.all(bin_features == np.array([1., 0.]), axis=1)))
    print("# of 1's: %d" % np.sum(np.all(bin_features == np.array([0., 1.]), axis=1)))
    return np.array(bin_features)

def clear_cloud_aerosol_mixed_classifiers(dataframe):
    """
    [0] -> Clear
    [1] -> Cloud only
    [2] -> Aerosol only
    [3] -> Mixed cloud and aerosol

    :param dataframe:
    :return:
    """
    features = list(dataframe['CALIOP Vertical Feature Mask'])
    bin_features = []
    for item in features:
        cloud_object = [i in range(1, 8 + 1) for i in item]
        cloud_in_pixel = sum(cloud_object) > 0
        aerosol_object = [i > 8 for i in item]
        aerosol_in_pixel = sum(aerosol_object) > 0
        if cloud_in_pixel and aerosol_in_pixel:
            bin_features.append(np.array([0., 0., 0., 1.]))
        elif cloud_in_pixel and not aerosol_in_pixel:
            bin_features.append(np.array([0., 1., 0., 0.]))
        elif not cloud_in_pixel and aerosol_in_pixel:
            bin_features.append(np.array([0., 0., 1., 0.]))
        else:
            bin_features.append(np.array([1., 0., 0., 0.]))
    bin_features = np.array(bin_features)
    print("# of Clear: %d" % np.sum(np.all(bin_features == np.array([1., 0., 0., 0.]), axis=1)))
    print("# of Cloud Only: %d" % np.sum(np.all(bin_features == np.array([0., 1., 0., 0.]), axis=1)))
    print("# of Aerosol Only: %d" % np.sum(np.all(bin_features == np.array([0., 0., 1., 0.]), axis=1)))
    print("# of Mixed: %d" % np.sum(np.all(bin_features == np.array([0., 0., 0., 1.]), axis=1)))
    return np.array(bin_features)

def complex_classifiers(dataframe):
    """
    [0] -> Clear
    [1] -> Cloud only
    [2] -> Aerosol only
    [3] -> Cloud over Aerosol (change to 1 OD over n OD and >0.1km difference in height)
    [4] -> Aerosol over Cloud (change to 1 OD over n OD and >0.1km difference in height)
    [5] -> Fully Mixed (overlap in layers)

    Future: Add thin over thin, thin over thick (and thick over thin, thick over thick?)

    :param dataframe:
    :return:
    """
    features = list(dataframe['CALIOP Vertical Feature Mask'])
    # ODs = list(dataframe['CALIOP ODs for 532nm'])
    tops = list(dataframe['CALIOP Feature Top Altitudes'])
    bases = list(dataframe['CALIOP Feature Base Altitudes'])
    bin_features = []
    for item, item_bases, item_tops in zip(features, tops, bases):
        cloud_object = [i in range(1, 8 + 1) for i in item]
        cloud_in_pixel = sum(cloud_object) > 0
        aerosol_object = [i > 8 for i in item]
        aerosol_in_pixel = sum(aerosol_object) > 0
        if sum(cloud_object) + sum(aerosol_object) > 1: # Objects in pixel?
            if cloud_in_pixel and aerosol_in_pixel: # Either object over object or mixed
                overlap = [item_bases[i] < item_tops[i+1] for i in range(len(item_bases)-1)]
                if not overlap[0]: # No overlap in the first layer
                    if cloud_object[0]: # If top object is cloud
                        bin_features.append(np.array([0., 0., 0., 1., 0., 0.])) # Cloud over aerosol
                    elif aerosol_object[0]: # If top object is aerosol
                        bin_features.append(np.array([0., 0., 0., 0., 1., 0.])) # Aerosol over cloud
                else:
                    bin_features.append(np.array([0., 0., 0., 0., 0., 1.])) # Fully mixed
            elif cloud_in_pixel and not aerosol_in_pixel:
                bin_features.append(np.array([0., 1., 0., 0., 0., 0.])) # Cloud only
            elif not cloud_in_pixel and aerosol_in_pixel:
                bin_features.append(np.array([0., 0., 1., 0., 0., 0.])) # Aerosol only
        else:
            bin_features.append(np.array([1., 0., 0., 0., 0., 0.])) # Clear air
    bin_features = np.array(bin_features)
    print("# of Clear: %d" % np.sum(np.all(bin_features == np.array([1., 0., 0., 0., 0., 0.]), axis=1)))
    print("# of Cloud Only: %d" % np.sum(np.all(bin_features == np.array([0., 1., 0., 0., 0., 0.]), axis=1)))
    print("# of Aerosol Only: %d" % np.sum(np.all(bin_features == np.array([0., 0., 1., 0., 0., 0.]), axis=1)))
    print("# of Cloud Over Aerosol: %d" % np.sum(np.all(bin_features == np.array([0., 0., 0., 1., 0., 0.]), axis=1)))
    print("# of Aerosol Over Cloud: %d" % np.sum(np.all(bin_features == np.array([0., 0., 0., 0., 1., 0.]), axis=1)))
    print("# of Mixed: %d" % np.sum(np.all(bin_features == np.array([0., 0., 0., 0., 0., 1.]), axis=1)))
    return np.array(bin_features)

def advanced_classifiers(dataframe):
    """
    [0] -> Clear (OD < 0.1)
    [1] -> Thick Cloud only (OD > 3. and top layer(s))
    [2] -> Thick Smoke only (OD > 3. and top layer(s))
    [3] -> Thick Dust only (OD > 3. and top layer(s))
    [4] -> Thick Other Aerosol (OD > 3. and contains layers of aerosol and cloud)
    [5] -> Mixed Thick Objects ( 0.1 < AOD < 3. over COD > 3.)
    [6] -> Thin Cloud only (0.1 < OD < 3. and only clouds)
    [7] -> Thin Smoke only (0.1 < OD < 3. and top layer(s))
    [8] -> Thin Dust only (0.1 < OD < 3. and top layer(s))
    [9] -> Thin Aerosol only (0.1 < OD < 3. and only aerosols)
    [10] -> Mixed Thin Objects (Each layer has 0.1 < OD < 3. and contains layers of aerosol and cloud)

    :param dataframe:
    :return:
    """
    features = list(dataframe['CALIOP Vertical Feature Mask'])
    ODs = list(dataframe['CALIOP ODs for 532nm'])
    tops = list(dataframe['CALIOP Feature Top Altitudes'])
    bases = list(dataframe['CALIOP Feature Base Altitudes'])
    bin_features = []
    for item, item_tops, item_bases, item_ODs in zip(features, tops, bases, ODs):
        bin_feature = np.zeros(11)
        if sum(item_ODs) > 0.1:
            if sum(item_ODs) >= 3.:
                col_OD = 0.
                last_index = 0
                while col_OD < 3.:
                    col_OD += item_ODs[last_index]
                    last_index += 1
                top_features = item[:last_index]
                if np.all([i in range(1, 8 + 1) for i in top_features]): # Cloud only
                    bin_feature[1] = 1. # Thick cloud
                elif np.all([i in [11, 14, 19] for i in top_features]): # Smoke only
                    bin_feature[2] = 1. # Thick smoke
                elif np.all([i in [10, 13, 15] for i in top_features]): # Dust only
                    bin_feature[3] = 1. # Thick dust
                elif np.all([i > 8 for i in top_features]): # All aerosols
                    bin_feature[4] = 1. # Thick aerosols
                else:
                    bin_feature[5] = 1. # Thick mixed objects
            else: # 0.1 < OD < 3.
                if np.all([i in range(1, 8 + 1) for i in item]):
                    bin_feature[6] = 1. # Thin cloud
                elif np.any([i in [11, 14, 19, 10, 13, 15] for i in item]):
                    smoke_mask = (np.array(item) == 11) | (np.array(item) == 14) | (np.array(item) == 19)
                    dust_mask = (np.array(item) == 10) | (np.array(item) == 13) | (np.array(item) == 15)
                    if np.any(smoke_mask) and not np.any(dust_mask): # Smoke and no dust
                        if sum(item_ODs[smoke_mask]) > 0.5*sum(item_ODs): # Smoke makes up more than 50% of the pixel OD
                            bin_feature[7] = 1. # Thin smoke
                        elif np.all([i > 8 for i in item]):
                            bin_feature[9] = 1. # Thin aerosol
                        else:
                            bin_feature[10] = 1. # Mixed thin objects
                    elif np.any(dust_mask) and not np.any(smoke_mask): # Dust and no smoke
                        if sum(item_ODs[dust_mask]) > 0.5*sum(item_ODs): # Dust makes up more than 50% of the pixel OD
                            bin_feature[8] = 1. # Thin dust
                        elif np.all([i > 8 for i in item]):
                            bin_feature[9] = 1. # Thin aerosol
                        else:
                            bin_feature[10] = 1. # Mixed thin objects
                    else: # Both smoke and dust
                        if sum(item_ODs[smoke_mask]) > 0.5*sum(item_ODs): # Smoke makes up more than 50% of the pixel OD
                            bin_feature[7] = 1. # Thin smoke
                        elif sum(item_ODs[dust_mask]) > 0.5*sum(item_ODs): # Dust makes up more than 50% of the pixel OD
                            bin_feature[8] = 1. # Thin dust
                        elif np.all([i > 8 for i in item]):
                            bin_feature[9] = 1. # Thin aerosol
                        else:
                            bin_feature[10] = 1. # Mixed thin objects
                elif np.all([i > 8 for i in item]):
                    bin_feature[9] = 1. # Thin aerosol
                else:
                    bin_feature[10] = 1. # Mixed thin objects
        else:
            bin_feature[0] = 1. # Clear
        bin_features.append(bin_feature)
    bin_features = np.array(bin_features)
    print("# of Clear: %d" % np.sum(bin_features[:, 0]))
    print("# of Thick Cloud Only: %d" % np.sum(bin_features[:, 1]))
    print("# of Thick Smoke Only: %d" % np.sum(bin_features[:, 2]))
    print("# of Thick Dust Only: %d" % np.sum(bin_features[:, 3]))
    print("# of Thick Other Aerosols: %d" % np.sum(bin_features[:, 4]))
    print("# of Mixed Thick Objects: %d" % np.sum(bin_features[:, 5]))
    print("# of Thick Cloud Only: %d" % np.sum(bin_features[:, 6]))
    print("# of Thick Smoke Only: %d" % np.sum(bin_features[:, 7]))
    print("# of Thick Dust Only: %d" % np.sum(bin_features[:, 8]))
    print("# of Thick Other Aerosols: %d" % np.sum(bin_features[:, 9]))
    print("# of Mixed Thick Objects: %d" % np.sum(bin_features[:, 10]))
    return np.array(bin_features)

def advanced_classifiers_v1(dataframe):
    """
    [0] -> Clear (OD < 0.1) and/or only clean marine and continental
    [1] -> Thick Cloud only (OD > 3. and top layer(s))
    [2] -> Thick Aerosol (OD > 3. and contains layers of aerosol and cloud)
    [3] -> Mixed Thick Objects ( 0.1 < AOD < 3. over COD > 3.)
    [4] -> Thin Cloud only (0.1 < OD < 3. and only clouds)
    [5] -> Thin Aerosol only (0.1 < OD < 3. and only aerosols)
    [6] -> Mixed Thin Objects (Each layer has 0.1 < OD < 3. and contains layers of aerosol and cloud)

    :param dataframe:
    :return:
    """
    ### Define ID number lists ###
    clear_ids = [0, 9, 12]
    cloud_ids = [i for i in range(1, 8 + 1)]
    aerosol_ids = [10, 11] + [i for i in range(13, 19 + 1)]
    ### Load the data for collocted pixels ###
    features = list(dataframe['CALIOP Vertical Feature Mask'])
    ODs = list(dataframe['CALIOP ODs for 532nm'])
    ### Define the list of classifiers to be returned at the end of the function ###
    bin_features = []
    ### Classify each pixel based on the CALIOP VFM information ###
    for item, item_ODs in zip(features, ODs):
        ### Change clear ODs (-9999.) to 0. ###
        item_ODs = np.array(item_ODs)
        item_ODs[item_ODs == -9999.] = 0.
        item_ODs = list(item_ODs)
        ### Create empty classifier ###
        bin_feature = np.zeros(7)
        if np.all([i in clear_ids for i in item]):
            bin_feature[0] = 1.  # Clear
        else:
            non_clear_indices = [i for i, j in enumerate(item) if j not in clear_ids]
            item = [item[i] for i in non_clear_indices]
            item_ODs = [item_ODs[i] for i in non_clear_indices]
            if sum(item_ODs) > 3.:
                col_OD = 0.
                last_index = 0
                while col_OD < 3.:
                    col_OD += item_ODs[last_index]
                    last_index += 1
                top_features = item[:last_index]
                if np.all([i in cloud_ids for i in top_features]): # Cloud only
                    bin_feature[1] = 1. # Thick cloud
                elif np.all([i in aerosol_ids for i in top_features]): # Aerosols only
                    bin_feature[2] = 1. # Thick aerosol
                else:
                    bin_feature[3] = 1. # Mixed Thick Objects
            else:
                col_OD = 0.
                last_index = 0
                while col_OD < 0.5 * sum(item_ODs):
                    col_OD += item_ODs[last_index]
                    last_index += 1
                top_features = item[:last_index]
                if np.all([i in cloud_ids for i in top_features]): # Cloud only
                    bin_feature[4] = 1. # Thin cloud
                elif np.all([i in aerosol_ids for i in top_features]): # Aerosols only
                    bin_feature[5] = 1. # Thin aerosol
                else:
                    bin_feature[6] = 1. # Mixed Thin Objects
        bin_features.append(bin_feature)
    bin_features = np.array(bin_features)
    print("# of Clear: %d" % np.sum(bin_features[:, 0]))
    print("# of Thick Cloud Only: %d" % np.sum(bin_features[:, 1]))
    print("# of Thick Aerosols: %d" % np.sum(bin_features[:, 2]))
    print("# of Mixed Thick Objects: %d" % np.sum(bin_features[:, 3]))
    print("# of Thin Cloud Only: %d" % np.sum(bin_features[:, 4]))
    print("# of Thin Aerosols: %d" % np.sum(bin_features[:, 5]))
    print("# of Mixed Thin Objects: %d" % np.sum(bin_features[:, 6]))
    return np.array(bin_features)

def advanced_cloud_binary(dataframe, CAD_lim=30.):
    """
    [0] -> Non-cloud (CAD < 30)
    [1] -> Cloud (CAD >= 30)

    NB// Both classifiers are for the "top layer" of the CALIOP column,
    i.e. the more than half the column optical depth or up to OD ~ 3.

    :param dataframe:
    :return:
    """
    ### Load the data for collocted pixels ###
    CADs = list(dataframe['CALIOP CAD Scores'])
    ODs = list(dataframe['CALIOP ODs for 532nm'])
    ### Define the list of classifiers to be returned at the end of the function ###
    bin_features = []
    ### Classify each pixel based on the CALIOP CAD information ###
    for item_CADs, item_ODs in zip(CADs, ODs):
        ### Change clear ODs (-9999.) to 0. ###
        item_ODs = np.array(item_ODs)
        item_ODs[item_ODs == -9999.] = 0.
        item_ODs = list(item_ODs)
        ### Create empty classifier ###
        bin_feature = np.zeros(2)
        if np.all([i < CAD_lim for i in item_CADs]):
            bin_feature[0] = 1.  # Non-cloud
        else:
            col_OD = 0.
            last_index = 0
            while col_OD < 3. and col_OD < 0.5*sum(item_ODs):
                col_OD += item_ODs[last_index]
                last_index += 1
            top_features = item_CADs[:last_index]
            if np.all([i >= CAD_lim for i in top_features]): # Cloud only
                bin_feature[1] = 1. # Cloud
            else:
                bin_feature[0] = 1. # Mixed Thick Objects
        bin_features.append(bin_feature)
    bin_features = np.array(bin_features)
    print("# of Non-cloud: %d" % np.sum(bin_features[:, 0]))
    print("# of Cloud: %d" % np.sum(bin_features[:, 1]))
    return np.array(bin_features)

def advanced_cloud_binary_v2(dataframe, CAD_lim=30.):
    """
    [0] -> Non-cloud (CAD < 30)
    [1] -> Cloud (CAD >= 30)

    NB// Both classifiers are for the "top layer" of the CALIOP column,
    i.e. the more than half the column optical depth or up to OD ~ 3.

    :param dataframe:
    :return:
    """
    ### Load the data for collocted pixels ###
    CADs = list(dataframe['CALIOP CAD Scores'])
    ODs = list(dataframe['CALIOP ODs for 532nm'])
    tops = list(dataframe['CALIOP Feature Top Altitudes'])
    tropos = list(dataframe['CALIOP Tropopause Altitudes'])
    features = list(dataframe['CALIOP Vertical Feature Mask'])
    JMA_classiefiers = list(dataframe['Himawari JMA Cloud Mask Binary'])
    ### Define the list of classifiers to be returned at the end of the function ###
    bin_features = []
    ### Classify each pixel based on the CALIOP CAD information ###
    for item_CADs, item_ODs, item_tops, item_tropos, item_vfms, jma in zip(CADs, ODs, tops, tropos, features, JMA_classiefiers):
        ### Clean out thin, stratospheric objects ###
        stratospheric = item_tops > item_tropos
        item_ODs[item_ODs == -9999.] = 0. # Change clear ODs (-9999.) to 0.
        thin = item_ODs < 0.1
        thin_strato = stratospheric
        # print(thin_strato)
        item_CADs = item_CADs[~thin_strato] # Only want to look at ODs and CADs in th rest of the algorithm
        item_ODs = item_ODs[~thin_strato]
        item_ODs = list(item_ODs)
        ### Create empty classifier ###
        bin_feature = np.zeros(2)
        if len(item_CADs) == 0:
            bin_feature[0] = 1.  # Non-cloud
        elif np.all([i <= -CAD_lim for i in item_CADs]):
            bin_feature[0] = 1.  # Non-cloud
        elif np.sum(item_ODs) < 0.1:
            bin_feature[0] = 1.  # Too thin to worry about for now
        else:
            # col_OD = 0.
            # last_index = 0
            # while col_OD < 3. and col_OD < 0.5*sum(item_ODs):
            #     col_OD += item_ODs[last_index]
            #     last_index += 1
            # top_features = item_CADs[:last_index]
            # top_features = [item_CADs[0]]
            # if np.all([i >= CAD_lim for i in top_features]): # Cloud only
            #     bin_feature[1] = 1. # Cloud
            # else:
            #     bin_feature[0] = 1. # Mixed Thick Objects
            col_OD = 0.
            last_index = 0
            for layer_CAD, layer_OD in zip(item_CADs, item_ODs):
                if col_OD < 0.1 and col_OD < 0.5 * sum(item_ODs):
                    if layer_OD >= 0.1:
                        if layer_CAD >= CAD_lim:
                            bin_feature[1] = 1.  # Cloud
                            break
                        elif layer_CAD <= -CAD_lim:
                            bin_feature[0] = 1.  # Aerosol or surface
                            break
                        else:
                            bin_feature[0] = np.nan
                            bin_feature[1] = np.nan # Data not suitable for training or validation
                            break
                    else:
                        if last_index == len(item_CADs)-1:
                            bin_feature[0] = 1.  # Too thin to be picked up as cloud
                            break
                        else:
                            col_OD += layer_OD
                            last_index += 1
                else:
                    if item_CADs[0] >= CAD_lim:  # Cloud only
                        bin_feature[1] = 1. # Cloud
                        break
                    elif item_CADs[0] <= -CAD_lim:
                        bin_feature[0] = 1.  # Aerosol or surface
                        break
                    else:
                        bin_feature[0] = np.nan
                        bin_feature[1] = np.nan  # Data not suitable for training or validation
                        break
        if bin_feature[1] != jma:
            print('#-#-#-#-#')
            print(bin_feature, jma)
            print(item_CADs)
            print(item_ODs)
            print(item_vfms[~thin_strato])
            print('|=|=|=|=|')
        bin_features.append(bin_feature)
    bin_features = np.array(bin_features)
    bad_data_mask = np.isnan(bin_features[:, 0])
    print("# of Non-cloud: %d" % np.sum(bin_features[:, 0][~bad_data_mask]))
    print("# of Cloud: %d" % np.sum(bin_features[:, 1][~bad_data_mask]))
    return np.array(bin_features)

def advanced_cloud_binary_v3(dataframe, CAD_lim=30.):
    """
    [0] -> Non-cloud (CAD < 30)
    [1] -> Cloud (CAD >= 30)

    NB// Both classifiers are for the "top layer" of the CALIOP column,
    i.e. the more than half the column optical depth or up to OD ~ 3.

    :param dataframe:
    :return:
    """
    ### Load the data for collocted pixels ###
    CADs = list(dataframe['CALIOP CAD Scores'])
    ODs = list(dataframe['CALIOP ODs for 532nm'])
    tops = list(dataframe['CALIOP Feature Top Altitudes'])
    tropos = list(dataframe['CALIOP Tropopause Altitudes'])
    features = list(dataframe['CALIOP Vertical Feature Mask'])
    # JMA_classiefiers = list(dataframe['Himawari JMA Cloud Mask Binary'])
    ### Define the list of classifiers to be returned at the end of the function ###
    bin_features = []
    ### Classify each pixel based on the CALIOP CAD information ###
    for item_CADs, item_ODs, item_tops, item_tropos, item_vfms in zip(CADs, ODs, tops, tropos, features):
        ### Clean out thin, stratospheric objects ###
        stratospheric = item_tops > item_tropos
        item_ODs[item_ODs == -9999.] = 0. # Change clear ODs (-9999.) to 0.
        thin = item_ODs < 0.1
        thin_strato = stratospheric & thin
        # print(thin_strato)
        item_CADs = item_CADs[~thin_strato] # Only want to look at ODs and CADs in th rest of the algorithm
        item_ODs = item_ODs[~thin_strato]
        item_ODs = list(item_ODs)
        item_vfms = item_vfms[~thin_strato]
        ### Create empty classifier ###
        bin_feature = np.zeros(2)
        if len(item_vfms) > 0:
            if item_vfms[0] in np.arange(1,8+1):
                if item_CADs[0] >= CAD_lim:
                    bin_feature[1] = 1.  # Cloud
                else:
                    bin_feature[0] = np.nan
                    bin_feature[1] = np.nan  # Data not suitable for training or validation
            else:
                if item_CADs[0] <= -CAD_lim:
                    bin_feature[0] = 1.  # Non-cloud
                else:
                    bin_feature[0] = np.nan
                    bin_feature[1] = np.nan  # Data not suitable for training or validation
        else:
            bin_feature[0] = 1. # Non-cloud
        # if bin_feature[1] != jma:
        #     print('#-#-#-#-#')
        #     print(bin_feature, jma)
        #     print(item_CADs)
        #     print(item_ODs)
        #     print(item_vfms)
        #     print('|=|=|=|=|')
        bin_features.append(bin_feature)
    bin_features = np.array(bin_features)
    bad_data_mask = np.isnan(bin_features[:, 0])
    print("# of Non-cloud: %d" % np.sum(bin_features[:, 0][~bad_data_mask]))
    print("# of Cloud: %d" % np.sum(bin_features[:, 1][~bad_data_mask]))
    return np.array(bin_features)

def basic_height_regression_classifiers(dataframe):
    all_top_heights = list(dataframe['CALIOP Feature Top Altitudes'])
    ODs = list(dataframe['CALIOP ODs for 532nm'])
    top_heights = []
    for i, j in zip(ODs, all_top_heights):
        top_heights += [j[np.argmax(i)]]
    top_heights = np.array(top_heights)
    top_heights = (top_heights + 0.5) / 30.6
    top_heights = top_heights.reshape(len(top_heights), 1)
    return top_heights

def generate_input_data(dataframe):
    norm_data = normalise_data(dataframe)
    inputs = format_inputs(norm_data)
    return inputs

def generate_random_mask(length_of_mask, frac_of_trues=0.7):
    mask = np.full(length_of_mask, False)
    number_of_trues = int(frac_of_trues * length_of_mask)
    mask[:number_of_trues] = True
    np.random.shuffle(mask)
    return mask

def generate_training_and_validation_data(dataframe, training_frac=0.7, classifier_type='simple_binary', add_JMA=False,
                                          force_even_split=False, CAD_lim=30., max_OD=5., input_to_noisify=None,
                                          spatial_filter=False):
    acceptable_classifiers = [
        'simple_binary',
        'simple_high_OD',
        'simple_cloud',
        'cloud_aerosol_mixed',
        'complex',
        'advanced',
        'advanced_cloud_binary',
        'height_regression',
        'OD_regression'
    ]
    no_split_classifiers =[
        'cloud_aerosol_mixed',
        'height_regression',
        'OD_regression',
        'complex',
        'advanced'
    ]
    if classifier_type in acceptable_classifiers:
        if classifier_type == 'simple_binary':
            classifiers = basic_binary_classifiers(dataframe)
            df = dataframe.copy()
        elif classifier_type == 'simple_high_OD':
            classifiers = high_OD_binary_classifiers(dataframe)
            df = dataframe.copy()
        elif classifier_type == 'height_regression':
            classifiers = basic_height_regression_classifiers(dataframe)
            df = dataframe.copy()
        elif classifier_type == 'OD_regression':
            classifiers = basic_OD_regression_classifiers(dataframe, max_OD)
            df = dataframe.copy()
        elif classifier_type == 'simple_cloud':
            classifiers = basic_binary_cloud_classifiers(dataframe)
            df = dataframe.copy()
        elif classifier_type == 'cloud_aerosol_mixed':
            classifiers = clear_cloud_aerosol_mixed_classifiers(dataframe)
            df = dataframe.copy()
        elif classifier_type == 'complex':
            classifiers = complex_classifiers(dataframe)
            df = dataframe.copy()
        elif classifier_type == 'advanced':
            classifiers = advanced_classifiers_v1(dataframe)
            df = dataframe.copy()
        elif classifier_type == 'advanced_cloud_binary':
            classifiers = advanced_cloud_binary_v3(dataframe, CAD_lim=CAD_lim)
            bad_data_mask = np.isnan(classifiers[:, 0])
            classifiers = classifiers[~bad_data_mask]
            df = dataframe.copy()[~bad_data_mask]
        if spatial_filter:
            dspatial = get_spatial_diff(df)
            ozas = np.array(list(df['Himawari Observer Zenith Angle']))
            ozas = np.abs(ozas)
            comp_values = dspatial / np.sin(np.deg2rad(ozas))
            good_values = (comp_values < 22.) | (dspatial <= 5.)
            classifiers = classifiers[good_values]
            df = df.copy()[good_values]
        inputs = generate_input_data(df)
        if type(input_to_noisify) == type(1):
            inputs[:, input_to_noisify] = convert_channel_to_noise(inputs[:, input_to_noisify])
        if force_even_split and classifier_type not in no_split_classifiers:
            zeros = np.all(classifiers == [1., 0.], axis=1)
            ones = np.all(classifiers == [0., 1.], axis=1)
            if sum(zeros) > sum(ones):
                where_true = np.where(zeros == True)[0]
                red_mask = np.full(len(where_true), False)
                number_to_remove = int(sum(zeros) - sum(ones))
                red_mask[:number_to_remove] = True
                np.random.shuffle(red_mask)
                remove_mask = where_true[red_mask]
                zeros[remove_mask] = False
            elif sum(ones) > sum(zeros):
                where_true = np.where(ones == True)[0]
                red_mask = np.full(len(where_true), False)
                number_to_remove = int(sum(ones) - sum(zeros))
                red_mask[:number_to_remove] = True
                np.random.shuffle(red_mask)
                remove_mask = where_true[red_mask]
                ones[remove_mask] = False
            print("# of 0's w/ even split: %d" % sum(zeros))
            print("# of 1's w/ even split: %d" % sum(ones))
            even_split_mask = zeros + ones
            inputs = inputs[even_split_mask]
            classifiers = classifiers[even_split_mask]
            df = df.copy()[even_split_mask]
        training_mask = generate_random_mask(len(classifiers), training_frac)
        data_dict = {}
        data_dict['Training Inputs'] = inputs[training_mask]
        data_dict['Training Classifiers'] = classifiers[training_mask]
        data_dict['Validation Inputs'] = inputs[~training_mask]
        data_dict['Validation Classifiers'] = classifiers[~training_mask]
        ODs, surface_types, heights, lats, SZAs, OZAs, spatial_diffs = process_auxiliaries(df)
        data_dict['Training Optical Depths'] = ODs[training_mask]
        data_dict['Training Surface Types'] = surface_types[training_mask]
        data_dict['Training Top Heights'] = heights[training_mask]
        data_dict['Training Latitudes'] = lats[training_mask]
        data_dict['Training Solar Zenith Angles'] = SZAs[training_mask]
        data_dict['Training Observer Zenith Angles'] = OZAs[training_mask]
        data_dict['Training Spatial Differences'] = spatial_diffs[training_mask]
        data_dict['Validation Optical Depths'] = ODs[~training_mask]
        data_dict['Validation Surface Types'] = surface_types[~training_mask]
        data_dict['Validation Top Heights'] = heights[~training_mask]
        data_dict['Validation Latitudes'] = lats[~training_mask]
        data_dict['Validation Solar Zenith Angles'] = SZAs[~training_mask]
        data_dict['Validation Observer Zenith Angles'] = OZAs[~training_mask]
        data_dict['Validation Spatial Differences'] = spatial_diffs[~training_mask]
        if add_JMA:
            jma_binary = np.array(list(df['Himawari JMA Cloud Mask Binary']))
            jma_probs = np.array(list(df['Himawari JMA Cloud Mask Probability']))
            data_dict['Training JMA Binary'] = jma_binary[training_mask]
            data_dict['Validation JMA Binary'] = jma_binary[~training_mask]
            data_dict['Training JMA Probabilities'] = jma_probs[training_mask]
            data_dict['Validation JMA Probabilities'] = jma_probs[~training_mask]
        return data_dict
    else:
        raise Exception('classifier_type is invalid')

def convert_channel_to_noise(input_arr):
    from scipy.stats import truncnorm
    noise = truncnorm(
        a = np.nanmin(input_arr),
        b = np.nanmax(input_arr),
        loc = np.mean(input_arr),
        scale = np.std(input_arr),
        size = input_arr.shape
    )
    return noise

### For NN Prediction for a Scene ###

def normalise_scene_data(himawari_scene):
    norm_vals = {'1': [50., 100.],  # 120.,
                 '2': [50., 100.],  # 135.,
                 '3': [50., 100.],  # 165.,
                 '4': [50., 100.],  # 205.,
                 '5': [50., 100.],  # 121.,
                 '6': [50., 100.],  # 132.,
                 '7': [273.15, 70.],  # 401.,
                 '8': [273.15, 70.],  # 317.,
                 '9': [273.15, 70.],  # 327.,
                 '10': [273.15, 70.],  # 327.,
                 '11': [273.15, 70.],  # 344.,
                 '12': [273.15, 70.],  # 328.,
                 '13': [273.15, 70.],  # 371.,
                 '14': [273.15, 70.],  # 348.,
                 '15': [273.15, 70.],  # 403.,
                 '16': [273.15, 70.],  # 410.,
                 'LAT': [90., 180.],
                 'LON': [180., 360.],
                 'DATE': 366.,
                 'TIME': (24. * 3600 + 1.),
                 'ANGLES': 360.,
                 'SZA': 90.,
                 'OZA': 45.}
    normalised_data = {}
    ### Normalise Band Inputs ###
    print('Normalising Bands')
    for band_number in range(1, 16 + 1):
        #
        str_num = str(band_number)
        if len(str_num) == 1:
            str_num = '0' + str_num
        band_identifier = 'B' + str_num
        band_arr = h8a.band_to_array(himawari_scene, band_identifier)
        # 1km Resolution Bands
        if band_identifier in ['B01', 'B02', 'B04']:
            mean_values, sigma_values = h8a.halve_res(band_arr)
            mean_values -= norm_vals[str(band_number)][0]
            mean_values /= norm_vals[str(band_number)][1]
            normalised_data['Himawari Band %s %s at 2km Resolution' % (str(band_number), 'Mean')] = mean_values
            sigma_values -= norm_vals[str(band_number)][0]
            sigma_values /= norm_vals[str(band_number)][1]
            normalised_data['Himawari Band %s %s at 2km Resolution' % (str(band_number), 'Sigma')] = sigma_values
        # 0.5km Resolution Bands
        elif band_identifier == 'B03':
            mean_values, sigma_values = h8a.quarter_res(band_arr)
            mean_values -= norm_vals[str(band_number)][0]
            mean_values /= norm_vals[str(band_number)][1]
            normalised_data['Himawari Band %s %s at 2km Resolution' % (str(band_number), 'Mean')] = mean_values
            sigma_values -= norm_vals[str(band_number)][0]
            sigma_values /= norm_vals[str(band_number)][1]
            normalised_data['Himawari Band %s %s at 2km Resolution' % (str(band_number), 'Sigma')] = sigma_values
        # 2km Resolution Bands
        else:
            values = band_arr - norm_vals[str(band_number)][0]
            values /= norm_vals[str(band_number)][1]
            normalised_data['Himawari Band %s %s at 2km Resolution' % (str(band_number), 'Value')] = values
    ### Get Latitudes and Longitudes ###
    print('Normalising Coordinates')
    lon, lat = himawari_scene['B16'].area.get_lonlats()
    ### Normalise Latitudes ###
    norm_lats = (lat + norm_vals['LAT'][0]) / norm_vals['LAT'][-1]
    normalised_data['Latitude'] = norm_lats
    ### Normalise Longitudes ###
    norm_lons = (lon + norm_vals['LON'][0]) / norm_vals['LON'][-1]
    normalised_data['Longitude'] = norm_lons
    ### Normalise Date and Time Inputs ###
    # Date #
    print('Normalising Date')
    start_time = himawari_scene.start_time
    end_time = himawari_scene.end_time
    dtime_delta = end_time - start_time
    avg_time = start_time + dtime_delta / 2
    yday = avg_time.timetuple().tm_yday
    yday = yday / norm_vals['DATE']
    normalised_data['Date'] = np.full(lat.shape, yday)
    # Time #
    print('Normalising Time')
    dsec = (avg_time.hour * 3600) + (avg_time.minute * 60) + (avg_time.second)
    dsec = dsec / norm_vals['TIME']
    normalised_data['Time'] = np.full(lat.shape, dsec)
    # Angles #
    print('Normalising Observation Angles')
    avg_time = avg_time.replace(tzinfo=timezone.utc)
    avg_times = np.full(lat.shape, avg_time)
    SatAziAs, SatElvAs = get_observer_look(sat_lon=140.7, sat_lat=0.0, sat_alt=35793.,
                                           utc_time=avg_times, lon=lon, lat=lat, alt=np.zeros(len(lat)))
    normalised_data['Himawari Observer Zenith Angle'] = ((90. - SatElvAs) - norm_vals['OZA']) / norm_vals['OZA']
    normalised_data['Himawari Observer Azimuth Angle'] = SatAziAs / norm_vals['ANGLES']
    print('Normalising Solar Angles')
    SolarElvAs, SolarAziAs = get_alt_az(utc_time=avg_times, lon=lon, lat=lat)
    SolarElvAs = np.rad2deg(SolarElvAs)
    SolarAziAs = np.rad2deg(SolarAziAs)
    SolarZenAs = 90. - SolarElvAs
    normalised_data['Himawari Solar Zenith Angle'] = (SolarZenAs - norm_vals['SZA']) / norm_vals['SZA']
    normalised_data['Himawari Solar Azimuth Angle'] = (SolarAziAs + 180.) / norm_vals['ANGLES']
    print('Normalisation complete')
    # print(normalised_data)
    return normalised_data

def normalise_scene_data_v2(himawari_scene):
    norm_vals = {'1': 120.,
                 '2': 135.,
                 '3': 165.,
                 '4': 205.,
                 '5': 121.,
                 '6': 132.,
                 '7': 401.,
                 '8': 317.,
                 '9': 327.,
                 '10': 327.,
                 '11': 344.,
                 '12': 328.,
                 '13': 371.,
                 '14': 348.,
                 '15': 403.,
                 '16': 410.,
                 'LAT': [90., 180.],
                 'LON': [180., 360.],
                 'DATE': 366.,
                 'TIME': (24. * 3600 + 1.),
                 'ANGLES': 360.}
    normalised_data = {}
    ### Normalise Band Inputs ###
    print('Normalising Bands')
    for band_number in range(1, 16 + 1):
        #
        str_num = str(band_number)
        if len(str_num) == 1:
            str_num = '0' + str_num
        band_identifier = 'B' + str_num
        band_arr = h8a.band_to_array(himawari_scene, band_identifier)
        # 1km Resolution Bands
        if band_identifier in ['B01', 'B02', 'B04']:
            divide_by = 5
        # 0.5km Resolution Bands
        elif band_identifier == 'B03':
            divide_by = 10
        # 2km Resolution Bands
        else:
            band_arr = h8a.upsample_array(band_arr, times_by=2)
            divide_by = 5
        print('Adding ', band_identifier)
        mean_values, sigma_values = h8a.downsample_array(band_arr, divide_by=divide_by)
        mean_values /= norm_vals[str(band_number)]
        normalised_data['Himawari Band %s %s at 5km Resolution' % (str(band_number), 'Mean')] = mean_values
        sigma_values /= norm_vals[str(band_number)]
        normalised_data['Himawari Band %s %s at 5km Resolution' % (str(band_number), 'Sigma')] = sigma_values
    ### Get Latitudes and Longitudes ###
    print('Normalising Coordinates')
    lon, lat = himawari_scene['B16'].area.get_lonlats()
    lon = h8a.downsample_him_lons(h8a.upsample_array(lon, times_by=2),
                                  divide_by=5,
                                  central_longitude=himawari_scene['B16'].attrs['satellite_longitude'])[0]
    lat = h8a.downsample_array(h8a.upsample_array(lat, times_by=2), divide_by=5)[0]
    ### Normalise Latitudes ###
    norm_lats = (lat + norm_vals['LAT'][0]) / norm_vals['LAT'][-1]
    normalised_data['Latitude'] = norm_lats
    ### Normalise Longitudes ###
    norm_lons = (lon + norm_vals['LON'][0]) / norm_vals['LON'][-1]
    normalised_data['Longitude'] = norm_lons
    ### Normalise Date and Time Inputs ###
    # Date #
    print('Normalising Date')
    start_time = himawari_scene.start_time
    end_time = himawari_scene.end_time
    dtime_delta = end_time - start_time
    avg_time = start_time + dtime_delta / 2
    yday = avg_time.timetuple().tm_yday
    yday = yday / norm_vals['DATE']
    normalised_data['Date'] = np.full(lat.shape, yday)
    # Time #
    print('Normalising Time')
    dsec = (avg_time.hour * 3600) + (avg_time.minute * 60) + (avg_time.second)
    dsec = dsec / norm_vals['TIME']
    normalised_data['Time'] = np.full(lat.shape, dsec)
    # Angles #
    print('Normalising Observation Angles')
    avg_time = avg_time.replace(tzinfo=timezone.utc)
    avg_times = np.full(lat.shape, avg_time)
    SatAziAs, SatElvAs = get_observer_look(sat_lon=140.7, sat_lat=0.0, sat_alt=35793.,
                                           utc_time=avg_times, lon=lon, lat=lat, alt=np.zeros(len(lat)))
    normalised_data['Himawari Observer Zenith Angle'] = (90. - SatElvAs) / norm_vals['ANGLES']
    normalised_data['Himawari Observer Azimuth Angle'] = SatAziAs / norm_vals['ANGLES']
    print('Normalising Solar Angles')
    SolarElvAs, SolarAziAs = get_alt_az(utc_time=avg_times, lon=lon, lat=lat)
    SolarElvAs = np.rad2deg(SolarElvAs)
    SolarAziAs = np.rad2deg(SolarAziAs)
    SolarZenAs = 90. - SolarElvAs
    normalised_data['Himawari Solar Zenith Angle'] = SolarZenAs / norm_vals['ANGLES']
    normalised_data['Himawari Solar Azimuth Angle'] = (SolarAziAs + 180.) / norm_vals['ANGLES']
    print('Normalisation complete')
    # print(normalised_data)
    return normalised_data

def prep_scene_inputs(himawari_scene):
    print('Preprocessing scene')
    normalised_data = normalise_scene_data(himawari_scene)
    inputs = [
              'Himawari Band 1 Mean at 2km Resolution',
              'Himawari Band 1 Sigma at 2km Resolution',
              'Himawari Band 2 Mean at 2km Resolution',
              'Himawari Band 2 Sigma at 2km Resolution',
              'Himawari Band 3 Mean at 2km Resolution',
              'Himawari Band 3 Sigma at 2km Resolution',
              'Himawari Band 4 Mean at 2km Resolution',
              'Himawari Band 4 Sigma at 2km Resolution',
              'Himawari Band 5 Value at 2km Resolution',
              'Himawari Band 6 Value at 2km Resolution',
              'Himawari Band 7 Value at 2km Resolution',
              'Himawari Band 8 Value at 2km Resolution',
              'Himawari Band 9 Value at 2km Resolution',
              'Himawari Band 10 Value at 2km Resolution',
              'Himawari Band 11 Value at 2km Resolution',
              'Himawari Band 12 Value at 2km Resolution',
              'Himawari Band 13 Value at 2km Resolution',
              'Himawari Band 14 Value at 2km Resolution',
              'Himawari Band 15 Value at 2km Resolution',
              'Himawari Band 16 Value at 2km Resolution',
              # 'Latitude',
              # 'Longitude',
              # 'Date',
              # 'Time',
              'Himawari Solar Zenith Angle',
              # 'Himawari Solar Azimuth Angle',
              # 'Himawari Observer Elevation Angle',
              'Himawari Observer Zenith Angle',
              # 'Himawari Observer Azimuth Angle'
              ]
    data_dict = {}
    data_dict['Original Shape'] = normalised_data['Latitude'].shape
    # print(original_shape)
    for input in inputs:
        print('Adding ' + input, end='                                             \r')
        data = normalised_data[input].flatten()
        data[data == np.inf] = np.nan
        normalised_data[input] = data.reshape(len(data), 1)
    print('All Inputs Prepared                                             ')
    arr_list = [normalised_data[input] for input in inputs]
    full_scene_inputs = np.hstack(tuple(arr_list))
    # print(full_scene_inputs)
    nan_mask = np.any(np.isnan(full_scene_inputs), axis = 1)
    data_dict['NaN Mask'] = nan_mask
    # print(nan_mask)
    # print(full_scene_inputs[nan_mask])
    processable_inputs = full_scene_inputs[~nan_mask]
    data_dict['Normalised Processable Inputs'] = processable_inputs
    print('Scene prepared')
    return data_dict

def plot_scene_inputs(himawari_scene, scn_folder, apply_mask=False):
    normalised_data = normalise_scene_data(himawari_scene)
    inputs = ['Himawari Band 1 Mean at 2km Resolution',
              'Himawari Band 1 Sigma at 2km Resolution',
              'Himawari Band 2 Mean at 2km Resolution',
              'Himawari Band 2 Sigma at 2km Resolution',
              'Himawari Band 3 Mean at 2km Resolution',
              'Himawari Band 3 Sigma at 2km Resolution',
              'Himawari Band 4 Mean at 2km Resolution',
              'Himawari Band 4 Sigma at 2km Resolution',
              'Himawari Band 5 Value at 2km Resolution',
              'Himawari Band 6 Value at 2km Resolution',
              'Himawari Band 7 Value at 2km Resolution',
              'Himawari Band 8 Value at 2km Resolution',
              'Himawari Band 9 Value at 2km Resolution',
              'Himawari Band 10 Value at 2km Resolution',
              'Himawari Band 11 Value at 2km Resolution',
              'Himawari Band 12 Value at 2km Resolution',
              'Himawari Band 13 Value at 2km Resolution',
              'Himawari Band 14 Value at 2km Resolution',
              'Himawari Band 15 Value at 2km Resolution',
              'Himawari Band 16 Value at 2km Resolution',
              'Himawari Solar Zenith Angle',
              'Himawari Solar Azimuth Angle',
              # 'Himawari Observer Elevation Angle',
              'Himawari Observer Zenith Angle',
              'Himawari Observer Azimuth Angle'
              ]
    fig = plt.figure(figsize = [16, 9])
    for row in range(4):
        for col in range(6):
            ax_number = row*6 + col + 1
            ax = fig.add_subplot(4, 6, ax_number, projection=ccrs.Geostationary(140.735785863))
            ax.add_feature(feature.COASTLINE, edgecolor='yellow')
            # ax.set_global()
            input = inputs[ax_number - 1]
            title = ' '.join(input.split()[1:4])
            ax.set_title(title)
            input_data = normalised_data[input]
            if apply_mask:
                mask = np.full(input_data.shape, False)
                min_y, max_y = int(5500 / 11000 * input_data.shape[0]), int(10500 / 11000 * input_data.shape[0])
                min_x, max_x = int(2500 / 11000 * input_data.shape[0]), int(10000 / 11000 * input_data.shape[0])
                mask[min_y:max_y, min_x:max_x] = True
                new_shape = (int(max_y - min_y), int(max_x - min_x))
                input_data = input_data[mask].reshape(new_shape)
                xmin, xmax = 2500 * 1000 - 5500000, 10000 * 1000 - 5500000
                ymin, ymax = 5500000 - 10500 * 1000, 5500000 - 5500 * 1000
            else:
                xmin, xmax = -5500000, 5500000
                ymin, ymax = -5500000, 5500000
            pcm = ax.imshow(input_data, transform=ccrs.Geostationary(140.735785863),
                            extent=(xmin, xmax, ymin, ymax), cmap='bone')
            fig.colorbar(pcm, ax=ax)
    plt.savefig(os.path.join(scn_folder, 'normalised_inputs.png'), bbox_inches='tight', dpi=1200)

def plot_scene_inputs_v2(himawari_scene, scn_folder, apply_mask=False):
    normalised_data = normalise_scene_data_v2(himawari_scene)
    inputs = ['Himawari Band 1 Mean at 5km Resolution',
              'Himawari Band 1 Sigma at 5km Resolution',
              'Himawari Band 2 Mean at 5km Resolution',
              'Himawari Band 2 Sigma at 5km Resolution',
              'Himawari Band 3 Mean at 5km Resolution',
              'Himawari Band 3 Sigma at 5km Resolution',
              'Himawari Band 4 Mean at 5km Resolution',
              'Himawari Band 4 Sigma at 5km Resolution',
              'Himawari Band 5 Mean at 5km Resolution',
              'Himawari Band 5 Sigma at 5km Resolution',
              'Himawari Band 6 Mean at 5km Resolution',
              'Himawari Band 6 Sigma at 5km Resolution',
              'Himawari Band 7 Mean at 5km Resolution',
              'Himawari Band 7 Sigma at 5km Resolution',
              'Himawari Band 8 Mean at 5km Resolution',
              'Himawari Band 8 Sigma at 5km Resolution',
              'Himawari Band 9 Mean at 5km Resolution',
              'Himawari Band 9 Sigma at 5km Resolution',
              'Himawari Band 10 Mean at 5km Resolution',
              'Himawari Band 10 Sigma at 5km Resolution',
              'Himawari Band 11 Mean at 5km Resolution',
              'Himawari Band 11 Sigma at 5km Resolution',
              'Himawari Band 12 Mean at 5km Resolution',
              'Himawari Band 12 Sigma at 5km Resolution',
              'Himawari Band 13 Mean at 5km Resolution',
              'Himawari Band 13 Sigma at 5km Resolution',
              'Himawari Band 14 Mean at 5km Resolution',
              'Himawari Band 14 Sigma at 5km Resolution',
              'Himawari Band 15 Mean at 5km Resolution',
              'Himawari Band 15 Sigma at 5km Resolution',
              'Himawari Band 16 Mean at 5km Resolution',
              'Himawari Band 16 Sigma at 5km Resolution',
              'Himawari Solar Zenith Angle',
              # 'Himawari Solar Azimuth Angle',
              # 'Himawari Observer Elevation Angle',
              'Himawari Observer Zenith Angle',
              # 'Himawari Observer Azimuth Angle'
              ]
    fig = plt.figure(figsize = [32, 18])
    for row in range(6):
        for col in range(6):
            ax_number = row*6 + col + 1
            if ax_number > len(inputs):
                break
            else:
                ax = fig.add_subplot(6, 6, ax_number, projection=ccrs.Geostationary(140.735785863))
                ax.add_feature(feature.COASTLINE, edgecolor='yellow')
                # ax.set_global()
                input = inputs[ax_number - 1]
                title = ' '.join(input.split()[1:4])
                ax.set_title(title)
                input_data = normalised_data[input]
                if apply_mask:
                    mask = np.full(input_data.shape, False)
                    min_y, max_y = int(5500 / 11000 * input_data.shape[0]), int(10500 / 11000 * input_data.shape[0])
                    min_x, max_x = int(2500 / 11000 * input_data.shape[0]), int(10000 / 11000 * input_data.shape[0])
                    mask[min_y:max_y, min_x:max_x] = True
                    new_shape = (int(max_y - min_y), int(max_x - min_x))
                    input_data = input_data[mask].reshape(new_shape)
                    xmin, xmax = 2500 * 1000 - 5500000, 10000 * 1000 - 5500000
                    ymin, ymax = 5500000 - 10500 * 1000, 5500000 - 5500 * 1000
                else:
                    xmin, xmax = -5500000, 5500000
                    ymin, ymax = -5500000, 5500000
                pcm = ax.imshow(input_data, transform=ccrs.Geostationary(140.735785863),
                                extent=(xmin, xmax, ymin, ymax), cmap='bone')
                fig.colorbar(pcm, ax=ax)
    plt.savefig(os.path.join(scn_folder, 'normalised_inputs_v2.png'), bbox_inches='tight', dpi=1200)


if __name__ == '__main__':
    scn = h8a.read_h8_folder('/mnt/c/Users/drob0013/PhD/Data/20200105_0300')
    plot_scene_inputs(scn, '/mnt/c/Users/drob0013/PhD/Data/Images/20200105_0300')





