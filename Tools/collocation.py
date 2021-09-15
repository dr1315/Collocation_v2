from satpy import Scene, find_files_and_readers
import sys
import os
import subprocess as sp
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyhdf.SD import SD, SDC
from itertools import repeat
import pandas as pd
import pickle
import datetime
from time import time
from pyorbital.orbital import get_observer_look
from pyorbital.astronomy import get_alt_az
sys.path.append(
  os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "Tools"
  )
)
from him8analysis import read_h8_folder, halve_res, quarter_res, generate_band_arrays
from caliop_tools import number_to_bit, custom_feature_conversion, calipso_to_datetime


### Global Variables ###

band_dict={
    '1': 0.4703, # all channels are in microns
    '2': 0.5105,
    '3': 0.6399,
    '4': 0.8563,
    '5': 1.6098,
    '6': 2.257,
    '7': 3.8848,
    '8': 6.2383,
    '9': 6.9395,
    '10': 7.3471,
    '11': 8.5905,
    '12': 9.6347,
    '13': 10.4029,
    '14': 11.2432,
    '15': 12.3828,
    '16': 13.2844
}

### General Tools ###

def write_list(input_list, list_filename, list_dir):
    """
    Writes a list to a .txt file. Specify the directory it is stored in.

    :param input_list: list type.
    :param list_filename: str type. The filename of the .txt file to be written.
    :param list_dir: str type. Full path to the directory the file is stored in.
    :return: list type. List of strings from the .txt file.
    """
    if list_filename[-4:] != '.txt':  # Check if the .txt file extension
        list_filename += '.txt'
    full_filename = os.path.join(list_dir, list_filename)
    print(full_filename)
    with open(full_filename, 'w') as f:
        f.writelines('%s\n' % item for item in input_list)
    print('List stored')

def read_list(list_filename, list_dir):
    """
    Reads a list from a .txt file. Specify the directory it is stored in.

    :param list_filename: str type. The filename of the .txt file to be read.
    :param list_dir: str type. Full path to the directory the file is stored in.
    :return: list type. List of strings from the .txt file.
    """
    if list_filename[-4:] != '.txt': # Check if the .txt file extension
        list_filename += '.txt'
    full_name = os.path.join(list_dir, list_filename)
    with open(full_name, 'r') as f: # Open and read the .txt file
        list_of_lines = [line.rstrip() for line in f.readlines()] # For each line, remove newline character and store in a list
    return list_of_lines

### Processing tools ###

def find_possible_collocated_him_folders(caliop_overpass):
    """
    Will find Himawari folders that fall within the time range of the given CALIOP profile.

    :param caliop_overpass: Loaded CALIOP .hdf file to collocate with Himawari data
    :return: list of str type. Names of the folders that should collocate with the
             given CALIOP profile
    """
    cal_time = caliop_overpass.select('Profile_UTC_Time').get()
    cal_time = calipso_to_datetime(cal_time)
    start = cal_time[0][0]
    end = cal_time[-1][-1]
    print('Raw Start: %s' % datetime.datetime.strftime(start, '%Y%m%d_%H%M'))
    print('Raw End: %s' % datetime.datetime.strftime(end, '%Y%m%d_%H%M'))
    cal_lats = caliop_overpass.select('Latitude').get()
    cal_lons = caliop_overpass.select('Longitude').get()
    hemisphere_mask = (cal_lats <= 81.1) & (cal_lats >= -81.1) & \
                      (((cal_lons >= 60.6) & (cal_lons <= 180.)) | \
                       ((cal_lons >= -180.) & (cal_lons <= -138.0))) # Due to looking at Eastern Hemisphere
    cal_time = cal_time[hemisphere_mask]
    print('MASKED START LAT/LON: (%s, %s)' % (cal_lats[hemisphere_mask][0], cal_lons[hemisphere_mask][0]))
    print('MASKED END LAT/LON: (%s, %s)' % (cal_lats[hemisphere_mask][-1], cal_lons[hemisphere_mask][-1]))
    if len(cal_time) == 0:
        return None
    print(len(cal_time))
    start = cal_time[0]
    end = cal_time[-1]
    print('Masked Start: %s' % datetime.datetime.strftime(start, '%Y%m%d_%H%M'))
    print('Masked End: %s' % datetime.datetime.strftime(end, '%Y%m%d_%H%M'))
    start -= datetime.timedelta(minutes=start.minute % 10,
                                seconds=start.second,
                                microseconds=start.microsecond)
    end -= datetime.timedelta(minutes=end.minute % 10,
                              seconds=end.second,
                              microseconds=end.microsecond)
    print('First Folder: %s' % start)
    print('Last Folder: %s' % end)
    folder_names = []
    while start <= end:
        folder_name = datetime.datetime.strftime(start, '%Y%m%d_%H%M')
        folder_names.append(folder_name)
        start += datetime.timedelta(minutes=10)
    return folder_names

def get_him_folders(him_names, data_dir):
    """
    Finds the Himawari folders given in the list in the mdss on NCI.

    :param him_names: list of str types of Himawari folder names.
    :param data_dir: str type. Full path to the directory where the
                     data will be stored.
    :return: Saves and un-tars Himawari data from mdss into a readable
             folder system for further analysis
    """
    for name in him_names:
        year = name[:4]
        month = name[4:6]
        day = name[6:8]
        filename = 'HS_H08_%s_FLDK.tar' % name
        path = os.path.join('satellite/raw/ahi/FLDK', year, month, day, filename)
        if sp.getoutput('mdss -P rr5 ls %s' % path) == path:
            print('%s available' % name)
            destination = os.path.join(data_dir, name)
            if not os.path.isdir(destination):
                os.mkdir(destination)
            os.system('mdss -P rr5 get %s %s' % (path, destination))
        else:
            print('%s unavailable' % name)

def clear_him_folders(him_names, data_dir):
    """
    Finds the Himawari folders given in the list in the mdss on NCI.

    :param him_names: list of str types of Himawari folder names.
    :param data_dir: str type. Full path to the directory where the
                     data will be stored.
    :return: Removes the Himawari data folders w/n the him_name list
             from /g/data/k10/dr1709/ahi/ directory.
    """
    for name in him_names:
        destination = os.path.join(data_dir, name)
        if os.path.isdir(destination):
            os.system('rm -r %s' % destination)

def define_collocation_area(geo_lons, geo_lats, central_geo_lon,
                            lidar_lons, lidar_lats, spatial_tolerance):
    ### Shift meridian to be defined by geostationary satellite ###
    shifted_geo_lons = geo_lons - central_geo_lon # For geostationary satellite coordinates
    shifted_geo_lons[shifted_geo_lons < -180.] += 360.
    shifted_geo_lons[shifted_geo_lons > 180.] -= 360.
    shifted_lidar_lons = lidar_lons - central_geo_lon # For active satellite coordinates
    shifted_lidar_lons[shifted_lidar_lons < -180.] += 360.
    shifted_lidar_lons[shifted_lidar_lons > 180.] -= 360.
    ### Find limits defined by active satellite ###
    min_lidar_lat, max_lidar_lat = np.nanmin(lidar_lats), np.nanmax(lidar_lats)
    min_lidar_lon, max_lidar_lon = np.nanmin(shifted_lidar_lons), np.nanmax(shifted_lidar_lons)
    ### Find area of geostationary satellite defined by limits ###
    loc_mask = (geo_lats > (min_lidar_lat - spatial_tolerance)) & \
               (geo_lats < (max_lidar_lat + spatial_tolerance)) & \
               (shifted_geo_lons > (min_lidar_lon - spatial_tolerance)) & \
               (shifted_geo_lons < (max_lidar_lon + spatial_tolerance))
    ### Return spatial mask for the geostationary data ###
    return loc_mask

def small_angle_region(latitudes, longitudes, central_geo_lon, small_angle_value):
    ### Shift meridian to be defined by geostationary satellite ###
    shifted_lons = longitudes - central_geo_lon  # For geostationary satellite coordinates
    shifted_lons[shifted_lons < -180.] += 360.
    shifted_lons[shifted_lons > 180.] -= 360.
    region = (shifted_lons < small_angle_value) & (shifted_lons > -small_angle_value) & \
             (latitudes < small_angle_value) & (latitudes > -small_angle_value)
    return region

def load_obs_angles(root_dir):
    """
    Loads satellite surface observation angles from the root directory provided.

    :param root_dir: str type. Full path to the root directory where
                     the observation angle arrays are stored.
    :return: Two np.ndarrays of angles --> (azimuth, elevation)
    """
    sat_azimuths = np.load(os.path.join(root_dir, 'him8-sat_azimuth_angle.npy'))
    sat_elevations = np.load(os.path.join(root_dir, 'him8-sat_elevation_angle.npy'))
    return sat_azimuths, sat_elevations

def load_era_dataset(him_name, var_name='t', multilevel=True):
    """
    Finds and loads the corresponding era5 data for the Himawari-8 scene.

    :param him_name: str type. Himawari-8 scene name.
    :return:
    """
    from glob import glob
    from netCDF4 import Dataset
    if multilevel:
        level_dir = 'pressure-levels'
        level_marker = 'pl'
    else:
        level_dir = 'single-levels'
        level_marker = 'sfc'
    path_to_data = os.path.join(
        f'/g/data/rt52/era5/{level_dir}/monthly-averaged-by-hour/{var_name}',
        him_name[:4],
        f'{var_name}_era5_mnth_{level_marker}_{him_name[:6]}01-{him_name[:6]}??.nc'
    )
    print(path_to_data)
    fname = glob(path_to_data)[0]
    print(fname)
    dst = Dataset(fname)
    print(dst)
    time_stamp = int(him_name[-4:-2]) - 1
    if var_name == '2t':
        var_name_mod = 't2m'
    elif var_name == 'ci':
        var_name_mod = 'siconc'
    else:
        var_name_mod = var_name
    if multilevel:
        data_arr_l = dst[var_name_mod][time_stamp, :, 35:686, 958:]
        data_arr_r = dst[var_name_mod][time_stamp, :, 35:686, :168]
        data_arr = np.dstack((data_arr_l, data_arr_r))
        data_arr = np.dstack(tuple([data_arr[n, :, :] for n in range(37)]))
    else:
        data_arr_l = dst[var_name_mod][time_stamp, 35:686, 958:]
        data_arr_r = dst[var_name_mod][time_stamp, 35:686, :168]
        data_arr = np.hstack((data_arr_l, data_arr_r))
        data_arr[data_arr < 0] = np.nan
    lons, lats = np.meshgrid(
        np.concatenate((dst['longitude'][958:], dst['longitude'][:168])),
        dst['latitude'][35:686],
    )
    return data_arr, lats, lons

def get_closest_era_profile_mask(him_lat, him_lon, era_lats, era_lons):
    """
    Generate a mask for era5 data to locate the closest matching profile to
    the input Himawari-8 coordinate.

    :param him_lat:
    :param him_lon:
    :param era_lats:
    :param era_lons:
    :return:
    """
    shifted_him_lon = him_lon - 140.7
    if shifted_him_lon < -180:
        shifted_him_lon += 360.
    shifted_era_lons = era_lons - 140.7
    shifted_era_lons[shifted_era_lons < -180.] += 360.
    comp_lats = np.abs(era_lats - him_lat)
    comp_lons = np.abs(shifted_era_lons - shifted_him_lon)
    total_comp = comp_lons + comp_lats
    return total_comp == np.nanmin(total_comp)


def parallax_collocation(caliop_overpass, him_scn, caliop_name, him_name):
    """
    Collocates a Himawari scene w/ a given CALIOP overpass and returns the
    collocated data as a pandas dataframe. The dataframe from this function will
    contain repeats and therefore needs further processing to clean the data.
    
    :param caliop_overpass: 
    :param him_scn: 
    :param caliop_name: 
    :param him_name: 
    :return: 
    """
    ### Load Himawari variables from the scene ###
    him_lon = him_scn['B16'].attrs['satellite_longitude']  # Himawari central longitude
    him_lat = him_scn['B16'].attrs['satellite_latitude']  # Himawari central latitude
    him_alt = him_scn['B16'].attrs['satellite_altitude'] / 1000. # Himawari altitude in km (taken from Earth's CoM)
    start_time = him_scn.start_time # Himawari scene scan start time
    end_time = him_scn.end_time # Himawari scene scan end time
    avg_time = start_time + (end_time - start_time) / 2.  # Himawari scene scan middle time
    him_lons, him_lats = him_scn['B16'].area.get_lonlats()  # Himawari surface lats & lons
    him_lons[him_lons == np.inf] = np.nan # Set np.inf values to NaNs
    him_lats[him_lats == np.inf] = np.nan # Set np.inf values to NaNs
    ### Load and hold CALIOP data ###
    caliop_data = {}
    caliop_triplet_datasets = ['Profile_UTC_Time', # CALIOP pixel scan times
                               'Latitude', # CALIOP pixel latitudes
                               'Longitude'] # CALIOP pixel longitudes
    caliop_fifteen_datasets = ['Feature_Classification_Flags', # CALIOP pixel vertical feature flags
                               'Feature_Optical_Depth_532', # CALIOP pixel features' optical depths (532nm)
                               'Feature_Optical_Depth_1064', # CALIOP pixel features' optical depths (1064nm)
                               'Layer_IAB_QA_Factor', # CALIOP pixel features' quality assurance values
                               'CAD_Score', # CALIOP pixel features' cloud vs aerosol score
                               'Layer_Top_Altitude', # CALIOP pixel features' top altitudes
                               'Layer_Base_Altitude'] # CALIOP pixel features' base altitudes
    caliop_singlet_datasets = ['Tropopause_Height', # CALIOP pixel tropopause heights
                               'IGBP_Surface_Type', # CALIOP pixel surface types
                               'DEM_Surface_Elevation'] # CALIOP pixel surface elevation values
    all_datasets = caliop_triplet_datasets + \
                   caliop_fifteen_datasets + \
                   caliop_singlet_datasets  # Full list of ordered dataset names
    special_case_datasets = ['Profile_UTC_Time', # Easy-to-reference special case storage
                             'DEM_Surface_Elevation']
    for dataset in caliop_triplet_datasets: # Expand pixel data with 3 sub-values to standard format
        dst = caliop_overpass.select(dataset).get().flatten()
        if dataset in special_case_datasets:
            dst = calipso_to_datetime(dst)  # Ensure times are datetime objects
        dst = np.repeat(dst, np.full((dst.shape[0]), 15), axis=0).reshape(dst.shape[0], 15)
        caliop_data[dataset] = dst
    for dataset in caliop_fifteen_datasets: # Expand pixel data with 15 sub-values to standard format
        dst = caliop_overpass.select(dataset).get()
        dst = np.hstack((dst, dst, dst)).reshape(dst.shape[0]*3, 15)
        if dataset == 'Layer_Top_Altitude': # Set fill values to NaNs
            dst[dst < -1000.] = np.nan
            all_nans = np.all(np.isnan(dst), axis=1) # If the a row is all fill, implies it is clear air
            dst[all_nans, 0] = 0. # Set clear air height to 0 for calculating observation angles
        caliop_data[dataset] = dst
    for dataset in caliop_singlet_datasets: # Expand pixel data with single sub-value to standard format
        dst = caliop_overpass.select(dataset).get()
        if dataset in special_case_datasets: # Special case; sub-value is array, not single float or int
            dst = np.repeat(dst, np.full((dst.shape[0]), 3), axis=0)
            dst = np.repeat(dst, np.full((dst.shape[0]), 15), axis=0).reshape(dst.shape[0], 15, 4)
            if dataset == 'Profile_UTC_Time':
                dst = dst.astype('str')
        else:
            dst = np.repeat(dst, 3)
            dst = np.repeat(dst, np.full((dst.shape[0]), 15), axis=0).reshape(dst.shape[0], 15)
        caliop_data[dataset] = dst
    ### Define temporal mask for CALIOP data and starting points for mask w/n the data ###
    time_mask = (caliop_data['Profile_UTC_Time'] >= start_time) & \
                (caliop_data['Profile_UTC_Time'] <= end_time)
    if np.sum(time_mask) == 0:
        print('No matches in time between %s and %s' % (him_name, caliop_name))
        return None # If none of the values fall within the designated time period,
                    # stop the whole process and return None
    ### Apply temporal mask to CALIOP data ###
    else:
        for dataset in caliop_data.keys():
            dst = caliop_data[dataset][time_mask]
            caliop_data[dataset] = dst
            # print(dataset, dst.shape)
    ### Define spatial mask for Himawari data ###
    spatial_mask = define_collocation_area(geo_lons = him_lons,
                                           geo_lats = him_lats,
                                           central_geo_lon = him_lon,
                                           lidar_lons = caliop_data['Longitude'],
                                           lidar_lats = caliop_data['Latitude'],
                                           spatial_tolerance = 10./111.)
    ### Apply spatial mask to Himawari coordinates ###
    him_lons = him_lons[spatial_mask]
    him_lats = him_lats[spatial_mask]
    ### Load Himawari band data and apply spatial mask ###
    him_band_values = generate_band_arrays(him_scn, spatial_mask)
    ### Load surface observation angles and apply spatial mask ###
    # angle_dir = '/mnt/c/Users/drob0013/PhD/Data' # Directory where angle information is held
    angle_dir = '/g/data/k10/dr1709/.angles'  # Directory where angle information is held
    sat_azimuth, sat_elevation = load_obs_angles(root_dir = angle_dir)
    sat_azimuth = sat_azimuth[spatial_mask] # Apply spatial mask
    sat_azimuth *= 1e6 # Floating point correction
    sat_azimuth = sat_azimuth.astype('int') # Floating point correction
    sat_elevation = sat_elevation[spatial_mask] # Apply spatial mask
    sat_elevation *= 1e6 # Floating point correction
    sat_elevation = sat_elevation.astype('int') # Floating point correction
    ### Define small-angle region ###
    cal_lats = caliop_data['Latitude'].copy()
    cal_lons = caliop_data['Longitude'].copy()
    cal_heights = caliop_data['Layer_Top_Altitude'].copy()
    small_angles = small_angle_region(latitudes = cal_lats,
                                      longitudes = cal_lons,
                                      central_geo_lon = him_lon,
                                      small_angle_value = 2.)
    cal_heights[small_angles] = 0.
    ### Get observation angles for given objects ###
    obsa, obsl = get_observer_look(him_lon,
                                   him_lat,
                                   him_alt,
                                   avg_time,
                                   cal_lons,
                                   cal_lats,
                                   cal_heights)
    ### Carry out collocation ###
    data = [] # Temporary storage for collocated data that is later converted into a dataframe
    for cal_idx in np.ndindex(obsa.shape):
        if not np.isnan(caliop_data['Layer_Top_Altitude'][cal_idx]) \
                and not np.isnan(obsa[cal_idx]) \
                and not np.isnan(obsl[cal_idx]): # Skip any fill objects
            azi = obsa[cal_idx]  # Extract satellite azimuth angle for given object
            azi_comp_arr = np.abs(sat_azimuth - int(azi*1e6))
            elv = obsl[cal_idx]  # Extract satellite elevation angle for given object
            elv_comp_arr = np.abs(sat_elevation - int(elv*1e6))
            comp_scan_arr = azi_comp_arr + elv_comp_arr  # Generate a comparative coord array
            him_idx = np.nanargmin(comp_scan_arr)  # Find flattened index of min value
            ### Extract collocated values ###
            in_data_p1 = [caliop_name] # Start w/ caliop filename
            in_data_p2 = [caliop_data[dataset][cal_idx] for dataset in all_datasets] # Add caliop data
            in_data_p3 = ['HS_H08_' + him_name + '_FLDK', # Add Himawari folder name
                          float(him_lats[him_idx]), # Add Himawari pixel latitude
                          float(him_lons[him_idx])] # Add Himawari pixel longitude
            in_data_p4 = [float(v) for v in him_band_values[him_idx]] # Add Himawari pixel band data
            in_data_p5 = [start_time, # Add Himawari scene start time
                          end_time] # Add Himawari scene start time
            in_data = in_data_p1 + in_data_p2 + in_data_p3 + in_data_p4 + in_data_p5
            data.append(in_data) # Add all collocated data to temporary storage
    ### Convert list of data into dataframe ###
    columns=['CALIOP Filenames', # Set column names
             'CALIOP Pixel Scan Times',
             'CALIOP Latitudes',
             'CALIOP Longitudes',
             'CALIOP Vertical Feature Mask',
             'CALIOP ODs for 532nm',
             'CALIOP ODs for 1064nm',
             'CALIOP QA Scores',
             'CALIOP CAD Scores',
             'CALIOP Feature Top Altitudes',
             'CALIOP Feature Base Altitudes',
             'CALIOP Tropopause Altitudes',
             'CALIOP IGBP Surface Types',
             'CALIOP DEM Surface Elevation Statistics',
             'Himawari Folder Name',
             'Himawari Latitude',
             'Himawari Longitude',
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
             'Himawari Scene Start Time',
             'Himawari Scene End Time']
    data = pd.DataFrame(data, columns = columns) # Convert data
    ### Remove entries which contain NaN values from the dataframe ###
    data = data.dropna().reset_index(drop = True) # Reset indicies of entries
    ### Return dataframe of collocated data ###
    return data

def load_5km_caliop_data(caliop_overpass):
    caliop_data = {}
    caliop_triplet_datasets = [
        'Profile_UTC_Time',  # CALIOP pixel scan times
        'Latitude',  # CALIOP pixel latitudes
        'Longitude'  # CALIOP pixel longitudes
    ]
    caliop_fifteen_datasets = [
        'Feature_Classification_Flags',  # CALIOP pixel vertical feature flags
        'Feature_Optical_Depth_532',  # CALIOP pixel features' optical depths (532nm)
        'Feature_Optical_Depth_1064',  # CALIOP pixel features' optical depths (1064nm)
        'Layer_IAB_QA_Factor',  # CALIOP pixel features' quality assurance values
        'CAD_Score',  # CALIOP pixel features' cloud vs aerosol score
        'Layer_Top_Altitude',  # CALIOP pixel features' top altitudes
        'Layer_Base_Altitude'  # CALIOP pixel features' base altitudes
    ]
    caliop_singlet_datasets = [
        'Tropopause_Height',  # CALIOP pixel tropopause heights
        'IGBP_Surface_Type',  # CALIOP pixel surface types
        'DEM_Surface_Elevation'  # CALIOP pixel surface elevation values
    ]
    special_case_datasets = [
        'Profile_UTC_Time',  # Easy-to-reference special case storage
        'DEM_Surface_Elevation'
    ]
    for dataset in caliop_triplet_datasets:  # Expand pixel data with 3 sub-values to standard format
        dst = caliop_overpass.select(dataset).get().flatten()
        if dataset in special_case_datasets:
            dst = calipso_to_datetime(dst)  # Ensure times are datetime objects
        dst = np.repeat(dst, np.full((dst.shape[0]), 15), axis=0).reshape(dst.shape[0], 15)
        caliop_data[dataset] = dst
    for dataset in caliop_fifteen_datasets:  # Expand pixel data with 15 sub-values to standard format
        dst = caliop_overpass.select(dataset).get()
        dst = np.hstack((dst, dst, dst)).reshape(dst.shape[0] * 3, 15)
        if dataset == 'Layer_Top_Altitude':  # Set fill values to NaNs
            dst[dst < -1000.] = np.nan
            all_nans = np.all(np.isnan(dst), axis=1)  # If the a row is all fill, implies it is clear air
            dst[all_nans, 0] = 0.  # Set clear air height to 0 for calculating observation angles
        caliop_data[dataset] = dst
    for dataset in caliop_singlet_datasets:  # Expand pixel data with single sub-value to standard format
        dst = caliop_overpass.select(dataset).get()
        if dataset in special_case_datasets:  # Special case; sub-value is array, not single float or int
            dst = np.repeat(dst, np.full((dst.shape[0]), 3), axis=0)
            dst = np.repeat(dst, np.full((dst.shape[0]), 15), axis=0).reshape(dst.shape[0], 15, 4)
            if dataset == 'Profile_UTC_Time':
                dst = dst.astype('str')
        else:
            dst = np.repeat(dst, 3)
            dst = np.repeat(dst, np.full((dst.shape[0]), 15), axis=0).reshape(dst.shape[0], 15)
        caliop_data[dataset] = dst
    return caliop_data

def load_1km_caliop_data(caliop_overpass):
    caliop_data = {}
    caliop_singlet_datasets_a = [
        'Profile_UTC_Time',  # CALIOP pixel scan times
        'Latitude',  # CALIOP pixel latitudes
        'Longitude',  # CALIOP pixel longitudes
    ]
    caliop_ten_datasets = [
        'Feature_Classification_Flags',  # CALIOP pixel vertical feature flags
        'Layer_IAB_QA_Factor',  # CALIOP pixel features' quality assurance values
        'CAD_Score',  # CALIOP pixel features' cloud vs aerosol score
        'Layer_Top_Altitude',  # CALIOP pixel features' top altitudes
        'Layer_Base_Altitude'  # CALIOP pixel features' base altitudes
    ]
    caliop_singlet_datasets_b = [
        'Tropopause_Height',  # CALIOP pixel tropopause heights
        'IGBP_Surface_Type',  # CALIOP pixel surface types
        'DEM_Surface_Elevation'  # CALIOP pixel surface elevation values
    ]
    ### Only singlet datasets need to be expanded ###
    for dataset in caliop_singlet_datasets_a:  # Expand pixel data with single sub-value to standard format
        dst = caliop_overpass.select(dataset).get()
        if dataset == 'Profile_UTC_Time':
            dst = calipso_to_datetime(dst)
        dst = np.repeat(dst, 10).reshape(dst.shape[0], 10)
        caliop_data[dataset] = dst
    for dataset in caliop_ten_datasets:  # Expand pixel data with 15 sub-values to standard format
        dst = caliop_overpass.select(dataset).get()
        if dataset == 'Layer_Top_Altitude':  # Set fill values to NaNs
            dst[dst < -1000.] = np.nan
            all_nans = np.all(np.isnan(dst), axis=1)  # If the a row is all fill, implies it is clear air
            dst[all_nans, 0] = 0.  # Set clear air height to 0 for calculating observation angles
        caliop_data[dataset] = dst
    for dataset in caliop_singlet_datasets_b:  # Expand pixel data with single sub-value to standard format
        dst = caliop_overpass.select(dataset).get()
        dst = np.repeat(dst, 10).reshape(dst.shape[0], 10)
        caliop_data[dataset] = dst
    return caliop_data

def parallax_collocation_v2(caliop_overpass, him_scn, caliop_name, him_name):
    """
    Collocates a Himawari scene w/ a given CALIOP overpass and returns the
    collocated data as a pandas dataframe. The dataframe from this function will
    contain repeats and therefore needs further processing to clean the data.

    :param caliop_overpass:
    :param him_scn:
    :param caliop_name:
    :param him_name:
    :return:
    """
    ### Check resolution of caliop data ###
    if '1km' in caliop_name:
        caliop_res = '1km'
    elif '5km' in caliop_name:
        caliop_res = '5km'
    else:
        print('CALIOP file resolution is incompatible')
        return None # End the process and return None if the resolution
    ### Load Himawari variables from the scene ###
    him_lon = him_scn['B16'].attrs['satellite_longitude']  # Himawari central longitude
    him_lat = him_scn['B16'].attrs['satellite_latitude']  # Himawari central latitude
    him_alt = him_scn['B16'].attrs['satellite_altitude'] / 1000.  # Himawari altitude in km (taken from Earth's CoM)
    start_time = him_scn.start_time  # Himawari scene scan start time
    end_time = him_scn.end_time  # Himawari scene scan end time
    avg_time = start_time + (end_time - start_time) / 2.  # Himawari scene scan middle time
    him_lons, him_lats = him_scn['B16'].area.get_lonlats()  # Himawari surface lats & lons
    him_lons[him_lons == np.inf] = np.nan  # Set np.inf values to NaNs
    him_lats[him_lats == np.inf] = np.nan  # Set np.inf values to NaNs
    ### Load and hold CALIOP data ###
    if caliop_res == '1km':
        caliop_data = load_1km_caliop_data(caliop_overpass)
    elif caliop_res == '5km':
        caliop_data = load_5km_caliop_data(caliop_overpass)
    ### Define temporal mask for CALIOP data and starting points for mask w/n the data ###
    time_mask = (caliop_data['Profile_UTC_Time'] >= start_time) & \
                (caliop_data['Profile_UTC_Time'] <= end_time)
    if np.sum(time_mask) == 0:
        print('No matches in time between %s and %s' % (him_name, caliop_name))
        return None  # If none of the values fall within the designated time period,
                     # stop the whole process and return None
    ### Apply temporal mask to CALIOP data ###
    else:
        for dataset in caliop_data.keys():
            dst = caliop_data[dataset][time_mask]
            caliop_data[dataset] = dst
            # print(dataset, dst.shape)
    ### Define spatial mask for Himawari data ###
    spatial_mask = define_collocation_area(
        geo_lons=him_lons,
        geo_lats=him_lats,
        central_geo_lon=him_lon,
        lidar_lons=caliop_data['Longitude'],
        lidar_lats=caliop_data['Latitude'],
        spatial_tolerance=10. / 111.
    )
    ### Apply spatial mask to Himawari coordinates ###
    him_lons = him_lons[spatial_mask]
    him_lats = him_lats[spatial_mask]
    ### Get original array locations and mask them ###
    y, x = np.meshgrid(np.arange(5500), np.arange(5500))
    y = y[spatial_mask]
    x = x[spatial_mask]
    ### Load Himawari band data and apply spatial mask ###
    him_band_values = generate_band_arrays(him_scn, spatial_mask)
    ### Load surface observation angles and apply spatial mask ###
    angle_dir = '/g/data/k10/dr1709/.angles'  # Directory where angle information is held
    sat_azimuth, sat_elevation = load_obs_angles(root_dir=angle_dir)
    sat_azimuth = sat_azimuth[spatial_mask]  # Apply spatial mask
    # sat_azimuth *= 1e6  # Floating point correction
    #sat_azimuth = sat_azimuth.astype('int')  # Floating point correction
    sat_elevation = sat_elevation[spatial_mask]  # Apply spatial mask
    # sat_elevation *= 1e6  # Floating point correction
    # sat_elevation = sat_elevation.astype('int')  # Floating point correction
    ### NEW: Add surface-type mask ###
    surface_mask = np.load('/g/data/k10/dr1709/code/Personal/Collocation/v2/auxiliary_data/himawari_surface_mask.npy')
    surface_mask = surface_mask[spatial_mask]
    ### NEW: Load ERA5 datasets into a dictionary to be read ###
    sl_keys = [  # Single level keys for ERA5 datasets
        'fal',  # Forecast albedo
        'aluvp',  # UV albedo (direct)
        'aluvd',  # UV albedo (diffuse)
        'alnip',  # NIR albedo (direct)
        'alnid',  # NIR albedo (diffuse)
        '2t',  # 2m temperature
        'skt',  # Skin temperature
        'ci',  # Sea Ice Fraction
    ]
    pl_keys = [  # Multilevel (pressure level) keys for ERA5 datasets
        'u',  # U component of wind
        'v',  # V component of wind
        't',  # Temperature profile of atmosphere
        'clwc',  # Specific cloud liquid water content
        'ciwc',  # Specific cloud ice water content
        'r',  # Relative humidity
    ]
    fal_data, era_lats, era_lons = load_era_dataset(him_name, 'fal', False)  # Generate initial dataset
    era_dict = {'fal': fal_data}  # Initialise dictionary
    era_dict.update({key: load_era_dataset(him_name, key, False)[0] for key in sl_keys[1:]})  # Add rest of sl keys
    era_dict.update({key: load_era_dataset(him_name, key, True)[0] for key in pl_keys})  # Add pl keys
    ### Define small-angle region ###
    cal_lats = caliop_data['Latitude'].copy()
    cal_lons = caliop_data['Longitude'].copy()
    cal_heights = caliop_data['Layer_Top_Altitude'].copy()
    small_angles = small_angle_region(
        latitudes=cal_lats,
        longitudes=cal_lons,
        central_geo_lon=him_lon,
        small_angle_value=2.
    )
    cal_heights[small_angles] = 0.
    ### Get observation angles for given objects ###
    obsa, obsl = get_observer_look(
        him_lon,
        him_lat,
        him_alt,
        avg_time,
        cal_lons,
        cal_lats,
        cal_heights
    )
    ### Carry out collocation ###
    data = []  # Temporary storage for collocated data that is later converted into a dataframe
    for cal_idx in np.ndindex(obsa.shape):
        if not np.isnan(caliop_data['Layer_Top_Altitude'][cal_idx]) \
                and not np.isnan(obsa[cal_idx]) \
                and not np.isnan(obsl[cal_idx]):  # Skip any fill objects
            azi = obsa[cal_idx]  # Extract satellite azimuth angle for given object
            azi_comp_arr = np.abs((sat_azimuth * 1e6).astype('int') - int(azi * 1e6))
            elv = obsl[cal_idx]  # Extract satellite elevation angle for given object
            elv_comp_arr = np.abs((sat_elevation * 1e6).astype('int') - int(elv * 1e6))
            comp_scan_arr = azi_comp_arr + elv_comp_arr  # Generate a comparative coord array
            him_idx = np.nanargmin(comp_scan_arr)  # Find flattened index of min value
            original_coord = [x[him_idx], y[him_idx]]
            collocated_him_lat = float(him_lats[him_idx])
            collocated_him_lon = float(him_lons[him_idx])
            SolarElvAngle, SolarAziAngle = get_alt_az(
                utc_time=start_time + (end_time - start_time)/2.,
                lon=collocated_him_lon,
                lat=collocated_him_lat
            )
            SolarElvAngle = np.rad2deg(SolarElvAngle)
            SolarAziAngle = np.rad2deg(SolarAziAngle)
            ### Extract collocated values ###
            in_data_p1 = [caliop_name]  # Start w/ caliop filename
            in_data_p2 = [caliop_data[dataset][cal_idx] for dataset in caliop_data.keys()]  # Add caliop data
            in_data_p3 = [
                'HS_H08_' + him_name + '_FLDK',  # Add Himawari folder name
                collocated_him_lat,  # Add Himawari pixel latitude
                collocated_him_lon,  # Add Himawari pixel longitude
                original_coord,  # Add original location from full array
                surface_mask[him_idx],  # Add surface type from Himawari data
                sat_azimuth[him_idx],  # Add observation azimuth angle
                sat_elevation[him_idx],  # Add observation elevation angle
                SolarAziAngle,  # Add solar azimuth angle
                90. - SolarElvAngle  # Add solar zenith angle
            ]
            in_data_p4 = [float(v) for v in him_band_values[him_idx]]  # Add Himawari pixel band data
            in_data_p5 = [
                start_time,  # Add Himawari scene start time
                end_time  # Add Himawari scene start time
            ]
            ### NEW: Add in ERA5 data ###
            col_era_mask = get_closest_era_profile_mask(
                collocated_him_lat,
                collocated_him_lon,
                era_lats,
                era_lons
            )
            in_data_p6 = [
                era_lats[col_era_mask],  # Corresponding ERA5 latitude
                era_lons[col_era_mask],  # Corresponding ERA5 longitude
            ]
            in_data_p7 = [era_dict[key][col_era_mask] for key in sl_keys] + [era_dict[key][col_era_mask][0] for key in pl_keys]  # Add collocated ERA5 data
            in_data = in_data_p1 + in_data_p2 + in_data_p3 + in_data_p4 + in_data_p5 + in_data_p6 + in_data_p7
            data.append(in_data)  # Add all collocated data to temporary storage
    ### Convert list of data into dataframe ###
    if caliop_res == '5km':
        cal_columns = [  # Set CALIOP column names
            'CALIOP Filenames',
            'CALIOP Pixel Scan Times',
            'CALIOP Latitudes',
            'CALIOP Longitudes',
            'CALIOP Vertical Feature Mask (Binary Format)',
            'CALIOP ODs for 532nm',
            'CALIOP ODs for 1064nm',
            'CALIOP QA Scores',
            'CALIOP CAD Scores',
            'CALIOP Feature Top Altitudes',
            'CALIOP Feature Base Altitudes',
            'CALIOP Tropopause Altitudes',
            'CALIOP IGBP Surface Types',
            'CALIOP DEM Surface Elevation Statistics'
        ]
    elif caliop_res == '1km':
        cal_columns = [  # Set CALIOP column names
            'CALIOP Filenames',
            'CALIOP Pixel Scan Times',
            'CALIOP Latitudes',
            'CALIOP Longitudes',
            'CALIOP Vertical Feature Mask (Binary Format)',
            'CALIOP QA Scores',
            'CALIOP CAD Scores',
            'CALIOP Feature Top Altitudes',
            'CALIOP Feature Base Altitudes',
            'CALIOP Tropopause Altitudes',
            'CALIOP IGBP Surface Types',
            'CALIOP DEM Surface Elevation Statistics'
        ]
    him_columns = [  # Set Himawari-8 column names
        'Himawari Folder Name',
        'Himawari Latitude',
        'Himawari Longitude',
        'Himawari Original Array Position',
        'Himawari Surface Type',
        'Himawari Observation Azimuth Angle',
        'Himawari Observation Elevation Angle',
        'Himawari Solar Azimuth Angle',
        'Himawari Solar Zenith Angle',
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
        'Himawari Scene Start Time',
        'Himawari Scene End Time'
        ]
    ### POTENTIAL: Add T-10mins and T-20mins Himawari-8 data
    era_columns = [  # Set ERA5 column names
        'ERA5 Latitude',
        'ERA5 Longitude',
        'ERA5 Forecast Albedo',
        'ERA5 UV Visible Albedo (Direct)',
        'ERA5 UV Visible Albedo (Diffuse)',
        'ERA5 NIR Visible Albedo (Direct)',
        'ERA5 NIR Visible Albedo (Diffuse)',
        'ERA5 2m Temperature',
        'ERA5 Skin Temperature',
        'ERA5 Sea-Ice Fraction',  # End of single level names
        'ERA5 U Wind Component Profile',
        'ERA5 V Wind Component Profile',
        'ERA5 Atmosphere Temperature Profile',
        'ERA5 Specific Cloud Liquid Water Content Profile',
        'ERA5 Specific Cloud Ice Water Content Profile',
        'ERA5 Relative Humidity Profile'  # End of pressure level names
    ]
    columns = cal_columns + him_columns + era_columns
    for n, column in enumerate(columns):
        print('- - - - -')
        print(column, data[0][n])
        print('# # # # #')
    for n, entry in enumerate(data[0]):
        print('- - - - -')
        print(n, data[0][n])
        print('# # # # #')
    data = pd.DataFrame(data, columns=columns)  # Convert data
    ### Remove entries which contain NaN values from the dataframe ###
    data = data.dropna().reset_index(drop=True)  # Reset indicies of entries
    ### Return dataframe of collocated data ###
    return data

def omit_sub_duplicates(dataframe_of_collocated_data):
    columns = [
        'CALIOP Filenames',
        'CALIOP Vertical Feature Mask (Binary Format)',
        'CALIOP Feature Top Altitudes',
        'CALIOP Feature Base Altitudes',
        'CALIOP Tropopause Altitudes',
        'CALIOP IGBP Surface Types'
    ]
    comp_df = dataframe_of_collocated_data[columns]
    for i in np.arange(2,5,1): # For float entries defined by index in column names
        comp_df[columns[i]] *= 1e6
        comp_df[columns[i]] = comp_df[columns[i]].astype('int')
    where_duplicates = comp_df.duplicated()
    return dataframe_of_collocated_data[where_duplicates == False]

def clean_up_df(dataframe_of_collocated_data):
    clean_dataframe = pd.DataFrame(columns = list(dataframe_of_collocated_data.keys())) # Create empty dataframe to be filled
    column_names = ['Himawari Latitude', 'Himawari Longitude'] # Want to find repeated pixels defined by coords
    comp_df = dataframe_of_collocated_data[column_names] # Look only at the Himawari coords
    comp_df *= 1e6 # Try to prevent floating point error
    uniques = comp_df.duplicated(subset = column_names) # Find duplicated coords
    unique_him_coords = dataframe_of_collocated_data[uniques == False][column_names] # Take only single incidences
    for original_pos in range(len(unique_him_coords)): # For each pixel, extract values needed
        lat = unique_him_coords.iloc[original_pos]['Himawari Latitude']
        lon = unique_him_coords.iloc[original_pos]['Himawari Longitude']
        mask = (dataframe_of_collocated_data['Himawari Latitude'] * 1e6 == lat * 1e6) & \
               (dataframe_of_collocated_data['Himawari Longitude'] * 1e6 == lon * 1e6) # Find repeated entries of the given pixel
        # print(lat, lon, 'count: ', sum(mask))
        pixel_df = dataframe_of_collocated_data[mask] # Look at only that pixel's data
        caliop_object_dict = {
            key: [] for key in dataframe_of_collocated_data.keys() if 'CALIOP' in key  # Create temp dictionary for storage
        }
        # print(pixel_df)
        pixel_df = pixel_df.sort_values(by=['CALIOP Feature Top Altitudes'], ascending=False) # Order objects from highest to lowest top height
        # print(pixel_df)
        pixel_df = omit_sub_duplicates(pixel_df)
        # print(pixel_df)
        for sub_pos in range(len(pixel_df)): # For each object
            object = pixel_df.iloc[sub_pos] # Get object CALIOP data
            for caliop_key in caliop_object_dict.keys(): # For each CALIOP data entry
                caliop_object_dict[caliop_key] += [object[caliop_key]] # Add object to pixel dictionary
        for caliop_key in caliop_object_dict.keys(): # For each CALIOP data entry
            caliop_object_dict[caliop_key] = np.array(caliop_object_dict[caliop_key]) # Convert lists into numpy arrays
        him_and_era_dict = {
            key: np.nan for key in dataframe_of_collocated_data.keys() if 'Himawari' in key or 'ERA5' in key # Take the pixel's Himawari and ERA data; temporarily store in another dictionary
        }
        for him_and_era_key in him_and_era_dict.keys(): # For each Himawari/ERA5 data entry
            him_and_era_dict[him_and_era_key] = pixel_df.iloc[0][him_and_era_key] # Store the data in the dictionary
        pixel_dict = {**caliop_object_dict, **him_and_era_dict} # Merge dictionaries
        clean_dataframe = clean_dataframe.append(pixel_dict, ignore_index = True) # Add pixel data to the clean DataFrame
    return clean_dataframe

def expand_df(clean_dataframe):
    columns = list(clean_dataframe.keys())
    expanded_df = pd.DataFrame(columns = columns) # Empty dataframe that will be filled with induvidual objects
    # him_columns = [i for i in columns if 'Himawari' in i]
    # cal_columns = [i for i in columns if 'CALIOP' in i]
    # for column in cal_columns:
    #
    # expanded_data = [i for ]
    # print('#-#-#-#-#-#-#-#-#-#-#-#-#-#-#')
    for i in range(len(clean_dataframe)): # For each Himawari pixel
        # print('Him Coords: ',
        #       clean_dataframe.iloc[i]['Himawari Latitude'],
        #       clean_dataframe.iloc[i]['Himawari Longitude'])
        for j in range(len(clean_dataframe.iloc[i]['CALIOP Vertical Feature Mask (Binary Format)'])): # For each collocated object
            # if clean_dataframe.iloc[i]['CALIOP Feature Top Altitudes'][j] < 1.:
                # print('----------------')
                # print('Caliop Coords: ',
                #       clean_dataframe.iloc[i]['CALIOP Latitudes'][j],
                #       clean_dataframe.iloc[i]['CALIOP Longitudes'][j])
                # print('Object Height: ',
                #       clean_dataframe.iloc[i]['CALIOP Feature Top Altitudes'][j])
                # print('Shift in Lat: ',
                #       np.abs(clean_dataframe.iloc[i]['Himawari Latitude'] -
                #              clean_dataframe.iloc[i]['CALIOP Latitudes'][j]))
                # print('Shift in Lon: ',
                #       np.abs(clean_dataframe.iloc[i]['Himawari Longitude'] -
                #              clean_dataframe.iloc[i]['CALIOP Longitudes'][j]))
                # print('----------------')
            object_dict = {} # Create temporary storage for object information
            for column in columns: # For each dataset per object
                if 'CALIOP' in column: # For CALIOP datasets
                    object_dict[column] = clean_dataframe.iloc[i][column][j]
                else: # For Himawari datasets
                    object_dict[column] = clean_dataframe.iloc[i][column]
            expanded_df = expanded_df.append(object_dict, ignore_index = True) # Add object to expanded dataframe
        # print('#-#-#-#-#-#-#-#-#-#-#-#-#-#-#')
    print(expanded_df)
    return expanded_df

def brute_force(caliop_file, him_dataset, caliop_name, him_name):
    """
    Will collocate CALIOP pixels w/ a Himawari-8 by brute force comparison of
    each CALIOP pixel with the whole Himawari-8 scene.

    :param caliop_overpass: Loaded CALIOP .hdf file to collocate with Himawari data
    :param him_dataset: satpy Scene of Himawari data to be collocated with CALIOP profile
    :param caliop_name: str type. Name of the given CALIOP .hdf file
    :param him_name: str type. Name of the folder containing the given Himawari data
    :return: pandas dataframe of collocated data
    """
    print('Starting collocation by brute force method')
    over_start = time()
    # filename = 'HS_H08_%s_FLDK.tar' % him_name
    # destination = '/g/data/k10/dr1709/ahi/%s' % him_name
    # sp.run('tar -C %s -xvf %s' % (destination, os.path.join(destination, filename)), shell=True)
    # sp.run('rm %s' % os.path.join(destination, filename), shell=True)
    him_scn = read_h8_folder(him_dataset)
    caliop_overpass = SD(caliop_file, SDC.READ)
    ### Run parallax-corrected collocation ###
    df = parallax_collocation(
        caliop_overpass = caliop_overpass,
        him_scn = him_scn,
        caliop_name = caliop_name,
        him_name = him_name
    )
    minutes_taken = (time() - over_start) / 60.
    print('Time taken for collocation: %0.2fmins' % minutes_taken)
    return df

def brute_force_parallel(him_dataset, him_name, caliop_file, caliop_name, return_dict, process_num):
    """
    Will collocate CALIOP pixels w/ a Himawari-8 by brute force comparison of
    each CALIOP pixel with the whole Himawari-8 scene.

    :param him_dataset: satpy Scene of Himawari data to be collocated with CALIOP profile
    :param him_name: str type. Name of the folder containing the given Himawari data
    :param caliop_file: Loaded CALIOP .hdf file to collocate with Himawari data
    :param caliop_name: str type. Name of the given CALIOP .hdf file
    :param return_dict: dict type. Dictionary where the collocated dataframe will be stored.
    :param process_num: int type. Number assigned to a collocation process and used as
                        the key in the return dictionary.
    :return: pandas dataframe of collocated data.
    """
    from glob import glob
    print('Starting collocation by brute force method for %s' % him_name)
    over_start = time()
    # filename = 'HS_H08_%s_FLDK.tar' % him_name
    if len(glob(os.path.join(him_dataset, '*.DAT'))) == 160:
        # sp.run('tar -C %s -xf %s' % (him_dataset, os.path.join(him_dataset, filename)), shell=True)
        # sp.run('rm %s' % os.path.join(him_dataset, filename), shell=True)
        him_scn = read_h8_folder(him_dataset)
        caliop_overpass = SD(caliop_file, SDC.READ)
        caliop_fname = caliop_file.split('/')
        caliop_fname = caliop_fname[-1]
        ### Run parallax-corrected collocation ###
        df = parallax_collocation_v2(
            caliop_overpass = caliop_overpass,
            him_scn = him_scn,
            caliop_name = caliop_fname,
            him_name = him_name
        )
        minutes_taken = (time() - over_start) / 60.
        print('Time taken for collocation: %0.2fmins' % minutes_taken)
        return_dict[str(process_num)] = df
    else:
        print('Himawari folder %s has not been retrieved' % him_name)
        return_dict[str(process_num)] = None

def isolate_closest(dataframe):
    """
    Identify repeated Himawari pixels and take keep the pixel closest to
    its collocated CALIOP pixel, removing the other duplicates.

    :param dataframe: pandas dataframe of collocated data.
    :return: pandas dataframe with only closest collocations.
    """
    print('Removing duplicated Himawari pixels...')
    comp_dataframe = dataframe[['Himawari Latitude', 'Himawari Longitude',
                               'CALIOP Latitude', 'CALIOP Longitude']]*1e6
    comp_dataframe = comp_dataframe.astype('int')
    all_duplicates = comp_dataframe.duplicated(['Himawari Latitude',
                                                'Himawari Longitude'],
                                               keep=False)
    all_duplicates = comp_dataframe[['Himawari Latitude', 'Himawari Longitude']][all_duplicates]
    dups = all_duplicates.duplicated(['Himawari Latitude',
                                      'Himawari Longitude'])
    all_duplicates = all_duplicates[dups!=True]
    uniques = zip(np.array(list(all_duplicates['Himawari Latitude'])),
                  np.array(list(all_duplicates['Himawari Longitude'])))
    rows_to_delete = []
    for dup_lat, dup_lon in uniques:
        where_lats = (dataframe['Himawari Latitude']*1e6).astype('int') == dup_lat
        where_lons = (dataframe['Himawari Longitude']*1e6).astype('int') == dup_lon
        where_dups = where_lats & where_lons
        duplicates = comp_dataframe[where_dups]
        idxs = duplicates.index.to_list()
        if len(idxs) == 0 or len(idxs) == 1:
            pass
        else:
            him_arr = np.array(duplicates[['Himawari Latitude', 'Himawari Longitude']], dtype='float32')
            cal_arr = np.array(duplicates[['CALIOP Latitude', 'CALIOP Longitude']], dtype='float32')
            comp_arr = np.average(np.abs(him_arr - cal_arr), axis=1)
            idxs.pop(comp_arr.argmin())
            rows_to_delete+=idxs
    rows_to_delete = list(dict.fromkeys(rows_to_delete))
    corrected_df = dataframe.drop(index=rows_to_delete)
    corrected_df = corrected_df.reset_index(drop=True)
    print('Done')
    return corrected_df

def save_df(dataframe, dataframe_name, base_dir):
    """
    Save the given dataframe as an HDF5 file called collocated_data_{cal_name}.h5
    for later use.

    :param dataframe: pandas dataframe.
    :param dataframe_name: str type. Name to be given to the saved dataframe.
    :param base_dir: str type. Directory where dataframe is to be stored.
    :return:
    """
    filename = 'collocated_data-%s.h5' %  dataframe_name
    full_name = os.path.join(base_dir, filename)
    print('Saving dataframe as %s.h5 in %s' % (filename, base_dir))
    df = dataframe.copy()
    df.to_hdf(path_or_buf = full_name,
              key = 'df',
              mode = 'w')
    print('Dataframe saved')

def load_df(path_to, dataframe_name):
    data_name = os.path.join(path_to, 'collocated_data-%s.h5' % dataframe_name)
    collocated_df = pd.read_hdf(data_name, 'df')
    # print(collocated_df.shape)
    # print(collocated_df.keys())
    print('Dataframe loaded')
    return collocated_df

def individual_collocation(caliop_file, him_names):
    """
    Carries out the full collocation between a CALIOP file and the
    corresponding Himawari-8 data.

    :param caliop_overpass: str type. Full path to CALIOP .hdf file,
                           stored as compressed .gz file.
    :param him_names: lst type. List of collocated Himawari folder names.
    :return: pandas dataframe of collocated data
    """
    ### Load CALIOP file ###
    sp.run('gunzip %s' % caliop_file, shell=True)
    caliop_overpass = SD(caliop_file[:-3], SDC.READ)
    sp.run('gzip %s' % caliop_file[:-3], shell=True)
    ### Carry out parallel collocation ###
    dst_list = [os.path.join('/g/data/k10/dr1709/ahi/', fn) for fn in him_names]  # Define paths to Himawari data
    args = zip(repeat(caliop_overpass), dst_list, repeat(caliop_file[-63:-7]), him_names)
    pool = mp.Pool(processes=len(him_names))
    df_list = pool.starmap(func=brute_force, iterable=args)
    pool.close()
    pool.join()
    ### Concatenate dataframes into single dataframe ###
    df = pd.concat(df_list, ignore_index=True)
    ### Convert CALIOP VFM integer numbers to custom feature numbers ###
    df['CALIOP Vertical Feature Mask'] = df['CALIOP Vertical Feature Mask'].apply(number_to_bit)
    df['CALIOP Vertical Feature Mask'] = df['CALIOP Vertical Feature Mask'].apply(custom_feature_conversion)
    ### Remove repeated Himawari pixels that are furthest from collocated CALIOP pixel ###
    df = clean_up_df(df)
    print('Number of entries: %s' % len(df))
    ### Return properly formatted dataframe of collocated data ###
    return df

### Collocated Data Analysis Tools ###

def plot_caliop_data(feature_dataframe, feature_type, feature_dict, fig_number):
    """
    Produces a 4x4 figure of CALIOP data:
     - AOD
     - QA
     - Feature Thickness
     - Feature Top Altitude
    :param feature_dataframe: pandas dataframe of feature data.
    :param feature_type: str type. Feature type from feature_dict to be plotted.
    :param feature_dict: dict type. Dictionary of features specifying the feature number.
    :param fig_number: int type. Number which is associated with the produced figure.
    :return: matplotlib.pyplot figure of histograms.
    """
    ### Identify where only the specified feature type is ###
    # where_only_feature = np.asarray(list(feature_dataframe['CALIOP Vertical Feature Mask']))
    where_only_feature = feature_dataframe['CALIOP Vertical Feature Mask'] == feature_dict[feature_type]
    ### Generate Figure ###
    fig, axes = plt.subplots(2, 2, num=fig_number, figsize=(16, 16))
    fig.suptitle('CALIOP Data for ' + feature_type, fontsize=24)
    ### Add AOD to plot ###
    data = np.asarray(list(feature_dataframe['CALIOP ODs for 532nm']))
    data = data[where_only_feature]
    bin_arr = np.arange(0, 5.1, 0.1)
    heights, bins = np.histogram(np.around(data, 1), bins=bin_arr)
    percent = np.asarray([i / sum(heights) * 100 for i in heights]).astype(float)
    axes[0, 0].bar(bins[:-1], percent, align="center", color='b', width=0.1)
    axes[0, 0].plot(bins[:-1], percent, 'r')
    axes[0, 0].tick_params(axis='both', which='major', labelsize=15)
    axes[0, 0].set_ylim((0, np.nanmax(percent) + 1))
    axes[0, 0].set_xlim((bin_arr[0], bin_arr[-1]))
    axes[0, 0].grid(b=True, which='major', color='#666666', linestyle='-')
    axes[0, 0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axes[0, 0].set_title('$CALIOP$ $Feature$ $Optical$ $Depth$ $for$ $532nm$', fontsize=15)
    axes[0, 0].set_xlabel('Optical Depth', fontsize=15)
    ### Add QA to plot ###
    data = np.asarray(list(feature_dataframe['CALIOP QA Scores']))
    data = data[where_only_feature]
    bin_arr = np.arange(0, 1.01, 0.01)
    heights, bins = np.histogram(np.around(data, 2), bins=bin_arr)
    percent = np.asarray([i / sum(heights) * 100 for i in heights]).astype(float)
    axes[0, 1].bar(bins[:-1], percent, align="center", color='b', width=0.01)
    axes[0, 1].plot(bins[:-1], percent, 'r')
    axes[0, 1].tick_params(axis='both', which='major', labelsize=15)
    axes[0, 1].set_ylim((0, np.nanmax(percent) + 1))
    axes[0, 1].set_xlim((bin_arr[0], bin_arr[-1]))
    axes[0, 1].grid(b=True, which='major', color='#666666', linestyle='-')
    axes[0, 1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axes[0, 1].set_title('$CALIOP$ $QA$ $Score$ $for$ $Feature$', fontsize=15)
    axes[0, 1].set_xlabel('QA Score', fontsize=15)
    ### Add Feature Top Height to plot ###
    data = np.asarray(list(feature_dataframe['CALIOP Feature Top Altitudes']))
    data = data[where_only_feature]
    bin_arr = np.arange(-0.5, 30.5, 0.5)
    heights, bins = np.histogram(np.around(data * 2., 0) / 2., bins=bin_arr)
    percent = np.asarray([i / sum(heights) * 100 for i in heights]).astype(float)
    axes[1, 1].bar(bins[:-1], percent, align="center", color='b', width=0.5)
    axes[1, 1].plot(bins[:-1], percent, 'r')
    axes[1, 1].tick_params(axis='both', which='major', labelsize=15)
    axes[1, 1].set_ylim((0, np.nanmax(percent) + 1))
    axes[1, 1].set_xlim((bin_arr[0], bin_arr[-1]))
    axes[1, 1].grid(b=True, which='major', color='#666666', linestyle='-')
    axes[1, 1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axes[1, 1].set_title('$Feature$ $Top$ $Altitude$', fontsize=15)
    axes[1, 1].set_xlabel('Altitude [km]', fontsize=15)
    ### Add Feature Thickness to plot ###
    data1 = np.asarray(list(feature_dataframe['CALIOP Feature Top Altitudes']))
    data1 = data1[where_only_feature]
    data2 = np.asarray(list(feature_dataframe['CALIOP Feature Base Altitudes']))
    data2 = data2[where_only_feature]
    data = data1 - data2
    data[data == np.nanmin(data)] = np.nan
    data = np.around(data, 1)
    bin_arr = np.arange(0., np.nanmax(data) + 0.1, 0.1)
    heights, bins = np.histogram(data, bins=bin_arr)
    percent = np.asarray([i / sum(heights) * 100 for i in heights]).astype(float)
    axes[1, 0].bar(bins[:-1], percent, align="center", color='b', width=0.1)
    axes[1, 0].plot(bins[:-1], percent, 'r')
    axes[1, 0].tick_params(axis='both', which='major', labelsize=15)
    axes[1, 0].set_ylim((0, np.nanmax(percent) + 1))
    axes[1, 0].set_xlim((bin_arr[0], bin_arr[-1]))
    axes[1, 0].grid(b=True, which='major', color='#666666', linestyle='-')
    axes[1, 0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axes[1, 0].set_title('$Feature$ $Thickness$', fontsize=15)
    axes[1, 0].set_xlabel('Thickness [km]', fontsize=15)
    ### Add common axis label ###
    fig.text(0.015, 0.5, 'Normalised Occurance within Dataset [%]',
              fontsize=18, va='center', rotation='vertical')
    fig.tight_layout(rect=[0.03, 0.02, 0.96, 0.95])
    return fig

def histogram_him_collocated_single(dataframe, feature_type):
    """
    Produces a figure containing histograms of Himawari band data
    from the given dataframe for the given feature type (single layer only).

    :param dataframe: pandas Dataframe of collocated data.
    :param feature_type: str type. Label for collocated feature type.
    :return: figure of histograms
    """
    feature_dict = {'Clear Air': 0,
                    'Low Overcast (Transparent)': 1,
                    'Low Overcast (Opaque)': 2,
                    'Transition Stratocumulus': 3,
                    'Low Broken Cumulus': 4,
                    'Altocumulus (Transparent)': 5,
                    'Altostratus (Opaque)': 6,
                    'Cirrus (Transparent)': 7,
                    'Deep Convective (Opaque)': 8,
                    'Clean Marine': 9,
                    'Dust': 10,
                    'Polluted Continental/Smoke': 11,
                    'Clean Continental': 12,
                    'Polluted Dust': 13,
                    'Elevated Smoke (Tropospheric)': 14,
                    'Dusty Marine': 15,
                    'PSC Aerosol': 16,
                    'Volcanic Ash': 17,
                    'Sulfate/Other': 18,
                    'Elevated Smoke (Stratospheric)': 19}
    ### Take only feature_type data ###
    where_feature = dataframe['CALIOP Vertical Feature Mask'] == feature_dict[feature_type]
    feature_df = dataframe[where_feature]
    # raw_data = np.asarray(list(feature_df['Himawari Band Values at 2km Resolution']))
    ### Generate dictionary of band data ###
    him_keys = ['Himawari Band 1 Mean at 2km Resolution',
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
                'Himawari Band 16 Value at 2km Resolution']
    data = {}
    for key in him_keys:
        data[key] = np.array(list(feature_df[key]))
    # for i in range(1, 21, 1):
    #     if i == 2 or i == 4 or i == 6 or i == 8:
    #         band = '$Band$ %s $\sigma$' % str(int(i / 2))
    #         data[band] = raw_data[:, i - 1]
    #     elif i > 7:
    #         band = '$Band$ %s' % str(i - 4)
    #         data[band] = raw_data[:, i - 1]
    #     else:
    #         band = '$Band$ %s' % str(int((i + 1) / 2))
    #         data[band] = raw_data[:, i - 1]
    #     print(band)
    ### Create figure with 4x4 subplots ###
    fig1, axes = plt.subplots(4, 5, figsize=(20, 16), num=1)
    fig1.suptitle('Himawari Data for ' + feature_type, fontsize=24)
    for i in range(4):  # Row
        for j in range(5):  # Column
            ### Calculate band to be plotted based on position w/n figure ###
            num = int(j + 1 + (i * 5))
            data_key = him_keys[num - 1]
            ### Assign band unit and "data width" based on which band it is ###
            if num <= 10:
                unit = 'Reflectance [%]'
                bin_arr = np.arange(0, 101, 1.)
            else:
                unit = 'Brightness Temperature [K]'
                bin_arr = np.arange(190, 301, 1.)
            ### Plot histogram of data w/ axes labelled ###
            if num == 2 or num == 4 or num == 6 or num == 8:
                num = int(num / 2)
                band = '$Band$ %s $\sigma$' % str(num)
            elif num > 7:
                num = num - 4
                band = '$Band$ %s' % str(num)
            else:
                num = int((num + 1) / 2)
                band = '$Band$ %s' % str(num)
            heights, bins = np.histogram(np.around(data[data_key], 0), bins = bin_arr)
            percent = np.asarray([i / sum(heights) * 100 for i in heights]).astype(float)
            axes[i,j].bar(bins[:-1], percent, align="center", color='b', width=1.)
            axes[i,j].plot(bins[:-1], percent, 'r')
            axes[i,j].set_ylim(0, 100.)
            axes[i, j].set_xlim((bin_arr[0], bin_arr[-1]))
            axes[i,j].tick_params(axis='both', which='major', labelsize=15)
            axes[i,j].grid(b=True, which='major', color='#666666', linestyle='-')
            axes[i,j].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            axes[i,j].set_title('%s (%s$\mu m$)' % (band, np.around(band_dict[str(num)],2)),
                                fontsize=15)
            axes[i,j].set_xlabel(unit, fontsize=15)
    ### Add common axis label ###
    fig1.text(0.015, 0.5, 'Normalised Occurance within Dataset [%]',
             fontsize=18, va='center', rotation='vertical')
    fig1.tight_layout(rect=[0.03, 0.02, 0.96, 0.95])
    return fig1

def histogram_CALIOP_collocated_single(dataframe, feature_type):
    """
    Produces a figure containing histograms of CALIOP data
    from the given dataframe for the given feature type (single layer only).

    :param dataframe: pandas Dataframe of collocated data.
    :param feature_type: str type. Label for collocated feature type.
    :return: figure of histograms
    """
    feature_dict = {'Clear Air': 0,
                    'Low Overcast (Transparent)': 1,
                    'Low Overcast (Opaque)': 2,
                    'Transition Stratocumulus': 3,
                    'Low Broken Cumulus': 4,
                    'Altocumulus (Transparent)': 5,
                    'Altostratus (Opaque)': 6,
                    'Cirrus (Transparent)': 7,
                    'Deep Convective (Opaque)': 8,
                    'Clean Marine': 9,
                    'Dust': 10,
                    'Polluted Continental/Smoke': 11,
                    'Clean Continental': 12,
                    'Polluted Dust': 13,
                    'Elevated Smoke (Tropospheric)': 14,
                    'Dusty Marine': 15,
                    'PSC Aerosol': 16,
                    'Volcanic Ash': 17,
                    'Sulfate/Other': 18,
                    'Elevated Smoke (Stratospheric)': 19}
    # vfm = np.asarray(list(dataframe['CALIOP Vertical Feature Mask']))
    # where_feature = (np.sum(vfm > 0., axis=1) == 1) & \
    #                 (np.sum(vfm, axis=1) == feature_dict[feature_type])
    where_feature = dataframe['CALIOP Vertical Feature Mask'] == feature_dict[feature_type]
    feature_df = dataframe[where_feature]
    fig2 = plot_caliop_data(feature_df, feature_type, feature_dict, fig_number=2)
    return fig2

def histogram_him_collocated_double(dataframe, top_feature, base_feature):
    """
    Produces a figure containing histograms of Himawari band data
    from the given dataframe where the top_feture type is above the
    base_feature type (double only).
    NB// All cloud types are reduced into a single cloud category.

    :param dataframe: pandas Dataframe of collocated data.
    :param top_feature: str type. Label for collocated feature type that sits
                        at the higher altitude.
    :param base_feature: str type. Label for collocated feature type that sits
                         at the lower altitude.
    :return: figure of histograms.
    """
    feature_dict = {'Clear Air': 0,
                    'Cloud': 1,
                    'Clean Marine': 9,
                    'Dust': 10,
                    'Polluted Continental/Smoke': 11,
                    'Clean Continental': 12,
                    'Polluted Dust': 13,
                    'Elevated Smoke (Tropospheric)': 14,
                    'Dusty Marine': 15,
                    'PSC Aerosol': 16,
                    'Volcanic Ash': 17,
                    'Sulfate/Other': 18,
                    'Elevated Smoke (Stratospheric)': 19}
    ### Carry out conversion of cloud types ###
    vfm = np.asarray(list(dataframe['CALIOP Vertical Feature Mask']))
    where_cloud = (vfm == 2) | (vfm == 3) | (vfm == 4) | (vfm == 5) | \
                  (vfm == 6) | (vfm == 7) | (vfm == 8)
    vfm[where_cloud] = 1
    ### Take double layers containing both feature types ###
    where_double_feature = (np.sum(vfm > 0., axis=1) == 2)
    where_good_combo = (vfm[:, 0] == feature_dict[top_feature]) & \
                       (vfm[:, 1] == feature_dict[base_feature])
    feature_df = dataframe[(where_good_combo) & (where_double_feature)]
    raw_data = np.asarray(list(feature_df['Himawari Band Values at 2km Resolution']))
    ### Generate dictionary of band data ###
    data = {}
    for i in range(1,21,1):
        if i == 2 or i == 4 or i == 6 or i == 8:
            band = '$Band$ %s $\sigma$' % str(int(i / 2))
            data[band] = raw_data[:, i - 1]
        elif i > 7:
            band = '$Band$ %s' % str(i - 4)
            data[band] = raw_data[:, i - 1]
        else:
            band = '$Band$ %s' % str(int((i + 1) / 2))
            data[band] = raw_data[:, i - 1]
    ### Create figure with 4x4 subplots ###
    fig1, axes = plt.subplots(4, 5, figsize=(20, 16), num=1)
    fig1.suptitle('Himawari Data for '+top_feature+' on '+base_feature, fontsize=24)
    for i in range(4):  # Row
        for j in range(5):  # Column
            ### Calculate band to be plotted based on position w/n figure ###
            num = int(j + 1 + (i * 5))
            print(num)
            ### Assign band unit and "data width" based on which band it is ###
            if num <= 10:
                unit = 'Reflectance [%]'
                bin_arr = np.arange(0, 101, 1.)
            else:
                unit = 'Brightness Temperature [K]'
                bin_arr = np.arange(190, 301, 1.)
            ### Plot histogram of data w/ axes labelled ###
            if num == 2 or num == 4 or num == 6 or num == 8:
                num = int(num  / 2)
                band = '$Band$ %s $\sigma$' % str(num)
            elif num > 7:
                num = num - 4
                band = '$Band$ %s' % str(num)
            else:
                num = int((num + 1) / 2)
                band = '$Band$ %s' % str(num)
            heights, bins = np.histogram(np.around(data[band], 0), bins=bin_arr)
            percent = np.asarray([i / sum(heights) * 100 for i in heights]).astype(float)
            axes[i, j].bar(bins[:-1], percent, align="center", color='b', width=1.)
            axes[i, j].plot(bins[:-1], percent, 'r')
            axes[i, j].set_ylim((0, np.nanmax(percent) + 1))
            axes[i, j].set_xlim((bin_arr[0], bin_arr[-1]))
            axes[i, j].tick_params(axis='both', which='major', labelsize=15)
            axes[i, j].grid(b=True, which='major', color='#666666', linestyle='-')
            axes[i, j].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            axes[i, j].set_title('$%s$ (%s$\mu m$)' % (band, np.around(band_dict[str(num)], 2)),
                                 fontsize=15)
            axes[i, j].set_xlabel(unit, fontsize=15)
    ### Add common axis label ###
    fig1.text(0.015, 0.5, 'Normalised Occurance within Dataset [%]',
              fontsize=18, va='center', rotation='vertical')
    fig1.tight_layout(rect=[0.03, 0.02, 0.96, 0.95])
    return fig1

def histogram_CALIOP_collocated_double(dataframe, top_feature, base_feature):
    """
    Produces a figure containing histograms of CALIOP data
    from the given dataframe where the top_feture type is above the
    base_feature type (double only).
    NB// All cloud types are reduced into a single cloud category.

    :param dataframe: pandas Dataframe of collocated data.
    :param top_feature: str type. Label for collocated feature type that sits
                        at the higher altitude.
    :param base_feature: str type. Label for collocated feature type that sits
                         at the lower altitude.
    :return: figure of histograms.
    """
    feature_dict = {'Clear Air': 0,
                    'Cloud': 1,
                    'Clean Marine': 9,
                    'Dust': 10,
                    'Polluted Continental/Smoke': 11,
                    'Clean Continental': 12,
                    'Polluted Dust': 13,
                    'Elevated Smoke (Tropospheric)': 14,
                    'Dusty Marine': 15,
                    'PSC Aerosol': 16,
                    'Volcanic Ash': 17,
                    'Sulfate/Other': 18,
                    'Elevated Smoke (Stratospheric)': 19}
    ### Carry out conversion of cloud types ###
    vfm = np.asarray(list(dataframe['CALIOP Vertical Feature Mask']))
    where_cloud = (vfm == 2) | (vfm == 3) | (vfm == 4) | (vfm == 5) | \
                  (vfm == 6) | (vfm == 7) | (vfm == 8)
    vfm[where_cloud] = 1
    ### Take double layers containing both feature types ###
    where_double_feature = (np.sum(vfm > 0., axis=1) == 2)
    where_good_combo = (vfm[:, 0] == feature_dict[top_feature]) & \
                       (vfm[:, 1] == feature_dict[base_feature])
    feature_df = dataframe[(where_good_combo) & (where_double_feature)]
    top_fig = plot_caliop_data(feature_df, top_feature, feature_dict, fig_number=2)
    base_fig = plot_caliop_data(feature_df, base_feature, feature_dict, fig_number=3)
    return top_fig, base_fig


# if __name__ == '__main__':
    ### Print Himawari folder names that fall w/n CALIOP profile time ###
    # names = find_possible_collocated_him_folders(cal_profile)
    # for name in names:
    #     print(name)
    # get_him_folders(names)
    ### Histogram Feature Type Data ###
    # f_type = 'Cloud'
    # other_f_type = 'Dusty Marine'
    # fig1 = histogram_him_collocated_double(collocated_data, f_type, other_f_type)
    # top_fig, base_fig = histogram_CALIOP_collocated_double(collocated_data, f_type, other_f_type)
    # fig1.savefig('%s Himawari Data.png' % f_type)
    # top_fig.savefig('%s on %s (%s) CALIOP Data.png' % (f_type, other_f_type, f_type))
    # base_fig.savefig('%s on %s (%s) CALIOP Data.png' % (f_type, other_f_type, other_f_type))
