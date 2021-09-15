import sys
import os
import numpy as np
import pandas as pd
from datetime import timezone
from pyorbital.orbital import get_observer_look
from pyorbital.astronomy import get_alt_az
sys.path.append("/g/data/k10/dr1709/code/Tools")
from collocation import load_df, save_df


def main(dataframe_dir, dataframe_marker):
    dataframe = load_df(path_to=dataframe_dir,
                        dataframe_name=dataframe_marker)
    lats = list(dataframe['Himawari Latitude'])
    lons = list(dataframe['Himawari Longitude'])
    dtime_start = dataframe['Himawari Scene Start Time']
    dtime_delta = dataframe['Himawari Scene End Time'] - dtime_start
    dtime_avg = list(dtime_start + dtime_delta / 2)
    SatElvAs, SatAziAs = get_observer_look(sat_lon=140.7,
                                           sat_lat=0.0,
                                           sat_alt=35793.,
                                           utc_time=np.array(dtime_avg),
                                           lon=np.array(lons),
                                           lat=np.array(lats),
                                           alt=np.zeros(len(lats)))
    SolarElvAs, SolarAziAs = get_alt_az(utc_time=np.array(dtime_avg),
                                        lon=np.array(lons),
                                        lat=np.array(lats))
    SolarElvAs = np.rad2deg(SolarElvAs)
    SolarAziAs = np.rad2deg(SolarAziAs)
    SolarZenAs = 90. - SolarElvAs
    dataframe['Himawari Solar Zenith Angle'] = SolarZenAs
    dataframe['Himawari Solar Azimuth Angle'] = SolarAziAs
    dataframe['Himawari Observer Elevation Angle'] = SatElvAs
    dataframe['Himawari Observer Azimuth Angle'] = SatAziAs
    save_df(dataframe=dataframe,
            dataframe_name=dataframe_marker,
            base_dir=dataframe_dir)


if __name__ == '__main__':
    df_dir = sys.argv[-1]
    list_of_dfs = [df[-24:-3] for df in os.listdir(df_dir) if df.endswith(".h5")]
    for n, df in enumerate(list_of_dfs):
        print('Updating %d of %d: %s' % (n+1, len(list_of_dfs), df))
        main(dataframe_dir=df_dir,
             dataframe_marker=df)
