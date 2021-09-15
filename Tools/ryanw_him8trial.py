import satpy
from satpy import Scene
from glob import glob
from pathlib import Path
from him8analysis import read_h8_folder 
import os
import numpy as np
import pandas as pd
from ReadInPoly import read_vaac
import matplotlib
# matplotlib.use('pdf')
from matplotlib.path import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings('ignore')


#read in Himwari satellite data from here
indir = '/g/data/k10/dr1709/AHI_comp/20200105_0500/'

# write out data from here 
oudir = '/g/data/k10/ryanw/plots/'

# lons_poly = [108.444817,112.444817]
# lats_poly = [-9.540831,-5.540831]

directory_in = "/g/data/k10/ryanw/"

# Predefining output dataframe
vaacframe = pd.DataFrame()

for item in glob(directory_in + '*.txt'):

	temp = pd.DataFrame()
		
	lons_poly, lats_poly, year_poly, month_poly, day_poly, time_poly = read_vaac(item)

	temp['Longitude'] = lons_poly
	temp['Latitude'] = lats_poly
	temp['Year'] = year_poly
	temp['Day'] = day_poly
	temp['Month'] = month_poly
	temp['Time'] = time_poly
		
	vaacframe=vaacframe.append(temp,ignore_index = True)


lons_poly = vaacframe.Longitude
lats_poly = vaacframe.Latitude
print(lons_poly)

#INSERT TEST GRAPH###
def set_global(self):
        """

        """
        self.set_xlim(self.projection.x_limits)
        self.set_ylim(self.projection.y_limits)

ax = plt.axes(projection=ccrs.PlateCarree())

x = lons_poly
y = lats_poly
x_ex = [x + 2 for x in x]
y_ex = [y + 2 for y in y]
set_global(ax)
# ax.set_extent([112, 154, -44, -5.6], ccrs.PlateCarree())
ax.coastlines(resolution='110m')
# plt.plot(x, y,  markersize=2, marker='o', color='red')
# plt.fill(x, y, color='coral')
# plt.show()


polygon = [(lon, lat) for lon, lat in zip(lons_poly, lats_poly)]

scn = read_h8_folder(indir,True)

# print(scn.keys())
# print('Print here 111111')
# print(type(scn))
# print('Print here 222222')
# print(np.shape(scn['B10']))
# print(scn['B10'])
# print('Print here 333333')
# plt.plot(scn['B10'])
# print('Print here 444444')
# plt.show()
# plt.imshow(scn['B10'])

lons_arr, lats_arr = scn['B10'].area.get_lonlats()
x, y = lons_arr.flatten(), lats_arr.flatten()
points = np.vstack((x,y)).T
print('Print here 55555')

p = Path(polygon)  # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(lats_arr.shape)
print(mask.shape, np.sum(mask))
# exit()
b10_ma = np.ma.array(scn['B10'], mask=mask==False)
# plt.plot(b10_ma)
plt.imshow(np.array(scn['B10']), cmap = 'Greys', origin='upper', transform=ccrs.Geostationary(140.735785863), extent=(-5500000, 5500000, -5500000, 5500000))
plt.imshow(mask.astype('float'), alpha=mask.astype('float'), vmin=0., vmax=1., cmap='Reds', origin='upper', transform=ccrs.Geostationary(140.735785863), extent=(-5500000, 5500000, -5500000, 5500000))
plt.plot(lons_poly, lats_poly, transform=ccrs.Geodetic(), marker='o', color='red')
plt.show() 






