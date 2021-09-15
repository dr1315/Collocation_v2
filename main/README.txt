Data format for collocated files using:
	1km CALIOP products:
		CALIOP data:
			- Filenames
			- Pixel scan times
			- Latitudes
			- Longitudes
			- Vertical feature mask (binary format; see CALIOP documentation)
			- QA scores
			- CAD scores
			- Feature top altitudes
			- Feature base altitudes
			- IGBP surface types
			- DEM surface elevation 
		Himawari-8 (AHI) data:
			- Folder name
			- Latitude
			- Longitude
			- Original position in full-disk array
			- Land-sea mask
			- Band data at 2km resolution:
				* Bands 1-4 -> mean and standard deviation of downscaled pixels
				* Bands 5-16 -> original value
			- Scene start time
			- Scene end time 
		ERA5 data:
			- Single level data:
				* Latitude
				* Longitude
				* Forecast albedo
				* UV albedo (direct)
				* UV albedo (diffuse)
				* NIR albedo (direct)
				* NIR albedo (diffuse)
				* 2m temperature
				* Skin temperature
				* Sea-ice fraction
			- Pressure level data:
				* U Wind Component Profile
				* V Wind Component Profile
				* Atmosphere Temperature Profile
				* Specific Cloud Liquid Water Content Profile
                                * Specific Cloud Ice Water Content Profile
                                * Relative Humidity Profile

