from pathlib import Path
import os
import pandas as pd



def read_vaac(fn_vaac):

	import re
	longt = []
	lat = []
	year = []
	month = []
	day = []
	time =[]

	fullName = fn_vaac

	with open(fullName) as file:

		# Importing file as a single string
		lines = file.read().replace('        '," ").replace('\n','').replace('-','')

		#Checnking if VA cloud coordinates exists
		if 'VA NOT IDENTIFIABLE' in lines:
			return None, None, None, None, None, None

		if 'VA NOT IDENTIFIABLE' not in lines:
			#Collecting year, data and time data
			dateStr = re.search('DTG:(.+?)Z',lines).group(0)
			year = dateStr[5:9]
			month = dateStr[9:11]
			day = dateStr[11:13]
			time = str(dateStr[14:-3]+":"+dateStr[-3:-1])

			coordStr = re.search('CLD:(.+?)MOV',lines).group(0)

			#Checking if coordinates 'North' or 'South'
			if coordStr[15] == 'S':
				latde = re.findall(r'\bS\d+',coordStr)
				test = 1

			elif coordStr[15] == "N":
				latde = re.findall(r'\bN\d+',coordStr)
				test = 0

			east = re.findall(r'\bE\d+',coordStr)

			#Formating coordinates
			for x in range(len(latde)):
				tempN = latde[x]
				coord_decimalN = float(tempN[1:]) / 100.
				coord_stringN = '{0:.2f}'.format(coord_decimalN).split('.')
				coord_floatN = float(coord_stringN[0]) + float(coord_stringN[1])/60.
				if test == 1:
					#South coordinates become negative
					coord_floatN = -coord_floatN
					lat.append(coord_floatN)
				elif test == 0:
					lat.append(coord_floatN)

				tempE = east[x]
				coord_decimalE = float(tempE[1:]) / 100.
				coord_stringE = '{0:.2f}'.format(coord_decimalE).split('.')
				coord_floatE = float(coord_stringE[0]) + float(coord_stringE[1])/60.
				longt.append(coord_floatE)

	return longt, lat, year, month, day, time

