#!/bin/bash
YEAR=$1
#YEAR='2019'

for MONTH in {1..12}
do
	len=`expr length $MONTH`
	if [ $len -lt 2 ]; then
		MONTH='0'$MONTH
	fi
	echo ${YEAR}-${MONTH}
	/g/data/k10/dr1709/code/Personal/Collocation/v2/main/run_collocation_processor_v2.sh -l /g/data/k10/dr1709/CALIOP/1km_Cloud/${YEAR}/${MONTH}/caliop_files_${YEAR}_${MONTH}.txt -d /g/data/k10/dr1709/CALIOP/1km_Cloud/${YEAR}/${MONTH}
done
