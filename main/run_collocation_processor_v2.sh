#!/bin/bash

while getopts "l:f:d:" opt; do 
	case "${opt}" in
		l) list_of_files="${OPTARG}" ;;
		f) single_file="${OPTARG}" ;;
		d) target_dir="${OPTARG}" ;;
	esac
done

qsub -v "CALIOP_LIST=$list_of_files,TARGET_DIR=$target_dir,CALIOP_FILE=$single_file" -o '/g/data/k10/dr1709/code/Personal/Collocation/v2/main/.o+e_archive/' -e '/g/data/k10/dr1709/code/Personal/Collocation/v2/main/.o+e_archive/' /g/data/k10/dr1709/code/Personal/Collocation/v2/main/collocation_processor_v2.qsub
