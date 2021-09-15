import sys
import os
from pyhdf.SD import SD, SDC
sys.path.append("/g/data/k10/dr1709/code/Personal/Tools")
from collocation import find_possible_collocated_him_folders, write_list


def main(fname, target_dir):
    caliop_overpass = SD(fname, SDC.READ)
    list_of_him_folders = find_possible_collocated_him_folders(caliop_overpass)
    split_fname = fname.split('/')
    list_name = f'{split_fname[-1][:-4]}_him_folders.txt'
    write_list(list_of_him_folders, list_name, target_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-f",
            "--filename",
            type=str,
            help="name of file to be downloaded"
            )
    parser.add_argument(
            "-d",
            "--target_directory",
            default=os.getcwd(),
            type=str,
            help="full path to the directory where the files will be stored"
            )
    args = parser.parse_args()
    main(args.filename, args.target_directory)
