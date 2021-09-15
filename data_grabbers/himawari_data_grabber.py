'''
Will retrieve Himawari-8 folders via MDSS from a .txt file containing 
a list of folder names and put them into the target directory for the
CALIOP file specified.
'''

import sys
import os
sys.path.append("/g/data/k10/dr1709/code/Personal/Tools")
from collocation import get_him_folders, read_list


def main(caliop_fname, target_dir):
    proc_data_dir = os.path.join(target_dir, '.proc_data')
    split_fname = caliop_fname.split('/')
    folder_list_name = f'{split_fname[-1][:-4]}_him_folders.txt'
    collocated_folders = read_list(
        list_filename = folder_list_name,
        list_dir = target_dir
    )
    get_him_folders(
        him_names = collocated_folders,
        data_dir = proc_data_dir
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        nargs="?",
        type=str,
        help="name of file to be downloaded"
    )
    parser.add_argument(
        "-d",
        "--target_directory",
        nargs="?",
        default=os.getcwd(),
        type=str,
        help="full path to the directory where the files will be stored"
    )
    args = parser.parse_args()
    main(args.filename, args.target_directory)
