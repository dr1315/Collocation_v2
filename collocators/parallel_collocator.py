import sys
import os
import numpy as np
import subprocess as sp
import multiprocessing as mp
import pandas as pd
from pyhdf.SD import SD, SDC
from itertools import repeat
import pickle
import datetime
from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
sys.path.append(
  os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "Tools"
  )
)
from collocation import brute_force_parallel, clean_up_df, save_df, read_list
from caliop_tools import number_to_bit, custom_feature_conversion, calipso_to_datetime

def build_args(himawari_folder_names, himawari_base_dir,
               caliop_filename, caliop_base_dir):
    """
    Takes the list of Himawari-8 scenes and collocates them with the given CALIOP file.

    :param himawari_folder_names: list type. List of folder names for the Himawari-8
                                  scenes to be collocated.
    :param himawari_base_dir: str type. The full path to the directory in which the
                              Himawari-8 data is held.
    :param caliop_filename: str type. Name of the CALIOP file to be collocated with the given
                            Himawari-8 scenes.
    :param caliop_base_dir: str type. The full path to the directory in which the
                            CALIOP file is held.
    :return: list type. A list of lists, with each entry corresponding to a Himawari-8
             scene and CALIOP overpass to be collocated.
    """
    args = [] # Create empty list of args to be filled
    for himawari_name in himawari_folder_names: # For each folder containing a Himawari-8 scene
        arg = (os.path.join(himawari_base_dir, himawari_name), # Himawari-8 folder from list
               himawari_name,  # Name of the Himawari-8 folder to be collocated
               os.path.join(caliop_base_dir, caliop_filename),  # CALIOP file to be collocated
               caliop_filename[-25:-4]) # CALIOP overpass "name", I.E. the date-time marker given in the filename.
        args.append(arg) # Add sublist of args for collocation to args list
    return args

def parallel_collocation(brute_force_args):
    """
    Will take the input args and collocate each set of files in parallel.

    :param brute_force_args: list type. Output of build_args function for collocation args.
    :return: list of collocated dataframes
    """
    manager = mp.Manager() # Start Manager to track parallel processes
    return_dict = manager.dict() # Create dictionary for temporarily storing collocated data
    processes = [] # Create empty list of processes to be filled
    for num, arg_set in enumerate(brute_force_args): # Set each collocation task running
        proc_num = num + 1
        print('Starting Process %s' % proc_num)
        p = mp.Process(target = brute_force_parallel, # Create a collocation process
                       args = tuple(list(arg_set) + [return_dict, proc_num]))
        processes.append(p) # Add task to processes list
        p.start() # Start the collocation process
    for process in processes: # For each process started
        process.join() # Connect all processes so that the next line in this
                       # function will not run until al processes have been completed
    print('Processes completed')
    print('Returned process keys: ', return_dict.keys())
    process_nums = return_dict.keys()
    process_nums.sort()
    df_list = [] # Create a final list for storing the collocated data
    for num in process_nums: # For each process's collocated dataframe
        df_list.append(return_dict[str(num)]) # Add dataframe to df_list; ensures ordering is correct
    return df_list # Return the ordered list of collocated dataframes

def post_process(list_of_df):
    """
    Takes a list of collocated dataframes and processes them into a single dataframe.

    :param list_of_df: list type. List of dataframes to be concatenated and processed.
    :return: pandas dataframe of collocated data.
    """
    list_of_df = tuple(list_of_df) #Ensure list is converted to tuple type
    df = pd.concat(list_of_df, ignore_index=True) # Concatenate dataframes
    # df['CALIOP Vertical Feature Mask'] = df['CALIOP Vertical Feature Mask'].apply(number_to_bit) # Convert CALIOP VFM to custom
    # df['CALIOP Vertical Feature Mask'] = df['CALIOP Vertical Feature Mask'].apply(custom_feature_conversion) # version pf VFM
    df = clean_up_df(df) # Remove duplicated entries (keeps closest matches)
    print(df.shape) # Print final shape of the dataframe
    return df # Return the single dataframe of collocated data

def parallel_collocator(himawari_folder_names, himawari_base_dir,
                        caliop_filename, caliop_base_dir,
                        save_final_dataframe=False, df_dir='path/to/storage/dir'):
    """
    Takes a list of folder names for Himawari-8 scenes which have been approximately
    collocated with a CALIOP file and explicitly collocates each scene with the given overpass.
    The final data is given as a pandas dataframe.

    :param himawari_folder_names: list type. List of folder names for the Himawari-8
                                  scenes to be collocated.
    :param himawari_base_dir: str type. The full path to the directory in which the
                              Himawari-8 data is held.
    :param caliop_filename: str type. Name of the CALIOP file to be collocated with the given
                            Himawari-8 scenes.
    :param caliop_base_dir: str type. The full path to the directory in which the
                            CALIOP file is held.
    :param save_final_dataframe: bool type. If True, will save the dataframe in the directory
                                 set using the df_dir arg.
    :param df_dir: str type. The full path to the directory in which the final datframe will be saved.
    :return: pandas dataframe of collocated data.
    """
    #-#-# Build args for the parallel collocation processes #-#-#
    args = build_args(himawari_folder_names=himawari_folder_names,
                      himawari_base_dir=himawari_base_dir,
                      caliop_filename=caliop_filename,
                      caliop_base_dir=caliop_base_dir)
    #-#-# Carry out collocation and return the list of collocated dataframes #-#-#
    df_list = parallel_collocation(brute_force_args=args)
    #-#-# Post-process the collocated dataframes and concatenate them into a single dataframe #-#-#
    final_df = post_process(list_of_df=df_list)
    #-#-# Save the final dataframe if the save_df option is activated #-#-#
    if save_final_dataframe:
        if df_dir == 'path/to/storage/dir':
            raise Exception('Please set the path to the directory in which the final dataframe will be stored.')
        else:
            if '1km' in caliop_filename:
                cres = '1km'
            elif '5km' in caliop_filename:
                cres = '5km'
            save_df(dataframe=final_df,
                    dataframe_name=cres+'_'+caliop_filename[-25:-4],
                    base_dir=df_dir)
    #-#-# Return the final dataframe #-#-#
    return final_df

def main(caliop_filename, staging_dir):
    split_caliop_fname = caliop_filename.split('/')
    print('CALIOP name: ', split_caliop_fname)
    data_dir = os.path.join(staging_dir, '.proc_data')
    print('Directory for processable data: ', data_dir)
    folder_list_name = f'{split_caliop_fname[-1][:-4]}_him_folders.txt' 
    him_name_list = read_list(list_filename = folder_list_name, list_dir = staging_dir)
    for hname in him_name_list:
        print(hname, 'in .proc_dir: ', os.path.exists(os.path.join(data_dir, hname)))
    parallel_collocator(himawari_folder_names = him_name_list,
                        himawari_base_dir = data_dir,
                        caliop_filename = split_caliop_fname[-1],
                        caliop_base_dir = data_dir,
                        save_final_dataframe = True,
                        df_dir = staging_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="name of file to be processed"
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
    
