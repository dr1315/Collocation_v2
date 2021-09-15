'''
Main processor for v2 of the collocation between CALIOP and Himawari-8.
'''

import os
import sys
import traceback
from datetime import datetime
from pyhdf.SD import SD, SDC


def log_w_message(message):
    dt_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")
    dt_message = f'{dt_now} - {message}'
    print(dt_message)
    return dt_message
    
def build_temp_folders(fname, target_dir):
    staging_dir = os.path.join(target_dir, fname[:-4])
    if not os.path.exists(staging_dir) and not os.path.isdir(staging_dir):
        os.mkdir(staging_dir)
    archive_dir = os.path.join(staging_dir, '.o+e_archive')
    if not os.path.exists(archive_dir) and not os.path.isdir(archive_dir):
        os.mkdir(archive_dir)
    proc_data_dir = os.path.join(staging_dir, '.proc_data')
    if not os.path.exists(proc_data_dir) and not os.path.isdir(proc_data_dir):
        os.mkdir(proc_data_dir)
    return staging_dir, archive_dir, proc_data_dir

def read_list(filename):
    with open(filename, 'r') as f:
        all_lines = [line[:-1] for line in f.readlines()]
    header = [line for line in all_lines if line.startswith('#')]
    good_lines = [line for line in all_lines if not line.startswith('#')]
    return header, good_lines

def decompress_h8_data(data_dir):
    h8_dirs = [dirs[1] for dirs in os.walk(data_dir)][0]
    print(h8_dirs)
    for h8_dir in h8_dirs:
        full_dir = os.path.join(data_dir, h8_dir)
        os.chdir(full_dir)
        os.system(f'tar -xf {os.path.join(full_dir, "HS_H08_"+h8_dir+"_FLDK.tar")}')
        os.system(f'rm {os.path.join(full_dir, "HS_H08_"+h8_dir+"_FLDK.tar")}')
        os.system(f'bunzip2 {full_dir}/*.bz2')

def full_collocation(fname, target_dir):
    caliop_filename = fname.split('/')[-1]
    if 'CAL' not in caliop_filename:
        raise Exception('Filename is not an acceptable format')
    if caliop_filename[-4:] != '.hdf':
        caliop_filename = caliop_filename + '.hdf'
    if 'V4-21' in caliop_filename:
        caliop_filename = caliop_filename.replace('V4-21', 'V4-20')
        fname = fname.replace('V4-21', 'V4-20')
    print(caliop_filename)
    log_w_message(f'Initiating collocation for {caliop_filename}')
    staging_dir, archive_dir, proc_data_dir = build_temp_folders(fname, target_dir)
    os.chdir(staging_dir)
    log_w_message(f'Getting CALIOP file: {caliop_filename}')
    os.system(f'qsub -W block=True -v "FNAME={fname},TARGET_DIR={proc_data_dir}" -o {os.path.join(archive_dir, "get_caliop_output.txt")} -e {os.path.join(archive_dir, "get_caliop_error.txt")} /g/data/k10/dr1709/code/Personal/Collocation/v2/data_grabbers/caliop_data_grabber.qsub')
    if not os.path.exists(os.path.join(proc_data_dir, caliop_filename)):
                raise Exception('CALIOP file %s cannot be retrieved' % caliop_filename)
    log_w_message('CALIOP file retrieved')
    log_w_message(f'Finding Himawari-8 scenes that collocate with {caliop_filename}')
    sys.path.append('/g/data/k10/dr1709/code/Personal/Collocation/v2/collocators')
    from rough_collocator import main as rc_main
    rc_main(os.path.join(proc_data_dir, fname), staging_dir)
    log_w_message('Collocated Himawari-8 scenes found')
    log_w_message('Getting collocated Himawari-8 folders from mdss')
    os.system(f'qsub -W block=True -v "FNAME={fname},TARGET_DIR={staging_dir}" -o {os.path.join(archive_dir, "get_himawari_output.txt")} -e {os.path.join(archive_dir, "get_himawari_error.txt")} /g/data/k10/dr1709/code/Personal/Collocation/v2/data_grabbers/himawari_data_grabber.qsub')
    log_w_message('All collocated Himawari-8 folders retrieved')
    log_w_message('Decompressing compressed Himawari-8 data')
    decompress_h8_data(proc_data_dir)
    log_w_message('Decompression complete')
    log_w_message('Carrying out collocation of Himawari-8 and CALIOP data')
    os.system(f'qsub -W block=True -v "FNAME={fname},TARGET_DIR={staging_dir}" -o {os.path.join(archive_dir, "parallel_collocation_output.txt")} -e {os.path.join(archive_dir, "parallel_collocation_error.txt")} /g/data/k10/dr1709/code/Personal/Collocation/v2/collocators/parallel_collocator.qsub')
    log_w_message(f'Collocated data stored in {staging_dir}')
    log_w_message('Cleaning out Himawari-8 and CALIOP data')
    os.system(f'rm -r {proc_data_dir}')

def main(list_of_files, target_dir):
    log_w_message('Reading in filenames')
    header, fnames = read_list(list_of_files)
    log_w_message(f'Running for {len(fnames)} files')
    #num_failures = 0
    for n, fname in enumerate(fnames):
        try:
            full_collocation(fname=fname, target_dir=target_dir)
            fnames[n] = '# ' + fname
            with open(list_of_files,'w') as f:
                f.writelines([head_line + '\n' for head_line in header] + [lst_fname + '\n' for lst_fname in fnames])
        except Exception as e:
            log_w_message(f'{fname} collocation failed')
            traceback.print_exc()
            #if num_failures < 2:
            #    num_failures += 1
            #else:
            #    break
            staging_dir_name = fname.split('/')[-1][:-4]
            os.system(f'rm -r {os.path.join(target_dir, staging_dir_name)}')
            pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--list_of_files",
        nargs="?",
        type=str,
        help="name of .txt file listing all the files to be downloaded"
        )
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
    if args.list_of_files is not None:
        main(args.list_of_files, args.target_directory)
    elif args.filename is not None:
        full_collocation(args.filename, args.target_directory)
    else:
        raise Exception('Need to provide a filename or a text file containing a list of filenames')
