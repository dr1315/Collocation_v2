"""
Get CALIOP files from NASA using a list of filenames
(For use in copyq job using qsub)
"""

import os
import sys
from pyhdf.SD import SD, SDC

def read_list(filename):
  with open(filename, 'r') as f:
      good_lines = [line[:-1] for line in f.readlines() if not line.startswith('#')]
  return good_lines

def grab_CALIOP_file(caliop_fname, target_dir='.'):
  if os.path.exists(os.path.join(target_dir, caliop_fname)):
    print(f'{caliop_fname} already exists in {target_dir}')
  else:
    print(f'Downloading {caliop_fname}')
    root_dir = "https://asdc.larc.nasa.gov/data/CALIPSO"
    ftype_dir = caliop_fname[4:34]
    year_dir = caliop_fname[35:39]
    month_dir = caliop_fname[40:42]
    url = os.path.join(root_dir, ftype_dir, year_dir, month_dir, caliop_fname)
    # os.chdir(target_dir)
    n_attempts = 0
    while not os.path.exists(os.path.join(target_dir, caliop_fname)) and n_attempts < 3:
      os.system(f'curl -o {os.path.join(target_dir, caliop_fname)} -b ~/.urs_cookies -c ~/.urs_cookies --remote-name --location --netrc {url}')
      try:
          SD(os.path.join(target_dir, caliop_fname), SDC.READ)
      except:
          n_attempts += 1
    if not os.path.exists(os.path.join(target_dir, caliop_fname)):
        raise Execption('CALIOP file could not be retrieved')
    else:
        print(f'{caliop_fname} saved to {target_dir}')

def main(filename, target_dir='.'):
  caliop_fnames = read_list(filename)
  for caliop_fname in caliop_fnames:
    grab_CALIOP_file(caliop_fname, target_dir)


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
    grab_CALIOP_file(args.filename, args.target_directory)
  else:
    raise Exception('Need to provide a filename or a text file containing a list of filenames')
