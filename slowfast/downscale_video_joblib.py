# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import uuid

from joblib import delayed
from joblib import Parallel
import pandas as pd

#file_src = 'trainlist.txt'
#folder_path = 'YOUR_DATASET_FOLDER/train/'
#output_path = 'YOUR_DATASET_FOLDER/train_256/'


# file_src = '/vallist.txt'
# folder_path = 'YOUR_DATASET_FOLDER/val/'
# output_path = 'YOUR_DATASET_FOLDER/val_256/'


'''file_list = []

f = open(file_src, 'r')

for line in f:
    rows = line.split()
    fname = rows[0]
    file_list.append(fname)

f.close()'''


def downscale_clip(inname, outname):

    status = False
    inname = '"%s"' % inname
    outname = '"%s"' % outname
    command = "ffmpeg  -loglevel panic -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -q:v 1 -c:a copy {}".format( inname, outname)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output

    status = os.path.exists(outname)
    return status, 'Downscaled'


def downscale_clip_wrapper(row):

    nameset  = row.split('/')
    videoname = nameset[-1]
    classname = nameset[-2]

    output_folder = output_path + classname
    if os.path.isdir(output_folder) is False:
        try:
            os.mkdir(output_folder)
        except:
            print(output_folder)

    inname = folder_path + classname + '/' + videoname
    outname = output_path + classname + '/' +videoname

    downscaled, log = downscale_clip(inname, outname)
    return downscaled

def main(input_csv, output_dir):
    f_classes = open("classnames_train.txt","w")
    f = open("test_train.txt","w")

    #output_dir = "/data/Peter/Data/Kinetics600_rescaled/"
    #input_dir = "/data/Peter/Data/Kinetics600/"

    output_dir = "/data/Peter/Data/Kinetics600_train_rescaled/"
    input_dir = "/data/Peter/Data/Kinetics600_train/"

    dirlist = os.listdir(input_dir) 
    mehetovabb = True
    for class_number, classname in enumerate(sorted(dirlist)):
        print(classname)
        '''if classname != "smoking pipe" and mehetovabb:
            continue
        mehetovabb = False'''

        input_class_dir_path = os.path.join(input_dir, classname)
        videonames = os.listdir(input_class_dir_path)

        output_classname = classname.replace(" ", "_")
        output_class_dir_path = os.path.join(output_dir, output_classname)

        f_classes.write("\t\"")
        f_classes.write(output_classname)
        f_classes.write("\": " + str(class_number)+",\n")

        if os.path.exists(output_class_dir_path) is False:
            os.mkdir(output_class_dir_path)
        
        for videoname in videonames:
            inname = os.path.join(input_class_dir_path, videoname)
            print(inname)
            outname = os.path.join(output_class_dir_path, videoname)
            f.write(outname + " " + str(class_number) + "\n")
            downscaled, log = downscale_clip(inname, outname)

if __name__ == '__main__':
    description = 'Helper script for trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_dir', type=str,
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.')
    main(**vars(p.parse_args()))