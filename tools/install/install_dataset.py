#!/usr/bin/python
###############################################################################
#    (C) Copyright 2022 CEA LIST. All Rights Reserved.
#    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)
#
#    This software is governed by the CeCILL-C license under French law and
#    abiding by the rules of distribution of free software.  You can  use,
#    modify and/ or redistribute the software under the terms of the CeCILL-C
#    license as circulated by CEA, CNRS and INRIA at the following URL
#    "http://www.cecill.info".
#
#    As a counterpart to the access to the source code and  rights to copy,
#    modify and redistribute granted by the license, users are provided only
#    with a limited warranty  and the software's author,  the holder of the
#    economic rights,  and the successive licensors  have only  limited
#    liability.
#
#    The fact that you are presently reading this means that you have had
#    knowledge of the CeCILL-C license and that you accept its terms.
###############################################################################

import os
import sys
import shutil
import argparse
import requests
import tarfile, zipfile, gzip

###############################################################################

# Archive extensions
ZIP_EXTENSION = ".zip"
TAR_EXTENSION = ".tar"
TAR_GZ_EXTENSION = ".tar.gz"
TGZ_EXTENSION = ".tgz"
TAR_BZ2_EXTENSION = ".tar.bz2"
GZ_EXTENSION = ".gz"
DAT_EXTENSION = ".dat"
EMPTY_URL_ERROR = "ERROR: URL should not be empty."
FILENAME_ERROR = "ERROR: Filename should not be empty."
UNKNOWN_FORMAT = "ERROR: Unknown file format. Can't extract."

# Structure of datasets
# key: Name of the dataset
# value: 0 - Name of the folder to put the dataset at the install path
#        1 - Size of the dataset (MB)
#        2 - List of the urls required to download the dataset

datasets = {
    "MNIST" : ["mnist", 110, 
        ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]],

    "CIFAR-10" : ["", 150, 
        ["http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"]],

    "CIFAR-100" : ["", 150, 
        ["http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"]],

    "Caltech-101" : ["", 139, 
        ["http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"]],

    "Caltech-256" : ["", 1300, 
        ["http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"]],

    "Food-101" : ["food101", 5000,
        ["http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"]],

    "GSTRB" : ["", 490, 
        ["https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"]],

    "GTSRB-FinalTest" : ["GTSRB", 0, 
        ["https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"]],

    "Speech" : ["speech", 100,
        ["http://www.festvox.org/examples/cstr_us_ked_timit/packed/cmu_us_ked_timit.tar.bz2"]],

    "IMDB" : ["IMDB-WIKI", 295000,
        ["https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_0.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_1.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_2.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_3.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_4.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_5.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_6.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_7.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_8.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_9.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"]],

    "WIKI" : ["IMDB-WIKI", 4000,
        ["https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"]],
        
    "DVS" : ["dvs", 110, 
        ["http://www.ini.uzh.ch/~tobi/dvs/events20051221T014416 freeway.mat.dat",
        "http://www.ini.uzh.ch/~tobi/dvs/events-2005-12-28T11-14-28-0800 drive SC postoffice.dat",
        "http://www.ini.uzh.ch/~tobi/dvs/Tmpdiff128-2006-06-17T10-48-06+0200-0 vogelsang saturday monring #2.dat",
        "http://www.ini.uzh.ch/~tobi/dvs/Tmpdiff128-2007-02-28T15-08-15-0800-0 3 flies 2m 1f.dat",
        "http://www.ini.uzh.ch/~tobi/dvs/events20051219T171855 driving pasa freeway.mat.dat"]]
}

###############################################################################

def display_dataset_info():
    """Display the available datasets to download"""
    datasetsName = list(datasets.keys())

    for i in range(len(datasets)):
        size = datasets[datasetsName[i]][1]
        unity = "MB"
        if size > 1000:
            size = size / 1000
            unity = "GB"

        print(f"{datasetsName[i]} - {size} {unity}")


def get_install_path():
    """Define the install path from 
    the N2D2_DATA environment variable
    """

    if os.name == 'nt':
        N2D2_DATA = os.getenv("N2D2_DATA", "C:\\n2d2_data")
    else:
        N2D2_DATA = os.getenv("N2D2_DATA", "/local/" + os.getenv("USER") + "/n2d2_data")

    installPath = input(f"Installation path of the dataset [default is {N2D2_DATA}]: ")

    if installPath == "":
        installPath = N2D2_DATA

    return installPath


def get_datasetname():
    """Display the available datasets to download
    and impose the user to select one of them
    """
    print("Please choose the dataset to install among:")
    display_dataset_info()

    isDatasetAvailable = False
    dataset_name = ""
    while not isDatasetAvailable:
        dataset_name = input("Type the dataset you want to install: ")

        if dataset_name in list(datasets.keys()):
            isDatasetAvailable = True
        else:
            print("Please choose an available dataset name")
    
    return dataset_name


def get_filename(url):
    """Extract filename from file url"""
    filename = os.path.basename(url)
    return filename


def get_file_location(target_path, filename):
    """Concatenate download directory and filename"""
    return target_path + filename


def parse_option():
    parser = argparse.ArgumentParser(description='N2D2 dataset installation script')
    parser.add_argument('--dataset', metavar='dataset', type=str, default='',
                        choices=list(datasets.keys()), help='Dataset to install')
    parser.add_argument('--installPath', metavar='path', type=str, 
                        default='', help='Path where to install the dataset')
    parser.add_argument("--no_keep_download", action='store_true',
                        help="Don't keep the original file after extraction")
    parser.add_argument("--overwrite_download", action='store_true', 
                        help="Download dataset even if it already exists")
    parser.add_argument("--info_datasets", action='store_true', 
                        help="Display information about the datasets")
    args = parser.parse_args()

    if args.info_datasets:
        display_dataset_info()
        sys.exit(0)

    if args.dataset == "":
        args.dataset = get_datasetname()

    if args.installPath == "":
        args.installPath = get_install_path()

    return args


def extract_file(target_path, filename):
    """Extract file based on file extension
    target_path: string, location where data will be extracted
    filename: string, name of the file along with extension
    """
    if filename == "" or filename is None:
        raise Exception(FILENAME_ERROR)

    file_location = get_file_location(target_path, filename)

    if filename.endswith(ZIP_EXTENSION):
        print("Extracting zip file.")
        zipf = zipfile.ZipFile(file_location, 'r')
        zipf.extractall(target_path)
        zipf.close()
    elif filename.endswith(TAR_EXTENSION) or \
         filename.endswith(TAR_GZ_EXTENSION) or \
         filename.endswith(TGZ_EXTENSION) or \
         filename.endswith(TAR_BZ2_EXTENSION):
        print("Extracting tar file.")
        tarf = tarfile.open(file_location, 'r')
        tarf.extractall(target_path)
        tarf.close()
    elif filename.endswith(GZ_EXTENSION):
        print("Extracting gz file.")
        raw = gzip.open(file_location, 'rb').read()
        open(os.path.splitext(file_location)[0], 'wb').write(raw)
    elif filename.endswith(DAT_EXTENSION):
        print("No extraction for dat files.")
    else:
        print(UNKNOWN_FORMAT)


def download_dataset(dataset, target_path, no_keep_download, overwrite_download):
    """Download and extract dataset
    dataset: string, name of the dataset to install
    target_path: string, location where dataset will be downloaded and extracted
    no_keep_download: boolean, option to delete downloaded files
    overwrite_download: boolean, option to download even if the file exists
    """

    # Add subdirectory to the target path
    target_subdir_path = target_path + "/" + datasets[dataset][0]
    if not datasets[dataset][0] == "":
        target_subdir_path = target_subdir_path + "/"

    os.makedirs(target_subdir_path, exist_ok=True)

    for url in datasets[dataset][2]:

        if url == "" or url is None:
            raise Exception(EMPTY_URL_ERROR)

        filename = get_filename(url)
        file_location = get_file_location(target_subdir_path, filename)

        if os.path.exists(file_location) and not overwrite_download:
            print(f"File already exists at {file_location}")
            print("Use: 'overwrite_download=True' to overwrite download")
        else:
            print(f"Downloading file from {url} to {file_location}.")
            # Download
            with open(file_location, 'wb') as f:
                with requests.get(url, allow_redirects=True, stream=True) as resp:
                    for chunk in resp.iter_content(chunk_size = 512):  #chunk_size in bytes
                        if chunk:
                            f.write(chunk)
            print("Finished downloading.")

        print("Extracting the file.")
        extract_file(target_subdir_path, filename)

        if no_keep_download:
            os.remove(file_location)

    print(f"{dataset} dataset installed!")


if __name__ == "__main__":

    opt = parse_option()
    download_dataset(opt.dataset, opt.installPath, 
                     opt.no_keep_download, opt.overwrite_download)    
