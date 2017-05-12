#!/usr/bin/python
################################################################################
#    (C) Copyright 2010 CEA LIST. All Rights Reserved.
#    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
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
################################################################################

import os
import sys
import urllib
import tarfile
import gzip, zipfile

if os.name == 'nt':
    N2D2_DATA = os.getenv("N2D2_DATA", "C:\\n2d2_data")
else:
    N2D2_DATA = os.getenv("N2D2_DATA", "/local/" + os.getenv("USER")
        + "/n2d2_data")

installPath = raw_input("Installation path of the stimuli [default is %s]: "
    % (N2D2_DATA))

if installPath == "":
    installPath = N2D2_DATA

data = {
    "http://kitti.is.tue.mpg.de/kitti/data_road.zip" : "KITTI",
    "http://kitti.is.tue.mpg.de/kitti/data_tracking_label_2.zip" : "KITTI",
    "http://kitti.is.tue.mpg.de/kitti/data_tracking_image_2.zip" : "KITTI",
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz": "mnist",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz": "mnist",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz": "mnist",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz": "mnist",
    "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz": "",
    "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz": "",
    "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar": "",
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip": "",
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip": "",
    "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip": "GTSRB",
    "http://www.ini.uzh.ch/~tobi/dvs/events20051221T014416 freeway.mat.dat": "dvs",
    "http://www.ini.uzh.ch/~tobi/dvs/events-2005-12-28T11-14-28-0800 drive SC postoffice.dat": "dvs",
    "http://www.ini.uzh.ch/~tobi/dvs/Tmpdiff128-2006-06-17T10-48-06+0200-0 vogelsang saturday monring #2.dat": "dvs",
    "http://www.ini.uzh.ch/~tobi/dvs/Tmpdiff128-2007-02-28T15-08-15-0800-0 3 flies 2m 1f.dat": "dvs",
    "http://www.ini.uzh.ch/~tobi/dvs/events20051219T171855 driving pasa freeway.mat.dat": "dvs",
    "http://www.festvox.org/examples/cstr_us_ked_timit/packed/cmu_us_ked_timit.tar.bz2": "speech"
}

################################################################################

def progress(chunksSoFar, chunkSize, totalSize):
    sizeSoFar = min(totalSize, chunksSoFar*chunkSize)
    print "Downloaded %d of %d bytes (%3.1f%%)\r" \
        % (sizeSoFar, totalSize, 100.0*float(sizeSoFar)/totalSize),
    if sizeSoFar == totalSize:
        sys.stdout.write("\n")
    sys.stdout.flush()

for url, dirName in data.iteritems():
    (baseUrl, fileName) = url.rsplit('/', 1)
    target = os.path.join(installPath, dirName)
    if not os.path.exists(target):
        os.makedirs(target)
    target = os.path.join(target, fileName)
    if not os.path.exists(target):
        print url + " -> " + target
        urllib.urlretrieve(baseUrl + "/"
            + urllib.quote(fileName), target, progress)

        if fileName.endswith(".tar.gz") or fileName.endswith(".tar.bz2") \
          or fileName.endswith(".tar"):
            raw = tarfile.open(target)
            for m in raw.getmembers():
                raw.extract(m, os.path.dirname(target))
            #os.unlink(target)
        elif fileName.endswith(".gz"):
            raw = gzip.open(target, 'rb').read()
            open(os.path.splitext(target)[0], 'wb').write(raw)
            #os.unlink(target)
        elif fileName.endswith(".zip"):
            raw = zipfile.ZipFile(target, 'r')
            raw.extractall(os.path.dirname(target))
            #os.unlink(target)

print "Done!"
