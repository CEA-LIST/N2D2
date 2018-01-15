#!/usr/bin/python
################################################################################
#    (C) Copyright 2010 CEA LIST. All Rights Reserved.
#    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
#                    David BRIAND (david.briand@cea.fr)
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

import gtk
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

data = {
    "KITTI road" : ["KITTI",5,"http://kitti.is.tue.mpg.de/kitti/data_road.zip"],

    "KITTI segmentation" : ["KITTI",15800, "http://kitti.is.tue.mpg.de/kitti/data_tracking_image_2.zip",\
        "http://kitti.is.tue.mpg.de/kitti/data_tracking_label_2.zip"],

    "KITTI object" : ["KITTI_OBJECT",12600, " http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip",\
        " http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip"],

    "MNIST" : ["mnist",110, "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",\
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",\
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",\
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"],

    "CIFAR-10" : ["", 150, "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"],

    "Caltech-101" : ["", 139, "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"],

    "Caltech-256" : ["", 1300, "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"],

    "GSTRB" : ["", 490, "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip",\
        "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"],

    "GTSRB-FinalTest" : ["GTSRB", 0, "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"],

    "Speech" : ["speech",100,"http://www.festvox.org/examples/cstr_us_ked_timit/packed/cmu_us_ked_timit.tar.bz2"],
    "IMDB" : ["IMDB-WIKI",295000,"https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_0.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_1.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_2.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_3.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_4.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_5.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_6.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_7.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_8.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_9.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"],

    "WIKI" : ["IMDB-WIKI",4000,"https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz",\
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"],

    "ILSVRC2012" : ["ILSVRC2012", 1400000, "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar",\
        "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train_t3.tar",\
        "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar",\
        "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar"],
        
    "DVS" : ["dvs",110, "http://www.ini.uzh.ch/~tobi/dvs/events20051221T014416 freeway.mat.dat",\
        "http://www.ini.uzh.ch/~tobi/dvs/events-2005-12-28T11-14-28-0800 drive SC postoffice.dat",\
        "http://www.ini.uzh.ch/~tobi/dvs/Tmpdiff128-2006-06-17T10-48-06+0200-0 vogelsang saturday monring #2.dat",\
        "http://www.ini.uzh.ch/~tobi/dvs/Tmpdiff128-2007-02-28T15-08-15-0800-0 3 flies 2m 1f.dat",\
        "http://www.ini.uzh.ch/~tobi/dvs/events20051219T171855 driving pasa freeway.mat.dat"]
}

dataToLoad=dict()
global dataSize
dataSize=0

def build_interface():
    main_layout = gtk.Table(6, 3, True)
    index = 0
    labelSize = gtk.Label()
    loadButton = gtk.Button(label='Load')

    for dbName, dbAttribute in data.iteritems():
        check_1 = gtk.CheckButton(dbName + '\n~' + str(dbAttribute[1]) + ' MBytes')
        check_1.connect('clicked', checkClick, dbAttribute[1],labelSize)
        loadButton.connect('clicked', chekAllButton, check_1, dbName)
        main_layout.attach(check_1, index/4, index/4+1, index%4, index%4+1)
        index += 1
    loadButton.connect('clicked', load_database)

    main_layout.attach(loadButton, 0, 1, index/2, index/2+1)
    main_layout.attach(labelSize, index/8, index/8+1, index/2, index/2+1)
    return main_layout

def chekAllButton(widget, button, name):
    if button.get_active():
        global dbToLoad
        dbToLoad = data.get(name, None)
        for k in range(2, len(dbToLoad)):
            dataToLoad[dbToLoad[k]] = dbToLoad[0]

def checkClick(button, size, label):
    global dataSize
    if button.get_active():
        dataSize += size
    else :
        dataSize -= size
    label.set_markup('<span color="#c0392b" weight="bold" font="FreeSerif">Space required:\n'\
        + str(dataSize) + ' MBytes </span>')

def load_database(widget):
    for url, dirName in dataToLoad.iteritems():
        print(url)
        print(dirName)
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


def progress(chunksSoFar, chunkSize, totalSize):
    sizeSoFar = min(totalSize, chunksSoFar*chunkSize)

    print "Downloaded %d of %d bytes (%3.1f%%)\r" \
        % (sizeSoFar, totalSize, 100.0*float(sizeSoFar)/totalSize),
    if sizeSoFar == totalSize:
        sys.stdout.write("\n")

    sys.stdout.flush()

if __name__ == '__main__':
    installPath = raw_input("Installation path of the stimuli [default is %s]: "
        % (N2D2_DATA))
    if installPath == "":
        installPath = N2D2_DATA

    window = gtk.Window()

    window.set_title('N2D2 Database Selection Menu')
    window.set_border_width(10)

    window.connect('delete-event', gtk.main_quit)

    main_layout = build_interface()
    window.add(main_layout)
    window.show_all()
    gtk.main()

