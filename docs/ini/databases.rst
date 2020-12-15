Databases
=========

The tool integrates pre-defined modules for several well-known database
used in the deep learning community, such as MNIST, GTSRB, CIFAR10 and
so on. That way, no extra step is necessary to be able to directly build
a network and learn it on these database.

MNIST
-----

MNIST :cite:`LeCun1998` is already fractionned into a
learning set and a testing set, with:

- 60,000 digits in the learning set;

- 10,000 digits in the testing set.

Example:

.. code-block:: ini

    [database]
    Type=MNIST_IDX_Database
    Validation=0.2  ; Fraction of learning stimuli used for the validation [default: 0.0]

+--------------------------+----------------------------------------------------+
| Option [default value]   | Description                                        |
+==========================+====================================================+
| ``Validation`` [0.0]     | Fraction of the learning set used for validation   |
+--------------------------+----------------------------------------------------+
| ``DataPath``             | Path to the database                               |
+--------------------------+----------------------------------------------------+
| [``$N2D2_DATA``/mnist]   |                                                    |
+--------------------------+----------------------------------------------------+

GTSRB
-----

GTSRB :cite:`Stallkamp2012` is already fractionned into a
learning set and a testing set, with:

- 39,209 digits in the learning set;

- 12,630 digits in the testing set.

Example:

.. code-block:: ini

    [database]
    Type=GTSRB_DIR_Database
    Validation=0.2  ; Fraction of learning stimuli used for the validation [default: 0.0]

+--------------------------+----------------------------------------------------+
| Option [default value]   | Description                                        |
+==========================+====================================================+
| ``Validation`` [0.0]     | Fraction of the learning set used for validation   |
+--------------------------+----------------------------------------------------+
| ``DataPath``             | Path to the database                               |
+--------------------------+----------------------------------------------------+
| [``$N2D2_DATA``/GTSRB]   |                                                    |
+--------------------------+----------------------------------------------------+

Directory
---------

Hand made database stored in files directories are directly supported
with the ``DIR_Database`` module. For example, suppose your database is
organized as following (in the path specified in the ``N2D2_DATA``
environment variable):

- ``GST/airplanes``: 800 images

- ``GST/car_side``: 123 images

- ``GST/Faces``: 435 images

- ``GST/Motorbikes``: 798 images

You can then instanciate this database as input of your neural network
using the following parameters:

.. code-block:: ini

    [database]
    Type=DIR_Database
    DataPath=\${N2D2_DATA}/GST
    Learn=0.4 ; 40\% of images of the smallest category = 49 (0.4x123) images for each category will be used for learning
    Validation=0.2 ; 20\% of images of the smallest category = 25 (0.2x123) images for each category will be used for validation
    ; the remaining images will be used for testing

Each subdirectory will be treated as a different label, so there will be
4 different labels, named after the directory name.

The stimuli are equi-partitioned for the learning set and the validation
set, meaning that the same number of stimuli for each category is used.
If the learn fraction is 0.4 and the validation fraction is 0.2, as in
the example above, the partitioning will be the following:

+-------------+------------------+-------------+------------------+------------+
| Label ID    | Label name       | Learn set   | Validation set   | Test set   |
+-------------+------------------+-------------+------------------+------------+
| [0.5ex] 0   | ``airplanes``    | 49          | 25               | 726        |
+-------------+------------------+-------------+------------------+------------+
| 1           | ``car_side``     | 49          | 25               | 49         |
+-------------+------------------+-------------+------------------+------------+
| 2           | ``Faces``        | 49          | 25               | 361        |
+-------------+------------------+-------------+------------------+------------+
| 3           | ``Motorbikes``   | 49          | 25               | 724        |
+-------------+------------------+-------------+------------------+------------+
|             | Total:           | 196         | 100              | 1860       |
+-------------+------------------+-------------+------------------+------------+

*Mandatory option*

+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]                    | Description                                                                                                                                                            |
+===========================================+========================================================================================================================================================================+
| ``DataPath``                              | Path to the root stimuli directory                                                                                                                                     |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Learn``                                 | If ``PerLabelPartitioning`` is true, fraction of images used for the learning; else, number of images used for the learning, regardless of their labels                |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``LoadInMemory`` [0]                      | Load the whole database into memory                                                                                                                                    |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Depth`` [1]                             | Number of sub-directory levels to include. Examples:                                                                                                                   |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                           | ``Depth`` = 0: load stimuli only from the current directory (``DataPath``)                                                                                             |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                           | ``Depth`` = 1: load stimuli from ``DataPath`` and stimuli contained in the sub-directories of ``DataPath``                                                             |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                           | ``Depth`` < 0: load stimuli recursively from ``DataPath`` and all its sub-directories                                                                                  |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``LabelName`` []                          | Base stimuli label name                                                                                                                                                |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``LabelDepth`` [1]                        | Number of sub-directory name levels used to form the stimuli labels. Examples:                                                                                         |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                           | ``LabelDepth`` = -1: no label for all stimuli (label ID = -1)                                                                                                          |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                           | ``LabelDepth`` = 0: uses ``LabelName`` for all stimuli                                                                                                                 |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                           | ``LabelDepth`` = 1: uses ``LabelName`` for stimuli in the current directory (``DataPath``) and ``LabelName``/*sub-directory name* for stimuli in the sub-directories   |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``PerLabelPartitioning`` [1]              | If true, the stimuli are equi-partitioned for the learn/validation/test sets, meaning that the same number of stimuli for each label is used                           |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Validation`` [0.0]                      | If ``PerLabelPartitioning`` is true, fraction of images used for the validation; else, number of images used for the validation, regardless of their labels            |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Test`` [1.0-``Learn``-``Validation``]   | If ``PerLabelPartitioning`` is true, fraction of images used for the test; else, number of images used for the test, regardless of their labels                        |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ValidExtensions`` []                    | List of space-separated valid stimulus file extensions (if left empty, any file extension is considered a valid stimulus)                                              |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``LoadMore`` []                           | Name of an other section with the same options to load a different ``DataPath``                                                                                        |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ROIFile`` []                            | File containing the stimuli ROIs. If a ROI file is specified, ``LabelDepth`` should be set to -1                                                                       |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``DefaultLabel`` []                       | Label name for pixels outside any ROI (default is no label, pixels are ignored)                                                                                        |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ROIsMargin`` [0]                        | Number of pixels around ROIs that are ignored (and not considered as ``DefaultLabel`` pixels)                                                                          |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

To load and partition more than one ``DataPath``, one can use the
``LoadMore`` option:

.. code-block:: ini

    [database]
    Type=DIR_Database
    DataPath=\${N2D2_DATA}/GST
    Learn=0.6
    Validation=0.4
    LoadMore=database.test

    ; Load stimuli from the "GST_Test" path in the test dataset
    [database.test]
    DataPath=\${N2D2_DATA}/GST_Test
    Learn=0.0
    Test=1.0
    ; The LoadMore option is recursive:
    ; LoadMore=database.more

    ; [database.more]
    ; Load even more data here

*Speech Commands Dataset*
~~~~~~~~~~~~~~~~~~~~~~~~~

Use with Speech Commands Data Set, released by the Google
:cite:`speechcommandsv2`.

.. code-block:: ini

    [database]
    Type=DIR_Database
    DataPath=\${N2D2_DATA}/speech_commands_v0.02
    ValidExtensions=wav
    IgnoreMasks=*/_background_noise_
    Learn=0.6
    Validation=0.2

Other built-in databases
------------------------

Actitracker\_Database
~~~~~~~~~~~~~~~~~~~~~

Actitracker database, released by the WISDM Lab
:cite:`Lockhart2011`.

+------------------------------------+---------------------------------------------------+
| Option [default value]             | Description                                       |
+====================================+===================================================+
| ``Learn`` [0.6]                    | Fraction of data used for the learning            |
+------------------------------------+---------------------------------------------------+
| ``Validation`` [0.2]               | Fraction of data used for the validation          |
+------------------------------------+---------------------------------------------------+
| ``UseUnlabeledForTest`` [0]        | If true, use the unlabeled dataset for the test   |
+------------------------------------+---------------------------------------------------+
| ``DataPath``                       | Path to the database                              |
+------------------------------------+---------------------------------------------------+
| [``$N2D2_DATA``/WISDM\_at\_v2.0]   |                                                   |
+------------------------------------+---------------------------------------------------+

CIFAR10\_Database
~~~~~~~~~~~~~~~~~

CIFAR10 database :cite:`Krizhevsky2009`.

+-----------------------------------------+----------------------------------------------------+
| Option [default value]                  | Description                                        |
+=========================================+====================================================+
| ``Validation`` [0.0]                    | Fraction of the learning set used for validation   |
+-----------------------------------------+----------------------------------------------------+
| ``DataPath``                            | Path to the database                               |
+-----------------------------------------+----------------------------------------------------+
| [``$N2D2_DATA``/cifar-10-batches-bin]   |                                                    |
+-----------------------------------------+----------------------------------------------------+

CIFAR100\_Database
~~~~~~~~~~~~~~~~~~

CIFAR100 database :cite:`Krizhevsky2009`.

+-------------------------------------+---------------------------------------------------------------+
| Option [default value]              | Description                                                   |
+=====================================+===============================================================+
| ``Validation`` [0.0]                | Fraction of the learning set used for validation              |
+-------------------------------------+---------------------------------------------------------------+
| ``UseCoarse`` [0]                   | If true, use the coarse labeling (10 labels instead of 100)   |
+-------------------------------------+---------------------------------------------------------------+
| ``DataPath``                        | Path to the database                                          |
+-------------------------------------+---------------------------------------------------------------+
| [``$N2D2_DATA``/cifar-100-binary]   |                                                               |
+-------------------------------------+---------------------------------------------------------------+

CKP\_Database
~~~~~~~~~~~~~

The Extended Cohn-Kanade (CK+) database for expression recognition
:cite:`Lucey2010`.

+---------------------------------------+----------------------------------------------+
| Option [default value]                | Description                                  |
+=======================================+==============================================+
| ``Learn``                             | Fraction of images used for the learning     |
+---------------------------------------+----------------------------------------------+
| ``Validation`` [0.0]                  | Fraction of images used for the validation   |
+---------------------------------------+----------------------------------------------+
| ``DataPath``                          | Path to the database                         |
+---------------------------------------+----------------------------------------------+
| [``$N2D2_DATA``/cohn-kanade-images]   |                                              |
+---------------------------------------+----------------------------------------------+

Caltech101\_DIR\_Database
~~~~~~~~~~~~~~~~~~~~~~~~~

Caltech 101 database :cite:`FeiFei2004`.

+--------------------------+----------------------------------------------------------------------+
| Option [default value]   | Description                                                          |
+==========================+======================================================================+
| ``Learn``                | Fraction of images used for the learning                             |
+--------------------------+----------------------------------------------------------------------+
| ``Validation`` [0.0]     | Fraction of images used for the validation                           |
+--------------------------+----------------------------------------------------------------------+
| ``IncClutter`` [0]       | If true, includes the BACKGROUND\_Google directory of the database   |
+--------------------------+----------------------------------------------------------------------+
| ``DataPath``             | Path to the database                                                 |
+--------------------------+----------------------------------------------------------------------+
| [``$N2D2_DATA``/         |                                                                      |
+--------------------------+----------------------------------------------------------------------+
| 101\_ObjectCategories]   |                                                                      |
+--------------------------+----------------------------------------------------------------------+

Caltech256\_DIR\_Database
~~~~~~~~~~~~~~~~~~~~~~~~~

Caltech 256 database :cite:`Griffin2007`.

+--------------------------+----------------------------------------------------------------------+
| Option [default value]   | Description                                                          |
+==========================+======================================================================+
| ``Learn``                | Fraction of images used for the learning                             |
+--------------------------+----------------------------------------------------------------------+
| ``Validation`` [0.0]     | Fraction of images used for the validation                           |
+--------------------------+----------------------------------------------------------------------+
| ``IncClutter`` [0]       | If true, includes the BACKGROUND\_Google directory of the database   |
+--------------------------+----------------------------------------------------------------------+
| ``DataPath``             | Path to the database                                                 |
+--------------------------+----------------------------------------------------------------------+
| [``$N2D2_DATA``/         |                                                                      |
+--------------------------+----------------------------------------------------------------------+
| 256\_ObjectCategories]   |                                                                      |
+--------------------------+----------------------------------------------------------------------+

CaltechPedestrian\_Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Caltech Pedestrian database :cite:`Dollar2009`.

Note that the images and annotations must first be extracted from the
seq video data located in the *videos* directory using the
``dbExtract.m`` Matlab tool provided in the “Matlab evaluation/labeling
code” downloadable on the dataset website.

Assuming the following directory structure (in the path specified in the
``N2D2_DATA`` environment variable):

- ``CaltechPedestrians/data-USA/videos/...`` (from the *setxx.tar* files)

- ``CaltechPedestrians/data-USA/annotations/...`` (from the *setxx.tar*
  files)

- ``CaltechPedestrians/tools/piotr_toolbox/toolbox`` (from the Piotr’s
  Matlab Toolbox archive)

- ``CaltechPedestrians/*.m`` including ``dbExtract.m`` (from the Matlab
  evaluation/labeling code)

Use the following command in Matlab to generate the images and
annotations:

.. code-block:: matlab

    cd([getenv('N2D2_DATA') '/CaltechPedestrians'])
    addpath(genpath('tools/piotr_toolbox/toolbox')) % add the Piotr's Matlab Toolbox in the Matlab path
    dbInfo('USA')
    dbExtract()

+--------------------------------------------+-------------------------------------------------------------------------------------+
| Option [default value]                     | Description                                                                         |
+============================================+=====================================================================================+
| ``Validation`` [0.0]                       | Fraction of the learning set used for validation                                    |
+--------------------------------------------+-------------------------------------------------------------------------------------+
| ``SingleLabel`` [1]                        | Use the same label for “person” and “people” bounding box                           |
+--------------------------------------------+-------------------------------------------------------------------------------------+
| ``IncAmbiguous`` [0]                       | Include ambiguous bounding box labeled “person?” using the same label as “person”   |
+--------------------------------------------+-------------------------------------------------------------------------------------+
| ``DataPath``                               | Path to the database images                                                         |
+--------------------------------------------+-------------------------------------------------------------------------------------+
| [``$N2D2_DATA``/                           |                                                                                     |
+--------------------------------------------+-------------------------------------------------------------------------------------+
| CaltechPedestrians/data-USA/images]        |                                                                                     |
+--------------------------------------------+-------------------------------------------------------------------------------------+
| ``LabelPath``                              | Path to the database annotations                                                    |
+--------------------------------------------+-------------------------------------------------------------------------------------+
| [``$N2D2_DATA``/                           |                                                                                     |
+--------------------------------------------+-------------------------------------------------------------------------------------+
| CaltechPedestrians/data-USA/annotations]   |                                                                                     |
+--------------------------------------------+-------------------------------------------------------------------------------------+

Cityscapes\_Database
~~~~~~~~~~~~~~~~~~~~

Cityscapes database :cite:`Cordts2016Cityscapes`.

+----------------------------------------+----------------------------------------------------------------------------------------------------------+
| Option [default value]                 | Description                                                                                              |
+========================================+==========================================================================================================+
| ``IncTrainExtra`` [0]                  | If true, includes the left 8-bit images - trainextra set (19,998 images)                                 |
+----------------------------------------+----------------------------------------------------------------------------------------------------------+
| ``UseCoarse`` [0]                      | If true, only use coarse annotations (which are the only annotations available for the trainextra set)   |
+----------------------------------------+----------------------------------------------------------------------------------------------------------+
| ``SingleInstanceLabels`` [1]           | If true, convert group labels to single instance labels (for example, ``cargroup`` becomes ``car``)      |
+----------------------------------------+----------------------------------------------------------------------------------------------------------+
| ``DataPath``                           | Path to the database images                                                                              |
+----------------------------------------+----------------------------------------------------------------------------------------------------------+
| [``$N2D2_DATA``/                       |                                                                                                          |
+----------------------------------------+----------------------------------------------------------------------------------------------------------+
| Cityscapes/leftImg8bit] or             |                                                                                                          |
+----------------------------------------+----------------------------------------------------------------------------------------------------------+
| [``$CITYSCAPES_DATASET``] if defined   |                                                                                                          |
+----------------------------------------+----------------------------------------------------------------------------------------------------------+
| ``LabelPath`` []                       | Path to the database annotations (deduced from ``DataPath`` if left empty)                               |
+----------------------------------------+----------------------------------------------------------------------------------------------------------+

Daimler\_Database
~~~~~~~~~~~~~~~~~

Daimler Monocular Pedestrian Detection Benchmark (Daimler Pedestrian).

+--------------------------+------------------------------------------------------------------------------+
| Option [default value]   | Description                                                                  |
+==========================+==============================================================================+
| ``Learn`` [1.0]          | Fraction of images used for the learning                                     |
+--------------------------+------------------------------------------------------------------------------+
| ``Validation`` [0.0]     | Fraction of images used for the validation                                   |
+--------------------------+------------------------------------------------------------------------------+
| ``Test`` [0.0]           | Fraction of images used for the test                                         |
+--------------------------+------------------------------------------------------------------------------+
| ``Fully`` [0]            | When activate it use the test dataset to learn. Use only on fully-cnn mode   |
+--------------------------+------------------------------------------------------------------------------+

DOTA\_Database
~~~~~~~~~~~~~~

DOTA database :cite:`DOTA`.

+--------------------------+--------------------------------------------+
| Option [default value]   | Description                                |
+==========================+============================================+
| ``Learn``                | Fraction of images used for the learning   |
+--------------------------+--------------------------------------------+
| ``DataPath``             | Path to the database                       |
+--------------------------+--------------------------------------------+
| [``$N2D2_DATA``/DOTA]    |                                            |
+--------------------------+--------------------------------------------+
| ``LabelPath``            | Path to the database labels list file      |
+--------------------------+--------------------------------------------+
| []                       |                                            |
+--------------------------+--------------------------------------------+

FDDB\_Database
~~~~~~~~~~~~~~

Face Detection Data Set and Benchmark (FDDB)
:cite:`Jain2010`.

+--------------------------+---------------------------------------------------------+
| Option [default value]   | Description                                             |
+==========================+=========================================================+
| ``Learn``                | Fraction of images used for the learning                |
+--------------------------+---------------------------------------------------------+
| ``Validation`` [0.0]     | Fraction of images used for the validation              |
+--------------------------+---------------------------------------------------------+
| ``DataPath``             | Path to the images (decompressed originalPics.tar.gz)   |
+--------------------------+---------------------------------------------------------+
| [``$N2D2_DATA``/FDDB]    |                                                         |
+--------------------------+---------------------------------------------------------+
| ``LabelPath``            | Path to the annotations (decompressed FDDB-folds.tgz)   |
+--------------------------+---------------------------------------------------------+
| [``$N2D2_DATA``/FDDB]    |                                                         |
+--------------------------+---------------------------------------------------------+

GTSDB\_DIR\_Database
~~~~~~~~~~~~~~~~~~~~

GTSDB database :cite:`Houben2013`.

+----------------------------------+----------------------------------------------+
| Option [default value]           | Description                                  |
+==================================+==============================================+
| ``Learn``                        | Fraction of images used for the learning     |
+----------------------------------+----------------------------------------------+
| ``Validation`` [0.0]             | Fraction of images used for the validation   |
+----------------------------------+----------------------------------------------+
| ``DataPath``                     | Path to the database                         |
+----------------------------------+----------------------------------------------+
| [``$N2D2_DATA``/FullIJCNN2013]   |                                              |
+----------------------------------+----------------------------------------------+

ILSVRC2012\_Database
~~~~~~~~~~~~~~~~~~~~

ILSVRC2012 database :cite:`ILSVRC15`.

+-------------------------------------------+--------------------------------------------+
| Option [default value]                    | Description                                |
+===========================================+============================================+
| ``Learn``                                 | Fraction of images used for the learning   |
+-------------------------------------------+--------------------------------------------+
| ``DataPath``                              | Path to the database                       |
+-------------------------------------------+--------------------------------------------+
| [``$N2D2_DATA``/ILSVRC2012]               |                                            |
+-------------------------------------------+--------------------------------------------+
| ``LabelPath``                             | Path to the database labels list file      |
+-------------------------------------------+--------------------------------------------+
| [``$N2D2_DATA``/ILSVRC2012/synsets.txt]   |                                            |
+-------------------------------------------+--------------------------------------------+

KITTI\_Database
~~~~~~~~~~~~~~~

The KITTI Database provide ROI which can be use for autonomous driving
and environment perception. The database provide 8 labeled different
classes. Utilization of the KITTI Database is under licensing conditions
and request an email registration. To install it you have to follow this
link: http://www.cvlibs.net/datasets/kitti/eval_tracking.php and
download the left color images (15 GB) and the trainling labels of
tracking data set (9 MB). Extract the downloaded archives in your
``$N2D2_DATA/KITTI`` folder.

+--------------------------+----------------------------------------------+
| Option [default value]   | Description                                  |
+==========================+==============================================+
| ``Learn`` [0.8]          | Fraction of images used for the learning     |
+--------------------------+----------------------------------------------+
| ``Validation`` [0.2]     | Fraction of images used for the validation   |
+--------------------------+----------------------------------------------+

KITTI\_Road\_Database
~~~~~~~~~~~~~~~~~~~~~

The KITTI Road Database provide ROI which can be used to road
segmentation. The dataset provide 1 labeled class (road) on 289 training
images. The 290 test images are not labeled. Utilization of the KITTI
Road Database is under licensing conditions and request an email
registration. To install it you have to follow this link:
http://www.cvlibs.net/datasets/kitti/eval_road.php and download the
“base kit” of (0.5 GB) with left color images, calibration and training
labels. Extract the downloaded archive in your ``$N2D2_DATA/KITTI``
folder.

+--------------------------+----------------------------------------------+
| Option [default value]   | Description                                  |
+==========================+==============================================+
| ``Learn`` [0.8]          | Fraction of images used for the learning     |
+--------------------------+----------------------------------------------+
| ``Validation`` [0.2]     | Fraction of images used for the validation   |
+--------------------------+----------------------------------------------+

KITTI\_Object\_Database
~~~~~~~~~~~~~~~~~~~~~~~

The KITTI Object Database provide ROI which can be use for autonomous
driving and environment perception. The database provide 8 labeled
different classes on 7481 training images. The 7518 test images are not
labeled. The whole database provide 80256 labeled objects. Utilization
of the KITTI Object Database is under licensing conditions and request
an email registration. To install it you have to follow this link:
http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark and
download the “lef color images” (12 GB) and the training labels of
object data set (5 MB). Extract the downloaded archives in your
``$N2D2_DATA/KITTI_Object`` folder.

+--------------------------+----------------------------------------------+
| Option [default value]   | Description                                  |
+==========================+==============================================+
| ``Learn`` [0.8]          | Fraction of images used for the learning     |
+--------------------------+----------------------------------------------+
| ``Validation`` [0.2]     | Fraction of images used for the validation   |
+--------------------------+----------------------------------------------+

LITISRouen\_Database
~~~~~~~~~~~~~~~~~~~~

LITIS Rouen audio scene dataset :cite:`Rakotomamonjy2014`.

+--------------------------------+----------------------------------------------+
| Option [default value]         | Description                                  |
+================================+==============================================+
| ``Learn`` [0.4]                | Fraction of images used for the learning     |
+--------------------------------+----------------------------------------------+
| ``Validation`` [0.4]           | Fraction of images used for the validation   |
+--------------------------------+----------------------------------------------+
| ``DataPath``                   | Path to the database                         |
+--------------------------------+----------------------------------------------+
| [``$N2D2_DATA``/data\_rouen]   |                                              |
+--------------------------------+----------------------------------------------+

Dataset images slicing
~~~~~~~~~~~~~~~~~~~~~~

It is possible to automatically slice images from a dataset, with a
given slice size and stride, using the ``.slicing`` attribute. This
effectively increases the number of stimuli in the set.

.. code-block:: ini

    [database.slicing]
    ApplyTo=NoLearn
    Width=2048
    Height=1024
    StrideX=2048
    StrideY=1024
    RandomShuffle=1  ; 1 is the default value

The ``RandomShuffle`` option, enabled by default, randomly shuffle the
dataset after slicing. If disabled, the slices are added in order at the
end of the dataset.

