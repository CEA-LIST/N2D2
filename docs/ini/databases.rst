Databases
=========

Introduction
------------

A ``Database`` handles the raw data, annotations and how the datasets 
(learn, validation or test) should be build.
N2D2 integrates pre-defined modules for several well-known database
used in the deep learning community, such as MNIST, GTSRB, CIFAR10 and
so on. That way, no extra step is necessary to be able to directly build
a network and learn it on these database.

All the database modules inherit from a base ``Database``, which contains some
generic configuration options:

+--------------------------+------------------------------------------------------------------+
| Option [default value]   | Description                                                      |
+==========================+==================================================================+
| ``DefaultLabel`` []      | Default label for composite image (for areas outside the ROIs).  |
|                          | If empty, no default label is created and default label ID is -1 |
+--------------------------+------------------------------------------------------------------+
| ``ROIsMargin`` [0]       | Margin around the ROIs, in pixels, with no label (label ID = -1) |
+--------------------------+------------------------------------------------------------------+
| ``RandomPartitioning``   | If true (1), the partitioning in the learn, validation and test  |
| [1]                      | sets is random, otherwise partitioning is in the order           |
+--------------------------+------------------------------------------------------------------+
| ``DataFileLabel`` [1]    | If true (1), load pixel-wise image labels, if they exist         |
+--------------------------+------------------------------------------------------------------+
| ``CompositeLabel``       | See the following ``CompositeLabel`` section                     |
| [``Auto``]               |                                                                  |
+--------------------------+------------------------------------------------------------------+
| ``TargetDataPath`` []    | Data path to target data, to be used in conjunction with the     |
|                          | ``DataAsTarget`` option in ``Target`` modules                    |
+--------------------------+------------------------------------------------------------------+
| ``MultiChannelMatch`` [] | See the following *multi-channel handling* section               |
+--------------------------+------------------------------------------------------------------+
| ``MultiChannelReplace``  | See the following *multi-channel handling* section               |
+--------------------------+------------------------------------------------------------------+


``CompositeLabel`` parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A label is said to be composite when it is not a single *labelID* for the 
stimulus (the stimulus label is a matrix of size > 1).
For the same stimulus, different type of labels can be specified,
i.e. the *labelID*, pixel-wise data and/or ROIs.
The way these different label types are handled is configured with the
``CompositeLabel`` parameter:

- ``None``: only the *labelID* is used, pixel-wise data are ignored and ROIs 
  are loaded but ignored as well by ``loadStimulusLabelsData()``.
- ``Auto``: the label is only composite when pixel-wise data are present
  or the stimulus *labelID* is -1 (in which case the ``DefaultLabel``
  is used for the whole label matrix). If the label is composite
  ROIs, if present, are applied. Otherwise, a single ROI is
  allowed and is automatically extracted when fetching the stimulus.
- ``Default``: the label is always composite. The *labelID* is ignored.
  If there is no pixel-wise data, the ``DefaultLabel`` is used.
  ROIs, if present, are applied.
- ``Disjoint``: the label is always composite. If there is no pixel-wise data:
  
  - the *labelID* is used if there is no ROI;
  - the ``DefaultLabel`` is used if there is any ROI.

  ROIs, if present, are applied.
- ``Combine``: the label is always composite.
  If there is no pixel-wise data, the *labelID* is used.
  ROIs, if present, are applied.

    
Multi-channel handling
~~~~~~~~~~~~~~~~~~~~~~

Multi-channel images are automatically handled and the default image format in 
N2D2 is **BGR**.

Any ``Database`` can also handle multi-channel data, where each channel is stored
in a different file. In order to be able to interpret a series of files as an 
additional data channel to a first series of files, the file names must follow
a simple yet arbitrary naming convention. A first parameter,
``MultiChannelMatch``, is used to match the files constituting a single
channel. Then, a second parameter, ``MultiChannelReplace`` is used to indicate
how the file names of the other channels are obtained. See the example below,
with the ``DIR_Database``:

.. code-block:: ini

    [database]
    Type=DIR_Database
    ...
    ; Multi-channel handling:
    ; MultiChannelMatch is a regular expression for matching a single channel (for example the first one).
    ; Here we match anything followed by "_0", followed by "." and anything except 
    ; ".", so we match "_0" before the file extension.
    MultiChannelMatch=(.*)_0(\.[^.]+)
    ; Replace what we matched to obtain the file name of the different channels.
    ; For the first channel, replace "_0" by "_0", so the name doesn't change.
    ; For the second channel, replace "_0" by "_1" in the file name.
    ; To disable the second channel, replace $1_1$2 by ""
    MultiChannelReplace=$1_0$2 $1_1$2

Note that when ``MultiChannelMatch`` is not empty, only files matching this parameter
regexp pattern (and the associated channels obtained with ``MultiChannelReplace``, 
when they exist) will be loaded. Other files in the dataset not matching the 
``MultiChannelMatch`` filter will be ignored.

Stimuli are loaded even if some channels are missing (in which case, "Notice" 
messages are issued for the missing channel(s) during database loading). Missing
channel values are set to 0.

Annotations are common to all channels. If annotations exist for a specific channel,
they are fused with the annotations of the other channels (for geometric annotations).
Pixel-wise annotations, obtained when ``DataFileLabel`` is 1 (true), through 
the ``Database::readLabel()`` virtual method, are only read for the match 
(``MultiChannelMatch``) channel.



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
    DataPath=${N2D2_DATA}/GST
    Learn=0.4 ; 40% of images of the smallest category = 49 (0.4x123) images for each category will be used for learning
    Validation=0.2 ; 20% of images of the smallest category = 25 (0.2x123) images for each category will be used for validation
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
| ``IgnoreMasks``                           | Space-separated list of mask strings to ignore. If any is present in a file path, the file gets ignored. The usual \* and \+ wildcards are allowed.                    |
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
| ``PerLabelPartitioning`` [1]              | If true (1), the ``Learn``, ``Validation`` and  ``Test`` parameters represent the fraction of the total stimuli to be partitioned in each set,                         |
|                                           | instead of a number of stimuli                                                                                                                                         |
+-------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``EquivLabelPartitioning`` [1]            | If true (1), the stimuli are equi-partitioned in the learn and validation sets, meaning that the same number of stimuli **for each label** is used                     |
|                                           | (only when ``PerLabelPartitioning`` is 1). The remaining stimuli are partitioned in the test set                                                                       |
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


.. Note::

    If ``EquivLabelPartitioning`` is 1 (default setting), the number of stimuli
    per label that will be partitioned in the learn and validation sets will 
    correspond to the number of stimuli from the label with the fewest stimuli.


To load and partition more than one ``DataPath``, one can use the
``LoadMore`` option:

.. code-block:: ini

    [database]
    Type=DIR_Database
    DataPath=${N2D2_DATA}/GST
    Learn=0.6
    Validation=0.4
    LoadMore=database.test

    ; Load stimuli from the "GST_Test" path in the test dataset
    [database.test]
    DataPath=${N2D2_DATA}/GST_Test
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
    DataPath=${N2D2_DATA}/speech_commands_v0.02
    ValidExtensions=wav
    IgnoreMasks=*/_background_noise_
    Learn=0.6
    Validation=0.2


CSV data files
--------------

``CSV_Database`` is a generic driver for handling CSV data files. It can be used
to load one or several CSV files where each line is a different stimulus and one
column contains the label.

The parameters are the following:

+------------------------------------+---------------------------------------------------+
| Option [default value]             | Description                                       |
+====================================+===================================================+
| ``DataPath``                       | Path to the database                              |
+------------------------------------+---------------------------------------------------+
| ``Learn`` [0.6]                    | Fraction of data used for the learning            |
+------------------------------------+---------------------------------------------------+
| ``Validation`` [0.2]               | Fraction of data used for the validation          |
+------------------------------------+---------------------------------------------------+
| ``PerLabelPartitioning`` [1]       | If true (1), the ``Learn``, ``Validation`` and    |
|                                    | ``Test`` parameters represent the fraction of the |
|                                    | total stimuli to be partitioned in each set,      |
|                                    | instead of a number of stimuli                    |
+------------------------------------+---------------------------------------------------+
| ``EquivLabelPartitioning`` [1]     | If true (1), the stimuli are equi-partitioned in  |
|                                    | the learn and validation sets, meaning that the   |
|                                    | same number of stimuli **for each label** is used |
|                                    | (only when ``PerLabelPartitioning`` is 1).        |
|                                    | The remaining stimuli are partitioned in the test |
|                                    | set                                               |
+------------------------------------+---------------------------------------------------+
| ``LabelColumn`` [-1]               | Index of the column containing the label (if < 0, |
|                                    | from the end of the row)                          |
+------------------------------------+---------------------------------------------------+
| ``NbHeaderLines`` [0]              | Number of header lines to skip                    |
+------------------------------------+---------------------------------------------------+
| ``Test`` [1.0-``Learn``-           | If ``PerLabelPartitioning`` is true, fraction of  |
| ``Validation``]                    | images used for the test; else, number of images  |
|                                    | used for the test, regardless of their labels     |
+------------------------------------+---------------------------------------------------+
| ``LoadMore`` []                    | Name of an other section with the same options to |
|                                    | load a different ``DataPath``                     |
+------------------------------------+---------------------------------------------------+


.. Note::

    If ``EquivLabelPartitioning`` is 1 (default setting), the number of stimuli
    per label that will be partitioned in the learn and validation sets will 
    correspond to the number of stimuli from the label with the fewest stimuli.



Usage example
~~~~~~~~~~~~~

In this example, we load the *Electrical Grid Stability Simulated Data Data Set*
(https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+).

The CSV data file (``Data_for_UCI_named.csv``) is the following:

::

    "tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4","stab","stabf"
    2.95906002455997,3.07988520422811,8.38102539191882,9.78075443222607,3.76308477206316,-0.782603630987543,-1.25739482958732,-1.7230863114883,0.650456460887227,0.859578105752345,0.887444920638513,0.958033987602737,0.0553474891727752,"unstable"
    9.3040972346785,4.90252411201167,3.04754072762177,1.36935735529605,5.06781210427845,-1.94005842705193,-1.87274168559721,-1.25501199162931,0.41344056837935,0.862414076352903,0.562139050527675,0.781759910653126,-0.00595746432603695,"stable"
    8.97170690932022,8.84842842134833,3.04647874898866,1.21451813833956,3.40515818001095,-1.20745559234302,-1.27721014673295,-0.92049244093498,0.163041039311334,0.766688656526962,0.839444015400588,0.109853244952427,0.00347087904838871,"unstable"
    0.716414776295121,7.66959964406565,4.48664083058949,2.34056298396795,3.96379106326633,-1.02747330413905,-1.9389441526466,-0.997373606480681,0.446208906537321,0.976744082924302,0.929380522872661,0.36271777426931,0.028870543444887,"unstable"
    3.13411155161342,7.60877161603408,4.94375930178099,9.85757326996638,3.52581081652096,-1.12553095451115,-1.84597485447561,-0.554305007534195,0.797109525792467,0.455449947148291,0.656946658473716,0.820923486481631,0.0498603734837059,"unstable"
    ...

There is one header line and the last column is the label, which is the default.

This file is loaded and the data is splitted between the learning set and the 
validation set with a 0.7/0.3 ratio in the INI file with the following section:

.. code-block:: ini

    [database]
    Type=CSV_Database
    Learn=0.7
    Validation=0.3
    DataPath=Data_for_UCI_named.csv
    NbHeaderLines=1



Other built-in databases
------------------------

Actitracker_Database
~~~~~~~~~~~~~~~~~~~~

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
| [``$N2D2_DATA``/WISDM_at_v2.0]     |                                                   |
+------------------------------------+---------------------------------------------------+

CIFAR10_Database
~~~~~~~~~~~~~~~~

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

CIFAR100_Database
~~~~~~~~~~~~~~~~~

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

CKP_Database
~~~~~~~~~~~~

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

Caltech101_DIR_Database
~~~~~~~~~~~~~~~~~~~~~~~

Caltech 101 database :cite:`FeiFei2004`.

+--------------------------+----------------------------------------------------------------------+
| Option [default value]   | Description                                                          |
+==========================+======================================================================+
| ``Learn``                | Fraction of images used for the learning                             |
+--------------------------+----------------------------------------------------------------------+
| ``Validation`` [0.0]     | Fraction of images used for the validation                           |
+--------------------------+----------------------------------------------------------------------+
| ``IncClutter`` [0]       | If true, includes the BACKGROUND_Google directory of the database    |
+--------------------------+----------------------------------------------------------------------+
| ``DataPath``             | Path to the database                                                 |
+--------------------------+----------------------------------------------------------------------+
| [``$N2D2_DATA``/         |                                                                      |
+--------------------------+----------------------------------------------------------------------+
| 101_ObjectCategories]    |                                                                      |
+--------------------------+----------------------------------------------------------------------+

Caltech256_DIR_Database
~~~~~~~~~~~~~~~~~~~~~~~

Caltech 256 database :cite:`Griffin2007`.

+--------------------------+----------------------------------------------------------------------+
| Option [default value]   | Description                                                          |
+==========================+======================================================================+
| ``Learn``                | Fraction of images used for the learning                             |
+--------------------------+----------------------------------------------------------------------+
| ``Validation`` [0.0]     | Fraction of images used for the validation                           |
+--------------------------+----------------------------------------------------------------------+
| ``IncClutter`` [0]       | If true, includes the BACKGROUND_Google directory of the database    |
+--------------------------+----------------------------------------------------------------------+
| ``DataPath``             | Path to the database                                                 |
+--------------------------+----------------------------------------------------------------------+
| [``$N2D2_DATA``/         |                                                                      |
+--------------------------+----------------------------------------------------------------------+
| 256_ObjectCategories]    |                                                                      |
+--------------------------+----------------------------------------------------------------------+

CaltechPedestrian_Database
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Cityscapes_Database
~~~~~~~~~~~~~~~~~~~

Cityscapes database :cite:`Cordts2016Cityscapes`.

``Warning`` Don't forget to install the **libjsoncpp-dev** package on your device if you wish to use this database.

.. code-block:: bash

        # To install JSON for C++ library on Ubuntu
        sudo apt-get install libjsoncpp-dev

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

Daimler_Database
~~~~~~~~~~~~~~~~

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

DOTA_Database
~~~~~~~~~~~~~

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

FDDB_Database
~~~~~~~~~~~~~

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

GTSDB_DIR_Database
~~~~~~~~~~~~~~~~~~

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

ILSVRC2012_Database
~~~~~~~~~~~~~~~~~~~

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

KITTI_Database
~~~~~~~~~~~~~~

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

KITTI_Road_Database
~~~~~~~~~~~~~~~~~~~

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

KITTI_Object_Database
~~~~~~~~~~~~~~~~~~~~~

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

LITISRouen_Database
~~~~~~~~~~~~~~~~~~~

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
| [``$N2D2_DATA``/data_rouen]    |                                              |
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

