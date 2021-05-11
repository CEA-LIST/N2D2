Databases
=========

.. testsetup:: *

   import N2D2
   path = "/nvme0/DATABASE/MNIST/raw/"


Introduction: 
-------------

N2D2 allow you to import default dataset or to load your own dataset. 
This can be done suing Database objects.

Download datasets:
------------------

To import Data you can use a python Script situated in ``./tools/install_stimuli_gui.py``.

This script will download the data in ``/local/$USER/n2d2_data/``. 
You can change this path with the environment variable ``N2D2_data``.

Once the dataset downloaded, you can load it with the appropriate class. 
Here is an example of the loading of the MNIST dataset :

.. testcode::
    
    database = N2D2.MNIST_IDX_Database()
    database.load(path)

In this example, the data are located in the folder **path**.

Use your own data:
------------------

TODO

Database:
---------

Database
~~~~~~~~

.. autoclass:: N2D2.Database
        :members:
        :inherited-members:

MNIST_IDX_Database
~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.MNIST_IDX_Database
        :members:
        :inherited-members:

Actitracker_Database
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.Actitracker_Database
        :members:
        :inherited-members:

AER_Database
~~~~~~~~~~~~

.. autoclass:: N2D2.AER_Database
        :members:
        :inherited-members:

Caltech101_DIR_Database
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.Caltech101_DIR_Database
        :members:
        :inherited-members:

Caltech256_DIR_Database
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.Caltech256_DIR_Database
        :members:
        :inherited-members:

CaltechPedestrian_Database
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.CaltechPedestrian_Database
        :members:
        :inherited-members:

CelebA_Database
~~~~~~~~~~~~~~~

.. autoclass:: N2D2.CelebA_Database
        :members:
        :inherited-members:

CIFAR_Database
~~~~~~~~~~~~~~

.. autoclass:: N2D2.CIFAR_Database
        :members:
        :inherited-members:

CKP_Database
~~~~~~~~~~~~

.. autoclass:: N2D2.CKP_Database
        :members:
        :inherited-members:

DIR_Database
~~~~~~~~~~~~

.. autoclass:: N2D2.DIR_Database
        :members:
        :inherited-members:

GTSRB_DIR_Database
~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.GTSRB_DIR_Database
        :members:
        :inherited-members:

GTSDB_DIR_Database
~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.GTSDB_DIR_Database
        :members:
        :inherited-members:

ILSVRC2012_Database
~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.ILSVRC2012_Database
        :members:
        :inherited-members:

IDX_Database
~~~~~~~~~~~~

.. autoclass:: N2D2.IDX_Database
        :members:
        :inherited-members:

IMDBWIKI_Database
~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.IMDBWIKI_Database
        :members:
        :inherited-members:

KITTI_Database
~~~~~~~~~~~~~~

.. autoclass:: N2D2.KITTI_Database
        :members:
        :inherited-members:

KITTI_Object_Database
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.KITTI_Object_Database
        :members:
        :inherited-members:

KITTI_Road_Database
~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.KITTI_Road_Database
        :members:
        :inherited-members:

LITISRouen_Database
~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.LITISRouen_Database
        :members:
        :inherited-members:

N_MNIST_Database
~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.N_MNIST_Database
        :members:
        :inherited-members:

DOTA_Database
~~~~~~~~~~~~~

.. autoclass:: N2D2.DOTA_Database
        :members:
        :inherited-members:

Fashion_MNIST_IDX_Database
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.Fashion_MNIST_IDX_Database
        :members:
        :inherited-members:

FDDB_Database
~~~~~~~~~~~~~

.. autoclass:: N2D2.FDDB_Database
        :members:
        :inherited-members:

Daimler_Database
~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.Daimler_Database
        :members:
        :inherited-members:


.. testcode::
   :hide:

   N2D2.Actitracker_Database()
   N2D2.Caltech101_DIR_Database(0.1)
   N2D2.Caltech256_DIR_Database(0.1)
   N2D2.CaltechPedestrian_Database()
   N2D2.CelebA_Database(True, True)
   N2D2.CIFAR10_Database()
   N2D2.CIFAR100_Database()
   N2D2.CKP_Database(0.1)
   # N2D2.Cityscapes_Database()
   N2D2.GTSDB_DIR_Database(0.1)
   N2D2.GTSRB_DIR_Database(0.1)
   N2D2.ILSVRC2012_Database(0.1)
   N2D2.IDX_Database()
   N2D2.IMDBWIKI_Database(True, True, True, True, 0.1, 1)
   N2D2.KITTI_Database(1)
   N2D2.KITTI_Object_Database(1)
   N2D2.KITTI_Road_Database(1)
   N2D2.LITISRouen_Database()
   N2D2.DOTA_Database(1)
   N2D2.MNIST_IDX_Database()
   N2D2.N_MNIST_Database()
   N2D2.Fashion_MNIST_IDX_Database()
   N2D2.FDDB_Database(1)
   N2D2.Daimler_Database(1,1,1,True)