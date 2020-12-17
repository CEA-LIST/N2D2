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



.. autoclass:: N2D2.Database
   :members:

..autoclass:: N2D2.MNIST_IDX_Database
   :members: