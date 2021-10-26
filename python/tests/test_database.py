"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 
                    Johannes THIELE (johannes.thiele@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
"""
from os import mkdir, rmdir, remove
from os.path import exists
import N2D2
import n2d2
import unittest



class test_DIR(unittest.TestCase):
    def setUp(self):
        self.x = 10
        self.y = 3
        print("Creating data")
        self.data_path = "data"
        self.label_path = "label"
        self.suffix = "/fake.txt"
        if not exists(self.data_path):
            mkdir(self.data_path)
        if not exists(self.label_path):
            mkdir(self.label_path)
        with open(self.data_path + self.suffix, "w") as f:
            f.write(str(self.x)+'\n')
        with open(self.label_path + self.suffix, "w") as f:
            f.write(str(self.y)+'\n')
        print("Init database object")
        self.db = n2d2.database.DIR(self.data_path, 1.0)
        self.provider = n2d2.provider.DataProvider(self.db, [1, 1, 1], batch_size=1)
        
        
        print("Set up done !")

    def tearDown(self):
        print("Cleaning data")
        remove(self.data_path + self.suffix)
        remove(self.label_path + self.suffix)
        rmdir(self.data_path)
        rmdir(self.label_path)

    def test_load(self):
        print('Loading data')

        self.provider.set_partition("Learn")
        self.provider.read_random_batch().htod()
        print(self.provider.read_random_batch()[0])
        self.assertEqual(self.provider.read_random_batch()[0], self.x)


if __name__ == '__main__':
    unittest.main()