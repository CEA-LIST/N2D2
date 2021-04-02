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
        self.db = n2d2.database.DIR()
        self.provider = n2d2.provider.DataProvider(self.db, [1, 1, 1], batch_size=1)
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
        print("Set up done !")

    def tearDown(self):
        print("Cleaning data")
        remove(self.data_path + self.suffix)
        remove(self.label_path + self.suffix)
        rmdir(self.data_path)
        rmdir(self.label_path)

    def test_load(self):
        print('Loading data')
        self.db.load(self.data_path, 0, self.label_path, 0)
        self.provider.set_partition("Learn")

        self.db.partition_stimuli(1,0,0)
        self.assertEqual(self.provider.read_random_batch().tensor[0], self.x)
        
 
@unittest.skipIf(not exists("/nvme0/DATABASE/MNIST/raw/"), "Data not found !")
class test_MNIST(unittest.TestCase):
    def setUp(self):
        self.db = n2d2.database.MNIST(data_path="/nvme0/DATABASE/MNIST/raw/")       
        self.size = [28, 28, 1]
        self.batch_size = 1
        self.provider = n2d2.provider.DataProvider(self.db, self.size, batch_size=self.batch_size)
    def tearDown(self):
        pass

    def test_size(self):
        self.assertEqual(self.provider.dims(), self.size + [self.batch_size])

    def test_label(self):
        self.assertEqual(self.db.get_label_name(0), "0")
        self.assertEqual(self.db.get_label_name(1), "1")
        self.assertEqual(self.db.get_label_name(2), "2")
        self.assertEqual(self.db.get_label_name(3), "3")
        self.assertEqual(self.db.get_label_name(4), "4")
        self.assertEqual(self.db.get_label_name(5), "5")
        self.assertEqual(self.db.get_label_name(6), "6")
        self.assertEqual(self.db.get_label_name(7), "7")
        self.assertEqual(self.db.get_label_name(8), "8")
        self.assertEqual(self.db.get_label_name(9), "9")

@unittest.skipIf(not (exists("/nvme0/DATABASE/ILSVRC2012") and exists("/nvme0/DATABASE/ILSVRC2012/synsets.txt")), "Data not found !")
class test_ILSVRC2012(test_MNIST):
    def setUp(self):
        self.db = n2d2.database.ILSVRC2012(learn=1.0)
        self.db.load("/nvme0/DATABASE/ILSVRC2012", label_path="/nvme0/DATABASE/ILSVRC2012/synsets.txt")
        self.size = [500, 334, 3]
        self.batch_size = 1
        provider = n2d2.provider.DataProvider(database=self.db, size=self.size, batch_size=self.batch_size)
        self.provider = n2d2.provider.DataProvider(self.db, self.size, batch_size=1)
    def test_label(self):
        self.assertEqual(self.db.get_label_name(0), "n01440764")
        self.assertEqual(self.db.get_label_name(1), "n01443537")
        self.assertEqual(self.db.get_label_name(811), "n04265275")
        self.assertEqual(self.db.get_label_name(998), "n13133613")
        self.assertEqual(self.db.get_label_name(999), "n15075141")



if __name__ == '__main__':
    unittest.main()