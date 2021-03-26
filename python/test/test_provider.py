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
import N2D2
import n2d2
import unittest



class test_provider(unittest.TestCase):
    def setUp(self):
        self.db = n2d2.database.MNIST(dataPath="/nvme0/DATABASE/MNIST/raw/")       
        self.size = [28, 28, 1]
        self.batch_size = 1
        self.provider = n2d2.provider.DataProvider(self.db, self.size, batchSize=self.batch_size)        
        
    def tearDown(self):
        pass

    def test_get_batch_size(self):
        self.assertEqual(self.batch_size, self.provider.get_batch_size())

    def test_loop():
        # TODO : Find a way to test interating over a provider
        pass
    
    def test_read_random_batch_error_partition(self):
        with self.assertRaises(ValueError):
            self.provider.read_random_batch(partition='Wrong string !')
        
    def test_read_random_batch_error_partition(self):
        with self.assertRaises(ValueError):
            self.provider.read_batch(partition='Wrong string !', idx=0)

    

    def test_read_random_batch(self):
        input_tensor = self.provider.get_data()
        for i in input_tensor:
            self.assertEqual(i, 0)
        self.provider.read_random_batch(partition='Test')
        input_tensor = self.provider.get_data()
        empty = True
        for i in input_tensor:
            if i != 0:
                empty = False
                break
        self.assertFalse(empty)

    def test_read_batch(self):
        input_tensor = self.provider.get_data()
        for i in input_tensor:
            self.assertEqual(i, 0)
        self.provider.read_batch(partition='Test', idx=0)
        input_tensor = self.provider.get_data()
        vide = True
        for i in input_tensor:
            if i != 0:
                vide = False
                break
        self.assertFalse(vide)
if __name__ == '__main__':
    unittest.main()