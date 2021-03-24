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
import unittest
import n2d2
import numpy as np

class test_tensor(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.z = (2,3,4)
        self.tensor = n2d2.tensor.Tensor([self.x, self.y, self.z], defaultDataType=int, value=0)
    def test_set_index(self):
        self.tensor[0] = 1
        self.assertEqual(self.tensor[0], 1)
        self.assertEqual(self.tensor[0, 0, 0], 1)

    def test_value_param(self):
        self.tensor = n2d2.tensor.Tensor([self.x, self.y, self.z], defaultDataType=int, value=5)
        for i in self.tensor:
            self.assertEqual(i, 5)

    def test_set_slice(self):
        self.tensor[1:3] = 1  
        self.assertEqual(self.tensor[1], 1)
        self.assertEqual(self.tensor[0, 0, 1], 1)
        self.assertEqual(self.tensor[2], 1)
        self.assertEqual(self.tensor[0, 0, 2], 1)

    def test_set_coordinate(self):
        self.tensor[0,0,3] = 1 # Using coordinates
        self.assertEqual(self.tensor[0, 0, 3], 1)
        self.assertEqual(self.tensor[3], 1)

    def test_fill(self):
        self.tensor[0:] = 1
        for i in self.tensor:
            self.assertEqual(i, 1)

    def test_len(self):
        self.assertEqual(len(self.tensor), self.x*self.y*self.z)

    def test_copy_equal(self):
        copy = self.tensor.copy()
        self.assertTrue(copy is not self.tensor and copy == self.tensor)

    def test_dim(self):
        self.x, self.y, self.z, self.b = (1, 2, 3, 4)
        self.tensor = n2d2.tensor.Tensor([self.b, self.z, self.y, self.x], defaultDataType=int, value=0)
        self.assertEqual(self.tensor.dimX(), self.x)
        self.assertEqual(self.tensor.dimY(), self.y)
        self.assertEqual(self.tensor.dimZ(), self.z)
        self.assertEqual(self.tensor.dimB(), self.b)

    def test_reshape(self):
        self.tensor.reshape([self.z, self.y, self.x])
        self.assertEqual(self.tensor.shape(), [self.z, self.y, self.x])

    def test_equal(self):
        same_tensor = n2d2.tensor.Tensor([self.x, self.y, self.z], defaultDataType=int, value=0)
        type_different = n2d2.tensor.Tensor([self.x, self.y, self.z], defaultDataType=float, value=0)
        different_tensor = n2d2.tensor.Tensor([self.x, self.y, self.z], defaultDataType=int, value=1)
        dim_different = n2d2.tensor.Tensor([self.x, self.y], defaultDataType=int, value=0)
        self.assertTrue(self.tensor == same_tensor)
        self.assertTrue(self.tensor == type_different)
        self.assertTrue(self.tensor != different_tensor)
        self.assertTrue(self.tensor != dim_different)

    def test_contain(self):
        self.tensor[0] = 5
        self.assertTrue(5 in self.tensor)

    def test_numpy(self):
        self.tensor = n2d2.tensor.Tensor([3, 2])
        self.tensor[0] = 1
        self.tensor[1] = 2
        self.tensor[2] = 3
        self.tensor[3] = 4
        self.tensor[4] = 5
        self.tensor[5] = 6
        
        np_tensor = self.tensor.to_numpy()
        equivalent_numpy = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(np_tensor, equivalent_numpy))
        tensor_numpy = n2d2.tensor.Tensor([3, 2])
        tensor_numpy.from_numpy(np_tensor)
        self.assertTrue(self.tensor == tensor_numpy)


class test_cudatensor(test_tensor):
    def setUp(self):
        self.x, self.y, self.z = (2,3,4)
        self.tensor = n2d2.tensor.CudaTensor([self.x, self.y, self.z], defaultDataType=int, value=0)

    def test_value_param(self):
            self.tensor = n2d2.tensor.CudaTensor([self.x, self.y, self.z], defaultDataType=int, value=5)
            for i in self.tensor:
                self.assertEqual(i, 5)


    def test_equal(self):
        same_tensor = n2d2.tensor.CudaTensor([self.x, self.y, self.z], defaultDataType=int, value=0)
        type_different = n2d2.tensor.CudaTensor([self.x, self.y, self.z], defaultDataType=float, value=0)
        different_tensor = n2d2.tensor.CudaTensor([self.x, self.y, self.z], defaultDataType=int, value=2)
        dim_different = n2d2.tensor.CudaTensor([self.x, self.y], defaultDataType=int, value=0)
        self.assertTrue(self.tensor == same_tensor)
        self.assertTrue(self.tensor == type_different)
        self.assertTrue(self.tensor != different_tensor)
        self.assertTrue(self.tensor != dim_different)


    def test_numpy(self):
        self.tensor = n2d2.tensor.Tensor([3, 2])
        self.tensor[0] = 1
        self.tensor[1] = 2
        self.tensor[2] = 3
        self.tensor[3] = 4
        self.tensor[4] = 5
        self.tensor[5] = 6
        
        np_tensor = self.tensor.to_numpy()
        equivalent_numpy = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(np_tensor, equivalent_numpy))
        tensor_numpy = n2d2.tensor.CudaTensor([3, 2])
        tensor_numpy.from_numpy(np_tensor)
        self.assertTrue(self.tensor == tensor_numpy)
    

    
if __name__ == '__main__':
    unittest.main()
