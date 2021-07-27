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
import N2D2

class test_ConvCell(unittest.TestCase):

    def setUp(self):
        self.cell = n2d2.cells.Conv(1, 10, kernel_dims=[2, 2], no_bias=True)

    def tearDown(self):
        pass

    def test_get_set_weight(self):
        weight = n2d2.Tensor([2, 2])
        weight[0] = 1
        weight[1] = 2
        self.cell.set_weight(0, 0, weight)
        self.assertTrue(weight == self.cell.get_weight(0, 0))

    def test_link(self):
        inputs = n2d2.Tensor([1, 1, 2, 2], cuda=True)
        new_cell = n2d2.cells.Fc(10, 5)
        output = new_cell(self.cell(inputs))
        N2D2_deep = output.get_deepnet().N2D2() 
        cells = N2D2_deep.getCells()
        first_cell = cells[N2D2_deep.getLayers()[1][0]] # The first layer is the env, so we get the second.
        last_cell =cells[N2D2_deep.getLayers()[-1][-1]]
        self.assertTrue(first_cell is self.cell.N2D2())
        self.assertTrue(last_cell is new_cell.N2D2())

class test_FcCell(unittest.TestCase):

    def setUp(self):
        self.cell = n2d2.cells.Fc(10, 10, no_bias=True)
    def tearDown(self):
        pass

    def test_get_set_weight(self):
        weight = n2d2.Tensor([1])
        weight[0] = 1
        self.cell.set_weight(0, 0, weight)
        self.assertTrue(weight == self.cell.get_weight(0, 0))

    def test_link(self):
        inputs = n2d2.Tensor([1, 10, 1, 1], cuda=True)
        new_cell = n2d2.cells.Fc(10, 20)
        x = self.cell(inputs)
        output = new_cell(x)
        N2D2_deep = output.get_deepnet().N2D2() 
        cells = N2D2_deep.getCells()
        first_cell = cells[N2D2_deep.getLayers()[1][0]] # The first layer is the env, so we get the second.
        last_cell =cells[N2D2_deep.getLayers()[-1][-1]]
        self.assertTrue(first_cell is self.cell.N2D2())
        self.assertTrue(last_cell is new_cell.N2D2())

    def test_getOutput(self):
        self.cell.get_outputs()


class test_SoftmaxCell(unittest.TestCase):

    def setUp(self):
        self.cell = n2d2.cells.Softmax()
    def tearDown(self):
        pass

    def test_link(self):
        inputs = n2d2.Tensor([1, 10, 1, 1], cuda=True)
        new_cell = n2d2.cells.Fc(10, 20)
        output = self.cell(new_cell(inputs))
        N2D2_deep = output.get_deepnet().N2D2() 
        cells = N2D2_deep.getCells()
        first_cell = cells[N2D2_deep.getLayers()[1][0]]
        last_cell = cells[N2D2_deep.getLayers()[-1][-1]]
        self.assertTrue(first_cell is new_cell.N2D2())
        self.assertTrue(last_cell is self.cell.N2D2())

# class test_PoolCell(unittest.TestCase):

#     def setUp(self):
#         self.cell = n2d2.cells.Pool([1, 1])
#     def tearDown(self):
#         pass

#     def test_link(self):
#         inputs = n2d2.Tensor([1, 1, 2, 2], cuda=True)
#         new_cell = n2d2.cells.Fc(1, 5)
#         output = new_cell(self.cell(inputs))
#         N2D2_deep = output.get_deepnet().N2D2() 
#         cells = N2D2_deep.getCells()
#         first_cell = cells[N2D2_deep.getLayers()[1][0]]
#         last_cell = cells[N2D2_deep.getLayers()[-1][-1]]
#         self.assertTrue(last_cell is new_cell.N2D2())
#         self.assertTrue(first_cell is self.cell.N2D2())

if __name__ == '__main__':
    unittest.main()