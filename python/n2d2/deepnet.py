"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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
import n2d2.cells

class Deepnet():

    """
    def __init__(self):
        self.cells = None
        net = N2D2.Network(1)
        self.deepnet = N2D2.DeepNet(net)
    """

    # TODO: Proper exeception handling
    def __init__(self, cells):
        if not isinstance(cells, list):
            print("Error: cells not list objects")
            exit()
        self.cells = cells
        net = N2D2.Network(1)
        self.deepnet = N2D2.DeepNet(net)
    
    # Prepares cells for computation and initializes certain members
    def initialize(self):
         self.deepnet.initialize()
         
    def N2D2(self):
        return self.deepnet

    def __str__(self):
        return ""
    
"""
This should be usable similar to torch.nn.Sequential.
That means also recursive (sequence of sequence etc.).
Allows simple creation of a deepnet based on standard
Python data structures, without using N2D2 binding functions 
    Input:
    * (List of Python Cell objects, model_type)
    * (List of Lists, model_type)
"""    
class Sequential(Deepnet):
    # cells is typically a python list 
    def __init__(self, cells, model_type='Frame'):
        super().__init__(cells)

        self._generate_model(cells, model_type)

    def _generate_model(self, cells, model_type):
        if isinstance(cells, list):
            for cell in cells:
                self._generate_model(cell, model_type)
        elif isinstance(cells, n2d2.cells.Cell):
            cells.generate_model(self.deepnet, model_type)
        else:
            print("Invalid argument for cells")
            exit()

    """ 
        addInput() sets recursively the Tensor dimensions
        of input and output tensors of all cells
    """
    def addStimulus(self, stimuli_provider):
        self._generate_cell_links(self.cells, stimuli_provider)
        self.initialize()

    def _generate_cell_links(self, cells, previous):
        if isinstance(cells, list):
            for cell in cells:
                previous = self._generate_cell_links(cell, previous)
            return previous
        else:
            cells.N2D2().addInput(previous.N2D2())
            return cells

    #def propagate(self):

    #def update(self)
   
    def __str__(self):
        indent_level = [0]
        output = "n2d2.deepnet.Sequential("
        output += self._generate_str(self.cells, indent_level, [0])
        output += "\n)"
        return output

    # TODO: Do without artificial mutable objects
    def _generate_str(self, cells, indent_level, block_idx):
        output = ""
        if isinstance(cells, list):
            if indent_level[0] > 0:
                output += "\n"+ (indent_level[0]*"\t") + "(" + str(block_idx[0]) + ") " +"'Block'"
            indent_level[0] += 1
            local_block_idx = [0]
            for idx, cell in enumerate(cells):
                output += self._generate_str(cell, indent_level, local_block_idx)
                local_block_idx[0] += 1
            indent_level[0] -= 1
        else:
            output += "\n"+ (indent_level[0]*"\t") + "(" + str(block_idx[0]) + ") " + cells.__str__()
        return output