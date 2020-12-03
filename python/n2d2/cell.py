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
import n2d2.activation
import n2d2.solver

class Cell():

    def __init__(self, Name, NbOutputs, **cell_parameters):
        self._Name = Name
        self._NbOutputs = NbOutputs
        
        self._cell = None
        self._cell_parameters = cell_parameters
        self._model_parameters = None
                
        self._model_key = ""
        
    def N2D2(self):
        if self._cell is None:
            raise n2d2.UndefinedModelError("N2D2 cell member has not been created. Did you run generate_model?")
        return self._cell

    def __str__(self):
        output = "NbOutputs: " + str(self._NbOutputs) + "; "
        #output += "Activation: " + str(self._Activation)
        #output += "; "
        output += "Cell parameters: "
        output += str(self._cell_parameters)
        output += "; "
        output += "Model parameters: "
        output += str(self._model_parameters)
        #output += "\n"
        return output
        
   

class Fc(Cell):

    """Static members"""
    _cell_generators = {
            'Frame<float>' : N2D2.FcCell_Frame_float,
            'Frame_CUDA<float>' : N2D2.FcCell_Frame_CUDA_float,
    }
     

    def __init__(self, Name, NbOutputs, Activation, **cell_parameters):
        super().__init__(Name=Name, NbOutputs=NbOutputs, **cell_parameters)

        self._Activation = Activation

        if "Solver" in cell_parameters:
            self._Solver = cell_parameters["Solver"]
        else:
            print("No solver")
            exit()

    #TODO: Add method that initialized based on INI file section
    """
    The n2d2 FcCell type could this way serve as a wrapper for both the FcCell constructor and the
    #FcCellGenerator bindings. 
    """
    """
    def __init__(self, file_INI):
        self._cell = N2D2.FcCellGenerator(file=file_INI)
    """
         
    """
    # Optional for the moment. Has to assure coherence between n2d2 and N2D2 values
    def set_model_parameter(self, key, value):
        self._model_parameters[key] = value
        #self.cell.setParameter() # N2D2 code
    """
    
    def generate_model(self, deepnet, Model='Frame', DataType='float', **model_parameters):
        self._model_key = Model + '<' + DataType + '>'

        self._Activation.generate_model(Model, DataType)
        if not self._Solver == None:
            self._Solver.generate_model(Model, DataType)
        self._cell = self._cell_generators[self._model_key](deepnet,
                                                            self._Name,
                                                            self._NbOutputs,
                                                            self._Activation.N2D2())
        # NOTE: There might be a special case for certain Spike models that take different parameters

        # TODO: Initialize model parameters

        # TODO: Saver to initialize this with the actual values in the N2D2 objects?
        self._model_parameters = model_parameters

        self._cell.setWeightsSolver(self._Solver.N2D2())
        
        
    def __str__(self):
        output = '\'' + self._Name + '\' '
        output += "FcCell(" + self._model_key + "): "
        output += "Activation: " + str(self._Activation.__str__()) + ", "
        output += super().__str__()
        return output


class Softmax(Cell):
    """Static members"""
    _cell_generators = {
        'Frame<float>': N2D2.SoftmaxCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SoftmaxCell_Frame_CUDA_float
    }

    def __init__(self, Name, NbOutputs, WithLoss=False, GroupSize=0, **cell_parameters):
        super().__init__(Name=Name, NbOutputs=NbOutputs, **cell_parameters)

        self._WithLoss = WithLoss
        self._GroupSize = GroupSize

    # TODO: Add method that initialized based on INI file section

    def generate_model(self, deepnet, Model='Frame', DataType='float', **model_parameters):
        self._model_key = Model + '<' + DataType + '>'

        self._cell = self._cell_generators[self._model_key](deepnet,
                                                            self._Name,
                                                            self._NbOutputs,
                                                            self._WithLoss,
                                                            self._GroupSize)
        # NOTE: There might be a special case for certain Spike models that take different parameters

        # TODO: Initialize model parameters

        # TODO: Saver to initialize this with the actual values in the N2D2 objects?
        self._model_parameters = model_parameters

    def __str__(self):
        output = '\'' + self._Name + '\' '
        output += "SoftmaxCell(" + self._model_key + "): "
        output += super().__str__()
        return output