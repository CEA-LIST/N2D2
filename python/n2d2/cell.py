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

    def __init__(self, Name, NbOutputs):

        self._cell_parameters = {
            'Name': Name,
            'NbOutputs': NbOutputs,
        }
        
        self._cell = None
        self._model_parameters = None
                
        self._model_key = ""

    def set_cell_parameters(self, cell_parameters):
        for key, value in cell_parameters.items():
            if key in self._cell_parameters:
                self._cell_parameters[key] = value
            else:
                raise n2d2.UndefinedParameterError(key, self)
        
    def N2D2(self):
        if self._cell is None:
            raise n2d2.UndefinedModelError("N2D2 cell member has not been created. Did you run generate_model?")
        return self._cell

    def __str__(self):
        output = ""
        for key, value in self._cell_parameters.items():
            output += key + ": " + str(value) + ", "
        output += "; "
        output += "Model parameters: "
        output += str(self._model_parameters)
        #output += "\n"
        return output
        
   

class Fc(Cell):

    _cell_generators = {
            'Frame<float>': N2D2.FcCell_Frame_float,
            'Frame_CUDA<float>': N2D2.FcCell_Frame_CUDA_float,
    }

    _weightsExportFormat = {
        'OC': N2D2.FcCell.WeightsExportFormat.OC,
        'CO': N2D2.FcCell.WeightsExportFormat.CO
    }

    _frame_model_parameters = {
        'DropConnect': 1.0
    }

    _frame_CUDA_model_parameters = {}



    def __init__(self, Name, NbOutputs, **cell_parameters):
        super().__init__(Name=Name, NbOutputs=NbOutputs)

        """Equivalent to N2D2 class generator defaults. 
           NOTE: These are not necessarily the default values of the constructors!
           The default objects are only abstract n2d2 objects with small memory footprint.
           ALL existing cell parameters (in N2D2) are declared here, which also permits to check 
           validity of **cell_parameters entries. For easier compatibility with INI files, we 
           use the same name convention and parameter names.
        """
        """self._cell_parameters.update({
            'ActivationFunction': n2d2.activation.Tanh(),
            'WeightsSolver': n2d2.solver.SGD(),
            'BiasSolver': n2d2.solver.SGD(),
            'WeightsFiller': n2d2.filler.Normal(Mean=0.0, StdDev=0.05),
            'BiasFiller': n2d2.filler.Normal(Mean=0.0, StdDev=0.05),
        })"""

        # TODO: Use real defaults
        self._cell_parameters.update({
            'ActivationFunction': n2d2.activation.Linear(),
            'WeightsSolver': n2d2.solver.SGD(),
            'BiasSolver': n2d2.solver.SGD(),
            'WeightsFiller': n2d2.filler.He(),
            'BiasFiller': n2d2.filler.He(),
            'NoBias': False,
            'Normalize': False,
            'BackPropagate': True,
            'WeightsExportFormat': 'OC',
            'OutputsRemap': "",
        })

        self.set_cell_parameters(cell_parameters)


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

        """TODO: This is  necessary at the moment because the default argument in the binding constructor
        cannot be set. """
        self._cell_parameters['ActivationFunction'].generate_model(Model, DataType)

        self._cell = self._cell_generators[self._model_key](deepnet,
                                                            self._cell_parameters['Name'],
                                                            self._cell_parameters['NbOutputs'],
                                                            self._cell_parameters['ActivationFunction'].N2D2())

        """Set and initialize here all complex cell members"""
        for key, value in self._cell_parameters.items():
            if key is 'ActivationFunction':
                self._cell_parameters['ActivationFunction'].generate_model(Model, DataType)
                self._cell.setActivation(self._cell_parameters['ActivationFunction'].N2D2())
            if key is 'WeightsSolver':
                self._cell_parameters['WeightsSolver'].generate_model(Model, DataType)
                self._cell.setWeightsSolver(self._cell_parameters['WeightsSolver'].N2D2())
            if key is 'BiasSolver':
                self._cell_parameters['BiasSolver'].generate_model(Model, DataType)
                self._cell.setBiasSolver(self._cell_parameters['BiasSolver'].N2D2())
            if key is 'WeightsFiller':
                self._cell_parameters['WeightsFiller'].generate_model(DataType)
                self._cell.setWeightsFiller(self._cell_parameters['WeightsFiller'].N2D2())
            if key is 'BiasFiller':
                self._cell_parameters['BiasFiller'].generate_model(DataType)
                self._cell.setBiasFiller(self._cell_parameters['BiasFiller'].N2D2())
            if key is 'WeightsExportFormat':
                self._cell_parameters['WeightsExportFormat'] = self._weightsExportFormat[value]

        if Model is 'Frame':
            for key, value in model_parameters:
                if key in self._frame_model_parameters:
                    self._frame_model_parameters[key] = value
                else:
                    raise n2d2.UndefinedParameterError(key, self)
        if Model is 'Frame_CUDA':
            for key, value in model_parameters:
                if key in self._frame_CUDA_model_parameters:
                    self._frame_CUDA_model_parameters[key] = value
                else:
                    raise n2d2.UndefinedParameterError(key, self)

        
    def __str__(self):
        output = "FcCell(" + self._model_key + "): "
        output += super().__str__()
        return output


class Softmax(Cell):
    """Static members"""
    _cell_generators = {
        'Frame<float>': N2D2.SoftmaxCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SoftmaxCell_Frame_CUDA_float
    }

    def __init__(self, Name, NbOutputs, **cell_parameters):
        super().__init__(Name=Name, NbOutputs=NbOutputs)

        """
            Equivalent to N2D2 class constructor defaults. 
            NOTE: These are not necessarily the default values set by the generators!
            The default objects are only abstract n2d2 objects with small memory footprint.
            ALL existing cell parameters (in N2D2) are declared here, which also permits to check 
            validity of **cell_parameters entries. For easier compatibility with INI files, we 
            use the same name convention and parameter names.
        """
        self._cell_parameters.update({
            'WithLoss': False,
            'GroupSize': 0,
        })

        self.set_cell_parameters(cell_parameters)


    # TODO: Add method that initialized based on INI file section

    def generate_model(self, deepnet, Model='Frame', DataType='float', **model_parameters):
        self._model_key = Model + '<' + DataType + '>'

        self._cell = self._cell_generators[self._model_key](deepnet,
                                                            self._cell_parameters['Name'],
                                                            self._cell_parameters['NbOutputs'],
                                                            self._cell_parameters['WithLoss'],
                                                            self._cell_parameters['GroupSize'],
                                                            )
        # NOTE: There might be a special case for certain Spike models that take different parameters

        # TODO: Initialize model parameters

        # TODO: Saver to initialize this with the actual values in the N2D2 objects?
        self._model_parameters = model_parameters

    def __str__(self):
        output = "SoftmaxCell(" + self._model_key + "): "
        output += super().__str__()
        return output
