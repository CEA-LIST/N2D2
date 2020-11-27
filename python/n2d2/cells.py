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

class Cell():

    cell_type = 'Cell'
    
    activation_generators = {
            'TanhActivation_Frame<float>' : N2D2.TanhActivation_Frame_float()
    }

    def __init__(self, name, nbOutputs, activation, **cell_parameters):
        self.cell = None
        self.name = name
        self.nbOutputs = nbOutputs
        self.activation = activation
        
        self.cell_parameters = cell_parameters
        self.model_parameters = None
                
        self.model_key = ""
        
    def N2D2(self):
        return self.cell

    def __str__(self):
        output = "nbOutputs: " + str(self.nbOutputs) + ", "
        output += "activation: " + str(self.activation)
        output += "; "
        output += "Cell parameters: "
        output += str(self.cell_parameters)
        output += "; "
        output += "Model parameters: "
        output += str(self.model_parameters)
        #output += "\n"
        return output
        
   

class FcCell(Cell):
    
    cell_type = 'FcCell'
    
    """Static members"""
    cell_generators = {
            'Frame<float>' : N2D2.FcCell_Frame_float,
            #'Frame_CUDA<float>' : N2D2.FcCell_Frame_CUDA_float,
            #'Frame<half>' : N2D2.FcCell_Frame_half,
            #'Frame_CUDA<half>' : N2D2.FcCell_Frame_CUDA,
            #'Frame<double>' : N2D2.FcCell_Frame_double,
            #'Frame_CUDA<double>' : N2D2.FcCell_Frame_CUDA,
    }
     
        
    def __init__(self, name, nbOutputs, activation='Linear', **cell_parameters):
        super().__init__(name, nbOutputs, activation, **cell_parameters)
         
    """
    # Optional for the moment. Has to assure coherence between n2d2 and N2D2 values
    def set_model_parameter(self, key, value):
        self.model_parameters[key] = value
        #self.cell.setParameter() # N2D2 code
    """
    
    def generate_model(self, deepnet, model_type='Frame', data_type='float', **model_parameters):
        self.model_key = model_type + '<' + data_type + '>'
        
        self.cell = self.cell_generators[self.model_key](deepnet, self.name, self.nbOutputs, self.activation_generators[self.activation + 'Activation_' + self.model_key])
        # NOTE: There might be a special case for certain Spike models that take different parameters
        
        # TODO: Saver to initialize this with the actual values in the N2D2 objects?
        self.model_parameters = model_parameters
        
        
    def __str__(self):
        output = '\'' + self.name + '\' '
        output += "[FcCell(" + self.model_key + ")]: "
        output += super().__str__()
        return output
        
    
    