"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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
try:    
    # If you try to import numpy after initialising a deepNet, the import cause a crash.
    # Importing numpy before the library is a temporary fix. 
    # I put this import in a try bloc to avoid adding an explicit dependance to numpy. 
    import numpy
except:
    pass

# It's better to access these variables without mentioning global_variable in the path
from n2d2.global_variables import * 

# It's better to access these methods without mentioning utils in the path
from n2d2.utils import *

from n2d2.n2d2_interface import ConventionConverter

from n2d2.tensor import *
import n2d2.database 
import n2d2.cells
import n2d2.provider
import n2d2.transform
import n2d2.deepnet
import n2d2.solver
import n2d2.filler
import n2d2.target
import n2d2.application
import n2d2.activation
import n2d2.mapping
import n2d2.error_handler
import n2d2.models
# IP functions
import n2d2.quantizer
import n2d2.converter

