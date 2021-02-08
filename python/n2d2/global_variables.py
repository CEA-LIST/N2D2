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

import N2D2
import n2d2.deepnet

# TODO: In final version this should be in the use home or the API launch folder
model_cache = "/local/is154584/jt251134/MODELS"

default_seed = 1
default_model = 'Frame'
default_dataType = 'float'
default_net = N2D2.Network(default_seed)
#default_deepNet = n2d2.deepnet.DeepNet(default_net, default_model, default_dataType)
default_deepNet = None

cell_counter = 0
target_counter = 0
provider_counter = 0

class Verbosity:
    short = 0  # Constructor arguments only
    detailed = 1  # Config parameters and their parameters

verbosity = Verbosity.detailed

def set_cuda_device(id):
    N2D2.CudaContext.setDevice(id)

