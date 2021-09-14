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
from os.path import expanduser

model_cache = expanduser("~") + "/MODELS"

default_seed = 1 # TODO : I am not sure this should be a global variable as modifying it doesn't change the seed used !
default_model = 'Frame'
default_datatype = 'float'
default_net = N2D2.Network(default_seed, saveSeed=False, printTimeElapsed=False)
cuda_compiled = N2D2.cuda_compiled
n2d2_ip_compiled = N2D2.N2D2_IP 

_objects_counter = {}


class Verbosity:
    graph_only = 0  # Only names, cell types and inputs
    short = 1  # Constructor arguments only
    detailed = 2  # Config parameters and their parameters

verbosity = Verbosity.detailed

# TODO : Move this function to utils ?
def set_cuda_device(id):
    N2D2.CudaContext.setDevice(id)

# TODO : Move this function to utils ?
def generate_name(obj):
    """
    Function used to generate name of an object
    """
    name = obj.__class__.__name__
    if name in _objects_counter:
        _objects_counter[name] += 1
    else:
        _objects_counter[name] = 0
    name += "_"+str(_objects_counter[name])
    return name

def set_random_seed(seed):
    N2D2.mtSeed(seed)
    global default_seed
    default_seed = seed