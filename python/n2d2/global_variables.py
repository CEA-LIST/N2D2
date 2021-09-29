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


class GlobalVariables:
    
    def __init__(self):
        self.model_cache = expanduser("~") + "/MODELS"
        self._seed = 1
        self.default_model = 'Frame'
        self.default_datatype = 'float'
        self.default_net = N2D2.Network(self._seed, saveSeed=False, printTimeElapsed=False)
        self.cuda_compiled = N2D2.cuda_compiled
        self.n2d2_ip_compiled = N2D2.N2D2_IP 
        self._cuda_device = 0
        class VerbosityClass:
            graph_only = 0  # Only names, cell types and inputs
            short = 1  # Constructor arguments only
            detailed = 2  # Config parameters and their parameters
        self.Verbosity = VerbosityClass()
        self.verbosity = self.Verbosity.detailed

    @property
    def seed(self):
        return self._seed
    @seed.setter
    def seed(self, value):
        self._default_seed = value
        N2D2.mtSeed(value)
    @property
    def cuda_device(self):
        return self._cuda_device
    @cuda_device.setter
    def cuda_device(self, value):
        self._cuda_device = value
        N2D2.CudaContext.setDevice(value)
    def set_cuda_device(self, device):
        raise RuntimeError(f"set_cuda_device should not be used anymore, please replace it with :\nn2d2.global_variables.cuda_device = {device}")
    def set_random_seed(self, seed):
        raise RuntimeError (f"set_random_seed should not be used anymore, please replace it with :\nn2d2.global_variables.seed = {seed}")
