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
from inspect import ismethod

class GlobalVariables:
    """
    This class handle global parameters.

    Here is a list of the global paramters :

    +--------------------------+-------------------------------------------------------------------+
    | Default parameters       | Description                                                       |
    +==========================+===================================================================+
    | ``default_model``        | If you have compiled N2D2 with **CUDA**, you                      |
    |                          | can use ``Frame_CUDA``, default= ``Frame``                        |
    +--------------------------+-------------------------------------------------------------------+
    | ``default_datatype``     | Datatype of the layer of the neural network. Can be ``double``or  |
    |                          | ``float``, default= ``float``                                     |
    |                          |                                                                   |
    |                          | **Important :** This variable doesn't affect the data type of     |
    |                          | :py:class:`n2d2.Tensor` objects.                                  |
    +--------------------------+-------------------------------------------------------------------+
    | ``verbosity``            | Level of verbosity, can be                                        |
    |                          | ``n2d2.global_variables.Verbosity.graph_only``,                   |
    |                          | ``n2d2.global_variables.Verbosity.short`` or                      |
    |                          | ``n2d2.global_variables.Verbosity.detailed``,                     |
    |                          | default= ``n2d2.global_variables.Verbosity.detailed``             |
    +--------------------------+-------------------------------------------------------------------+
    |``seed``                  | Seed used to generate random numbers(0 = time based),             |
    |                          | default = ``0``                                                   |
    +--------------------------+-------------------------------------------------------------------+
    |``cuda_device``           | Device to use for GPU computation with CUDA, default = ``0``      |
    +--------------------------+-------------------------------------------------------------------+
    """
    def __init__(self):
        self.model_cache = expanduser("~") + "/MODELS"
        self._seed = 1
        self.default_model = 'Frame'
        self.default_datatype = 'float'
        self.default_net = N2D2.Network(self._seed, saveSeed=False, printTimeElapsed=False)
        self._cuda_compiled = N2D2.cuda_compiled
        self._n2d2_ip_compiled = N2D2.N2D2_IP 
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
        if value > N2D2.CudaContext.nbDevice():
            raise RuntimeError(f"Cannot set device {value}, you have {N2D2.CudaContext.nbDevice()} devices")
        self._cuda_device = value
        N2D2.setCudaDeviceOption(value) # Setting this variable is mandatory to use the fit method otherwise, 
                                # the device used for learning would be 0 (default value)
        N2D2.CudaContext.setDevice(value)

    @property
    def cuda_compiled(self):
        return self._cuda_compiled
    @cuda_compiled.setter
    def cuda_compiled(self, _):
        raise RuntimeError("The parameter cuda_compiled is on read only !")

    @property
    def n2d2_ip_compiled(self):
        return self._n2d2_ip_compiled
    @cuda_compiled.setter
    def n2d2_ip_compiled(self, _):
        raise RuntimeError("The parameter n2d2_ip_compiled is on read only !")

    # Legacy : We send a deprecated error if these methods are still used :
    def set_cuda_device(self, device):
        raise RuntimeError(f"set_cuda_device should not be used anymore, please replace it with :\nn2d2.global_variables.cuda_device = {device}")
    def set_random_seed(self, seed):
        raise RuntimeError (f"set_random_seed should not be used anymore, please replace it with :\nn2d2.global_variables.seed = {seed}")

    def __str__(self):
        variables = [var for var in dir(self) if not (var.startswith("_") or var[0].isupper() or ismethod(getattr(self, var)))]
        string = "n2d2 global variables :\n"
        for variable in variables:
            string += f"\t- {variable} : {getattr(self, variable)}\n"
        string = string.rstrip("\n")
        return string