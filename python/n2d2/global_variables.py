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
from enum import Enum

class GlobalVariables: # pylint: disable=too-many-instance-attributes
    """
    This class handle global parameters.

    Here is a list of the global paramters :

    +--------------------------+-------------------------------------------------------------------+
    | Default parameters       | Description                                                       |
    +==========================+===================================================================+
    | ``default_model``        | If you have compiled N2D2 with **CUDA**, you                      |
    |                          | can use ``Frame_CUDA``, default= ``Frame``                        |
    +--------------------------+-------------------------------------------------------------------+
    | ``default_datatype``     | Datatype of the layer of the neural network. At the moment only   |
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
    |``cuda_available``         | Is True if you have compiled N2D2 with the CUDA library.          |
    |                          | If False, you can install CUDA and reinstall N2D2 ot make CUDA    |
    |                          | available.                                                        |
    +--------------------------+-------------------------------------------------------------------+
    """

    class Verbosity(Enum):
        graph_only = 0  # Only names, cell types and inputs
        short = 1  # Constructor arguments only
        detailed = 2  # Config parameters and their parameters

    def __init__(self):
        self.model_cache = expanduser("~") + "/MODELS"
        self._seed = 1
        self.default_model = 'Frame'
        self.default_datatype = 'float'
        self.default_net = N2D2.Network(self._seed, saveSeed=False, printTimeElapsed=False)
        # check if N2D2 is cuda compiled and if there is devices available
        self._cuda_available = N2D2.cuda_compiled and (N2D2.CudaContext.nbDevice() > 0)
        self._json_compiled = N2D2.json_compiled
        self._onnx_compiled = N2D2.onnx_compiled
        self._n2d2_ip_compiled = N2D2.N2D2_IP
        self._cuda_device = 0
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
        if not self.cuda_available:
            raise RuntimeError("N2D2 is not compiled with CUDA.")
        if isinstance(value, int):
            if value > N2D2.CudaContext.nbDevice():
                raise RuntimeError(f"Cannot set device {value}, N2D2 detected only {N2D2.CudaContext.nbDevice()} devices")
            self._cuda_device = value
            N2D2.setCudaDeviceOption(value) # Setting this variable is mandatory to use the fit method otherwise,
                                    # the device used for learning would be 0 (default value)
            N2D2.CudaContext.setDevice(value)
        elif isinstance(value, (tuple, list)):
            devices = ""
            first_device = value[0]
            for val in value:
                if not isinstance(val, int):
                    raise TypeError("Device should be of type 'int'")
                if val > N2D2.CudaContext.nbDevice():
                    raise RuntimeError(f"Cannot set device {val}, N2D2 detected only {N2D2.CudaContext.nbDevice()} devices")
                devices += str(val) +","
            devices = devices.strip(",")
            N2D2.setMultiDevices(devices)
            N2D2.CudaContext.setDevice(first_device)
        else:
            raise TypeError(f"Device should be of type 'int' or 'tuple' got {type(value).__name__} instead")

    @property
    def cuda_available(self):
        return self._cuda_available
    @cuda_available.setter
    def cuda_available(self, _): # pylint: disable=no-self-use
        raise RuntimeError("The parameter cuda_available is on read only !")

    @property
    def json_compiled(self):
        return self._json_compiled

    @json_compiled.setter
    def json_compiled(self, _): # pylint: disable=no-self-use
        raise RuntimeError("The parameter json_compiled is on read only !")

    @property
    def onnx_compiled(self):
        return self._onnx_compiled

    @onnx_compiled.setter
    def onnx_compiled(self, _): # pylint: disable=no-self-use
        raise RuntimeError("The parameter json_compiled is on read only !")

    @property
    def n2d2_ip_compiled(self):
        return self._n2d2_ip_compiled

    @n2d2_ip_compiled.setter
    def n2d2_ip_compiled(self, _): # pylint: disable=no-self-use
        raise RuntimeError("The parameter n2d2_ip_compiled is on read only !")

    # Legacy : We send a deprecated error if these methods are still used :
    def set_cuda_device(self, device): # pylint: disable=no-self-use
        raise RuntimeError(f"set_cuda_device should not be used anymore, please replace it with :\nn2d2.global_variables.cuda_device = {device}")
    def set_random_seed(self, seed): # pylint: disable=no-self-use
        raise RuntimeError (f"set_random_seed should not be used anymore, please replace it with :\nn2d2.global_variables.seed = {seed}")

    def __str__(self):
        variables = [var for var in dir(self) if not (var.startswith("_") or var[0].isupper() or ismethod(getattr(self, var)))]
        string = "n2d2 global variables :\n"
        for variable in variables:
            string += f"\t- {variable} : {getattr(self, variable)}\n"
        string = string.rstrip("\n")
        return string
