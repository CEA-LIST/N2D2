"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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
from n2d2.n2d2_interface import N2D2_Interface
import n2d2.global_variables
from abc import ABC, abstractmethod

def PTQ(deepnet_cell,
        nb_bits,
        provider=None,
        no_unsigned=False,
        cross_layer_equalization=True,
        wt_clipping_mode="NONE",
        act_clipping_mode="MSE",
        act_scaling_mode="FLOAT_MULT",
        **kwargs):
    """
    :param nb_bits: Number of bits per weight for exports (can be for example `-16` for float 16 bits or `8` int 8 bits)
    :type nb_bits: int
    :param provider: Data provider to use for calibration, default=None
    :type provider: :py:class:`n2d2.provider.DataProvider`, optional
    :param no_unsigned: If True, disable the use of unsigned data type in integer calibration, default=False
    :type no_unsigned: bool, optional
    :param cross_layer_equalization: If True, disable the use of cross layer equalization in integer calibration, default=False
    :type cross_layer_equalization: bool, optional
    :param wt_clipping_mode: Weights clipping mode on calibration, can be ``NONE``, ``MSE`` or ``KL_DIVERGENCE``, default="NONE"
    :type wt_clipping_mode: str, optional
    :param act_clipping_mode: activation clipping mode on calibration, can be ``NONE``, ``MSE`` or ``KL_DIVERGENCE`` or ``Quantile``, default="MSE"
    :type act_clipping_mode: str, optional
    """
    if deepnet_cell.get_embedded_deepnet().calibrated:
        raise RuntimeError("This network have already been calibrated.")
    if act_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise n2d2.error_handler.WrongValue("act_clipping_mode", act_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    N2D2_act_clipping_mode = N2D2.ClippingMode.__members__[act_clipping_mode]
    if wt_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise n2d2.error_handler.WrongValue("wt_clipping_mode", wt_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    N2D2_wt_clipping_mode = N2D2.ClippingMode.__members__[wt_clipping_mode]
    if act_scaling_mode not in N2D2.ScalingMode.__members__.keys():
        raise n2d2.error_handler.WrongValue("act_scaling_mode", act_scaling_mode, ", ".join(N2D2.ScalingMode.__members__.keys()))
    N2D2_act_scaling_mode = N2D2.ScalingMode.__members__[act_scaling_mode]
    parameters = n2d2.n2d2_interface.Options(
        nb_bits=nb_bits,
        export_no_unsigned=no_unsigned,
        calibration=True,
        qat_SAT=False,
        export_no_cross_layer_equalization=not cross_layer_equalization,
        wt_clipping_mode=N2D2_wt_clipping_mode,
        act_clipping_mode=N2D2_act_clipping_mode,
        act_scaling_mode=N2D2_act_scaling_mode,
    ).N2D2()

    N2D2_deepnet = deepnet_cell.get_embedded_deepnet().N2D2()
    N2D2_deepnet.initialize()

    if provider is not None:
        N2D2_provider = provider.N2D2()
        N2D2_database = N2D2_provider.getDatabase()
        N2D2_deepnet.setDatabase(N2D2_database)
        N2D2_deepnet.setStimuliProvider(N2D2_provider)

    if len(N2D2_deepnet.getTargets()) == 0:
        # No target associated to the DeepNet
        # We create a Target for the last cell of the network
        last_cell = deepnet_cell[-1].N2D2()
        N2D2_target =  N2D2.TargetScore("Target", last_cell, provider.N2D2())
        N2D2_deepnet.addTarget(N2D2_target)
    elif provider is not None:
        # We already have a Target, so we attach the new provider to it
        for target in N2D2_deepnet.getTargets():
            target.setStimuliProvider(provider.N2D2())

    if N2D2_deepnet.getDatabase().getNbStimuli(N2D2.Database.StimuliSet.__members__["Validation"]) > 0:
        N2D2_deepnet.exportNetworkFreeParameters("weights_validation")
    else:
        N2D2_deepnet.exportNetworkFreeParameters("weights")

    N2D2.calibNetwork(parameters, N2D2_deepnet)
    deepnet_cell.get_embedded_deepnet().calibrated = True

class Quantizer(N2D2_Interface, ABC):

    @abstractmethod
    def __init__(self, **config_parameters):
        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = n2d2.global_variables.default_model
        if 'datatype' in config_parameters:
            self._datatype = config_parameters.pop('datatype')
        else:
            self._datatype = n2d2.global_variables.default_datatype

        self._model_key = self._model + '<' + self._datatype + '>'


        N2D2_Interface.__init__(self, **config_parameters)

    def set_range(self, integer_range):
        self._N2D2_object.setRange(integer_range)

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class CellQuantizer(Quantizer, ABC):
    @abstractmethod
    def __init__(self, **config_parameters):
        Quantizer.__init__(self, **config_parameters)

    def add_weights(self, weights, diff_weights):
        """
        :arg weights: Weights
        :param weights: :py:class:`n2d2.Tensor`
        :arg diff_weights: Diff Weights
        :param diff_weights: :py:class:`n2d2.Tensor`
        """
        if not isinstance(diff_weights, n2d2.Tensor):
            raise n2d2.error_handler("diff_weights", str(type(diff_weights)), ["n2d2.Tensor"])
        if not isinstance(weights, n2d2.Tensor):
            raise n2d2.error_handler("weights", str(type(weights)), ["n2d2.Tensor"])
        self.N2D2().addWeights(weights.N2D2(), diff_weights.N2D2())

    def add_biases(self, biases, diff_biases):
        """
        :arg biases: Biases
        :param biases: :py:class:`n2d2.Tensor`
        :arg diff_biases: Diff Biases
        :param diff_biases: :py:class:`n2d2.Tensor`
        """
        if not isinstance(diff_biases, n2d2.Tensor):
            raise n2d2.error_handler("diff_biases", type(diff_biases) ["n2d2.Tensor"])
        if not isinstance(biases, n2d2.Tensor):
            raise n2d2.error_handler("biases", type(biases) ["n2d2.Tensor"])
        self.N2D2().addBiases(biases.N2D2(), diff_biases.N2D2())

class ActivationQuantizer(Quantizer, ABC):
    @abstractmethod
    def __init__(self, **config_parameters):
        Quantizer.__init__(self, **config_parameters)

