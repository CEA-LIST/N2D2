"""
(C) Copyright 2022 CEA LIST. All Rights Reserved.
Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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

import n2d2
import N2D2

def _parse_export_parameters(gen_export=None, nb_bits=8, qat_SAT=False,
                             export_no_unsigned=False, calibration=0,
                             export_no_cross_layer_equalization=False,
                             wt_clipping_mode="NONE", act_clipping_mode="MSE",
                             act_scaling_mode="FLOAT_MULT", act_quantile_value=0.9999,
                             act_rescale_per_output=False, calibration_reload=False, report=100,
                             export_nb_stimuli_max= -1, wt_round_mode = "NONE",
                             b_round_mode="NONE", c_round_mode="NONE", find_lr=0):
    if wt_round_mode not in N2D2.WeightsApprox.__members__.keys():
        raise n2d2.error_handler.WrongValue("wt_round_mode", wt_round_mode,
        ", ".join(N2D2.WeightsApprox.__members__.keys()))
    N2D2_wt_round_mode = N2D2.WeightsApprox.__members__[wt_round_mode]
    if b_round_mode not in N2D2.WeightsApprox.__members__.keys():
        raise n2d2.error_handler.WrongValue("b_round_mode", b_round_mode,
        ", ".join(N2D2.WeightsApprox.__members__.keys()))
    N2D2_b_round_mode = N2D2.WeightsApprox.__members__[b_round_mode]
    if c_round_mode not in N2D2.WeightsApprox.__members__.keys():
        raise n2d2.error_handler.WrongValue("c_round_mode", c_round_mode,
        ", ".join(N2D2.WeightsApprox.__members__.keys()))
    N2D2_c_round_mode = N2D2.WeightsApprox.__members__[c_round_mode]

    if act_scaling_mode not in N2D2.ScalingMode.__members__.keys():
        raise n2d2.error_handler.WrongValue("act_scaling_mode", act_scaling_mode, ", ".join(N2D2.ScalingMode.__members__.keys()))
    N2D2_act_scaling_mode = N2D2.ScalingMode.__members__[act_scaling_mode]
    if act_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise n2d2.error_handler.WrongValue("act_clipping_mode", act_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    N2D2_act_clipping_mode = N2D2.ClippingMode.__members__[act_clipping_mode]
    if wt_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise n2d2.error_handler.WrongValue("wt_clipping_mode", wt_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    N2D2_wt_clipping_mode = N2D2.ClippingMode.__members__[wt_clipping_mode]

    return n2d2.n2d2_interface.Options(
        gen_export=gen_export,
        nb_bits=nb_bits,
        qat_SAT=qat_SAT,
        export_no_unsigned=export_no_unsigned,
        calibration=calibration,
        export_no_cross_layer_equalization=export_no_cross_layer_equalization,
        wt_clipping_mode=N2D2_wt_clipping_mode,
        act_clipping_mode=N2D2_act_clipping_mode,
        act_scaling_mode=N2D2_act_scaling_mode,
        act_quantile_value=act_quantile_value,
        act_rescale_per_output=act_rescale_per_output,
        calibration_reload=calibration_reload,
        report=report,
        export_nb_stimuli_max=export_nb_stimuli_max,
        wt_round_mode=N2D2_wt_round_mode,
        b_round_mode=N2D2_b_round_mode,
        c_round_mode=N2D2_c_round_mode,
        find_lr=find_lr).N2D2()

def _export_deepnet_operation(deepnet_cell: n2d2.cells.DeepNetCell,
        provider: n2d2.provider.Provider=None,
        **kwargs) -> None :
    """Retrieve N2D2 DeepNet and prepare it for an export.
    """

    N2D2_deepnet = deepnet_cell.get_embedded_deepnet().N2D2()
    N2D2_deepnet.initialize()

    if provider is not None:
        N2D2_provider = provider.N2D2()
        N2D2_database = N2D2_provider.getDatabase()
        N2D2_deepnet.setDatabase(N2D2_database)
        N2D2_deepnet.setStimuliProvider(N2D2_provider)

    if "calibration" in kwargs and kwargs["calibration"] != 0 and \
    N2D2_deepnet.getDatabase().getNbStimuli() == 0:
        raise ValueError("Cannot calibrate the network with an empty database,\
             please add a dataprovider to run the calibration.")

    if len(N2D2_deepnet.getTargets()) == 0:
        # No target associated to the DeepNet
        # We create a Target for the last cell of the network
        last_cell = deepnet_cell[-1].N2D2()
        N2D2_target =  N2D2.TargetScore("Target", last_cell, provider.N2D2())
        N2D2_deepnet.addTarget(N2D2_target)
    elif provider is not None:
        for target in N2D2_deepnet.getTargets():
            target.setStimuliProvider(provider.N2D2())

    if (N2D2_deepnet.getDatabase().getNbStimuli(N2D2.Database.StimuliSet.__members__["Validation"]) > 0):
        N2D2_deepnet.exportNetworkFreeParameters("weights_validation")
    else:
        N2D2_deepnet.exportNetworkFreeParameters("weights")
    return N2D2_deepnet


def export_c(deepnet_cell: n2d2.cells.DeepNetCell,
             provider: n2d2.provider.Provider=None,
             **kwargs) -> None:
    """Generate a C export of the neural network.

    :param deepnet_cell: The Neural network you want to export.
    :type deepnet_cell: :py:class:`n2d2.cells.DeepNetCell`
    :param provider: Data provider to use for calibration, default=None
    :type provider: :py:class:`n2d2.provider.DataProvider`, optional
    :param nb_bits: Number of bits per weight for exports, default=8
    :type nb_bits: int, optional
    :param qat_sat: Fuse a QAT trained with SAT method, default=False
    :type qat_sat: bool, optional
    :param export_no_unsigned: If True, disable the use of unsigned data type in integer exports, default=False
    :type export_no_unsigned: bool, optional
    :param calibration: The number of stimuli used for the calibration (``0`` = no calibration, ``-1`` = use the full test dataset), default=0
    :type calibration: int, optional
    :param export_no_cross_layer_equalization: If True, disable the use of cross layer equalization in integer exports, default=False
    :type export_no_cross_layer_equalization: bool, optional
    :param wt_clipping_mode: Weights clipping mode on export, can be ``NONE``, ``MSE`` or ``KL_DIVERGENCE``, default="NONE"
    :type wt_clipping_mode: str, optional
    :param act_clipping_mode: activation clipping mode on export, can be ``NONE``, ``MSE`` or ``KL_DIVERGENCE`` or ``Quantile``, default="MSE"
    :type act_clipping_mode: str, optional
    :param act_scaling_mode: activation scaling mode on export, can be ``NONE``, ``FIXED_MULT16``, ``SINGLE_SHIFT`` or ``DOUBLE_SHIFT``, default="SINGLE_SHIFT"
    :type act_scaling_mode: str, optional
    :param act_quantile_value: Quantile value for ``Quantile`` clipping mode, default=0.9999
    :type act_quantile_value: float, optional
    :param act_rescale_per_output: If True, rescale activation per output on export, default=False
    :type act_rescale_per_output: bool, optional
    :param calibration_reload: If True, reload and reuse the data of a previous calibration, default=False
    :type calibration_reload: bool, optional
    :param report: Number of steps between reportings, default=100
    :type report: int, optional
    :param export_nb_stimuli_max: Maximum number of stimuli to export (0 = no dataset export, -1 = unlimited), default=-1
    :type export_nb_stimuli_max: int, optional
    :param wt_round_mode: Weights clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
    :type wt_round_mode: str, optional
    :param b_round_mode: Biases clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
    :type b_round_mode: str, optional
    :param c_round_mode: Clip clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
    :type c_round_mode: str, optional
    :param find_lr: Find an appropriate learning rate over a number of iterations, default=0
    :type find_lr: int, optional
    """

    if "act_scaling_mode" in kwargs:
        if kwargs["act_scaling_mode"]=="FLOAT_MULT":
            raise ValueError("C export doesn't support FLOAT_MULT scaling.")
    else:
        kwargs["act_scaling_mode"]="SINGLE_SHIFT" # Default value

    N2D2_option = _parse_export_parameters(gen_export="C", **kwargs)
    N2D2_deepnet = _export_deepnet_operation(deepnet_cell, provider, **kwargs)
    N2D2.generateExport(N2D2_option, N2D2_deepnet)

def export_cpp(deepnet_cell: n2d2.cells.DeepNetCell,
               provider: n2d2.provider.Provider=None,
               **kwargs) -> None:
    """Generate a CPP export of the neural network.

    :param deepnet_cell: The Neural network you want to export.
    :type deepnet_cell: :py:class:`n2d2.cells.DeepNetCell`
    :param provider: Data provider to use for calibration, default=None
    :type provider: :py:class:`n2d2.provider.DataProvider`, optional
    :param nb_bits: Number of bits per weight for exports, default=8
    :type nb_bits: int, optional
    :param qat_sat: Fuse a QAT trained with SAT method, default=False
    :type qat_sat: bool, optional
    :param export_no_unsigned: If True, disable the use of unsigned data type in integer exports, default=False
    :type export_no_unsigned: bool, optional
    :param calibration: The number of stimuli used for the calibration (``0`` = no calibration, ``-1`` = use the full test dataset), default=0
    :type calibration: int, optional
    :param export_no_cross_layer_equalization: If True, disable the use of cross layer equalization in integer exports, default=False
    :type export_no_cross_layer_equalization: bool, optional
    :param wt_clipping_mode: Weights clipping mode on export, can be ``NONE``, ``MSE`` or ``KL_DIVERGENCE``, default="NONE"
    :type wt_clipping_mode: str, optional
    :param act_clipping_mode: activation clipping mode on export, can be ``NONE``, ``MSE`` or ``KL_DIVERGENCE`` or ``Quantile``, default="MSE"
    :type act_clipping_mode: str, optional
    :param act_scaling_mode: activation scaling mode on export, can be ``NONE``, ``FLOAT_MULT``, ``FIXED_MULT16``, ``SINGLE_SHIFT`` or ``DOUBLE_SHIFT``, default="FLOAT_MULT"
    :type act_scaling_mode: str, optional
    :param act_quantile_value: Quantile value for ``Quantile`` clipping mode, default=0.9999
    :type act_quantile_value: float, optional
    :param act_rescale_per_output: If True, rescale activation per output on export, default=False
    :type act_rescale_per_output: bool, optional
    :param calibration_reload: If True, reload and reuse the data of a previous calibration, default=False
    :type calibration_reload: bool, optional
    :param report: Number of steps between reportings, default=100
    :type report: int, optional
    :param export_nb_stimuli_max: Maximum number of stimuli to export (0 = no dataset export, -1 = unlimited), default=-1
    :type export_nb_stimuli_max: int, optional
    :param wt_round_mode: Weights clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
    :type wt_round_mode: str, optional
    :param b_round_mode: Biases clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
    :type b_round_mode: str, optional
    :param c_round_mode: Clip clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
    :type c_round_mode: str, optional
    :param find_lr: Find an appropriate learning rate over a number of iterations, default=0
    :type find_lr: int, optional
    """
    N2D2_option = _parse_export_parameters(gen_export="CPP", **kwargs)
    N2D2_deepnet = _export_deepnet_operation(deepnet_cell, provider, **kwargs)

    N2D2.generateExport(N2D2_option, N2D2_deepnet)



def export_tensor_rt(deepnet_cell: n2d2.cells.DeepNetCell,
                provider: n2d2.provider.Provider=None,
                **kwargs) -> None:
    """Generate a TensorRT export of the neural network.

    :param deepnet_cell: The Neural network you want to export.
    :type deepnet_cell: :py:class:`n2d2.cells.DeepNetCell`
    :param provider: Data provider to use for calibration, default=None
    :type provider: :py:class:`n2d2.provider.DataProvider`, optional
    :param nb_bits: Number of bits per weight for exports, default=8
    :type nb_bits: int, optional
    :param qat_sat: Fuse a QAT trained with SAT method, default=False
    :type qat_sat: bool, optional
    :param export_no_unsigned: If True, disable the use of unsigned data type in integer exports, default=False
    :type export_no_unsigned: bool, optional
    :param calibration: The number of stimuli used for the calibration (``0`` = no calibration, ``-1`` = use the full test dataset), default=0
    :type calibration: int, optional
    :param export_no_cross_layer_equalization: If True, disable the use of cross layer equalization in integer exports, default=False
    :type export_no_cross_layer_equalization: bool, optional
    :param wt_clipping_mode: Weights clipping mode on export, can be ``NONE``, ``MSE`` or ``KL-Diveregence``, default="NONE"
    :type wt_clipping_mode: str, optional
    :param act_clipping_mode: activation clipping mode on export, can be ``NONE``, ``MSE`` or ``KL-Divergence`` or ``Quantile``, default="MSE"
    :type act_clipping_mode: str, optional
    :param act_scaling_mode: activation scaling mode on export, can be ``NONE``, ``FLOAT_MULT``, ``FIXED_MULT16``, ``SINGLE_SHIFT`` or ``DOUBLE_SHIFT``, default="FLOAT_MULT"
    :type act_scaling_mode: str, optional
    :param act_quantile_value: Quantile value for ``Quantile`` clipping mode, default=0.9999
    :type act_quantile_value: float, optional
    :param act_rescale_per_output: If True, rescale activation per output on export, default=False
    :type act_rescale_per_output: bool, optional
    :param calibration_reload: If True, reload and reuse the data of a previous calibration, default=False
    :type calibration_reload: bool, optional
    :param report: Number of steps between reportings, default=100
    :type report: int, optional
    :param export_nb_stimuli_max: Maximum number of stimuli to export (0 = no dataset export, -1 = unlimited), default=-1
    :type export_nb_stimuli_max: int, optional
    :param wt_round_mode: Weights clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
    :type wt_round_mode: str, optional
    :param b_round_mode: Biases clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
    :type b_round_mode: str, optional
    :param c_round_mode: Clip clipping mode on export, can be ``NONE``, ``RINTF``, default="NONE"
    :type c_round_mode: str, optional
    :param find_lr: Find an appropriate learning rate over a number of iterations, default=0
    :type find_lr: int, optional
    """
    N2D2_option = _parse_export_parameters(gen_export="CPP_TensorRT", **kwargs)
    N2D2_deepnet = _export_deepnet_operation(deepnet_cell, provider, **kwargs)

    N2D2.generateExport(N2D2_option, N2D2_deepnet)
