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

def _export(N2D2_deepnet, gen_export, nb_bits=8, qat_SAT=False, 
                 export_no_unsigned=False, calibration=0, 
                 export_no_cross_layer_equalization=False, 
                 wt_clipping_mode="MSE", act_clipping_mode="NONE", 
                 act_scaling_mode="FLOAT_MULT", act_quantile_value=0.9999,
                 act_rescale_per_output=False, calibration_reload=False, report=100, 
                 export_nb_stimuli_max= -1, wt_round_mode = "NONE", 
                 b_round_mode="NONE", c_round_mode="NONE", find_lr=0):
    if wt_round_mode not in N2D2.WeightsApprox.__members__.keys():
            raise n2d2.error_handler.WrongValue("wt_round_mode", wt_round_mode, ", ".join(N2D2.WeightsApprox.__members__.keys()))
    else:
        N2D2_wt_round_mode = N2D2.WeightsApprox.__members__[wt_round_mode]
    if b_round_mode not in N2D2.WeightsApprox.__members__.keys():
        raise n2d2.error_handler.WrongValue("b_round_mode", b_round_mode, ", ".join(N2D2.WeightsApprox.__members__.keys()))
    else:
        N2D2_b_round_mode = N2D2.WeightsApprox.__members__[b_round_mode]
    if c_round_mode not in N2D2.WeightsApprox.__members__.keys():
        raise n2d2.error_handler.WrongValue("c_round_mode", c_round_mode, ", ".join(N2D2.WeightsApprox.__members__.keys()))
    else:
        N2D2_c_round_mode = N2D2.WeightsApprox.__members__[c_round_mode]

    if act_scaling_mode not in N2D2.ScalingMode.__members__.keys():
        raise n2d2.error_handler.WrongValue("act_scaling_mode", act_scaling_mode, ", ".join(N2D2.ScalingMode.__members__.keys()))
    else:
        N2D2_act_scaling_mode = N2D2.ScalingMode.__members__[act_scaling_mode]


    if act_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise n2d2.error_handler.WrongValue("act_clipping_mode", act_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    else:
        N2D2_act_clipping_mode = N2D2.ClippingMode.__members__[act_clipping_mode]
    if wt_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise n2d2.error_handler.WrongValue("wt_clipping_mode", wt_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    else:
        N2D2_wt_clipping_mode = N2D2.ClippingMode.__members__[wt_clipping_mode]

    parameters = n2d2.n2d2_interface.Options(
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
                 find_lr=find_lr
                )
    N2D2_deepnet.initialize()
    if (N2D2_deepnet.getDatabase().getNbStimuli(N2D2.Database.StimuliSet.__members__["Validation"]) > 0):
        N2D2_deepnet.exportNetworkFreeParameters("weights_validation")
    else:
        N2D2_deepnet.exportNetworkFreeParameters("weights")
    N2D2.generateExport(parameters.N2D2(), N2D2_deepnet)


def export_c(N2D2_deepnet, nb_bits=8, qat_SAT=False, 
                 export_no_unsigned=False, calibration=0, 
                 export_no_cross_layer_equalization=False, 
                 wt_clipping_mode="MSE", act_clipping_mode="NONE", 
                 act_scaling_mode="FLOAT_MULT", act_quantile_value=0.9999,
                 act_rescale_per_output=False, calibration_reload=False, report=100, 
                 export_nb_stimuli_max= -1, wt_round_mode = "NONE", 
                 b_round_mode="NONE", c_round_mode="NONE", find_lr=0):
    _export(N2D2_deepnet, "C",nb_bits=nb_bits, 
                 qat_SAT=qat_SAT, export_no_unsigned=export_no_unsigned, 
                 calibration=calibration,
                 export_no_cross_layer_equalization=export_no_cross_layer_equalization, 
                 wt_clipping_mode=wt_clipping_mode, 
                 act_clipping_mode=act_clipping_mode, 
                 act_scaling_mode=act_scaling_mode, 
                 act_quantile_value=act_quantile_value,
                 act_rescale_per_output=act_rescale_per_output, 
                 calibration_reload=calibration_reload, 
                 report=report, 
                 export_nb_stimuli_max=export_nb_stimuli_max, 
                 wt_round_mode=wt_round_mode, 
                 b_round_mode=b_round_mode, 
                 c_round_mode=c_round_mode, 
                 find_lr=find_lr)

def export_cpp(N2D2_deepnet, nb_bits=8, qat_SAT=False, 
                 export_no_unsigned=False, calibration=0, 
                 export_no_cross_layer_equalization=False, 
                 wt_clipping_mode="MSE", act_clipping_mode="NONE", 
                 act_scaling_mode="FLOAT_MULT", act_quantile_value=0.9999,
                 act_rescale_per_output=False, calibration_reload=False, report=100, 
                 export_nb_stimuli_max= -1, wt_round_mode = "NONE", 
                 b_round_mode="NONE", c_round_mode="NONE", find_lr=0):
    _export(N2D2_deepnet, "CPP",nb_bits=nb_bits, 
                 qat_SAT=qat_SAT, export_no_unsigned=export_no_unsigned, 
                 calibration=calibration,
                 export_no_cross_layer_equalization=export_no_cross_layer_equalization, 
                 wt_clipping_mode=wt_clipping_mode, 
                 act_clipping_mode=act_clipping_mode, 
                 act_scaling_mode=act_scaling_mode, 
                 act_quantile_value=act_quantile_value,
                 act_rescale_per_output=act_rescale_per_output, 
                 calibration_reload=calibration_reload, 
                 report=report, 
                 export_nb_stimuli_max=export_nb_stimuli_max, 
                 wt_round_mode=wt_round_mode, 
                 b_round_mode=b_round_mode, 
                 c_round_mode=c_round_mode, 
                 find_lr=find_lr)