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

import N2D2
from os import mkdir, remove
from os.path import exists
from n2d2 import error_handler, add_docstring, check_types, template_docstring
from n2d2.n2d2_interface import Options
from n2d2.quantizer import PTQ
from n2d2.cells import DeepNetCell, NeuralNetworkCell
from n2d2.provider import Provider
from n2d2.deepnet import associate_provider_to_deepnet

available_export = ["C", "CPP", "CPP_TensorRT", "CPP_STM32"]

@check_types
def _gen_exportable_cell_matrix(export_name: str)-> str:
    if export_name not in available_export:
            raise error_handler.WrongValue("export_name", export_name, available_export)

    map_cell_exportable =  {}

    for cell in NeuralNetworkCell.__subclasses__():
        if hasattr(cell, "is_exportable_to"):
            map_cell_exportable[cell.__name__] = cell.is_exportable_to(export_name)
        else:
            print(f'Warning : {cell.__name__} does not have is_exportable_to method')

    max_len = max([(len(cell_name)) for cell_name in map_cell_exportable.keys()])
    
    # Configurable variables 
    title_1 = " Cell Name "
    title_2 = " Available "
    str_exportable = "Yes" + " "*(len(title_2)-4)
    str_not_exportable = "No"+ " "*(len(title_2)-3)
    max_len = max(max_len, len(title_1)) + 2
    sep_line = '+' + '-'*max_len + "+" + "-"*len(title_2) +"+\n"
    # Generating the matrix !
    matrix = sep_line
    matrix += '|' + title_1 + " " * (max_len-(len(title_1))) + "|" + title_2 + "|\n"
    matrix += '+' + '='*max_len + "+" + "="*len(title_2) +"+\n"
    for cell_name, exportable in map_cell_exportable.items():
        matrix += '| ' + cell_name + " " * (max_len-(len(cell_name)+1))
        matrix += "| " + (str_exportable if exportable else str_not_exportable) + "|\n"
        matrix += sep_line
    return matrix

@check_types
@template_docstring("export_list", ", ".join(available_export))
def list_exportable_cell(export_name: str)-> None:
    """Print a list of exportable cells.

    :param export_name: Can be one of : {export_list}.
    :type export_name: str
    """
    print(_gen_exportable_cell_matrix(export_name))

# This is the default docstring for export.
# Parameters description can be override by the docstring defined inside the export function.
# The docstring header is always the one defined in the function !
export_doc_string = \
"""
:param deepnet_cell: The Neural network you want to export.
:type deepnet_cell: :py:class:`n2d2.cells.DeepNetCell`
:param provider: Data provider to use for calibration, default=None
:type provider: :py:class:`n2d2.provider.DataProvider`, optional
:param nb_bits: Number of bits for the weights and signals. Must be ``8``, ``16``, ``32`` or ``64`` for integer export, or ``-32``, ``-64`` for floating point export, default=8
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

def _parse_export_parameters(gen_export:str=None, nb_bits:int=8, qat_SAT:bool=False,
                             export_no_unsigned:bool=False, calibration:int=0,
                             export_no_cross_layer_equalization:bool=False,
                             wt_clipping_mode:str="NONE", act_clipping_mode:str="MSE",
                             act_scaling_mode:str="FLOAT_MULT", act_quantile_value:float=0.9999,
                             act_rescale_per_output:bool=False, calibration_reload:bool=False, report:int=100,
                             export_nb_stimuli_max:int= -1, wt_round_mode:str= "NONE",
                             b_round_mode:str="NONE", c_round_mode:str="NONE", find_lr:int=0, log_kernels:bool=False):
    if wt_round_mode not in N2D2.WeightsApprox.__members__.keys():
        raise error_handler.WrongValue("wt_round_mode", wt_round_mode,
        ", ".join(N2D2.WeightsApprox.__members__.keys()))
    N2D2_wt_round_mode = N2D2.WeightsApprox.__members__[wt_round_mode]
    if b_round_mode not in N2D2.WeightsApprox.__members__.keys():
        raise error_handler.WrongValue("b_round_mode", b_round_mode,
        ", ".join(N2D2.WeightsApprox.__members__.keys()))
    N2D2_b_round_mode = N2D2.WeightsApprox.__members__[b_round_mode]
    if c_round_mode not in N2D2.WeightsApprox.__members__.keys():
        raise error_handler.WrongValue("c_round_mode", c_round_mode,
        ", ".join(N2D2.WeightsApprox.__members__.keys()))
    N2D2_c_round_mode = N2D2.WeightsApprox.__members__[c_round_mode]

    if act_scaling_mode not in N2D2.ScalingMode.__members__.keys():
        raise error_handler.WrongValue("act_scaling_mode", act_scaling_mode, ", ".join(N2D2.ScalingMode.__members__.keys()))
    N2D2_act_scaling_mode = N2D2.ScalingMode.__members__[act_scaling_mode]
    if act_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise error_handler.WrongValue("act_clipping_mode", act_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    N2D2_act_clipping_mode = N2D2.ClippingMode.__members__[act_clipping_mode]
    if wt_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise error_handler.WrongValue("wt_clipping_mode", wt_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    N2D2_wt_clipping_mode = N2D2.ClippingMode.__members__[wt_clipping_mode]

    return Options(
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
        find_lr=find_lr,
        log_kernels=log_kernels).N2D2()

@check_types
def _generate_export(deepnet_cell:DeepNetCell, provider:Provider=None, **kwargs):

    export_folder_name = None if "export_folder_name" not in kwargs else kwargs.pop("export_folder_name")

    N2D2_option = _parse_export_parameters(**kwargs)
    N2D2_deepnet = deepnet_cell.get_embedded_deepnet().N2D2()

    if N2D2_option.calibration != 0:
        if "nb_bits" not in kwargs:
            kwargs["nb_bits"] = N2D2_option.nb_bits
        PTQ(deepnet_cell, provider=provider, **kwargs)
    else:
        # Graph otpimisations are done during calibration.
        # If we do not call calibration, we do graph optimisation now !
        if provider is not None:
            associate_provider_to_deepnet(N2D2_deepnet, provider.N2D2())
        N2D2_deepnet.fuseBatchNorm()
        N2D2_deepnet.removeDropout()

    if N2D2_deepnet.getStimuliProvider().getDatabase().getNbStimuli() == 0 \
            and N2D2_option.export_nb_stimuli_max != 0:
        print("Warning : The export generated has no stimuli associated to it.")

    if not deepnet_cell.is_integral() and N2D2_option.nb_bits > 0:
        raise RuntimeError(f"You need to calibrate the network to export it in {abs(N2D2_option.nb_bits)} bits integer" \
                            "set the 'calibration' option to something else than 0 or quantize the deepnetcell before export.")

    if not export_folder_name:
        export_folder_name = f"export_{N2D2_option.gen_export}_{'int' if N2D2_option.nb_bits > 0 else 'float'}{abs(N2D2_option.nb_bits)}"

    if not exists(export_folder_name):
        mkdir(export_folder_name)

    N2D2.generateExportFromCalibration(N2D2_option, N2D2_deepnet, fileName=export_folder_name)

@template_docstring("exportable_cells", "\n"+_gen_exportable_cell_matrix("C"))
@add_docstring(export_doc_string)
@check_types
def export_c(deepnet_cell: DeepNetCell,
             provider: Provider=None,
             **kwargs) -> None:
    """Generate a C export of the neural network.

    List of exportable cells :{exportable_cells}

    :param act_scaling_mode: activation scaling mode on export, can be ``NONE``, ``FIXED_MULT16``, ``SINGLE_SHIFT`` or ``DOUBLE_SHIFT``, default="SINGLE_SHIFT"
    :type act_scaling_mode: str, optional
    """

    if "act_scaling_mode" in kwargs:
        if kwargs["act_scaling_mode"]=="FLOAT_MULT":
            raise ValueError("C export doesn't support FLOAT_MULT scaling.")
    else:
        kwargs["act_scaling_mode"]="SINGLE_SHIFT" # Default value

    kwargs["gen_export"] = "C"
    _generate_export(deepnet_cell, provider, **kwargs)

@template_docstring("exportable_cells", "\n"+_gen_exportable_cell_matrix("CPP"))
@add_docstring(export_doc_string)
@check_types
def export_cpp(deepnet_cell: DeepNetCell,
               provider: Provider=None,
               optimize_buffer_memory: bool=True,
               **kwargs) -> None:
    """Generate a CPP export of the neural network.

    List of exportable cells :{exportable_cells}

    :param optimize_buffer_memory: If False deactivate memory optimization, default=True
    :type optimize_buffer_memory: bool, optional
    """
    kwargs["gen_export"] = "CPP"

    if not optimize_buffer_memory:
        extra_params_path = "./tmp.ini"
        with open(extra_params_path, "w") as param_file:
            param_file.write("OptimizeBufferMemory=0\n")
        N2D2.DeepNetExport.setExportParameters(extra_params_path)
        remove(extra_params_path)

    _generate_export(deepnet_cell, provider, **kwargs)

@template_docstring("exportable_cells", "\n"+_gen_exportable_cell_matrix("CPP_TensorRT"))
@add_docstring(export_doc_string)
@check_types
def export_tensor_rt(deepnet_cell: DeepNetCell,
                provider: Provider=None,
                **kwargs) -> None:
    """Generate a TensorRT export of the neural network.

    List of exportable cells :{exportable_cells}

    :param nb_bits: Only 32 floating point precision is available for this export. You can calibrate your network later with the export tools, default=-32
    :type nb_bits: int, optional
    """
    kwargs["gen_export"] = "CPP_TensorRT"
    if "nb_bits" not in kwargs:
        kwargs["nb_bits"] = -32
    else:
        if kwargs["nb_bits"] != -32:
            raise ValueError("The TensorRT export only support 32 floating point precision.\
Calibration needs to be done once the export is generated (see : https://cea-list.github.io/N2D2-docs/export/TensorRT.html)")
    _generate_export(deepnet_cell, provider, **kwargs)

@template_docstring("exportable_cells", "\n"+_gen_exportable_cell_matrix("CPP_STM32"))
@add_docstring(export_doc_string)
@check_types
def export_cpp_stm32(deepnet_cell: DeepNetCell,
                    provider: Provider=None,
                    optimize_buffer_memory: bool=True,
                    **kwargs) -> None:
    """Generate a CPP export for STM32 of the neural network.

    List of exportable cells :{exportable_cells}

    :param optimize_buffer_memory: If False deactivate memory optimization, default=True
    :type optimize_buffer_memory: bool, optional
    """
    kwargs["gen_export"] = "CPP_STM32"

    if not optimize_buffer_memory:
        extra_params_path = "./tmp.ini"
        with open(extra_params_path, "w") as param_file:
            param_file.write("OptimizeBufferMemory=0\n")
        N2D2.DeepNetExport.setExportParameters(extra_params_path)
        remove(extra_params_path)

    _generate_export(deepnet_cell, provider, **kwargs)
