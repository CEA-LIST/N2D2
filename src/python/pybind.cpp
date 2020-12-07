/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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
*/

#ifdef PYBIND
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace N2D2 {
//utils
void init_Random(py::module&);
void init_Parameterizable(py::module&);
void init_WindowFunction(py::module&);
void init_Kernel(py::module&);
void init_CudaContext(py::module&);
void init_Tensor(py::module&);
void init_CudaTensor(py::module&);
void init_Network(py::module&);
void init_Database(py::module&);
void init_StimuliProvider(py::module&);
void init_Cell(py::module&);
void init_Cell_Frame_Top(py::module&);
void init_Cell_Frame(py::module&);
void init_Cell_Frame_CUDA(py::module&);
void init_Target(py::module&);
void init_TargetScore(py::module&);
void init_DeepNet(py::module&);
void init_DeepNetGenerator(py::module&);
void init_MNIST_IDX_Database(py::module&);
void init_Environment(py::module&);
void init_ConvCell(py::module&);
void init_ConvCell_Frame(py::module&);
void init_FcCell(py::module&);
void init_FcCell_Frame(py::module&);
void init_FcCell_Frame_CUDA(py::module&);
void init_SoftmaxCell(py::module&);
void init_SoftmaxCell_Frame(py::module&);
void init_SoftmaxCell_Frame_CUDA(py::module&);

//Ativation
void init_Activation(py::module&);
void init_TanhActivation(py::module&);
void init_TanhActivation_Frame(py::module&);
void init_TanhActivation_Frame_CUDA(py::module &m);
void init_RectifierActivation(py::module&);
void init_RectifierActivation_Frame(py::module&);
void init_RectifierActivation_Frame_CUDA(py::module&);
void init_LinearActivation(py::module&);
void init_LinearActivation_Frame(py::module&);
void init_LinearActivation_Frame_CUDA(py::module&);
void init_SwishActivation(py::module&);
void init_SwishActivation_Frame(py::module&);
void init_SwishActivation_Frame_CUDA(py::module&);
void init_SoftplusActivation(py::module&);
void init_SoftplusActivation_Frame(py::module&);
void init_SoftplusActivation_Frame_CUDA(py::module&);
void init_SaturationActivation(py::module&);
void init_SaturationActivation_Frame(py::module&);
void init_SaturationActivation_Frame_CUDA(py::module&);
void init_LogisticActivation(py::module&);
void init_LogisticActivation_Frame(py::module&);
void init_LogisticActivation_Frame_CUDA(py::module&);

// Transformation
void init_Transformation(py::module&);
void init_AffineTransformation(py::module&);
void init_ApodizationTransformation(py::module&);
void init_ChannelExtractionTransformation(py::module&);
void init_ColorSpaceTransformation(py::module&);
void init_CompressionNoiseTransformation(py::module&);
void init_DCTTransformation(py::module&);
void init_DFTTransformation(py::module&);
void init_DistortionTransformation(py::module&);
void init_EqualizeTransformation(py::module&);
void init_ExpandLabelTransformation(py::module&);
void init_FlipTransformation(py::module&);
void init_FilterTransformation(py::module &m);
void init_PadCropTransformation(py::module&);
void init_GradientFilterTransformation(py::module&);
void init_LabelExtractionTransformation(py::module&);
void init_LabelSliceExtractionTransformation(py::module&);
void init_MagnitudePhaseTransformation(py::module&);
void init_MorphologicalReconstructionTransformation(py::module&);
void init_MorphologyTransformation(py::module&);
void init_NormalizeTransformation(py::module&);
void init_RandomAffineTransformation(py::module&);
void init_RangeAffineTransformation(py::module&);
void init_RangeClippingTransformation(py::module&);
void init_RescaleTransformation(py::module&);
void init_ReshapeTransformation(py::module&);
void init_SliceExtractionTransformation(py::module&);
void init_ThresholdTransformation(py::module&);
// void init_TrimTransformation(py::module &m);
void init_WallisFilterTransformation(py::module&);
void init_CompositeTransformation(py::module &);


void init_Solver(py::module&);
void init_SGDSolver(py::module&);
void init_SGDSolver_Frame(py::module&);
void init_SGDSolver_Frame_CUDA(py::module&);
void init_Filler(py::module&);
void init_HeFiller(py::module&);



PYBIND11_MODULE(N2D2, m) {
    //utils
    init_WindowFunction(m);
    init_Random(m);
    init_Parameterizable(m);
    init_Kernel(m);

    init_CudaContext(m);
    init_Tensor(m);
    init_CudaTensor(m);
    init_Network(m);
    init_Database(m);
    init_StimuliProvider(m);
    init_Cell(m);
    init_Cell_Frame_Top(m);
    init_Cell_Frame(m);
    init_Cell_Frame_CUDA(m);
    init_Target(m);
    init_TargetScore(m);
    init_DeepNet(m);
    init_DeepNetGenerator(m);
    init_MNIST_IDX_Database(m);
    init_Environment(m);
    init_ConvCell(m);
    init_ConvCell_Frame(m);
    init_FcCell(m);
    init_FcCell_Frame(m);
    init_FcCell_Frame_CUDA(m);
    init_SoftmaxCell(m);
    init_SoftmaxCell_Frame(m);
    init_SoftmaxCell_Frame_CUDA(m);

    //Activation
    init_Activation(m);
    init_TanhActivation(m);
    init_TanhActivation_Frame(m);
    init_TanhActivation_Frame_CUDA(m);
    init_RectifierActivation(m);
    init_RectifierActivation_Frame(m);
    init_RectifierActivation_Frame_CUDA(m);
    init_LinearActivation(m);
    init_LinearActivation_Frame(m);
    init_LinearActivation_Frame_CUDA(m);
    init_SwishActivation(m);
    init_SwishActivation_Frame(m);
    init_SwishActivation_Frame_CUDA(m);
    init_SoftplusActivation(m);
    init_SoftplusActivation_Frame(m);
    init_SoftplusActivation_Frame_CUDA(m);
    init_SaturationActivation(m);
    init_SaturationActivation_Frame(m);
    init_SaturationActivation_Frame_CUDA(m);
    init_LogisticActivation(m);
    init_LogisticActivation_Frame(m);
    init_LogisticActivation_Frame_CUDA(m);


    // Transformation
    init_Transformation(m);
    init_AffineTransformation(m);
    init_ApodizationTransformation(m);
    init_ChannelExtractionTransformation(m);
    init_ColorSpaceTransformation(m);
    init_CompressionNoiseTransformation(m);
    init_DCTTransformation(m);
    init_DFTTransformation(m);
    init_DistortionTransformation(m);
    init_EqualizeTransformation(m);
    init_ExpandLabelTransformation(m);
    init_FlipTransformation(m);
    init_FilterTransformation(m);
    init_PadCropTransformation(m);
    init_GradientFilterTransformation(m);
    init_LabelExtractionTransformation(m);
    init_LabelSliceExtractionTransformation(m);
    init_MagnitudePhaseTransformation(m);
    init_MorphologicalReconstructionTransformation(m);
    init_MorphologyTransformation(m);
    init_NormalizeTransformation(m);
    init_RandomAffineTransformation(m);
    init_RangeAffineTransformation(m);
    init_RangeClippingTransformation(m);
    init_RescaleTransformation(m);
    init_ReshapeTransformation(m);
    init_SliceExtractionTransformation(m);
    init_ThresholdTransformation(m);
    init_WallisFilterTransformation(m);
    // init_TrimTransformation(m);
    init_CompositeTransformation(m);


    init_Solver(m);
    init_SGDSolver(m);
    init_SGDSolver_Frame(m);
    init_SGDSolver_Frame_CUDA(m);
    init_Filler(m);
    init_HeFiller(m);

}
}

#endif
