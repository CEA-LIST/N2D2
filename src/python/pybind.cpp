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
void init_ConfusionMatrix(py::module&);
void init_Random(py::module&);
void init_Parameterizable(py::module&);
void init_WindowFunction(py::module&);
void init_Kernel(py::module&);
void init_IniParser(py::module&);

void init_Tensor(py::module&);
void init_Network(py::module&);
void init_StimuliProvider(py::module&);
void init_Cell(py::module&);
void init_Cell_Frame_Top(py::module&);
void init_Cell_Frame(py::module&);
void init_Target(py::module&);
void init_TargetScore(py::module&);

// DeepNet
void init_DeepNet(py::module&);
void init_DeepNetQuantization(py::module&);
void init_DeepNetExport(py::module&);
void init_DrawNet(py::module&);
//Database
void init_Database(py::module&);
void init_DIR_Database(py::module&);
void init_AER_Database(py::module&);
void init_MNIST_IDX_Database(py::module&);
void init_Actitracker_Database(py::module&);
void init_Caltech101_DIR_Database(py::module&);
void init_Caltech256_DIR_Database(py::module&);
void init_CelebA_Database(py::module&);
void init_CIFAR_Database(py::module&);
void init_CKP_Database(py::module&);
#ifdef JSONCPP
void init_Cityscapes_Database(py::module&);
#endif
void init_GTSDB_DIR_Database(py::module&);
void init_GTSRB_DIR_Database(py::module&);
void init_ILSVRC2012_Database(py::module&);
void init_IDX_Database(py::module&);
void init_IMDBWIKI_Database(py::module&);
void init_KITTI_Database(py::module&);
void init_KITTI_Object_Database(py::module&);
void init_KITTI_Road_Database(py::module&);
void init_LITISRouen_Database(py::module&);
void init_N_MNIST_Database(py::module&);
void init_DOTA_Database(py::module&);
void init_Fashion_MNIST_IDX_Database(py::module&);
void init_FDDB_Database(py::module&);
void init_Daimler_Database(py::module&);
void init_CaltechPedestrian_Database(py::module&);

void init_Scaling(py::module&);
void init_ScalingMode(py::module&);
void init_Histogram(py::module&);

//Activation
void init_Activation(py::module&);
void init_TanhActivation(py::module&);
void init_TanhActivation_Frame(py::module&);
void init_RectifierActivation(py::module&);
void init_RectifierActivation_Frame(py::module&);
void init_LinearActivation(py::module&);
void init_LinearActivation_Frame(py::module&);
void init_SwishActivation(py::module&);
void init_SwishActivation_Frame(py::module&);
void init_SoftplusActivation(py::module&);
void init_SoftplusActivation_Frame(py::module&);
void init_SaturationActivation(py::module&);
void init_SaturationActivation_Frame(py::module&);
void init_LogisticActivation(py::module&);
void init_LogisticActivation_Frame(py::module&);


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
void init_RandomResizeCropTransformation(py::module&);
void init_CustomTransformation(py::module&);
void init_CompositeTransformation(py::module &);

// Solver
void init_Solver(py::module&);
void init_SGDSolver(py::module&);
void init_SGDSolver_Frame(py::module&);
void init_AdamSolver(py::module&);
void init_AdamSolver_Frame(py::module&);

// Filler
void init_Filler(py::module&);
void init_HeFiller(py::module&);
void init_NormalFiller(py::module&);
void init_UniformFiller(py::module&);
void init_XavierFiller(py::module&);
void init_ConstantFiller(py::module&);

// Cell
void init_CellGenerator(py::module&);
void init_Cell(py::module&);
void init_Cell_Frame_Top(py::module&);
void init_Cell_Frame(py::module&);
void init_ConvCell(py::module&);
void init_ConvCell_Frame(py::module&);
void init_FcCell(py::module&);
void init_FcCell_Frame(py::module&);
void init_SoftmaxCell(py::module&);
void init_SoftmaxCell_Frame(py::module&);
void init_AnchorCell_Frame_Kernels_struct(py::module&);
void init_AnchorCell(py::module&);
void init_AnchorCell_Frame(py::module&);
void init_BatchNormCell(py::module&);
void init_BatchNormCell_Frame(py::module&);
void init_Cell_Spike(py::module&);
void init_ConvCell_Frame_Kernels(py::module&);
void init_ConvCell_Spike(py::module&);
void init_ConvCell_Spike_Analog(py::module&);
void init_ConvCell_Spike_PCM(py::module&);
void init_ConvCell_Spike_RRAM(py::module&);
void init_ConvCell_Transcode(py::module&);
void init_DeconvCell(py::module&);
void init_DeconvCell_Frame(py::module&);
void init_DropoutCell(py::module&);
void init_DropoutCell_Frame(py::module&);
void init_ElemWiseCell(py::module&);
void init_ElemWiseCell_Frame(py::module&);
void init_FMPCell(py::module&);
void init_FMPCell_Frame(py::module&);
void init_FcCell_Spike(py::module&);
void init_FcCell_Spike_Analog(py::module&);
void init_FcCell_Spike_PCM(py::module&);
void init_FcCell_Spike_RRAM(py::module&);
void init_LRNCell(py::module&);
void init_LRNCell_Frame(py::module&);
void init_LSTMCell(py::module&);
void init_NodeIn(py::module&);
void init_NodeOut(py::module&);
void init_NormalizeCell(py::module&);
void init_NormalizeCell_Frame(py::module&);
void init_ObjectDetCell(py::module&);
void init_ObjectDetCell_Frame(py::module&);
void init_PaddingCell(py::module&);
void init_PaddingCell_Frame(py::module&);
void init_PaddingCell_Frame_Kernels(py::module&);
void init_PoolCell(py::module&);
void init_PoolCell_Spike(py::module&);
void init_PoolCell_Frame(py::module&);
void init_PoolCell_Frame_Kernels(py::module&);
void init_PoolCell_Spike(py::module&);
void init_PoolCell_Transcode(py::module&);
void init_ProposalCell(py::module&);
void init_ProposalCell_Frame(py::module&);
void init_ROIPoolingCell(py::module&);
void init_ROIPoolingCell_Frame(py::module&);
void init_RPCell(py::module&);
void init_RPCell_Frame(py::module&);
void init_ResizeCell(py::module&);
void init_ResizeCell_Frame(py::module&);
void init_ReshapeCell(py::module&);
void init_ReshapeCell_Frame(py::module&);
void init_ScalingCell(py::module&);
void init_ScalingCell_Frame(py::module&);
void init_TargetBiasCell(py::module&);
void init_TargetBiasCell_Frame(py::module&);
void init_ThresholdCell(py::module&);
void init_ThresholdCell_Frame(py::module&);
void init_TransformationCell(py::module&);
void init_TransformationCell_Frame(py::module&);
void init_TransposeCell(py::module&);
void init_TransposeCell_Frame(py::module&);
void init_UnpoolCell(py::module&);
void init_UnpoolCell_Frame(py::module&);
void init_MappingGenerator(py::module&);
void init_ActivationCell(py::module&);
void init_ActivationCell_Frame(py::module&);

//Quantizer
void init_QuantizerCell(py::module&);
void init_QuantizerCell_Frame(py::module&);
void init_QuantizerActivation(py::module&);
void init_QuantizerActivation_Frame(py::module&);

void init_DeepNetGenerator(py::module&);

void init_helper(py::module&);

#ifdef CUDA
void init_CudaContext(py::module&);
void init_CudaTensor(py::module&);
void init_TanhActivation_Frame_CUDA(py::module &m);
void init_RectifierActivation_Frame_CUDA(py::module&);
void init_LinearActivation_Frame_CUDA(py::module&);
void init_SwishActivation_Frame_CUDA(py::module&);
void init_SoftplusActivation_Frame_CUDA(py::module&);
void init_SaturationActivation_Frame_CUDA(py::module&);
void init_LogisticActivation_Frame_CUDA(py::module&);
void init_SGDSolver_Frame_CUDA(py::module&);
void init_AdamSolver_Frame_CUDA(py::module&);
void init_Cell_Frame_CUDA(py::module&);
void init_FcCell_Frame_CUDA(py::module&);
void init_SoftmaxCell_Frame_CUDA(py::module&);
void init_AnchorCell_Frame_CUDA(py::module&);
void init_BatchNormCell_Frame_CUDA(py::module&);
void init_ConvCell_Frame_CUDA(py::module&);
void init_DeconvCell_Frame_CUDA(py::module&);
void init_DropoutCell_Frame_CUDA(py::module&);
void init_ElemWiseCell_Frame_CUDA(py::module&);
void init_FMPCell_Frame_CUDA(py::module&);
void init_LRNCell_Frame_CUDA(py::module&);
void init_LSTMCell_Frame_CUDA(py::module&);
void init_NormalizeCell_Frame_CUDA(py::module&);
void init_ObjectDetCell_Frame_CUDA(py::module&);
void init_PaddingCell_Frame_CUDA(py::module&);
void init_PoolCell_Frame_CUDA(py::module&);
void init_PoolCell_Frame_EXT_CUDA(py::module&);
void init_ProposalCell_Frame_CUDA(py::module&);
void init_ROIPoolingCell_Frame_CUDA(py::module&);
void init_RPCell_Frame_CUDA(py::module&);
void init_ResizeCell_Frame_CUDA(py::module&);
void init_ReshapeCell_Frame_CUDA(py::module&);
void init_ScalingCell_Frame_CUDA(py::module&);
void init_TargetBiasCell_Frame_CUDA(py::module&);
void init_ThresholdCell_Frame_CUDA(py::module&);
void init_TransformationCell_Frame_CUDA(py::module&);
void init_TransposeCell_Frame_CUDA(py::module&);
void init_UnpoolCell_Frame_CUDA(py::module&);
void init_ActivationCell_Frame_CUDA(py::module&);
void init_QuantizerCell_Frame_CUDA(py::module&);
void init_QuantizerActivation_Frame_CUDA(py::module&);
#endif // CUDA

void init_N2D2(py::module& m) {
    
    //utils
    init_ConfusionMatrix(m);
    init_WindowFunction(m);
    init_Random(m);
    init_Parameterizable(m);
    init_Kernel(m);
    init_IniParser(m);
    
    init_Scaling(m);
    init_ScalingMode(m);
    init_Histogram(m);

    init_Tensor(m);
    init_Network(m);
    
    init_StimuliProvider(m);
    init_Target(m);
    init_TargetScore(m);

    // Database
    init_Database(m);
    init_DIR_Database(m);
    init_AER_Database(m);
    init_MNIST_IDX_Database(m);
    init_Actitracker_Database(m);
    init_Caltech101_DIR_Database(m);
    init_Caltech256_DIR_Database(m);
    init_CaltechPedestrian_Database(m);
    init_CelebA_Database(m);
    init_CIFAR_Database(m);
    init_CKP_Database(m);
    #ifdef JSONCPP
    init_Cityscapes_Database(m);
    #endif    
    init_GTSDB_DIR_Database(m);
    init_GTSRB_DIR_Database(m);
    init_ILSVRC2012_Database(m);
    init_IDX_Database(m);
    init_IMDBWIKI_Database(m);
    init_KITTI_Database(m);
    init_KITTI_Object_Database(m);
    init_KITTI_Road_Database(m);
    init_LITISRouen_Database(m);
    init_N_MNIST_Database(m);
    init_DOTA_Database(m);
    init_Fashion_MNIST_IDX_Database(m);
    init_FDDB_Database(m);
    init_Daimler_Database(m);


    init_DeepNet(m);
    init_DeepNetQuantization(m);
    init_DeepNetExport(m);

    init_DrawNet(m);

    //Activation
    init_Activation(m);
    init_TanhActivation(m);
    init_TanhActivation_Frame(m);
    init_RectifierActivation(m);
    init_RectifierActivation_Frame(m);
    init_LinearActivation(m);
    init_LinearActivation_Frame(m);
    init_SwishActivation(m);
    init_SwishActivation_Frame(m);
    init_SoftplusActivation(m);
    init_SoftplusActivation_Frame(m);
    init_SaturationActivation(m);
    init_SaturationActivation_Frame(m);
    init_LogisticActivation(m);
    init_LogisticActivation_Frame(m);

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
    init_RandomResizeCropTransformation(m);
    init_CustomTransformation(m);
    init_CompositeTransformation(m);

    // Solver
    init_Solver(m);
    init_SGDSolver(m);
    init_SGDSolver_Frame(m);
    init_AdamSolver(m);
    init_AdamSolver_Frame(m);

    //Filler
    init_Filler(m);
    init_HeFiller(m);
    init_NormalFiller(m);
    init_UniformFiller(m);
    init_XavierFiller(m);
    init_ConstantFiller(m);

    //Cell
    init_CellGenerator(m);
    init_Cell(m);
    init_Cell_Frame_Top(m);
    init_Cell_Frame(m);
    init_ConvCell(m);
    init_ConvCell_Frame(m);
    init_FcCell(m);
    init_FcCell_Frame(m);
    init_SoftmaxCell(m);
    init_SoftmaxCell_Frame(m);
    init_AnchorCell_Frame_Kernels_struct(m);
    init_AnchorCell(m);
    init_AnchorCell_Frame(m);
    init_BatchNormCell(m);
    init_BatchNormCell_Frame(m);
    init_Cell_Spike(m);
    init_ConvCell_Spike(m);
    init_ConvCell_Spike_Analog(m);
    init_ConvCell_Spike_PCM(m);
    init_ConvCell_Spike_RRAM(m);
    init_DeconvCell(m);
    init_DeconvCell_Frame(m);
    init_DropoutCell(m);
    init_DropoutCell_Frame(m);
    init_ElemWiseCell(m);
    init_ElemWiseCell_Frame(m);
    // init_FcCell_Spike(m);
    // init_FcCell_Spike_Analog(m);
    // init_FcCell_Spike_PCM(m);
    // init_FcCell_Spike_RRAM(m);
    init_FMPCell(m);
    init_FMPCell_Frame(m);
    init_LRNCell(m);
    init_LRNCell_Frame(m);
    init_LSTMCell(m);
    init_NormalizeCell(m);
    init_NormalizeCell_Frame(m);
    init_ObjectDetCell(m);
    init_ObjectDetCell_Frame(m);
    init_PaddingCell(m);
    init_PaddingCell_Frame(m);
    init_PoolCell(m);
    init_PoolCell_Spike(m);
    init_PoolCell_Frame(m);
    init_ProposalCell(m);
    init_ProposalCell_Frame(m);
    init_ROIPoolingCell(m);
    init_ROIPoolingCell_Frame(m);
    init_RPCell(m);
    init_RPCell_Frame(m);
    init_ResizeCell(m);
    init_ResizeCell_Frame(m);
    init_ReshapeCell(m);
    init_ReshapeCell_Frame(m);
    init_ScalingCell(m);
    init_ScalingCell_Frame(m);
    init_TargetBiasCell(m);
    init_TargetBiasCell_Frame(m);
    init_ThresholdCell(m);
    init_ThresholdCell_Frame(m);
    init_TransformationCell(m);
    init_TransformationCell_Frame(m);
    init_TransposeCell(m);
    init_TransposeCell_Frame(m);
    init_UnpoolCell(m);
    init_UnpoolCell_Frame(m);
    init_ActivationCell(m);
    init_ActivationCell_Frame(m);

    // Mapping object
    init_MappingGenerator(m);

    //Quantizer
    init_QuantizerCell(m);
    init_QuantizerCell_Frame(m);
    
    init_QuantizerActivation(m);
    init_QuantizerActivation_Frame(m);
    
    init_DeepNetGenerator(m);
    init_helper(m);
    // Creating a variable to know if CUDA have been used for the compilation.
    #ifdef CUDA
    m.attr("cuda_compiled") = true;
    init_CudaContext(m);
    init_CudaTensor(m);
    init_TanhActivation_Frame_CUDA(m);
    init_RectifierActivation_Frame_CUDA(m);
    init_LinearActivation_Frame_CUDA(m);
    init_SwishActivation_Frame_CUDA(m);
    init_SoftplusActivation_Frame_CUDA(m);
    init_SaturationActivation_Frame_CUDA(m);
    init_LogisticActivation_Frame_CUDA(m);
    init_SGDSolver_Frame_CUDA(m);
    init_AdamSolver_Frame_CUDA(m);
    init_Cell_Frame_CUDA(m);
    init_ConvCell_Frame_CUDA(m);
    init_FcCell_Frame_CUDA(m);
    init_SoftmaxCell_Frame_CUDA(m);
    init_AnchorCell_Frame_CUDA(m);
    init_BatchNormCell_Frame_CUDA(m);
    init_DeconvCell_Frame_CUDA(m);
    init_DropoutCell_Frame_CUDA(m);
    init_ElemWiseCell_Frame_CUDA(m);
    init_FMPCell_Frame_CUDA(m);
    init_LRNCell_Frame_CUDA(m);
    init_LSTMCell_Frame_CUDA(m);
    init_NormalizeCell_Frame_CUDA(m); 
    init_ObjectDetCell_Frame_CUDA(m);
    init_PaddingCell_Frame_CUDA(m);
    init_PoolCell_Frame_CUDA(m);
    init_PoolCell_Frame_EXT_CUDA(m);
    init_ProposalCell_Frame_CUDA(m);
    init_RPCell_Frame_CUDA(m);
    init_ROIPoolingCell_Frame_CUDA(m);
    init_ResizeCell_Frame_CUDA(m);
    init_ReshapeCell_Frame_CUDA(m);
    init_ScalingCell_Frame_CUDA(m);
    init_TargetBiasCell_Frame_CUDA(m);
    init_ThresholdCell_Frame_CUDA(m);
    init_TransformationCell_Frame_CUDA(m);
    init_TransposeCell_Frame_CUDA(m);
    init_UnpoolCell_Frame_CUDA(m);
    init_ActivationCell_Frame_CUDA(m);
    init_QuantizerCell_Frame_CUDA(m);
    init_QuantizerActivation_Frame_CUDA(m);
    #else
    m.attr("cuda_compiled") = false;
    #endif

    #ifdef JSONCPP
    m.attr("json_compiled") = true;
    #else
    m.attr("json_compiled") = false;
    #endif
}

}



#endif
