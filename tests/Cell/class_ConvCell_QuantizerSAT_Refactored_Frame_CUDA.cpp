/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Inna KUCHER (inna.kucher@cea.fr)

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

#ifdef CUDA

#include "N2D2.hpp"

#include "Cell/ConvCell_Frame_CUDA.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "DeepNet.hpp"
#include "Xnet/Environment.hpp"
#include "Xnet/Network.hpp"
#include "Cell/DropoutCell_Frame_CUDA.hpp"
#include "Cell/SoftmaxCell_Frame_CUDA.hpp"
#endif
#include "Transformation/RescaleTransformation.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"
#include "Quantizer/Activation/SATQuantizerActivation_Frame_CUDA.hpp"
#include "Quantizer/Cell/SATQuantizerCell_Frame_CUDA.hpp"
#include "Activation/LinearActivation.hpp"
#include "Activation/Activation.hpp"

using namespace N2D2;

template <class T>
class ConvCell_QuantizerSAT_Refactored_Frame_CUDA_Test : public ConvCell_Frame_CUDA<T> {
public:
    ConvCell_QuantizerSAT_Refactored_Frame_CUDA_Test(const DeepNet& deepNet, 
                             const std::string& name,
                             const std::vector<unsigned int>& kernelDims,
                             unsigned int nbOutputs,
                             const std::vector<unsigned int>& subSampleDims,
                             const std::vector<unsigned int>& strideDims,
                             const std::vector<int>& paddingDims,
                             const std::vector<unsigned int>& dilationDims,
                             const std::shared_ptr
                             <Activation>& activation)
        : Cell(deepNet, name, nbOutputs),
          ConvCell(deepNet, name,
                   kernelDims,
                   nbOutputs,
                   subSampleDims,
                   strideDims,
                   paddingDims,
                   dilationDims),
          ConvCell_Frame_CUDA<T>(deepNet, name,
                              kernelDims,
                              nbOutputs,
                              subSampleDims,
                              strideDims,
                              paddingDims,
                              dilationDims,
                              activation 
                              ) {};

    friend class UnitTest_ConvCell_QuantizerSAT_Refactored_Frame_CUDA_float_ConvOneLayer_WeightsQuant_Propagate;    
};

TEST_DATASET(ConvCell_QuantizerSAT_Refactored_Frame_CUDA_float,
             ConvOneLayer_WeightsQuant_Propagate,
             (unsigned int kernelWidth,
              unsigned int kernelHeight,
              unsigned int subSampleX,
              unsigned int subSampleY,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight,
              size_t range),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 3U, 3U, 15)
             )
{

    CudaContext::setDevice(0);
    std::cout<<"ConvOneLayer_WeightsQuant_Propagate"<<std::endl;

    const unsigned int nbOutputs = 3;
    const unsigned int nbChannels = 1;
          
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    Tensor<Float_T>& in = env.getData();
    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    //fill the input
    int counter = 0;
    for (unsigned int b = 0; b < in.dimB(); ++b) {
        for (unsigned int z = 0; z < in.dimZ(); ++z) {
            for (unsigned int y = 0; y < in.dimY(); ++y) {
                for (unsigned int x = 0; x < in.dimX(); ++x) {
                    in(x, y, z, b) = 0.125f*counter;
                    in(x, y, z, b) += 10;
                    counter++;
                }
            }
        }
    }

    std::cout << "********************SET_INPUT********************" << std::endl; 
    std::cout << "[Input]\n" << in << std::endl;
    std::cout << "********************SET_INPUT_END********************\n\n" << std::endl; 

#if CUDNN_VERSION >= 5000
    DropoutCell_Frame_CUDA<Float_T> drop1(dn, "drop1", 1);
    drop1.setParameter<double>("Dropout", 0.0);
#endif
    ConvCell_QuantizerSAT_Refactored_Frame_CUDA_Test<float> conv1(dn, "conv1",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1.setParameter("NoBias", true);

    SATQuantizerCell_Frame_CUDA<float> quantCell;
    quantCell.setRange(range);
    quantCell.setQuantization(true);
    quantCell.setScaling(false);
    std::shared_ptr<QuantizerCell> quantizerCell = std::shared_ptr<QuantizerCell>(&quantCell, [](QuantizerCell *) {});

    SoftmaxCell_Frame_CUDA<float> softmax1(dn, "softmax1", nbOutputs, true, 0);

#if CUDNN_VERSION >= 5000
    drop1.addInput(in,in);
    conv1.addInput(&drop1);
    softmax1.addInput(&conv1);
    drop1.initialize();
#else
    conv1.addInput(in,in);
    softmax1.addInput(&conv1);
#endif
    conv1.setQuantizer(quantizerCell);
    conv1.initialize();
    softmax1.initialize();

    if(conv1.getQuantizer()){
        std::cout << "Added " <<  conv1.getQuantizer()->getType() <<
        " quantizer to " << conv1.getName() << std::endl;
    }

    float weight_tmp = 0.0f;
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int channel = 0; channel < nbChannels;
             ++channel) {
            Tensor<float> kernel({kernelWidth,
                                   kernelHeight});

            for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                    weight_tmp = 0.0f;
                    if(output==0){
                        weight_tmp = 0.001;
                        if (sy==1 && sx==1) weight_tmp = 0.5;
                    }
                    if(output==1){
                        weight_tmp = 0.1;
                        if (sy==1 && sx==1) weight_tmp = 0.5;               
                    }
                    if(output==2){
                        weight_tmp = 0.5;
                    }
                    kernel(sx, sy) = weight_tmp;
                }
            }
            conv1.setWeight(output, channel, kernel);
        }
    }

    std::cout << "********************SET_INIT_WEIGHTS********************" << std::endl; 
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            Tensor<float> weight;
            conv1.getWeight(output, channel, weight);
            std::cout << "[" << output << "][" << channel << "]: \n" 
            << weight << std::endl;
        }
    }
    std::cout << "********************SET_INIT_WEIGHTS_END********************\n\n" << std::endl; 

    std::cout << "********************PROPAGATE********************" << std::endl;

#if CUDNN_VERSION >= 5000
    drop1.propagate(false);
#endif
    conv1.propagate(false);
    softmax1.propagate(false);

    quantCell.getQuantizedWeights(0).synchronizeDToH();
    CudaTensor<float> qWeights = cuda_tensor_cast<float>(quantCell.getQuantizedWeights(0));
    quantCell.getQuantizedWeights(0).synchronizeHToD();
    std::cout << "[Conv][QuantizedWeights]\n" << qWeights << std::endl;

    conv1.getOutputs().synchronizeDToH();
    const CudaTensor<float>& outputConv = cuda_tensor_cast<float>(conv1.getOutputs());
    conv1.getOutputs().synchronizeHToD();
    std::cout << "[Conv][Output]\n" << outputConv << std::endl;

    std::cout << "********************PROPAGATE_END********************" << std::endl;

    //for future backpropagate test, not tested yet
    /*
    softmax1.mDiffInputs.synchronizeDToH();
    softmax1.getOutputs().synchronizeDToH();
    const CudaTensor<float>& out_softmax1 = cuda_tensor_cast<float>(softmax1.getOutputs());
    double loss = 0.0f;

    for(unsigned int nout = 0; nout < nbOutputs; ++nout){
        for (unsigned int batchPos = 0; batchPos < 1; ++batchPos){
            std::cout << "out_softmax1(nout, batchPos) = " << out_softmax1(nout, batchPos) << std::endl;
            if(nout==0) {
                softmax1.mDiffInputs(nout, batchPos) = 1.0f;
            }
            if(nout==1) {
                softmax1.mDiffInputs(nout, batchPos) = 0.0f;
            }
            if(nout==3) {
                softmax1.mDiffInputs(nout, batchPos) = 1.8f;
            }
            std::cout << "softmax1.mDiffInputs(nout, batchPos) = " << softmax1.mDiffInputs(nout, batchPos) << std::endl;
        }
    }

    loss = softmax1.applyLoss();
    std::cout << "test loss = " << loss << std::endl;
    softmax1.mDiffInputs.synchronizeHToD();
    softmax1.getOutputs().synchronizeHToD();

    softmax1.backPropagate();   
    conv1.backPropagate();
    drop1.backPropagate();

    quantCell.getDiffFullPrecisionWeights(0).synchronizeDToH();
    CudaTensor<float> diffFullPrecisionWeights = cuda_tensor_cast<float>(quantCell.getDiffFullPrecisionWeights(0));
    quantCell.getDiffFullPrecisionWeights(0).synchronizeHToD();

    std::cout << "********************DIFF_FULL_PR_WEIGHTS********************" << std::endl;
    std::cout << "[Conv][DiffFullPrWeights]" << diffFullPrecisionWeights << std::endl;
    std::cout << "********************DIFF_FULL_PR_WEIGHTS_END********************" << std::endl;

    std::cout << "********************BACKPROPAGATE_END********************" << std::endl;

    conv1.update();
    std::cout << "********************UPDATED_WEIGHTS********************" << std::endl; 
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            Tensor<float> weight;
            conv1.getWeight(output, channel, weight);
            std::cout << "[" << output << "][" << channel << "]: \n" 
            << weight << std::endl;
        }
    }
    std::cout << "********************UPDATED_WEIGHTS_END********************\n\n" << std::endl; 
    */
    
}

RUN_TESTS()