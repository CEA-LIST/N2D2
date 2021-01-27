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
#if CUDNN_VERSION >= 5000
#include "Cell/DropoutCell_Frame_CUDA.hpp"
#include "Cell/SoftmaxCell_Frame_CUDA.hpp"
#endif
#include "Transformation/RescaleTransformation.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"
#include "Quantizer/SATQuantizer_Frame_CUDA.hpp"

using namespace N2D2;

template <class T>
class ConvCell_QuantizerSAT_Frame_CUDA_Test : public ConvCell_Frame_CUDA<T> {
public:
    ConvCell_QuantizerSAT_Frame_CUDA_Test(const DeepNet& deepNet, 
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

    //friend class UnitTest_ConvCell_QuantizerSAT_Frame_CUDA_float_check_one_layer_with_SAT;
    //friend class UnitTest_ConvCell_QuantizerSAT_Frame_CUDA_float_check_2conv_layers_with_SAT;
    friend class UnitTest_ConvCell_QuantizerSAT_Frame_CUDA_float_check_miniMobileNet_with_SAT;
    friend class UnitTest_ConvCell_QuantizerSAT_Frame_CUDA_double_check_miniMobileNet_with_SAT;
    //friend class UnitTest_ConvCell_QuantizerSAT_Frame_CUDA_float_check_gradient_SAT;


    
};

/*

TEST_DATASET(ConvCell_QuantizerSAT_Frame_CUDA_float,
             check_one_layer_with_SAT,
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
              size_t range,
              float alpha),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 3U, 3U, 15,8.0)
             )
{

    std::cout<<"check_one_layer_with_SAT"<<std::endl;

    CudaContext::setDevice(1);
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
                    std::cout << "b, z, y, x = " << b << " , " << z << " , " << y << " , " << x << std::endl;
                    in(x, y, z, b) = 0.125f*counter;
                    in(x, y, z, b) += 10;
                    counter++;
                    std::cout << "in(x, y, z, b) = " << in(x, y, z, b) << std::endl;
                }
            }
        }
    }

#if CUDNN_VERSION >= 5000
    DropoutCell_Frame_CUDA<Float_T> drop1(dn, "drop1", 1);
    drop1.setParameter<double>("Dropout", 0.0);
#endif
    ConvCell_QuantizerSAT_Frame_CUDA_Test<float> conv1(dn, "conv1",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1.setParameter("NoBias", true);

    SATQuantizer_Frame_CUDA<float> quant;
    quant.setRange(range);
    quant.setAlpha(alpha);
    quant.setQuantization(true);
    quant.setScaling(false);
    std::shared_ptr<Quantizer> quantizer = std::shared_ptr<Quantizer>(&quant, [](Quantizer *) {});

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
    conv1.setQuantizer(quantizer);
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
                    std::cout << "sx = " << sx << " , sy = " << sy << " , kernel(sx, sy) = " << kernel(sx, sy) << std::endl;
                }
            }
            conv1.setWeight(output, channel, kernel);
        }
    }

#if CUDNN_VERSION >= 5000
    drop1.propagate(false);
#endif
    conv1.propagate(false);
    softmax1.propagate(false);

    conv1.getOutputs().synchronizeDToH();
    const Tensor<float>& out_conv = tensor_cast<float>(conv1.getOutputs());

    for (unsigned int b = 0; b < out_conv.dimB(); ++b) {
        for (unsigned int z = 0; z < out_conv.dimZ(); ++z) {
            for (unsigned int y = 0; y < out_conv.dimY(); ++y) {
                for (unsigned int x = 0; x < out_conv.dimX(); ++x) {
                    std::cout << "b, z, y, x = " << b << " , " << z << " , " << y << " , " << x << std::endl;
                    std::cout << "out_conv(x, y, z, b) (quant) = " << out_conv(x, y, z, b) << std::endl;
                }
            }
        }
    }
    conv1.getOutputs().synchronizeHToD();
    std::cout <<"end of propagate" << std::endl;

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

    //backpropagate 
    softmax1.backPropagate();   
    conv1.backPropagate();
#if CUDNN_VERSION >= 5000
    drop1.backPropagate();
#endif

    // 3 kernels
    CudaTensor<float> my_DiffFullPrecisionWeights = cuda_tensor_cast<float>(quant.getDiffFullPrecisionWeights(0));
    my_DiffFullPrecisionWeights.synchronizeDToH();
    for (unsigned int i=0; i<my_DiffFullPrecisionWeights[0].dims().back(); ++i) {
        std::cout << "k = " << 0 << " , i = " << i << "  , my_DiffFullPrecisionWeights [0][i] = " << my_DiffFullPrecisionWeights[0][i] << std::endl;
    }
    for (unsigned int i=0; i<my_DiffFullPrecisionWeights[1].dims().back(); ++i) {
        std::cout << "k = " << 1 << " , i = " << i << "  , my_DiffFullPrecisionWeights [1][i] = " << my_DiffFullPrecisionWeights[1][i] << std::endl;
    }
    for (unsigned int i=0; i<my_DiffFullPrecisionWeights[2].dims().back(); ++i) {
        std::cout << "k = " << 1 << " , i = " << i << "  , my_DiffFullPrecisionWeights [2][i] = " << my_DiffFullPrecisionWeights[2][i] << std::endl;
    }

    CudaTensor<float> my_DiffFullPrecisionActivations = cuda_tensor_cast<float>(quant.getDiffFullPrecisionActivations(0));
    my_DiffFullPrecisionActivations.synchronizeDToH();
    for (unsigned int i=0; i<my_DiffFullPrecisionActivations[0].dims().back(); ++i) {
        std::cout << "k = " << 0 << " , i = " << i << "  , my_DiffFullPrecisionActivations [0][i] = " << my_DiffFullPrecisionActivations[0][i] << std::endl;
    }


    conv1.update();
    CudaTensor<float> alphaEstimated = quant.getAlpha(0);
    alphaEstimated.synchronizeDToH();
    std::cout << "alphaEstimated = " << alphaEstimated << std::endl;

    
    Tensor<float> kernel1({kernelWidth, kernelHeight});
    Tensor<float> kernel2({kernelWidth, kernelHeight});
    Tensor<float> kernel3({kernelWidth, kernelHeight});
    conv1.getWeight(0,0,kernel1);
    conv1.getWeight(1,0,kernel2);
    conv1.getWeight(2,0,kernel3);

    for (unsigned int i=0; i<kernel1.dims().back(); ++i) {
        std::cout << "dim i = " << i  << ", weights in kernel1 =  " << kernel1[i] << std::endl;
    }
    for (unsigned int i=0; i<kernel2.dims().back(); ++i) {
        std::cout << "dim i = " << i  << ", weights in kernel2 =  " << kernel2[i] << std::endl;
    }
    for (unsigned int i=0; i<kernel3.dims().back(); ++i) {
        std::cout << "dim i = " << i  << ", weights in kernel3 =  " << kernel3[i] << std::endl;
    }
    
}


// check 2 conv layers with 2 quantization levels
TEST_DATASET(ConvCell_QuantizerSAT_Frame_CUDA_float,
             check_2conv_layers_with_SAT,
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
              size_t range1,
              float alpha1,
              size_t range2,
              float alpha2),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 5U, 5U, 255,8.0,15,8.0)
             )
{

    std::cout<<"check_2conv_layers_with_SAT"<<std::endl;

    CudaContext::setDevice(1);
    const unsigned int nbOutputs_conv1 = 1;
    const unsigned int nbOutputs_conv2 = 2;
    const unsigned int nbChannels = 1;
          
    Network net;
    DeepNet dn(net);
    unsigned int batchSize = 2;
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1}, batchSize);

    Tensor<Float_T>& in = env.getData();
    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    std::cout << "in.dimB() = " << in.dimB() << std::endl;
    std::cout << "in.dimZ() = " << in.dimZ() << std::endl;
    std::cout << "in.dimX() = " << in.dimX() << std::endl;
    std::cout << "in.dimY() = " << in.dimY() << std::endl;
    ///
    //fill input image with 1s
    //int counter = 0;
    for (unsigned int b = 0; b < in.dimB(); ++b) {
        for (unsigned int z = 0; z < in.dimZ(); ++z) {
            for (unsigned int y = 0; y < in.dimY(); ++y) {
                for (unsigned int x = 0; x < in.dimX(); ++x) {
                    std::cout << "b, z, y, x = " << b << " , " << z << " , " << y << " , " << x << std::endl;
                    if(b==0) in(x, y, z, b) = 1.0f;
                    if(b==1) in(x, y, z, b) = 5.0f;
                    //counter++;
                    std::cout << "in(x, y, z, b) = " << in(x, y, z, b) << std::endl;
                }
            }
        }
    }

    ConvCell_QuantizerSAT_Frame_CUDA_Test<float> conv1(dn, "conv1",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv1,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1.setParameter("NoBias", true);

    std::cout << "conv1 set "<< std::endl;

    ConvCell_QuantizerSAT_Frame_CUDA_Test<float> conv2(dn, "conv2",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv2,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv2.setParameter("NoBias", true);

    std::cout << "conv2 set "<< std::endl;

    SATQuantizer_Frame_CUDA<float> quant1;
    quant1.setRange(range1);
    quant1.setAlpha(alpha1);
    quant1.setQuantization(true);
    quant1.setScaling(false);
    std::shared_ptr<Quantizer> quantizer1 = std::shared_ptr<Quantizer>(&quant1, [](Quantizer *) {});

    std::cout << "quant1 set "<< std::endl;

    SATQuantizer_Frame_CUDA<float> quant2;
    quant2.setRange(range2);
    quant2.setAlpha(alpha2);
    quant2.setQuantization(true);
    quant2.setScaling(false);
    std::shared_ptr<Quantizer> quantizer2 = std::shared_ptr<Quantizer>(&quant2, [](Quantizer *) {});

    std::cout << "quant2 set "<< std::endl;

    SoftmaxCell_Frame_CUDA<float> softmax1(dn, "softmax1", nbOutputs_conv2, true, 0);
    std::cout << "softmax1 "<< std::endl;

    Tensor<float> out_diff({channelsWidth, channelsHeight, 1, batchSize});
    conv1.addInput(in, out_diff);
    std::cout << "conv1 addInput "<< std::endl;
    conv2.addInput(&conv1);
    std::cout << "conv2 addInput "<< std::endl;
    softmax1.addInput(&conv2);

    std::cout << "add input "<< std::endl;

    conv1.setQuantizer(quantizer1);
    conv1.initialize();
    std::cout << "conv1 init "<< std::endl;
    conv2.setQuantizer(quantizer2);
    conv2.initialize();
    std::cout << "conv2 init "<< std::endl;
    softmax1.initialize();
    std::cout << "softmax init "<< std::endl;

    if(conv1.getQuantizer()){
        std::cout << "Added " <<  conv1.getQuantizer()->getType() <<
        " quantizer to " << conv1.getName() << std::endl;
    }

    if(conv2.getQuantizer()){
        std::cout << "Added " <<  conv2.getQuantizer()->getType() <<
        " quantizer to " << conv2.getName() << std::endl;
    }
    
    int count = 0;
    int count_ind = 0;
    float weight_tmp = 0.0f;

    // set weights for conv1

        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
        for (unsigned int channel = 0; channel < nbChannels;
             ++channel) {
            Tensor<float> kernel({kernelWidth,
                                   kernelHeight});
            for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                    kernel(sx, sy) = 0.1;
                    std::cout << "conv1 :: sx = " << sx << " , sy = " << sy << " , kernel(sx, sy) = " << kernel(sx, sy) << std::endl;
                }
            }
            conv1.setWeight(output, channel, kernel);
        }
    }

    // set weights for conv2

    for (unsigned int output = 0; output < nbOutputs_conv2; ++output) {
        for (unsigned int channel = 0; channel < nbChannels;
             ++channel) {
            Tensor<float> kernel({kernelWidth,
                                   kernelHeight});

            for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                    weight_tmp = 0.0f;
                    if(output==0){
                        weight_tmp = 0.01;
                        if (sy==1 && sx==1) weight_tmp = 0.5;
                    }
                    if(output==1){
                        weight_tmp = 0.1;         
                        if (sy==1 && sx==1) weight_tmp = 0.5;   
                    }
                    kernel(sx, sy) = weight_tmp;
                    std::cout << "conv2 :: sx = " << sx << " , sy = " << sy << " , kernel(sx, sy) = " << kernel(sx, sy) << std::endl;
                }
            }
            conv2.setWeight(output, channel, kernel);
        }
    }


    //several iterations for propagate, backpropagate, update
    for(int iter_index = 0; iter_index < 2; ++iter_index){

        conv1.propagate(false);
        conv2.propagate(false);
        softmax1.propagate(false);

        conv1.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv1 = tensor_cast<float>(conv1.getOutputs());

        for (unsigned int b = 0; b < out_conv1.dimB(); ++b) {
            for (unsigned int z = 0; z < out_conv1.dimZ(); ++z) {
                for (unsigned int y = 0; y < out_conv1.dimY(); ++y) {
                    for (unsigned int x = 0; x < out_conv1.dimX(); ++x) {
                        std::cout << "b, z, y, x = " << b << " , " << z << " , " << y << " , " << x << std::endl;
                        std::cout << "out_conv1(x, y, z, b) = " << out_conv1(x, y, z, b) << std::endl;
                    }
                }
            }
        }
        conv1.getOutputs().synchronizeHToD();

        conv2.getOutputs().synchronizeDToH();
        	const Tensor<float>& out_conv2 = tensor_cast<float>(conv2.getOutputs());

        for (unsigned int b = 0; b < out_conv2.dimB(); ++b) {
            for (unsigned int z = 0; z < out_conv2.dimZ(); ++z) {
                for (unsigned int y = 0; y < out_conv2.dimY(); ++y) {
                    for (unsigned int x = 0; x < out_conv2.dimX(); ++x) {
                        std::cout << "b, z, y, x = " << b << " , " << z << " , " << y << " , " << x << std::endl;
                        std::cout << "out_conv2(x, y, z, b) = " << out_conv2(x, y, z, b) << std::endl;
                    }
                }
            }
        }
        conv2.getOutputs().synchronizeHToD();

        softmax1.mDiffInputs.synchronizeDToH();
        softmax1.getOutputs().synchronizeDToH();
        const CudaTensor<float>& out_softmax1 = cuda_tensor_cast<float>(softmax1.getOutputs());
        double loss = 0.0f;

        for(unsigned int nout = 0; nout < nbOutputs_conv2; ++nout){
            for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos){
                std::cout << "out_softmax1(nout, batchPos) = " << out_softmax1(nout, batchPos) << std::endl;
                if(nout==0) {
                    softmax1.mDiffInputs(nout, batchPos) = 1.0f;
                }
                if(nout==1) {
                    softmax1.mDiffInputs(nout, batchPos) = 0.0f;
                }
                std::cout << "softmax1.mDiffInputs(nout, batchPos) = " << softmax1.mDiffInputs(nout, batchPos) << std::endl;
            }
        }

        loss = softmax1.applyLoss();
        std::cout << "test loss = " << loss << std::endl;
        softmax1.mDiffInputs.synchronizeHToD();
        softmax1.getOutputs().synchronizeHToD();

        std::cout <<"end of propagate" << std::endl;

        //backpropagate 
        softmax1.backPropagate();   
        conv2.backPropagate();
        conv1.backPropagate();

        std::cout << "backpropagate" << std::endl;

        CudaTensor<float> my_DiffFullPrecisionWeights_conv1 = cuda_tensor_cast<float>(quant1.getDiffFullPrecisionWeights(0));
        my_DiffFullPrecisionWeights_conv1.synchronizeDToH();
        for (unsigned int i=0; i<my_DiffFullPrecisionWeights_conv1[0].dims().back(); ++i) {
            std::cout << "conv1 :: k = " << 0 << " , i = " << i << "  , my_DiffFullPrecisionWeights [0][i] = " << my_DiffFullPrecisionWeights_conv1[0][i] << std::endl;
        }

        CudaTensor<float> my_DiffFullPrecisionWeights_conv2 = cuda_tensor_cast<float>(quant2.getDiffFullPrecisionWeights(0));
        my_DiffFullPrecisionWeights_conv2.synchronizeDToH();
        for (unsigned int i=0; i<my_DiffFullPrecisionWeights_conv2[0].dims().back(); ++i) {
            std::cout << "conv2 :: k = " << 0 << " , i = " << i << "  , my_DiffFullPrecisionWeights [0][i] = " << my_DiffFullPrecisionWeights_conv2[0][i] << std::endl;
        }
        for (unsigned int i=0; i<my_DiffFullPrecisionWeights_conv2[1].dims().back(); ++i) {
            std::cout << "conv2 :: k = " << 0 << " , i = " << i << "  , my_DiffFullPrecisionWeights [1][i] = " << my_DiffFullPrecisionWeights_conv2[1][i] << std::endl;
        }

        CudaTensor<float> my_DiffFullPrecisionActivations = cuda_tensor_cast<float>(quant2.getDiffFullPrecisionActivations(0));
        my_DiffFullPrecisionActivations.synchronizeDToH();
        for (unsigned int i=0; i<my_DiffFullPrecisionActivations[0].dims().back(); ++i) {
            std::cout << "conv2 :: k = " << 0 << " , i = " << i << "  , my_DiffFullPrecisionActivations [0][i] = " << my_DiffFullPrecisionActivations[0][i] << std::endl;
        }
        
        std::cout << "end check in the test" << std::endl;

        conv1.update();
        
        CudaTensor<float> alphaEstimated1 = quant1.getAlpha(0);
        alphaEstimated1.synchronizeDToH();
        std::cout << "conv1 :: alphaEstimated = " << alphaEstimated1 << std::endl;
        
        conv2.update();
        CudaTensor<float> alphaEstimated = quant2.getAlpha(0);
        alphaEstimated.synchronizeDToH();
        std::cout << "conv2 :: alphaEstimated = " << alphaEstimated << std::endl;

        
        Tensor<float> kernel1_conv1({kernelWidth, kernelHeight});
        conv1.getWeight(0,0,kernel1_conv1);
        for (unsigned int i=0; i<kernel1_conv1.dims().back(); ++i) {
            std::cout << "conv1 :: dim i = " << i  << ", weights in kernel1 =  " << kernel1_conv1[i] << std::endl;
        }
        Tensor<float> kernel1_conv2({kernelWidth, kernelHeight});
        Tensor<float> kernel2_conv2({kernelWidth, kernelHeight});
        conv2.getWeight(0,0,kernel1_conv2);
        conv2.getWeight(1,0,kernel2_conv2);

        for (unsigned int i=0; i<kernel1_conv2.dims().back(); ++i) {
            std::cout << "conv2 :: dim i = " << i  << ", weights in kernel1 =  " << kernel1_conv2[i] << std::endl;
        }
        for (unsigned int i=0; i<kernel2_conv2.dims().back(); ++i) {
            std::cout << "conv2 :: dim i = " << i  << ", weights in kernel2 =  " << kernel2_conv2[i] << std::endl;
        }    
    }
}
*/

// check 3 conv layers with 2 quantization levels, float

TEST_DATASET(ConvCell_QuantizerSAT_Frame_CUDA_float,
             check_miniMobileNet_with_SAT,
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
              size_t range1,
              float alpha1,
              size_t range2,
              float alpha2),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 5U, 5U, 255,8.0,15,8.0)
             )
{

    std::cout<<"float :: check_miniMobileNet_with_SAT"<<std::endl;
    //to avoid warning when compile
    std::cout << kernelWidth << kernelHeight << subSampleX << subSampleY 
                << strideX << strideY << paddingX << paddingY 
                << channelsWidth << channelsHeight 
                << range1 << alpha1 << range2 << alpha2 << std::endl; 
    /*
    bool doQuant = true;

    CudaContext::setDevice(0);
    const unsigned int nbOutputs_conv1 = 1;
    const unsigned int nbOutputs_conv2 = 4;
    const unsigned int nbOutputs_conv3 = 4;
    const unsigned int nbChannels = 1;

    unsigned int kernelWidth2 = 1;
    unsigned int kernelHeight2 = 1;
          
    Network net;
    DeepNet dn(net);
    unsigned int batchSize = 2;
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1}, batchSize);

    Tensor<Float_T>& in = env.getData();
    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);
    ///
    //fill input image
    float input_tmp = 0.0f;

    for (unsigned int b = 0; b < in.dimB(); ++b) {
        for (unsigned int z = 0; z < in.dimZ(); ++z) {
            for (unsigned int y = 0; y < in.dimY(); ++y) {
                for (unsigned int x = 0; x < in.dimX(); ++x) {
                    if(b==0) {
                        if(x==0 && y==0) input_tmp = 1.0f;
                        if(x==1 && y==0) input_tmp = 5.0f;
                        if(x==2 && y==0) input_tmp = 8.0f;
                        if(x==3 && y==0) input_tmp = 10.0f;
                        if(x==4 && y==0) input_tmp = 30.0f;

                        if(x==0 && y==1) input_tmp = 65.0f;
                        if(x==1 && y==1) input_tmp = 70.0f;
                        if(x==2 && y==1) input_tmp = 80.0f;
                        if(x==3 && y==1) input_tmp = 50.0f;
                        if(x==4 && y==1) input_tmp = 125.0f;

                        if(x==0 && y==2) input_tmp = 29.0f;
                        if(x==1 && y==2) input_tmp = 30.0f;
                        if(x==2 && y==2) input_tmp = 165.0f;
                        if(x==3 && y==2) input_tmp = 1.0f;
                        if(x==4 && y==2) input_tmp = 1.0f;

                        if(x==0 && y==3) input_tmp = 1.0f;
                        if(x==1 && y==3) input_tmp = 1.0f;
                        if(x==2 && y==3) input_tmp = 1.0f;
                        if(x==3 && y==3) input_tmp = 1.0f;
                        if(x==4 && y==3) input_tmp = 1.0f;

                        if(x==0 && y==4) input_tmp = 1.0f;
                        if(x==1 && y==4) input_tmp = 1.0f;
                        if(x==2 && y==4) input_tmp = 1.0f;
                        if(x==3 && y==4) input_tmp = 1.0f;
                        if(x==4 && y==4) input_tmp = 1.0f;
                    }
                    if(b==1) {
                        if(x==0 && y==0) input_tmp = 1.0f;
                        if(x==1 && y==0) input_tmp = 5.0f;
                        if(x==2 && y==0) input_tmp = 8.0f;
                        if(x==3 && y==0) input_tmp = 10.0f;
                        if(x==4 && y==0) input_tmp = 30.0f;

                        if(x==0 && y==1) input_tmp = 65.0f;
                        if(x==1 && y==1) input_tmp = 70.0f;
                        if(x==2 && y==1) input_tmp = 80.0f;
                        if(x==3 && y==1) input_tmp = 50.0f;
                        if(x==4 && y==1) input_tmp = 125.0f;

                        if(x==0 && y==2) input_tmp = 29.0f;
                        if(x==1 && y==2) input_tmp = 30.0f;
                        if(x==2 && y==2) input_tmp = 73.0f;
                        if(x==3 && y==2) input_tmp = 1.0f;
                        if(x==4 && y==2) input_tmp = 1.0f;

                        if(x==0 && y==3) input_tmp = 1.0f;
                        if(x==1 && y==3) input_tmp = 55.0f;
                        if(x==2 && y==3) input_tmp = 1.0f;
                        if(x==3 && y==3) input_tmp = 1.0f;
                        if(x==4 && y==3) input_tmp = 1.0f;

                        if(x==0 && y==4) input_tmp = 1.0f;
                        if(x==1 && y==4) input_tmp = 1.0f;
                        if(x==2 && y==4) input_tmp = 1.0f;
                        if(x==3 && y==4) input_tmp = 56.0f;
                        if(x==4 && y==4) input_tmp = 1.0f;
                    }

                    in(x, y, z, b) = input_tmp/255.0f;
                    //std::cout  << "b, z, y, x = " << b << ", " << z << ", " << y << ", " << x << ", input = " << in(x, y, z, b) << std::endl;
                }
            }
        }
    }

    std::cout << "[Input]\n" << in << std::endl;

    ConvCell_QuantizerSAT_Frame_CUDA_Test<float> conv1(dn, "conv1",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv1,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1.setParameter("NoBias", true);

    ConvCell_QuantizerSAT_Frame_CUDA_Test<float> conv2(dn, "conv2",
        std::vector<unsigned int>({kernelWidth2, kernelHeight2}),
        nbOutputs_conv2,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv2.setParameter("NoBias", true);

    ////create a map to make a conv depthwise layer
    Tensor<bool> mapping;
    mapping.resize({nbOutputs_conv3, nbOutputs_conv3});
    mapping.fill(0);
    for(size_t out = 0; out < nbOutputs_conv3; ++out)
    {
        mapping(out, out) = 1;
    }

    ConvCell_QuantizerSAT_Frame_CUDA_Test<float> conv3(dn, "conv3",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv3,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv3.setParameter("NoBias", true);

    SATQuantizer_Frame_CUDA<float> quant1;
    quant1.setWeightsRange(range1);
    quant1.setActivationsRange(range1);
    quant1.setAlpha(alpha1);
    quant1.setQuantization(true);
    quant1.setScaling(false);
    std::shared_ptr<Quantizer> quantizer1 = std::shared_ptr<Quantizer>(&quant1, [](Quantizer *) {});

    SATQuantizer_Frame_CUDA<float> quant2;
    quant2.setWeightsRange(range2);
    quant2.setActivationsRange(range2);
    quant2.setAlpha(alpha2);
    quant2.setQuantization(true);
    quant2.setScaling(false);
    std::shared_ptr<Quantizer> quantizer2 = std::shared_ptr<Quantizer>(&quant2, [](Quantizer *) {});

    SATQuantizer_Frame_CUDA<float> quant3;
    quant3.setWeightsRange(range2);
    quant3.setActivationsRange(range2);
    quant3.setAlpha(alpha2);
    quant3.setQuantization(true);
    quant3.setScaling(false);
    std::shared_ptr<Quantizer> quantizer3 = std::shared_ptr<Quantizer>(&quant3, [](Quantizer *) {});

    SoftmaxCell_Frame_CUDA<float> softmax1(dn, "softmax1", nbOutputs_conv3, true, 0);

    Tensor<float> out_diff({channelsWidth, channelsHeight, 1, batchSize});
    conv1.addInput(in, out_diff);
    conv2.addInput(&conv1);
    conv3.addInput(&conv2, mapping);
    softmax1.addInput(&conv3);

    if(doQuant) conv1.setQuantizer(quantizer1);
    conv1.initialize();
    if(doQuant) conv2.setQuantizer(quantizer2);
    conv2.initialize();
    if(doQuant) conv3.setQuantizer(quantizer3);
    conv3.initialize();
    softmax1.initialize();

    if(conv1.getQuantizer()){
        std::cout << "Added " <<  conv1.getQuantizer()->getType() <<
        " quantizer to " << conv1.getName() << std::endl;
    }
    if(conv2.getQuantizer()){
        std::cout << "Added " <<  conv2.getQuantizer()->getType() <<
        " quantizer to " << conv2.getName() << std::endl;
    }
    if(conv3.getQuantizer()){
        std::cout << "Added " <<  conv3.getQuantizer()->getType() <<
        " quantizer to " << conv3.getName() << std::endl;
    }
    
    float weight_tmp = 0.0f;

    // set weights for conv1
        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
        for (unsigned int channel = 0; channel < nbChannels;
             ++channel) {
            Tensor<float> kernel({kernelWidth,
                                   kernelHeight});
            for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                    if (sy==0 && sx==0) weight_tmp = 0.025;
                    if (sy==0 && sx==1) weight_tmp = 0.5;
                    if (sy==0 && sx==2) weight_tmp = 0.075;

                    if (sy==1 && sx==0) weight_tmp = -0.01;
                    if (sy==1 && sx==1) weight_tmp = 0.01;
                    if (sy==1 && sx==2) weight_tmp = -0.01;

                    if (sy==2 && sx==0) weight_tmp = 0.35;
                    if (sy==2 && sx==1) weight_tmp = -0.5;
                    if (sy==2 && sx==2) weight_tmp = 0.2;

                    kernel(sx, sy) = weight_tmp;
                    //std::cout << "conv1 :: sx = " << sx << " , sy = " << sy << " , weight = " << kernel(sx, sy) << std::endl;
                }
            }
            conv1.setWeight(output, channel, kernel);
        }
    }

    // set weights for conv2
    // [[[0.01]]], [[[0.01]]], [[[0.01]]], [[[0.01]]
    weight_tmp = 0.0f;
    for (unsigned int output = 0; output < nbOutputs_conv2; ++output) {
        for (unsigned int channel = 0; channel < nbChannels;
             ++channel) {
            Tensor<float> kernel({kernelWidth2,
                                   kernelHeight2});

            for (unsigned int sx = 0; sx < kernelWidth2; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight2; ++sy){
                    if(output==0){
                        weight_tmp = 0.01;
                    }
                    if(output==1){
                        weight_tmp = 0.01;            
                    }
                    if(output==2){
                        weight_tmp = 0.01;            
                    }
                    if(output==3){
                        weight_tmp = 0.01;            
                    }
                    kernel(sx, sy) = weight_tmp;
                    //std::cout << "conv2 :: sx = " << sx << " , sy = " << sy << " , weight = " << kernel(sx, sy) << std::endl;
                }
            }
            conv2.setWeight(output, channel, kernel);
        }
    }

    // set weights for conv3
    weight_tmp = 0.0f;
    for (unsigned int output = 0; output < nbOutputs_conv3; ++output) {
            Tensor<float> kernel({kernelWidth,
                                   kernelHeight});

            for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                    weight_tmp = 0.0f;
                    if(output==0){
                        if (sy==0 && sx==0) weight_tmp = 0.01;
                        if (sy==0 && sx==1) weight_tmp = -0.013;
                        if (sy==0 && sx==2) weight_tmp = 0.01;

                        if (sy==1 && sx==0) weight_tmp = -0.01;
                        if (sy==1 && sx==1) weight_tmp = 0.5;
                        if (sy==1 && sx==2) weight_tmp = -0.013;

                        if (sy==2 && sx==0) weight_tmp = 0.01;
                        if (sy==2 && sx==1) weight_tmp = -0.01;
                        if (sy==2 && sx==2) weight_tmp = 0.01;
                    }
                    if(output==1){
                        if (sy==0 && sx==0) weight_tmp = 0.01;
                        if (sy==0 && sx==1) weight_tmp = -0.01;
                        if (sy==0 && sx==2) weight_tmp = 0.01;

                        if (sy==1 && sx==0) weight_tmp = -0.01;
                        if (sy==1 && sx==1) weight_tmp = 0.5;
                        if (sy==1 && sx==2) weight_tmp = -0.01;

                        if (sy==2 && sx==0) weight_tmp = 0.01;
                        if (sy==2 && sx==1) weight_tmp = -0.01;
                        if (sy==2 && sx==2) weight_tmp = 0.01;
                    }
                    if(output==2){
                        if (sy==0 && sx==0) weight_tmp = 0.01;
                        if (sy==0 && sx==1) weight_tmp = -0.01;
                        if (sy==0 && sx==2) weight_tmp = 0.01;

                        if (sy==1 && sx==0) weight_tmp = -0.01;
                        if (sy==1 && sx==1) weight_tmp = 0.5;
                        if (sy==1 && sx==2) weight_tmp = -0.01;

                        if (sy==2 && sx==0) weight_tmp = 0.013;
                        if (sy==2 && sx==1) weight_tmp = -0.01;
                        if (sy==2 && sx==2) weight_tmp = 0.01;  
                    }
                    if(output==3){
                        if (sy==0 && sx==0) weight_tmp = 0.1;
                        if (sy==0 && sx==1) weight_tmp = -0.013;
                        if (sy==0 && sx==2) weight_tmp = 0.01;

                        if (sy==1 && sx==0) weight_tmp = -0.01;
                        if (sy==1 && sx==1) weight_tmp = 0.9;
                        if (sy==1 && sx==2) weight_tmp = -0.01;

                        if (sy==2 && sx==0) weight_tmp = 0.01;
                        if (sy==2 && sx==1) weight_tmp = -0.01;
                        if (sy==2 && sx==2) weight_tmp = 0.3;  
                    }
                    kernel(sx, sy) = weight_tmp;
                    //std::cout << "conv3 :: output = "<< output << ", sx = " << sx << " , sy = " << sy << " , weight = " << kernel(sx, sy) << std::endl;
                }
            }
            conv3.setWeight(output, output, kernel);
    }

   
    
    //several iterations for propagate, backpropagate, update
    for(unsigned int iter_index = 0; iter_index < 10000; ++iter_index){

        if(iter_index==9999) std::cout << "iteration #" << iter_index << std::endl;
        if(iter_index==9999) std::cout << "===============================================================" << std::endl;

        if(iter_index==9999) std::cout << "******************PROPAGATE*******************\n\n\n" << std::endl;

        conv1.propagate(false);
        conv2.propagate(false);
        conv3.propagate(false);
        softmax1.propagate(false);

        conv1.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv1 = tensor_cast<float>(conv1.getOutputs());
        if(iter_index==9999){
            std::cout << "[Conv1][Outputs]" << std::endl;
            std::cout << out_conv1 << std::endl;
        }
        conv1.getOutputs().synchronizeHToD();

        conv2.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv2 = tensor_cast<float>(conv2.getOutputs());
        if(iter_index==9999){
            std::cout << "[Conv2][Outputs]" << std::endl;
            std::cout << out_conv2 << std::endl;
        }
        conv2.getOutputs().synchronizeHToD();

        conv3.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv3 = tensor_cast<float>(conv3.getOutputs());
        if(iter_index==9999){
            std::cout << "[Conv3][Outputs]" << std::endl;
            std::cout << out_conv3 << std::endl;
        }
        conv3.getOutputs().synchronizeHToD();

        softmax1.mDiffInputs.synchronizeDToH();
        softmax1.getOutputs().synchronizeDToH();
        const CudaTensor<float>& out_softmax1 = cuda_tensor_cast<float>(softmax1.getOutputs());
        double loss = 0.0;
        if(iter_index==9999){
            std::cout << "[SoftMax][Outputs]" << std::endl;
            std::cout << out_softmax1 << std::endl;
        }


        for(unsigned int nout = 0; nout < nbOutputs_conv3; ++nout){
            for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos){
                if(batchPos == 0) {
                    if(nout==0) {
                        softmax1.mDiffInputs(nout, batchPos) = 1.0f;
                    }
                    else
                        softmax1.mDiffInputs(nout, batchPos) = 0.0f; 
                }
                if(batchPos == 1){
                    if(nout==3) {
                        softmax1.mDiffInputs(nout, batchPos) = 1.0f;
                    }
                    else
                        softmax1.mDiffInputs(nout, batchPos) = 0.0f; 
                }
                //std::cout << "softmax1.mDiffInputs(nout, batchPos) = " << softmax1.mDiffInputs(nout, batchPos) << std::endl;
            }
        }

        loss = softmax1.applyLoss();
        //std::cout << "test loss = " << loss << std::endl;
        softmax1.mDiffInputs.synchronizeHToD();
        softmax1.getOutputs().synchronizeHToD();

        //std::cout <<"end of propagate" << std::endl;

        //backpropagate 
        softmax1.backPropagate();  
        conv3.backPropagate(); 
        conv2.backPropagate();
        conv1.backPropagate();

        if(iter_index==9999) std::cout << "****************BACKPROPAGATE******************" << std::endl;

    if(doQuant){
        
        //quant1.getDiffFullPrecisionWeights(0).synchronizeDToH();
        //quant2.getDiffFullPrecisionWeights(0).synchronizeDToH();
        //quant3.getDiffFullPrecisionWeights(0).synchronizeDToH();
        //quant2.getDiffFullPrecisionActivations(0).synchronizeDToH();
        //quant3.getDiffFullPrecisionActivations(0).synchronizeDToH();

        //quant2.getDiffQuantizedActivations(0).synchronizeDToH();
        //quant3.getDiffQuantizedActivations(0).synchronizeDToH();
        //conv1, kernel1
        CudaTensor<float> my_DiffFullPrecisionWeights_conv1 = cuda_tensor_cast<float>(quant1.getDiffFullPrecisionWeights(0));
        my_DiffFullPrecisionWeights_conv1.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv1][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights_conv1 << std::endl;

        //conv2 weights diff
        CudaTensor<float> my_DiffFullPrecisionWeights_conv2 = cuda_tensor_cast<float>(quant2.getDiffFullPrecisionWeights(0));
        my_DiffFullPrecisionWeights_conv2.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv2][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights_conv2 << std::endl;
        //conv2, activations diff
        CudaTensor<float> my_DiffQuantActivations_conv2 = cuda_tensor_cast<float>(quant2.getDiffQuantizedActivations(0));
        my_DiffQuantActivations_conv2.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv2][DiffQuantActivation]\n" << my_DiffQuantActivations_conv2 << std::endl;
        CudaTensor<float> my_QWeights_conv2 = cuda_tensor_cast<float>(quant2.getQuantizedWeights(0));
        my_QWeights_conv2.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv2][QuantizedWeights]\n" << my_QWeights_conv2 << std::endl;

        //CudaTensor<float> my_DiffFullPrecisionActivations_conv2 = cuda_tensor_cast<float>(quant2.getDiffFullPrecisionActivations(0));
        //my_DiffFullPrecisionActivations_conv2.synchronizeDToH();
        //std::cout << "[Conv2][DiffFullPrecisionActivation]\n" << my_DiffFullPrecisionActivations_conv2 << std::endl;
         CudaTensor<float> my_DiffInputs_conv2 = cuda_tensor_cast<float>(conv2.getDiffInputs());
        my_DiffInputs_conv2.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv2][DiffINputs]\n" << my_DiffInputs_conv2 << std::endl;


        //conv3 weights diff
        CudaTensor<float> my_DiffFullPrecisionWeights_conv3 = cuda_tensor_cast<float>(quant3.getDiffFullPrecisionWeights(0));
        my_DiffFullPrecisionWeights_conv3.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv3][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights_conv3 << std::endl;
        //conv3, activations diff
        CudaTensor<float> my_DiffQuantActivations_conv3 = cuda_tensor_cast<float>(quant3.getDiffQuantizedActivations(0));
        my_DiffQuantActivations_conv3.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv3][DiffQuantActivation]\n" << my_DiffQuantActivations_conv3 << std::endl;

        //quant1.getDiffFullPrecisionWeights(0).synchronizeHToD();
        //quant2.getDiffFullPrecisionWeights(0).synchronizeHToD();
        //quant3.getDiffFullPrecisionWeights(0).synchronizeHToD();
        //quant2.getDiffFullPrecisionActivations(0).synchronizeHToD();
        //quant3.getDiffFullPrecisionActivations(0).synchronizeHToD();
        
    }

        if(iter_index==9999)  std::cout << "end of backpropagate" << std::endl;

        if(iter_index==9999)  std::cout << "*****************UPDATE***************" << std::endl;

        conv3.update();
        if(doQuant){
            quant3.getAlpha(0).synchronizeDToH();
            CudaTensor<float> alphaEstimated3 = quant3.getAlpha(0);
            alphaEstimated3.synchronizeDToH();
            if(iter_index==9999) std::cout << "conv3 :: alphaEstimated = " << alphaEstimated3 << std::endl;
            quant3.getAlpha(0).synchronizeHToD();
        }
        
        

        conv2.update();
        if(doQuant){
            quant2.getAlpha(0).synchronizeDToH();
            CudaTensor<float> alphaEstimated2 = quant2.getAlpha(0);
            alphaEstimated2.synchronizeDToH();
            if(iter_index==9999) std::cout << "conv2 :: alphaEstimated = " << alphaEstimated2 << std::endl;
            quant2.getAlpha(0).synchronizeHToD();
        }
        
        
        conv1.update(); 
        if(doQuant){  
            quant1.getAlpha(0).synchronizeDToH();
            CudaTensor<float> alphaEstimated1 = quant1.getAlpha(0);
            alphaEstimated1.synchronizeDToH();
            if(iter_index==9999) std::cout << "conv1 :: alphaEstimated = " << alphaEstimated1 << std::endl;
            quant1.getAlpha(0).synchronizeHToD();
        }
             
        if(iter_index==9999) std::cout << "end of update" << std::endl;  
    }
    */
}



// double

// check 3 conv layers with 2 quantization levels
TEST_DATASET(ConvCell_QuantizerSAT_Frame_CUDA_double,
             check_miniMobileNet_with_SAT,
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
              size_t range1,
              double alpha1,
              size_t range2,
              double alpha2),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 5U, 5U, 255,8.0,15,8.0)
             )
{

    std::cout<<"double :: check_miniMobileNet_with_SAT"<<std::endl;
    //to avoid warning when compile
    std::cout << kernelWidth << kernelHeight << subSampleX << subSampleY 
                << strideX << strideY << paddingX << paddingY 
                << channelsWidth << channelsHeight 
                << range1 << alpha1 << range2 << alpha2 << std::endl; 

    /*
    bool doQuant = true;

    CudaContext::setDevice(0);
    const unsigned int nbOutputs_conv1 = 1;
    const unsigned int nbOutputs_conv2 = 4;
    const unsigned int nbOutputs_conv3 = 4;
    const unsigned int nbChannels = 1;

    unsigned int kernelWidth2 = 1;
    unsigned int kernelHeight2 = 1;
          
    Network net;
    DeepNet dn(net);
    unsigned int batchSize = 2;
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1}, batchSize);

    //Tensor<Float_T>& in = env.getData();
    Tensor<double> in;
    in.resize({channelsWidth, channelsHeight, 1, batchSize});

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);
    ///
    //fill input image
    double input_tmp = 0.0;

    for (unsigned int b = 0; b < in.dimB(); ++b) {
        for (unsigned int z = 0; z < in.dimZ(); ++z) {
            for (unsigned int y = 0; y < in.dimY(); ++y) {
                for (unsigned int x = 0; x < in.dimX(); ++x) {
                    if(b==0) {
                        if(x==0 && y==0) input_tmp = 1.0;
                        if(x==1 && y==0) input_tmp = 5.0;
                        if(x==2 && y==0) input_tmp = 8.0;
                        if(x==3 && y==0) input_tmp = 10.0;
                        if(x==4 && y==0) input_tmp = 30.0;

                        if(x==0 && y==1) input_tmp = 65.0;
                        if(x==1 && y==1) input_tmp = 70.0;
                        if(x==2 && y==1) input_tmp = 80.0;
                        if(x==3 && y==1) input_tmp = 50.0;
                        if(x==4 && y==1) input_tmp = 125.0;

                        if(x==0 && y==2) input_tmp = 29.0;
                        if(x==1 && y==2) input_tmp = 30.0;
                        if(x==2 && y==2) input_tmp = 165.0;
                        if(x==3 && y==2) input_tmp = 1.0;
                        if(x==4 && y==2) input_tmp = 1.0;

                        if(x==0 && y==3) input_tmp = 1.0;
                        if(x==1 && y==3) input_tmp = 1.0;
                        if(x==2 && y==3) input_tmp = 1.0;
                        if(x==3 && y==3) input_tmp = 1.0;
                        if(x==4 && y==3) input_tmp = 1.0;

                        if(x==0 && y==4) input_tmp = 1.0;
                        if(x==1 && y==4) input_tmp = 1.0;
                        if(x==2 && y==4) input_tmp = 1.0;
                        if(x==3 && y==4) input_tmp = 1.0;
                        if(x==4 && y==4) input_tmp = 1.0;
                    }
                    if(b==1) {
                        if(x==0 && y==0) input_tmp = 1.0;
                        if(x==1 && y==0) input_tmp = 5.0;
                        if(x==2 && y==0) input_tmp = 8.0;
                        if(x==3 && y==0) input_tmp = 10.0;
                        if(x==4 && y==0) input_tmp = 30.0;

                        if(x==0 && y==1) input_tmp = 65.0;
                        if(x==1 && y==1) input_tmp = 70.0;
                        if(x==2 && y==1) input_tmp = 80.0;
                        if(x==3 && y==1) input_tmp = 50.0;
                        if(x==4 && y==1) input_tmp = 125.0;

                        if(x==0 && y==2) input_tmp = 29.0;
                        if(x==1 && y==2) input_tmp = 30.0;
                        if(x==2 && y==2) input_tmp = 73.0;
                        if(x==3 && y==2) input_tmp = 1.0;
                        if(x==4 && y==2) input_tmp = 1.0;

                        if(x==0 && y==3) input_tmp = 1.0;
                        if(x==1 && y==3) input_tmp = 55.0;
                        if(x==2 && y==3) input_tmp = 1.0;
                        if(x==3 && y==3) input_tmp = 1.0;
                        if(x==4 && y==3) input_tmp = 1.0;

                        if(x==0 && y==4) input_tmp = 1.0;
                        if(x==1 && y==4) input_tmp = 1.0;
                        if(x==2 && y==4) input_tmp = 1.0;
                        if(x==3 && y==4) input_tmp = 56.0;
                        if(x==4 && y==4) input_tmp = 1.0;
                    }

                    in(x, y, z, b) = input_tmp/255.0;
                    //std::cout  << "b, z, y, x = " << b << ", " << z << ", " << y << ", " << x << ", input = " << in(x, y, z, b) << std::endl;
                }
            }
        }
    }

    std::cout << "[Input]\n" << in << std::endl;

    ConvCell_QuantizerSAT_Frame_CUDA_Test<double> conv1(dn, "conv1",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv1,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1.setParameter("NoBias", true);

    ConvCell_QuantizerSAT_Frame_CUDA_Test<double> conv2(dn, "conv2",
        std::vector<unsigned int>({kernelWidth2, kernelHeight2}),
        nbOutputs_conv2,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv2.setParameter("NoBias", true);

    ////create a map to make a conv depthwise layer
    Tensor<bool> mapping;
    mapping.resize({nbOutputs_conv3, nbOutputs_conv3});
    mapping.fill(0);
    for(size_t out = 0; out < nbOutputs_conv3; ++out)
    {
        mapping(out, out) = 1;
    }

    ConvCell_QuantizerSAT_Frame_CUDA_Test<double> conv3(dn, "conv3",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv3,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv3.setParameter("NoBias", true);

    SATQuantizer_Frame_CUDA<double> quant1;
    quant1.setWeightsRange(range1);
    quant1.setActivationsRange(range1);
    quant1.setAlpha(alpha1);
    quant1.setQuantization(true);
    quant1.setScaling(false);
    std::shared_ptr<Quantizer> quantizer1 = std::shared_ptr<Quantizer>(&quant1, [](Quantizer *) {});

    SATQuantizer_Frame_CUDA<double> quant2;
    quant2.setWeightsRange(range2);
    quant2.setActivationsRange(range2);
    quant2.setAlpha(alpha2);
    quant2.setQuantization(true);
    quant2.setScaling(false);
    std::shared_ptr<Quantizer> quantizer2 = std::shared_ptr<Quantizer>(&quant2, [](Quantizer *) {});

    SATQuantizer_Frame_CUDA<double> quant3;
    quant3.setWeightsRange(range2);
    quant3.setActivationsRange(range2);
    quant3.setAlpha(alpha2);
    quant3.setQuantization(true);
    quant3.setScaling(false);
    std::shared_ptr<Quantizer> quantizer3 = std::shared_ptr<Quantizer>(&quant3, [](Quantizer *) {});

    SoftmaxCell_Frame_CUDA<double> softmax1(dn, "softmax1", nbOutputs_conv3, true, 0);

    Tensor<double> out_diff({channelsWidth, channelsHeight, 1, batchSize});
    conv1.addInput(in, out_diff);
    conv2.addInput(&conv1);
    conv3.addInput(&conv2, mapping);
    softmax1.addInput(&conv3);

    if(doQuant) conv1.setQuantizer(quantizer1);
    conv1.initialize();
    if(doQuant) conv2.setQuantizer(quantizer2);
    conv2.initialize();
    if(doQuant) conv3.setQuantizer(quantizer3);
    conv3.initialize();
    softmax1.initialize();

    if(conv1.getQuantizer()){
        std::cout << "Added " <<  conv1.getQuantizer()->getType() <<
        " quantizer to " << conv1.getName() << std::endl;
    }
    if(conv2.getQuantizer()){
        std::cout << "Added " <<  conv2.getQuantizer()->getType() <<
        " quantizer to " << conv2.getName() << std::endl;
    }
    if(conv3.getQuantizer()){
        std::cout << "Added " <<  conv3.getQuantizer()->getType() <<
        " quantizer to " << conv3.getName() << std::endl;
    }
    
    float weight_tmp = 0.0;

    // set weights for conv1
    for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
        for (unsigned int channel = 0; channel < nbChannels;
             ++channel) {
            Tensor<float> kernel({kernelWidth,
                                   kernelHeight});
            for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                    if (sy==0 && sx==0) weight_tmp = 0.025;
                    if (sy==0 && sx==1) weight_tmp = 0.5;
                    if (sy==0 && sx==2) weight_tmp = 0.075;

                    if (sy==1 && sx==0) weight_tmp = -0.01;
                    if (sy==1 && sx==1) weight_tmp = 0.01;
                    if (sy==1 && sx==2) weight_tmp = -0.01;

                    if (sy==2 && sx==0) weight_tmp = 0.35;
                    if (sy==2 && sx==1) weight_tmp = -0.5;
                    if (sy==2 && sx==2) weight_tmp = 0.2;

                    kernel(sx, sy) = weight_tmp;
                    //std::cout << "conv1 :: sx = " << sx << " , sy = " << sy << " , weight = " << kernel(sx, sy) << std::endl;
                }
            }
            conv1.setWeight(output, channel, kernel);
        }
    }

    // set weights for conv2
    // [[[0.01]]], [[[0.01]]], [[[0.01]]], [[[0.01]]
    weight_tmp = 0.0;
    for (unsigned int output = 0; output < nbOutputs_conv2; ++output) {
        for (unsigned int channel = 0; channel < nbChannels;
             ++channel) {
            Tensor<double> kernel({kernelWidth2,
                                   kernelHeight2});

            for (unsigned int sx = 0; sx < kernelWidth2; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight2; ++sy){
                    if(output==0){
                        weight_tmp = 0.01;
                    }
                    if(output==1){
                        weight_tmp = 0.01;            
                    }
                    if(output==2){
                        weight_tmp = 0.01;            
                    }
                    if(output==3){
                        weight_tmp = 0.01;            
                    }
                    kernel(sx, sy) = weight_tmp;
                    //std::cout << "conv2 :: sx = " << sx << " , sy = " << sy << " , weight = " << kernel(sx, sy) << std::endl;
                }
            }
            conv2.setWeight(output, channel, kernel);
        }
    }

    // set weights for conv3
    weight_tmp = 0.0;
    for (unsigned int output = 0; output < nbOutputs_conv3; ++output) {
            Tensor<double> kernel({kernelWidth,
                                   kernelHeight});

            for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                    weight_tmp = 0.0f;
                    if(output==0){
                        if (sy==0 && sx==0) weight_tmp = 0.01;
                        if (sy==0 && sx==1) weight_tmp = -0.013;
                        if (sy==0 && sx==2) weight_tmp = 0.01;

                        if (sy==1 && sx==0) weight_tmp = -0.01;
                        if (sy==1 && sx==1) weight_tmp = 0.5;
                        if (sy==1 && sx==2) weight_tmp = -0.013;

                        if (sy==2 && sx==0) weight_tmp = 0.01;
                        if (sy==2 && sx==1) weight_tmp = -0.01;
                        if (sy==2 && sx==2) weight_tmp = 0.01;
                    }
                    if(output==1){
                        if (sy==0 && sx==0) weight_tmp = 0.01;
                        if (sy==0 && sx==1) weight_tmp = -0.01;
                        if (sy==0 && sx==2) weight_tmp = 0.01;

                        if (sy==1 && sx==0) weight_tmp = -0.01;
                        if (sy==1 && sx==1) weight_tmp = 0.5;
                        if (sy==1 && sx==2) weight_tmp = -0.01;

                        if (sy==2 && sx==0) weight_tmp = 0.01;
                        if (sy==2 && sx==1) weight_tmp = -0.01;
                        if (sy==2 && sx==2) weight_tmp = 0.01;
                    }
                    if(output==2){
                        if (sy==0 && sx==0) weight_tmp = 0.01;
                        if (sy==0 && sx==1) weight_tmp = -0.01;
                        if (sy==0 && sx==2) weight_tmp = 0.01;

                        if (sy==1 && sx==0) weight_tmp = -0.01;
                        if (sy==1 && sx==1) weight_tmp = 0.5;
                        if (sy==1 && sx==2) weight_tmp = -0.01;

                        if (sy==2 && sx==0) weight_tmp = 0.013;
                        if (sy==2 && sx==1) weight_tmp = -0.01;
                        if (sy==2 && sx==2) weight_tmp = 0.01;  
                    }
                    if(output==3){
                        if (sy==0 && sx==0) weight_tmp = 0.1;
                        if (sy==0 && sx==1) weight_tmp = -0.013;
                        if (sy==0 && sx==2) weight_tmp = 0.01;

                        if (sy==1 && sx==0) weight_tmp = -0.01;
                        if (sy==1 && sx==1) weight_tmp = 0.9;
                        if (sy==1 && sx==2) weight_tmp = -0.01;

                        if (sy==2 && sx==0) weight_tmp = 0.01;
                        if (sy==2 && sx==1) weight_tmp = -0.01;
                        if (sy==2 && sx==2) weight_tmp = 0.3;  
                    }
                    kernel(sx, sy) = weight_tmp;
                    //std::cout << "conv3 :: output = "<< output << ", sx = " << sx << " , sy = " << sy << " , weight = " << kernel(sx, sy) << std::endl;
                }
            }
            conv3.setWeight(output, output, kernel);
    }
    
    //several iterations for propagate, backpropagate, update
    for(unsigned int iter_index = 0; iter_index < 10000; ++iter_index){

        if(iter_index==9999) std::cout << "iteration #" << iter_index << std::endl;
        if(iter_index==9999) std::cout << "===============================================================" << std::endl;

        if(iter_index==9999) std::cout << "******************PROPAGATE*******************\n\n\n" << std::endl;

        conv1.propagate(false);
        conv2.propagate(false);
        conv3.propagate(false);
        softmax1.propagate(false);

        conv1.getOutputs().synchronizeDToH();
        const Tensor<double>& out_conv1 = tensor_cast<double>(conv1.getOutputs());
        if(iter_index==9999) {
            std::cout << "[Conv1][Outputs]" << std::endl;
            std::cout << out_conv1 << std::endl;
        }
        conv1.getOutputs().synchronizeHToD();

        conv2.getOutputs().synchronizeDToH();
        const Tensor<double>& out_conv2 = tensor_cast<double>(conv2.getOutputs());
        if(iter_index==9999) {
            std::cout << "[Conv2][Outputs]" << std::endl;
            std::cout << out_conv2 << std::endl;
        }
        conv2.getOutputs().synchronizeHToD();

        conv3.getOutputs().synchronizeDToH();
        const Tensor<double>& out_conv3 = tensor_cast<double>(conv3.getOutputs());
        if(iter_index==9999) {
            std::cout << "[Conv3][Outputs]" << std::endl;
            std::cout << out_conv3 << std::endl;
        }
        conv3.getOutputs().synchronizeHToD();

        softmax1.mDiffInputs.synchronizeDToH();
        softmax1.getOutputs().synchronizeDToH();
        const CudaTensor<double>& out_softmax1 = cuda_tensor_cast<double>(softmax1.getOutputs());
        double loss = 0.0;
        if(iter_index==9999) {
            std::cout << "[SoftMax][Outputs]" << std::endl;
            std::cout << out_softmax1 << std::endl;
        }

        for(unsigned int nout = 0; nout < nbOutputs_conv3; ++nout){
            for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos){
                if(batchPos == 0) {
                    if(nout==0) {
                        softmax1.mDiffInputs(nout, batchPos) = 1.0;
                    }
                    else
                        softmax1.mDiffInputs(nout, batchPos) = 0.0; 
                }
                if(batchPos == 1){
                    if(nout==3) {
                        softmax1.mDiffInputs(nout, batchPos) = 1.0;
                    }
                    else
                        softmax1.mDiffInputs(nout, batchPos) = 0.0; 
                }
                //std::cout << "softmax1.mDiffInputs(nout, batchPos) = " << softmax1.mDiffInputs(nout, batchPos) << std::endl;
            }
        }

        loss = softmax1.applyLoss();
        //std::cout << "test loss = " << loss << std::endl;
        softmax1.mDiffInputs.synchronizeHToD();
        softmax1.getOutputs().synchronizeHToD();

        //std::cout <<"end of propagate" << std::endl;

        //backpropagate 
        softmax1.backPropagate();  
        conv3.backPropagate(); 
        conv2.backPropagate();
        conv1.backPropagate();

        if(iter_index==9999) std::cout << "****************BACKPROPAGATE******************" << std::endl;

    if(doQuant){
        
        //quant1.getDiffFullPrecisionWeights(0).synchronizeDToH();
        //quant2.getDiffFullPrecisionWeights(0).synchronizeDToH();
        //quant3.getDiffFullPrecisionWeights(0).synchronizeDToH();
        //quant2.getDiffFullPrecisionActivations(0).synchronizeDToH();
        //quant3.getDiffFullPrecisionActivations(0).synchronizeDToH();

        //quant2.getDiffQuantizedActivations(0).synchronizeDToH();
        //quant3.getDiffQuantizedActivations(0).synchronizeDToH();
        //conv1, kernel1
        CudaTensor<double> my_DiffFullPrecisionWeights_conv1 = cuda_tensor_cast<double>(quant1.getDiffFullPrecisionWeights(0));
        my_DiffFullPrecisionWeights_conv1.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv1][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights_conv1 << std::endl;

        //conv2 weights diff
        CudaTensor<double> my_DiffFullPrecisionWeights_conv2 = cuda_tensor_cast<double>(quant2.getDiffFullPrecisionWeights(0));
        my_DiffFullPrecisionWeights_conv2.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv2][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights_conv2 << std::endl;
        //conv2, activations diff
        CudaTensor<double> my_DiffQuantActivations_conv2 = cuda_tensor_cast<double>(quant2.getDiffQuantizedActivations(0));
        my_DiffQuantActivations_conv2.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv2][DiffQuantActivation]\n" << my_DiffQuantActivations_conv2 << std::endl;
        CudaTensor<double> my_QWeights_conv2 = cuda_tensor_cast<double>(quant2.getQuantizedWeights(0));
        my_QWeights_conv2.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv2][QuantizedWeights]\n" << my_QWeights_conv2 << std::endl;

        //CudaTensor<float> my_DiffFullPrecisionActivations_conv2 = cuda_tensor_cast<float>(quant2.getDiffFullPrecisionActivations(0));
        //my_DiffFullPrecisionActivations_conv2.synchronizeDToH();
        //std::cout << "[Conv2][DiffFullPrecisionActivation]\n" << my_DiffFullPrecisionActivations_conv2 << std::endl;
         CudaTensor<double> my_DiffInputs_conv2 = cuda_tensor_cast<double>(conv2.getDiffInputs());
        my_DiffInputs_conv2.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv2][DiffINputs]\n" << my_DiffInputs_conv2 << std::endl;


        //conv3 weights diff
        CudaTensor<double> my_DiffFullPrecisionWeights_conv3 = cuda_tensor_cast<double>(quant3.getDiffFullPrecisionWeights(0));
        my_DiffFullPrecisionWeights_conv3.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv3][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights_conv3 << std::endl;
        //conv3, activations diff
        CudaTensor<double> my_DiffQuantActivations_conv3 = cuda_tensor_cast<double>(quant3.getDiffQuantizedActivations(0));
        my_DiffQuantActivations_conv3.synchronizeDToH();
        if(iter_index==9999) std::cout << "[Conv3][DiffQuantActivation]\n" << my_DiffQuantActivations_conv3 << std::endl;

        //quant1.getDiffFullPrecisionWeights(0).synchronizeHToD();
        //quant2.getDiffFullPrecisionWeights(0).synchronizeHToD();
        //quant3.getDiffFullPrecisionWeights(0).synchronizeHToD();
        //quant2.getDiffFullPrecisionActivations(0).synchronizeHToD();
        //quant3.getDiffFullPrecisionActivations(0).synchronizeHToD();
        
    }

        if(iter_index==9999) std::cout << "end of backpropagate" << std::endl;

        if(iter_index==9999) std::cout << "*****************UPDATE***************" << std::endl;

        conv3.update();
        if(doQuant){
            quant3.getAlpha(0).synchronizeDToH();
            CudaTensor<double> alphaEstimated3 = quant3.getAlpha(0);
            alphaEstimated3.synchronizeDToH();
            if(iter_index==9999) std::cout << "conv3 :: alphaEstimated = " << alphaEstimated3 << std::endl;
            quant3.getAlpha(0).synchronizeHToD();
        }
        
        

        conv2.update();
        if(doQuant){
            quant2.getAlpha(0).synchronizeDToH();
            CudaTensor<double> alphaEstimated2 = quant2.getAlpha(0);
            alphaEstimated2.synchronizeDToH();
            if(iter_index==9999) std::cout << "conv2 :: alphaEstimated = " << alphaEstimated2 << std::endl;
            quant2.getAlpha(0).synchronizeHToD();
        }
        
        
        conv1.update(); 
        if(doQuant){  
            quant1.getAlpha(0).synchronizeDToH();
            CudaTensor<double> alphaEstimated1 = quant1.getAlpha(0);
            alphaEstimated1.synchronizeDToH();
            if(iter_index==9999) std::cout << "conv1 :: alphaEstimated = " << alphaEstimated1 << std::endl;
            quant1.getAlpha(0).synchronizeHToD();
        }
            
        if(iter_index==9999) std::cout << "end of update" << std::endl;  
    }
    */
}


/*
TEST_DATASET(ConvCell_QuantizerSAT_Frame_CUDA_float,
             check_gradient_SAT,
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
              float full_weights,
              size_t range,
              float alpha),
             //std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 5U, 5U,1.0,255,2.0)
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 5U, 5U,1.0,15,2.0)
             )
{

    CudaContext::setDevice(0);
    const unsigned int nbOutputs = 1;
    const unsigned int nbChannels = 1;

    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    Tensor<Float_T>& in = env.getData();
    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    //fill one input image with 0.5
    for (unsigned int b = 0; b < in.dimB(); ++b) {
        for (unsigned int z = 0; z < in.dimZ(); ++z) {
            for (unsigned int y = 0; y < in.dimY(); ++y) {
                for (unsigned int x = 0; x < in.dimX(); ++x) {
                    //std::cout << "b, z, y, x = " << b << " , " << z << " , " << y << " , " << x << std::endl;
                    //std::cout << "in(x, y, z, b) = " << in(x, y, z, b) << std::endl;
                    in(x, y, z, b) = 0.5f;
                    //std::cout << "in(x, y, z, b) = " << in(x, y, z, b) << std::endl;
                }
            }
        }
    }

#if CUDNN_VERSION >= 5000
    DropoutCell_Frame_CUDA<Float_T> drop1(dn, "drop1", 1);
    drop1.setParameter<double>("Dropout", 0.0);
#endif

    ConvCell_QuantizerSAT_Frame_CUDA_Test<float> conv1(dn, "conv1",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1.setParameter("NoBias", true);

    SATQuantizer_Frame_CUDA<float> quant;
    quant.setRange(range);
    quant.setAlpha(alpha);
    quant.setQuantization(true);
    quant.setScaling(false);
    std::shared_ptr<Quantizer> quantizer = std::shared_ptr<Quantizer>(&quant, [](Quantizer *) {});

#if CUDNN_VERSION >= 5000
    drop1.addInput(in,in);
    conv1.addInput(&drop1);
    drop1.initialize();
#else
    conv1.addInput(in,in);
#endif
    conv1.setQuantizer(quantizer);
    conv1.initialize();
    
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        ASSERT_EQUALS(conv1.isConnection(0, output), true);
    }

    const unsigned int outputsWidth = std::ceil(
        std::floor((channelsWidth + 2 * paddingX - kernelWidth + strideX)
                   / (double)strideX) / (double)subSampleX);
    const unsigned int outputsHeight = std::ceil(
        std::floor((channelsHeight + 2 * paddingY - kernelHeight + strideY)
                   / (double)strideY) / (double)subSampleY);

    ASSERT_EQUALS(conv1.getNbSharedSynapses(),
                  kernelWidth * kernelHeight * nbOutputs);
    ASSERT_EQUALS(conv1.getNbChannels(), 1U);
    ASSERT_EQUALS(conv1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(conv1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(conv1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(conv1.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(conv1.getOutputsHeight(), outputsHeight);

    int count = 0;
    int count_ind = 0;

    
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int channel = 0; channel < nbChannels;
             ++channel) {
            Tensor<float> kernel({kernelWidth,
                                   kernelHeight});

            for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                    if(count_ind < 5) {
                        count_ind++;
                        count++;
                    }
                    else {
                        count_ind++;
                        count--;
                    }
                    kernel(sx, sy) = full_weights*count*0.1 + channel + conv1.getNbChannels()
                                                    * output;
                    //std::cout << "channel = " << channel << " , output = " << output << std::endl;
                    std::cout << "sx = " << sx << " , sy = " << sy << " , kernel(sx, sy) = " << kernel(sx,sy) << std::endl;
                }
            }
            conv1.setWeight(output, channel, kernel);
        }
    }
    

    conv1.checkGradient(alpha, range);
    //ASSERT_NOTHROW_ANY(conv1.checkGradient(alpha, range));
  
}
*/

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
