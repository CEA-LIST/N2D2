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
#include "Environment.hpp"
#include "Network.hpp"
#if CUDNN_VERSION >= 5000
#include "Cell/DropoutCell_Frame_CUDA.hpp"
#include "Cell/SoftmaxCell_Frame_CUDA.hpp"
#include "Cell/BatchNormCell_Frame_CUDA.hpp"
#endif
#include "Transformation/RescaleTransformation.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"
#include "Quantizer/SATQuantizer_Frame_CUDA.hpp"

using namespace N2D2;

template <class T>
class ConvCell_QuantizerSAT_BNFusion_Frame_CUDA_Test : public ConvCell_Frame_CUDA<T> {
public:
    ConvCell_QuantizerSAT_BNFusion_Frame_CUDA_Test(const DeepNet& deepNet, 
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

    friend class UnitTest_ConvCell_QuantizerSAT_BNFusion_Frame_CUDA_float_check_BNFusion_with_SAT;
};



// check 2 conv layers with 8 bits quantization, float
// insert bn1 between conv1 and conv2 and fuse conv1 and bn1

TEST_DATASET(ConvCell_QuantizerSAT_BNFusion_Frame_CUDA_float,
             check_BNFusion_with_SAT,
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
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 5U, 5U, 255,1.0,255,5.0)
             )
{

    std::cout<<"float :: check BN fusion with conv when using SAT quantizer"<<std::endl;

    bool doQuant = true;

    CudaContext::setDevice(0);
    const unsigned int nbOutputs_conv1 = 1;
    const unsigned int nbOutputs_conv2 = 4;
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

    ConvCell_QuantizerSAT_BNFusion_Frame_CUDA_Test<float> conv1(dn, "conv1",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv1,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1.setParameter("NoBias", true);

    BatchNormCell_Frame_CUDA<float> bn1(dn, "bn1", nbOutputs_conv1, std::shared_ptr<Activation>());

    ConvCell_QuantizerSAT_BNFusion_Frame_CUDA_Test<float> conv2(dn, "conv2",
        std::vector<unsigned int>({kernelWidth2, kernelHeight2}),
        nbOutputs_conv2,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv2.setParameter("NoBias", true);

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

    Tensor<float> out_diff({channelsWidth, channelsHeight, 1, batchSize});
    conv1.addInput(in, out_diff);
    bn1.addInput(&conv1);
    conv2.addInput(&bn1);

    if(doQuant) conv1.setQuantizer(quantizer1);
    conv1.initialize();
    bn1.initialize();
    if(doQuant) conv2.setQuantizer(quantizer2);
    conv2.initialize();

    if(conv1.getQuantizer()){
        std::cout << "Added " <<  conv1.getQuantizer()->getType() <<
        " quantizer to " << conv1.getName() << std::endl;
    }
    if(conv2.getQuantizer()){
        std::cout << "Added " <<  conv2.getQuantizer()->getType() <<
        " quantizer to " << conv2.getName() << std::endl;
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

    for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
        Tensor<float> scale({1}, Random::randNormal(1.0, 0.5));
        Tensor<float> bias({1}, Random::randUniform(-0.5, 0.5));
        Tensor<float> mean({1}, Random::randUniform(-0.5, 0.5));
        Tensor<float> variance({1}, Random::randUniform(0.0, 0.15));

        bn1.setScale(output, scale);
        bn1.setBias(output, bias);
        bn1.setMean(output, mean);
        bn1.setVariance(output, variance);
        std::cout << "bn1 :: scale = " << scale << " , bias = " << bias << " , mean = " << mean << " , variance = " << variance << std::endl;
    }
    
    for(unsigned int iter_index = 0; iter_index < 1; ++iter_index){

        std::cout << "iteration #" << iter_index << std::endl;
        std::cout << "===============================================================" << std::endl;

        std::cout << "******************PROPAGATE*******************\n\n\n" << std::endl;

        conv1.propagate(true);
        bn1.propagate(true);
        conv2.propagate(true);

        conv1.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv1 = tensor_cast<float>(conv1.getOutputs());
        std::cout << "[Conv1][Outputs]" << std::endl;
        std::cout << out_conv1 << std::endl;
        conv1.getOutputs().synchronizeHToD();

        bn1.getOutputs().synchronizeDToH();
        const Tensor<float>& out_bn1 = tensor_cast<float>(bn1.getOutputs());
        std::cout << "[BN1][Outputs]" << std::endl;
        std::cout << out_bn1 << std::endl;
        bn1.getOutputs().synchronizeHToD();

        quant2.getQuantizedActivations(0).synchronizeDToH();
        const Tensor<float>& quant_act_conv2 = tensor_cast<float>(quant2.getQuantizedActivations(0));
        std::cout << "[Conv2][Quant Input]" << std::endl;
        std::cout << quant_act_conv2 << std::endl;
        quant2.getQuantizedActivations(0).synchronizeHToD();

        /*
        conv2.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv2 = tensor_cast<float>(conv2.getOutputs());
        std::cout << "[Conv2][Outputs]" << std::endl;
        std::cout << out_conv2 << std::endl;
        conv2.getOutputs().synchronizeHToD();
        */

        // ===> fuse BN by hands following SAT paper logic in S7.

        // a1 and alpha1 are from conv1; a2 and alpha2 are from conv2;
        // 0. create conv1_quant layer with inputs in range [0,255] and quantized weights from conv1 rescaled to [-127, 128]
        // 1. get BN gamma and beta (as done in DeepNet.cpp), these parameters are "-per output", for now 1 output in conv1
        // 2. add bias : beta/gamma *  a1/alpha1
        // 3. clip : [0, alpha2/gamma * a1/alpha1]
        // 4. scale : alpha1/a1 * a2/alpha2 * gamma
        // 5. round
        // to compare the result after "round" with quant_act_conv2 :: /255 and * alpha2/a2
    }
}


RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
