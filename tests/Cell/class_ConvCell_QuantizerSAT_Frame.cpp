/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
                    David BRIAND (david.briand@cea.fr)
                    Inna KUCHER (inna.kucher@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include "N2D2.hpp"

#include "Cell/ConvCell_Frame.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "DeepNet.hpp"
#include "Environment.hpp"
#include "Network.hpp"
#if CUDNN_VERSION >= 5000
#include "Cell/DropoutCell_Frame.hpp"
#include "Cell/SoftmaxCell_Frame.hpp"
#endif
#include "Transformation/RescaleTransformation.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"
#include "Quantizer/SATQuantizer_Frame.hpp"

using namespace N2D2;

template <class T>
class ConvCell_QuantizerSAT_Frame_Test : public ConvCell_Frame<T> {
public:
    ConvCell_QuantizerSAT_Frame_Test(const DeepNet& deepNet, 
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
          ConvCell_Frame<T>(deepNet, name,
                              kernelDims,
                              nbOutputs,
                              subSampleDims,
                              strideDims,
                              paddingDims,
                              dilationDims,
                              activation 
                              ) {};

    //friend class UnitTest_ConvCell_QuantizerSAT_Frame_float_check_one_layer_with_SAT;
    //friend class UnitTest_ConvCell_QuantizerSAT_Frame_float_check_2conv_layers_with_SAT;
    friend class UnitTest_ConvCell_QuantizerSAT_Frame_float_check_miniMobileNet_with_SAT;
    //friend class UnitTest_ConvCell_QuantizerSAT_Frame_float_check_gradient_SAT;


    
};


// check 3 conv layers with 2 quantization levels
TEST_DATASET(ConvCell_QuantizerSAT_Frame_float,
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

    std::cout<<"check_miniMobileNet_with_SAT"<<std::endl;

    bool doQuant = true;

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

    ConvCell_QuantizerSAT_Frame_Test<float> conv1(dn, "conv1",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv1,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1.setParameter("NoBias", true);

    ConvCell_QuantizerSAT_Frame_Test<float> conv2(dn, "conv2",
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

    ConvCell_QuantizerSAT_Frame_Test<float> conv3(dn, "conv3",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv3,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv3.setParameter("NoBias", true);

    SATQuantizer_Frame<float> quant1;
    quant1.setWeightsRange(range1);
    quant1.setActivationsRange(range1);
    quant1.setAlpha(alpha1);
    quant1.setQuantization(true);
    quant1.setScaling(false);
    std::shared_ptr<Quantizer> quantizer1 = std::shared_ptr<Quantizer>(&quant1, [](Quantizer *) {});

    SATQuantizer_Frame<float> quant2;
    quant2.setWeightsRange(range2);
    quant2.setActivationsRange(range2);
    quant2.setAlpha(alpha2);
    quant2.setQuantization(true);
    quant2.setScaling(false);
    std::shared_ptr<Quantizer> quantizer2 = std::shared_ptr<Quantizer>(&quant2, [](Quantizer *) {});

    SATQuantizer_Frame<float> quant3;
    quant3.setWeightsRange(range2);
    quant3.setActivationsRange(range2);
    quant3.setAlpha(alpha2);
    quant3.setQuantization(true);
    quant3.setScaling(false);
    std::shared_ptr<Quantizer> quantizer3 = std::shared_ptr<Quantizer>(&quant3, [](Quantizer *) {});

    SoftmaxCell_Frame<float> softmax1(dn, "softmax1", nbOutputs_conv3, true, 0);

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
    /*
    [[0.025, 0.5, 0.075],
    [-0.01, 0.01, -0.01],
    [0.35, -0.5, 0.2]]
    */
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
    /*
    [[[0.01, -0.013, 0.01],
    [-0.01, 0.5, -0.013],
    [0.01, -0.01, 0.01]]],
                          
    [[[0.01, -0.01, 0.01],
    [-0.01, 0.5, -0.01],
    [0.01, -0.01, 0.01]]],

    [[[0.01, -0.01, 0.01],
    [-0.01, 0.5, -0.01],
    [0.013, -0.01, 0.01]]],
                          
    [[[0.10, -0.013, 0.01],
    [-0.01, 0.9, -0.01],
    [0.01, -0.01, 0.30]]],
    */

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

    /*
        check init weights for conv3
    */
    /*
    std::cout << "conv3 weights init (from conv cell) :" << std::endl;
    for (unsigned int output = 0; output < nbOutputs_conv3; ++output) {
            Tensor<float> weight;
            conv3.getWeight(output, output, weight);
            std::cout << "output = " << output << " , channel =  " << output << " , weight = " << weight << std::endl;
    }
    */
    
    
    //several iterations for propagate, backpropagate, update
    for(unsigned int iter_index = 0; iter_index < 10000; ++iter_index){

        std::cout << "iteration #" << iter_index << std::endl;
        std::cout << "===============================================================" << std::endl;

        std::cout << "******************PROPAGATE*******************\n\n\n" << std::endl;

        conv1.propagate(false);
        conv2.propagate(false);
        conv3.propagate(false);
        softmax1.propagate(false);

        const Tensor<float>& out_conv1 = tensor_cast<float>(conv1.getOutputs());
        std::cout << "[Conv1][Outputs]" << std::endl;
        std::cout << out_conv1 << std::endl;

        const Tensor<float>& out_conv2 = tensor_cast<float>(conv2.getOutputs());
        std::cout << "[Conv2][Outputs]" << std::endl;
        std::cout << out_conv2 << std::endl;

        const Tensor<float>& out_conv3 = tensor_cast<float>(conv3.getOutputs());
        std::cout << "[Conv3][Outputs]" << std::endl;
        std::cout << out_conv3 << std::endl;

        const Tensor<float>& out_softmax1 = tensor_cast<float>(softmax1.getOutputs());
        double loss = 0.0f;
        std::cout << "[SoftMax][Outputs]" << std::endl;
        std::cout << out_softmax1 << std::endl;

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
            }
        }

        loss = softmax1.applyLoss();
        //std::cout << "test loss = " << loss << std::endl;
        softmax1.mDiffInputs.setValid();
        softmax1.mDiffInputs.synchronizeHToD();
        //softmax1.getOutputs().synchronizeHToD();

        std::cout <<"end of propagate" << std::endl;

        //backpropagate 
        softmax1.backPropagate();  
        conv3.backPropagate(); 
        conv2.backPropagate();
        conv1.backPropagate();

        std::cout << "****************BACKPROPAGATE******************" << std::endl;

    if(doQuant){
        
        //conv1, kernel1
        Tensor<float> my_DiffFullPrecisionWeights_conv1 = tensor_cast<float>(quant1.getDiffFullPrecisionWeights(0));
        std::cout << "[Conv1][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights_conv1 << std::endl;

        //conv2 weights diff
        Tensor<float> my_DiffFullPrecisionWeights_conv2 = tensor_cast<float>(quant2.getDiffFullPrecisionWeights(0));
        std::cout << "[Conv2][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights_conv2 << std::endl;
        //conv2, activations diff
        Tensor<float> my_DiffQuantActivations_conv2 = tensor_cast<float>(quant2.getDiffQuantizedActivations(0));
        std::cout << "[Conv2][DiffQuantActivation]\n" << my_DiffQuantActivations_conv2 << std::endl;
        Tensor<float> my_QWeights_conv2 = tensor_cast<float>(quant2.getQuantizedWeights(0));
        std::cout << "[Conv2][QuantizedWeights]\n" << my_QWeights_conv2 << std::endl;

        //std::cout << "[Conv2][DiffFullPrecisionActivation]\n" << my_DiffFullPrecisionActivations_conv2 << std::endl;
        Tensor<float> my_DiffInputs_conv2 = tensor_cast<float>(conv2.getDiffInputs());
        std::cout << "[Conv2][DiffINputs]\n" << my_DiffInputs_conv2 << std::endl;


        //conv3 weights diff
        Tensor<float> my_DiffFullPrecisionWeights_conv3 = tensor_cast<float>(quant3.getDiffFullPrecisionWeights(0));
        std::cout << "[Conv3][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights_conv3 << std::endl;
        //conv3, activations diff
        Tensor<float> my_DiffQuantActivations_conv3 = tensor_cast<float>(quant3.getDiffQuantizedActivations(0));
        std::cout << "[Conv3][DiffQuantActivation]\n" << my_DiffQuantActivations_conv3 << std::endl;
        
    }

        std::cout << "end of backpropagate" << std::endl;

        std::cout << "*****************UPDATE***************" << std::endl;

        conv3.update();
        if(doQuant){
            Tensor<float> alphaEstimated3 = quant3.getAlpha(0);
            std::cout << "conv3 :: alphaEstimated = " << alphaEstimated3 << std::endl;
        }
        
        

        conv2.update();
        if(doQuant){
            Tensor<float> alphaEstimated2 = quant2.getAlpha(0);
            std::cout << "conv2 :: alphaEstimated = " << alphaEstimated2 << std::endl;
        }
        
        
        conv1.update(); 
        if(doQuant){  
            Tensor<float> alphaEstimated1 = quant1.getAlpha(0);
            std::cout << "conv1 :: alphaEstimated = " << alphaEstimated1 << std::endl;
        }
             
    /*
        std::cout << "conv1 weights after update : " << std::endl;
        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                Tensor<float> weight;
                conv1.getWeight(output, channel, weight);
                std::cout << weight << std::endl;
            }
        }

        std::cout << "conv2 weights after update :" << std::endl;
        for (unsigned int output = 0; output < nbOutputs_conv2; ++output) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                Tensor<float> weight;
                conv2.getWeight(output, channel, weight);
                std::cout <<  weight << std::endl;
            }
        }

        std::cout << "conv3 weights after update :" << std::endl;
        for (unsigned int output = 0; output < nbOutputs_conv3; ++output) {
                Tensor<float> weight;
                conv3.getWeight(output, output, weight);
                std::cout << weight << std::endl;
        }
*/
        std::cout << "end of update" << std::endl;  
    }
}

RUN_TESTS()

