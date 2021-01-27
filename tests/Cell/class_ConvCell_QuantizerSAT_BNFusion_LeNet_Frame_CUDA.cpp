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
#include "Cell/BatchNormCell_Frame_CUDA.hpp"
#endif
#include "Transformation/RescaleTransformation.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"
#include "Quantizer/SATQuantizer_Frame_CUDA.hpp"

using namespace N2D2;

static MNIST_IDX_Database& getDatabase() {
    static MNIST_IDX_Database database(N2D2_DATA("mnist"));
    return database;
}

template <class T>
class ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test : public ConvCell_Frame_CUDA<T> {
public:
    ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test(const DeepNet& deepNet, 
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

    friend class UnitTest_ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_float_check_BNFusion_with_SAT;
    friend class UnitTest_ConvCell_QuantizerSAT_BNFusion_Approx_LeNet_TestDatabase_Frame_CUDA_float_check_BNFusion_with_SAT;
    friend class UnitTest_ConvCell_QuantizerSAT_BNFusion_Exact_LeNet_TestDatabase_Frame_CUDA_float_check_BNFusion_with_SAT;
};



// check 2 conv layers with 8 bits quantization, float
// insert bn1 between conv1 and conv2 and fuse conv1 and bn1
TEST_DATASET(ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_float,
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
             std::make_tuple(5U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 32U, 32U, 255,1.0,255,7.9986)
             )
{

    std::cout<<"BN fusion with CONV, SAT quantizer and trained LeNet parameters"<<std::endl;
    //to avoid warning when compile
    std::cout << kernelWidth << kernelHeight << subSampleX << subSampleY 
                << strideX << strideY << paddingX << paddingY 
                << channelsWidth << channelsHeight 
                << range1 << alpha1 << range2 << alpha2 << std::endl; 
    /*
    bool doQuant = true;

    CudaContext::setDevice(0);
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs_conv1 = 6;
    const unsigned int nbOutputs_conv2 = 16;
    const unsigned int nbChannels = 1;
          
    Network net;
    DeepNet dn(net);
    unsigned int batchSize = 1;

    //Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, batchSize, false);
    //env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    //env.setCachePath();
    //env.readRandomBatch(Database::Test);
    //Tensor<Float_T>& in = env.getData();

    StimuliProvider sp(getDatabase(), {channelsWidth, channelsHeight, 1}, batchSize);
    sp.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    sp.setCachePath();
    sp.readStimulusBatch(0, Database::Test);
    Tensor<Float_T>& in = sp.getData();

    size_t dimsInput = in.dimX()*in.dimY()*in.dimZ()*batchSize;
    Tensor<Float_T> in_fused;
    in_fused.resize({in.dimX(),in.dimY(),in.dimZ(),in.dimB()});
    in_fused.fill(0.0);

    for(unsigned int i = 0; i < dimsInput; ++i)
    {
        in_fused(i) = in(i) * 255;
    }

    std::cout << "********************SET_INPUT_0_1********************" << std::endl; 
    std::cout << "[Input]\n" << in << std::endl;
    std::cout << "********************SET_INPUT_0_1_END********************\n\n" << std::endl; 

    std::cout << "********************SET_INPUT_0_255********************" << std::endl; 
    std::cout << "[Input]\n" << in_fused << std::endl;
    std::cout << "********************SET_INPUT_0_255_END********************\n\n" << std::endl; 

    
    ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv1(dn, "conv1",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv1,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1.setParameter("NoBias", true);

    ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv1_fused(dn, "conv1_fused",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv1,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
    conv1_fused.setParameter("NoBias", true);

    BatchNormCell_Frame_CUDA<float> bn1(dn, "bn1", nbOutputs_conv1, std::shared_ptr<Activation>());

    ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv2(dn, "conv2",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
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

    std::vector<float> v_w11 = {0.184185, 0.173881, 0.112021, 0.0815895, -0.173858, 
                                -0.0106287, -0.112002, -0.343679, -0.312961, -0.0172239, 
                                -0.171215, -0.291387, -0.217766, -0.177112, 0.37201, 
                                -0.144616, -0.24736, -0.0290122, 0.000954552, 0.390555, 
                                -0.226636, -0.0951204, 0.231468, 0.144098, 0.367307};

    Tensor<float> conv1_w1({kernelWidth, kernelHeight}, v_w11.begin(),v_w11.end());

    std::vector<float> v_w12 = {-0.177018, 0.139442, 0.288037, 0.158078, 0.0445046, 
                                -0.385413, -0.191178, -5.84598e-07, 0.0907883, 0.11398, 
                                0.0205704, -0.352544, -0.278075, -0.0861968, -0.0952327, 
                                0.283583, -0.0489072, -0.09156, -0.126722, -0.179714, 
                                0.034634, 0.0141254, -0.00935394, -0.071578, -0.11565};
    Tensor<float> conv1_w2({kernelWidth, kernelHeight}, v_w12.begin(),v_w12.end());

    std::vector<float> v_w13 = {-0.159753, 0.0313658, 0.145843, -0.228753, 0.153917, 
                                0.0871841, 0.216748, -0.0137101, -0.62059, -0.337779, 
                                -0.0262911, 0.179648, -0.0586273, -0.625386, -0.0737472, 
                                -0.193735, 0.309977, 0.441459, 0.304995, 0.221394, 
                                0.0382392, -0.158866, 0.486351, 0.351878, -0.0907166};
    Tensor<float> conv1_w3({kernelWidth, kernelHeight}, v_w13.begin(),v_w13.end());

    std::vector<float> v_w14 = {-0.0414309, -0.0289402, 0.0325286, -0.0799899, 0.0532336, 
                                -0.209418, 0.0343616, -0.156739, -0.1015, -0.133251, 
                                -0.178087, -0.0740313, -0.0604055, -0.0632791, -0.115115, 
                                -0.0816227, -0.155578, -0.00729363, -0.167025, -0.322911, 
                                -0.254545, -0.244077, -0.056426, -0.273549, -0.363163};
    Tensor<float> conv1_w4({kernelWidth, kernelHeight}, v_w14.begin(),v_w14.end());

    std::vector<float> v_w15 = {0.309687, -0.197089, 0.0191584, 3.49737e-05, 0.0382037, 
                                0.0226852, 0.0331087, 0.122251, 0.0578961, -0.0379796, 
                                -0.0278929, 0.387234, 0.387587, 0.105464, -0.186279, 
                                0.0619652, 0.0902182, 0.0114893, -0.315561, -0.476871, 
                                0.233503, -0.0460472, 0.0427327, -0.177359, -0.13211};
    Tensor<float> conv1_w5({kernelWidth, kernelHeight}, v_w15.begin(),v_w15.end());

    std::vector<float> v_w16 = {-0.205494, -0.252992, -0.149511, -0.141254, -0.220744, 
                                -0.115654, -0.094326, -0.207501, -0.0156246, -0.2556, 
                                -0.0641069, -0.108896, -0.0255057, -0.126213, -0.0662551, 
                                -0.0450158, 0.0129077, 0.0074881, -0.0830555, -0.282294, 
                                0.0423633, -0.141093, -0.247252, -0.151985, -0.224323};
    Tensor<float> conv1_w6({kernelWidth, kernelHeight}, v_w16.begin(),v_w16.end());

    for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            Tensor<float> kernel({kernelWidth, kernelHeight});

            if(output==0) kernel=conv1_w1;
            if(output==1) kernel=conv1_w2;
            if(output==2) kernel=conv1_w3;
            if(output==3) kernel=conv1_w4;
            if(output==4) kernel=conv1_w5;
            if(output==5) kernel=conv1_w6;

            conv1.setWeight(output, channel, kernel);
        }

    }

    //set fake weights for conv2, doesn't matter; we need only quant input of conv2 for comparison
    for (unsigned int output = 0; output < nbOutputs_conv2; ++output) {
        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            Tensor<float> kernel({kernelWidth, kernelHeight}, 1.0);
            conv2.setWeight(output, channel, kernel);
        }

    }

    std::vector<float> v_scale = {1.5143, 1.49043, 1.69947, 1.17242, 1.7597, 1.1505};
    std::vector<float> v_bias = {-0.0654494, -0.0264921, 0.494825, 0.0508658, 0.196104, 0.0267999};
    std::vector<float> v_mean = {-0.166211, -0.312195, 0.143546, -0.920134, 0.0963226, -0.955784};
    std::vector<float> v_variance = {0.697349, 0.323682, 0.430773, 1.60627, 0.465582, 1.65451};
 
    for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
           
        Tensor<float> scale({1},v_scale.at(output));
        Tensor<float> bias({1}, v_bias.at(output));
        Tensor<float> mean({1}, v_mean.at(output));
        Tensor<float> variance({1}, v_variance.at(output));

        bn1.setScale(output, scale);
        bn1.setBias(output, bias);
        bn1.setMean(output, mean);
        bn1.setVariance(output, variance);

        
        std::cout << "********************BN_PARAMS********************" << std::endl; 
        std::cout << "bn1 with output # " << output << " :: scale = " << scale << " , bias = " << bias << " , mean = " << mean << " , variance = " << variance << std::endl;
        std::cout << "********************BN_PARAMS_END********************\n\n" << std::endl; 
        
    }

    
    conv1.propagate(true);
    bn1.propagate(true);
    conv2.propagate(true);

    std::cout << "********************CONV1_AND_BN_OUTPUTS********************" << std::endl;
    conv1.getOutputs().synchronizeDToH();
    const Tensor<float>& out_conv1 = tensor_cast<float>(conv1.getOutputs());
    //std::cout << "[Conv1][Outputs]" << std::endl;
    //std::cout << out_conv1 << std::endl;
    conv1.getOutputs().synchronizeHToD();

    bn1.getOutputs().synchronizeDToH();
    const Tensor<float>& out_bn1 = tensor_cast<float>(bn1.getOutputs());
    //std::cout << "[BN1][Outputs]" << std::endl;
    //std::cout << out_bn1 << std::endl;
    bn1.getOutputs().synchronizeHToD();
    std::cout << "********************CONV1_AND_BN_OUTPUTS_END********************\n\n" << std::endl;
    

    std::cout << "********************CONV2_QUANT_INPUT********************" << std::endl; 
    quant2.getQuantizedActivations(0).synchronizeDToH();
    const Tensor<float>& quant_act_conv2 = tensor_cast<float>(quant2.getQuantizedActivations(0));
    //std::cout << "[Conv2][Quant Input]" << std::endl;
    //std::cout << quant_act_conv2 << std::endl;
    quant2.getQuantizedActivations(0).synchronizeHToD();
    std::cout << "********************CONV2_QUANT_INPUT_END********************\n\n" << std::endl;
   
    // ===> fuse BN by hands following SAT paper logic in S7.

    // a1 and alpha1 are from conv1; a2 and alpha2 are from conv2;
    // beta and gamma are from bn1

    // 0. create conv1_fused layer with inputs in range [0,255] and quantized weights from conv1 rescaled to [-127, 128]
    //set input and init

    std::cout << "********************SET_INPUT_8-BITS********************" << std::endl; 
    std::cout << "[Input for fused conv]\n" << in_fused << std::endl;
    //std::cout << "[Input for fused conv]\n" << in << std::endl;
    conv1_fused.addInput(in_fused, out_diff);
    conv1_fused.initialize();
    std::cout << "********************SET_INPUT_8-BITS_END********************\n\n" << std::endl; 

    //set weights for conv1_fused : quantized weights are in the quantizer of conv1

    std::cout << "********************SET_WEIGHTS_8-BITS********************" << std::endl; 
    quant1.getQuantizedWeights(0).synchronizeDToH();
    const Tensor<float>& quant_weights_conv1 = tensor_cast<float>(quant1.getQuantizedWeights(0));
    std::cout << "[Conv1][Quant Weights]" << std::endl;
    std::cout << quant_weights_conv1 << std::endl;
    quant1.getQuantizedWeights(0).synchronizeHToD();

    for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
        for (unsigned int channel = 0; channel < nbChannels;
             ++channel) {

            Tensor<float> kernel_rescaled({kernelWidth,
                                   kernelHeight});
            
            for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                    //weights in range [-128,127]
                    //kernel_rescaled(sx, sy) = rintf(0.5*(quant_weights_conv1[output][channel](sx, sy)*range1-1));
                    kernel_rescaled(sx, sy) = rintf(std::min(0.5*(quant_weights_conv1[output][channel](sx, sy)*(range1+1)),127.0));
                    //kernel_rescaled(sx, sy) = quant_weights_conv1[output][channel](sx, sy);
                    std::cout << "conv1_fused :: sx = " << sx << " , sy = " << sy << " , weight = " << quant_weights_conv1[output][channel](sx, sy) << 
                    " ==> " << kernel_rescaled(sx, sy) << std::endl;
                }
            }
            conv1_fused.setWeight(output, channel, kernel_rescaled);
        }
    }

    std::cout << "conv1_fused weights after rescale : " << std::endl;
    for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            Tensor<float> weight;
            conv1_fused.getWeight(output, channel, weight);
            std::cout << weight << std::endl;
        }
    }

    std::cout << "********************SET_WEIGHTS_8-BITS_END********************\n\n" << std::endl; 

    // 1. propagate
    std::cout << "********************PROPAGATE********************" << std::endl; 
    conv1_fused.propagate(true);

    conv1_fused.getOutputs().synchronizeDToH();
    const Tensor<float>& out_conv1_fused_prop = tensor_cast<float>(conv1_fused.getOutputs());
    //std::cout << "[Conv1 Fusion][Output after propagate]" << std::endl;
    //std::cout << out_conv1_fused_prop << std::endl;
    conv1_fused.getOutputs().synchronizeHToD();

    std::cout << "********************PROPAGATE_END********************\n\n" << std::endl; 

    // 2. get BN gamma and beta (as done in DeepNet.cpp fuseBatchNormWithConv)
    std::cout << "********************BETA_GAMMA_COMPUTE********************" << std::endl; 

    const bool noBias = conv1.getParameter<bool>("NoBias");
    const Tensor<float>& bnScales
        = tensor_cast<float>(*(bn1.getScales()));
    const Tensor<float>& bnBiases
        = tensor_cast<float>(*(bn1.getBiases()));
    const Tensor<float>& bnMeans
        = tensor_cast<float>(*(bn1.getMeans()));
    const Tensor<float>& bnVariances
        = tensor_cast<float>(*(bn1.getVariances()));
    const double eps = bn1.getParameter<double>("Epsilon");

    float meanVariance = 0.0f;
    unsigned int count = 0;

    for (std::size_t output = 0; output < nbOutputs_conv1; ++output) {
        if (bnVariances(output) > 1.0e-12) {
            meanVariance += bnVariances(output);
            ++count;
        }
        else {
            std::cout << "    zero-variance " << conv1.getName()
                << "[" << output << "]" << std::endl;
        }
    }

    meanVariance /= count;
    std::cout << "mean variance = " << meanVariance << std::endl;

    Tensor<float> gamma;
    gamma.resize({nbOutputs_conv1}, 0.0);
    Tensor<float> beta;
    beta.resize({nbOutputs_conv1}, 0.0);
    Tensor<float> bias_fusion;
    bias_fusion.resize({nbOutputs_conv1}, 0.0);

    //tensor for comparison with the reference
    Tensor<float> clipPerOutput;
    clipPerOutput.resize({nbOutputs_conv1}, 0.0);
    Tensor<float> scalePerOutput;
    scalePerOutput.resize({nbOutputs_conv1}, 0.0);
    //tensor rounded in biases and cliping values
    Tensor<float> clipPerOutputRound;
    clipPerOutputRound.resize({nbOutputs_conv1}, 0.0);

    Tensor<float> bias_fusion_rounded;
    bias_fusion_rounded.resize({nbOutputs_conv1}, 0.0);

    for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
        //Biases adjustments
        Tensor<float> bias;
        if (noBias)
            bias.resize({1}, 0.0);
        else
            conv1.getBias(output, bias);

        //factor for weights adjustments
        float factor = bnScales(output)
                / std::sqrt(eps + ((bnVariances(output) > 1.0e-12)
                            ? bnVariances(output) : meanVariance));
        gamma(output) = factor;
        beta(output) = bnBiases(output) + (bias(0) - bnMeans(output)) * factor;

        bias_fusion(output) = (beta(output)/gamma(output)) * ((float)range1/alpha1) * ((float)(range1+1)/2.0);
        clipPerOutput(output) = (alpha2/gamma(output)) * ((float)range1/alpha1) * ((float)(range1+1)/2.0);
        bias_fusion_rounded(output) = rintf((beta(output)/gamma(output)) * ((float)range1/alpha1) * ((float)(range1+1)/2.0));
        clipPerOutputRound(output) = rintf((alpha2/gamma(output)) * ((float)range1/alpha1) * ((float)(range1+1)/2.0));
        scalePerOutput(output) = (alpha1/(float)range1) * ((float)range2/alpha2) * gamma(output) *(2.0/(float)(range1+1));
        
    }  

    std::cout << "********************BETA_GAMMA_COMPUTE_END********************\n\n" << std::endl; 
    std::cout << "bias" << std::endl;
    std::cout << bias_fusion << std::endl;
    std::cout << "clipPerOutput" << std::endl;
    std::cout << clipPerOutput << std::endl;
    std::cout << "scalePerOutput" << std::endl;
    std::cout << scalePerOutput << std::endl;



    //not cliped, with bias 
    Tensor<float> conv1_add_bias;
    conv1_add_bias.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

    //cliped output 
    Tensor<float> conv1_clipped;
    conv1_clipped.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

    //scaled output 
    Tensor<float> conv1_scaled;
    conv1_scaled.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

    //round output 
    Tensor<float> conv1_rounded;
    conv1_rounded.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

    std::cout << "********************BN_FUSION********************" << std::endl;

    for (unsigned int batch = 0; batch < batchSize; ++batch) {
        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            for (unsigned int oy = 0; oy < conv1_fused.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < conv1_fused.getOutputsWidth(); ++ox) {
                    //Add bias term
                    float value = out_conv1_fused_prop(ox, oy, output, batch) 
                                    + bias_fusion(output);
                    conv1_add_bias(ox, oy, output, batch) = value;

                    //Clip the value with a clip factor par output
                    float clippedValue = (value < 0.0f) ? 0.0f 
                                        : (value < clipPerOutput(output)) ? value 
                                        : clipPerOutput(output);
                    conv1_clipped(ox, oy, output, batch) = clippedValue;
                    conv1_scaled(ox, oy, output, batch) = clippedValue*scalePerOutput(output);

                    //scale and round the result
                    conv1_rounded(ox, oy, output, batch) = rintf(clippedValue*scalePerOutput(output));
                }
            }
        }
    }

    //not cliped, with bias 
    Tensor<float> conv1_add_bias_r;
    conv1_add_bias_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

    //cliped output 
    Tensor<float> conv1_clipped_r;
    conv1_clipped_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

    //scaled output 
    Tensor<float> conv1_scaled_r;
    conv1_scaled_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

    //round output 
    Tensor<float> conv1_rounded_r;
    conv1_rounded_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

    std::cout << "********************BN_FUSION********************" << std::endl;

    for (unsigned int batch = 0; batch < batchSize; ++batch) {
        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            for (unsigned int oy = 0; oy < conv1_fused.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < conv1_fused.getOutputsWidth(); ++ox) {
                    //Add bias term
                    float value = out_conv1_fused_prop(ox, oy, output, batch) 
                                    + bias_fusion_rounded(output);
                    conv1_add_bias_r(ox, oy, output, batch) = value;

                    //Clip the value with a clip factor par output
                    float clippedValue = (value < 0.0f) ? 0.0f 
                                        : (value < clipPerOutputRound(output)) ? value 
                                        : clipPerOutputRound(output);
                    conv1_clipped_r(ox, oy, output, batch) = clippedValue;
                    conv1_scaled_r(ox, oy, output, batch) = clippedValue*scalePerOutput(output);

                    //scale and round the result
                    conv1_rounded_r(ox, oy, output, batch) = rintf(clippedValue*scalePerOutput(output));
                }
            }
        }
    }

    
    std::cout << "[QConv1][NOT CLIPED]" << std::endl;
    std::cout << conv1_add_bias << std::endl;
    std::cout << "[QConv1][CLIPPED]" << std::endl;
    std::cout << conv1_clipped << std::endl;
    std::cout << "[QConv1][SCALED]" << std::endl;
    std::cout << conv1_scaled << std::endl;
    
    std::cout << "[QConv1][ROUNDED]" << std::endl;
    std::cout << conv1_rounded << std::endl;

    size_t dimsQ2 = quant_act_conv2.dimX()*quant_act_conv2.dimY()*quant_act_conv2.dimZ()*batchSize;
    Tensor<float> quant_conv2_unscaled;
    quant_conv2_unscaled.resize({quant_act_conv2.dimX(),quant_act_conv2.dimY(),quant_act_conv2.dimZ(),batchSize}, 0.0);
    for(unsigned int i = 0; i < dimsQ2; ++i)
    {
        quant_conv2_unscaled(i) = rintf(quant_act_conv2(i) * ((float)range2/alpha2));
    }
    //std::cout << "[EXPECTED RESULT]" << std::endl;
    //std::cout << quant_conv2_unscaled << std::endl;

    double mse = 0.0, mse_r = 0.0;
    double max_error = 0.0;
    for (unsigned int batch = 0; batch < batchSize; ++batch) {
        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            for (unsigned int oy = 0; oy < conv1_fused.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < conv1_fused.getOutputsWidth(); ++ox) {
                    double error = conv1_rounded(ox, oy, output, batch) - quant_conv2_unscaled(ox, oy, output, batch);
                    double error_r = conv1_rounded_r(ox, oy, output, batch) - quant_conv2_unscaled(ox, oy, output, batch);

                    mse += std::pow(error, 2);
                    mse_r += std::pow(error_r, 2);

                    if(std::abs(error_r) > max_error)
                        max_error = std::abs(error_r);
                }
            }
        }
    }
    mse /= (double) batchSize* nbOutputs_conv1 * conv1_fused.getOutputsHeight() * conv1_fused.getOutputsWidth();
    mse_r /= (double) batchSize* nbOutputs_conv1 * conv1_fused.getOutputsHeight() * conv1_fused.getOutputsWidth();
    std::cout << "MeanSquareError: [NoRound] " << mse 
            << " [Round] " << mse_r 
            << " Max error value: " << max_error
            << std::endl;
    std::cout << "********************BN_FUSION_END********************" << std::endl;
    
  */
}


//read the entire test database, and compute MSE

TEST_DATASET(ConvCell_QuantizerSAT_BNFusion_Approx_LeNet_TestDatabase_Frame_CUDA_float,
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
             std::make_tuple(5U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 32U, 32U, 255,1.0,255,7.9986)
             )
{

    std::cout<<"BN fusion (approx) with CONV, SAT quantizer and trained LeNet parameters"<<std::endl;
    //to avoid warning when compile
    std::cout << kernelWidth << kernelHeight << subSampleX << subSampleY 
                << strideX << strideY << paddingX << paddingY 
                << channelsWidth << channelsHeight 
                << range1 << alpha1 << range2 << alpha2 << std::endl; 
    /*
    bool doQuant = true;

    CudaContext::setDevice(0);
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs_conv1 = 6;
    const unsigned int nbOutputs_conv2 = 16;
    const unsigned int nbChannels = 1;
          
    Network net;
    DeepNet dn(net);
    unsigned int batchSize = 1;

    StimuliProvider sp(getDatabase(), {channelsWidth, channelsHeight, 1}, batchSize);

    sp.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    sp.setCachePath();
    const unsigned int nbTest = getDatabase().getNbStimuli(Database::Test);
    std::cout << "Database test size = " << nbTest << std::endl;
    const unsigned int nbBatch = std::ceil(nbTest / (double)batchSize);
    //const unsigned int nbBatch = 10;

    std::vector<double> v_mse, v_mse_r, v_mse_max, v_faulty_pixel, v_faulty_pixel_percent;

    for (unsigned int b = 0; b < nbBatch; ++b) {
        
        if(b % 1000 == 0) std::cout << "b = " << b << std::endl;

        const unsigned int idx = b * batchSize;
        sp.readBatch(Database::Test, idx);
 
        Tensor<Float_T>& in = sp.getData();

        size_t dimsInput = in.dimX()*in.dimY()*in.dimZ()*batchSize;
        Tensor<Float_T> in_fused;
        in_fused.resize({in.dimX(),in.dimY(),in.dimZ(),in.dimB()});
        in_fused.fill(0.0);
        for(unsigned int i = 0; i < dimsInput; ++i)
        {
            in_fused(i) = in(i) * 255;
        }
    
        ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv1(dn, "conv1",
            std::vector<unsigned int>({kernelWidth, kernelHeight}),
            nbOutputs_conv1,
            std::vector<unsigned int>({subSampleX, subSampleY}),
            std::vector<unsigned int>({strideX, strideY}),
            std::vector<int>({(int)paddingX, (int)paddingY}),
            std::vector<unsigned int>({1U, 1U}),
            std::shared_ptr<Activation>());
        conv1.setParameter("NoBias", true);

        ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv1_fused(dn, "conv1_fused",
            std::vector<unsigned int>({kernelWidth, kernelHeight}),
            nbOutputs_conv1,
            std::vector<unsigned int>({subSampleX, subSampleY}),
            std::vector<unsigned int>({strideX, strideY}),
            std::vector<int>({(int)paddingX, (int)paddingY}),
            std::vector<unsigned int>({1U, 1U}),
            std::shared_ptr<Activation>());
        conv1_fused.setParameter("NoBias", true);

        BatchNormCell_Frame_CUDA<float> bn1(dn, "bn1", nbOutputs_conv1, std::shared_ptr<Activation>());

        ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv2(dn, "conv2",
            std::vector<unsigned int>({kernelWidth, kernelHeight}),
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

        std::vector<float> v_w11 = {0.184185, 0.173881, 0.112021, 0.0815895, -0.173858, 
                                    -0.0106287, -0.112002, -0.343679, -0.312961, -0.0172239, 
                                    -0.171215, -0.291387, -0.217766, -0.177112, 0.37201, 
                                    -0.144616, -0.24736, -0.0290122, 0.000954552, 0.390555, 
                                    -0.226636, -0.0951204, 0.231468, 0.144098, 0.367307};

        Tensor<float> conv1_w1({kernelWidth, kernelHeight}, v_w11.begin(),v_w11.end());

        std::vector<float> v_w12 = {-0.177018, 0.139442, 0.288037, 0.158078, 0.0445046, 
                                    -0.385413, -0.191178, -5.84598e-07, 0.0907883, 0.11398, 
                                    0.0205704, -0.352544, -0.278075, -0.0861968, -0.0952327, 
                                    0.283583, -0.0489072, -0.09156, -0.126722, -0.179714, 
                                    0.034634, 0.0141254, -0.00935394, -0.071578, -0.11565};
        Tensor<float> conv1_w2({kernelWidth, kernelHeight}, v_w12.begin(),v_w12.end());

        std::vector<float> v_w13 = {-0.159753, 0.0313658, 0.145843, -0.228753, 0.153917, 
                                    0.0871841, 0.216748, -0.0137101, -0.62059, -0.337779, 
                                    -0.0262911, 0.179648, -0.0586273, -0.625386, -0.0737472, 
                                    -0.193735, 0.309977, 0.441459, 0.304995, 0.221394, 
                                    0.0382392, -0.158866, 0.486351, 0.351878, -0.0907166};
        Tensor<float> conv1_w3({kernelWidth, kernelHeight}, v_w13.begin(),v_w13.end());

        std::vector<float> v_w14 = {-0.0414309, -0.0289402, 0.0325286, -0.0799899, 0.0532336, 
                                    -0.209418, 0.0343616, -0.156739, -0.1015, -0.133251, 
                                    -0.178087, -0.0740313, -0.0604055, -0.0632791, -0.115115, 
                                    -0.0816227, -0.155578, -0.00729363, -0.167025, -0.322911, 
                                    -0.254545, -0.244077, -0.056426, -0.273549, -0.363163};
        Tensor<float> conv1_w4({kernelWidth, kernelHeight}, v_w14.begin(),v_w14.end());

        std::vector<float> v_w15 = {0.309687, -0.197089, 0.0191584, 3.49737e-05, 0.0382037, 
                                    0.0226852, 0.0331087, 0.122251, 0.0578961, -0.0379796, 
                                    -0.0278929, 0.387234, 0.387587, 0.105464, -0.186279, 
                                    0.0619652, 0.0902182, 0.0114893, -0.315561, -0.476871, 
                                    0.233503, -0.0460472, 0.0427327, -0.177359, -0.13211};
        Tensor<float> conv1_w5({kernelWidth, kernelHeight}, v_w15.begin(),v_w15.end());

        std::vector<float> v_w16 = {-0.205494, -0.252992, -0.149511, -0.141254, -0.220744, 
                                    -0.115654, -0.094326, -0.207501, -0.0156246, -0.2556, 
                                    -0.0641069, -0.108896, -0.0255057, -0.126213, -0.0662551, 
                                    -0.0450158, 0.0129077, 0.0074881, -0.0830555, -0.282294, 
                                    0.0423633, -0.141093, -0.247252, -0.151985, -0.224323};
        Tensor<float> conv1_w6({kernelWidth, kernelHeight}, v_w16.begin(),v_w16.end());

        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                Tensor<float> kernel({kernelWidth, kernelHeight});

                if(output==0) kernel=conv1_w1;
                if(output==1) kernel=conv1_w2;
                if(output==2) kernel=conv1_w3;
                if(output==3) kernel=conv1_w4;
                if(output==4) kernel=conv1_w5;
                if(output==5) kernel=conv1_w6;

                conv1.setWeight(output, channel, kernel);
            }

        }

        //set fake weights for conv2, doesn't matter; we need only quant input of conv2 for comparison
        for (unsigned int output = 0; output < nbOutputs_conv2; ++output) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                Tensor<float> kernel({kernelWidth, kernelHeight}, 1.0);
                conv2.setWeight(output, channel, kernel);
            }

        }

        std::vector<float> v_scale = {1.5143, 1.49043, 1.69947, 1.17242, 1.7597, 1.1505};
        std::vector<float> v_bias = {-0.0654494, -0.0264921, 0.494825, 0.0508658, 0.196104, 0.0267999};
        std::vector<float> v_mean = {-0.166211, -0.312195, 0.143546, -0.920134, 0.0963226, -0.955784};
        std::vector<float> v_variance = {0.697349, 0.323682, 0.430773, 1.60627, 0.465582, 1.65451};
    
        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            
            Tensor<float> scale({1},v_scale.at(output));
            Tensor<float> bias({1}, v_bias.at(output));
            Tensor<float> mean({1}, v_mean.at(output));
            Tensor<float> variance({1}, v_variance.at(output));

            bn1.setScale(output, scale);
            bn1.setBias(output, bias);
            bn1.setMean(output, mean);
            bn1.setVariance(output, variance);
        }

    
        conv1.propagate(true);
        bn1.propagate(true);
        conv2.propagate(true);

        conv1.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv1 = tensor_cast<float>(conv1.getOutputs());
        conv1.getOutputs().synchronizeHToD();

        bn1.getOutputs().synchronizeDToH();
        const Tensor<float>& out_bn1 = tensor_cast<float>(bn1.getOutputs());
        bn1.getOutputs().synchronizeHToD();
 
        quant2.getQuantizedActivations(0).synchronizeDToH();
        const Tensor<float>& quant_act_conv2 = tensor_cast<float>(quant2.getQuantizedActivations(0));
        quant2.getQuantizedActivations(0).synchronizeHToD();
   
        // ===> fuse BN by hands following SAT paper logic in S7.
        // a1 and alpha1 are from conv1; a2 and alpha2 are from conv2;
        // beta and gamma are from bn1
        // 0. create conv1_fused layer with inputs in range [0,255] and quantized weights from conv1 rescaled to [-127, 128]
        //set input and init
        conv1_fused.addInput(in_fused, out_diff);
        conv1_fused.initialize();
        //std::cout << "********************SET_INPUT_8-BITS_END********************\n\n" << std::endl; 

        //set weights for conv1_fused : quantized weights are in the quantizer of conv1
        quant1.getQuantizedWeights(0).synchronizeDToH();
        const Tensor<float>& quant_weights_conv1 = tensor_cast<float>(quant1.getQuantizedWeights(0));
        quant1.getQuantizedWeights(0).synchronizeHToD();

        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            for (unsigned int channel = 0; channel < nbChannels;
                ++channel) {

                Tensor<float> kernel_rescaled({kernelWidth,
                                    kernelHeight});
                
                for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                    for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                        //[-127, 128]
                        //kernel_rescaled(sx, sy) = rintf(0.5*(quant_weights_conv1[output][channel](sx, sy)*range1+1));
                        //[-128,127]
                        //kernel_rescaled(sx, sy) = rintf(0.5f*(quant_weights_conv1[output][channel](sx, sy)*range1-1));
                        kernel_rescaled(sx, sy) = rintf(std::min(0.5*(quant_weights_conv1[output][channel](sx, sy)*(range1+1)),127.0));
                        //[-127, 127]
                        //kernel_rescaled(sx, sy) = rintf(0.5f*(quant_weights_conv1[output][channel](sx, sy)*(range1-1)));
                    }
                }
                conv1_fused.setWeight(output, channel, kernel_rescaled);
            }
        }

        // 1. propagate
        conv1_fused.propagate(true);

        conv1_fused.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv1_fused_prop = tensor_cast<float>(conv1_fused.getOutputs());
        //std::cout << "[Conv1 Fusion][Output after propagate]" << std::endl;
        //std::cout << out_conv1_fused_prop << std::endl;
        conv1_fused.getOutputs().synchronizeHToD();

        // 2. get BN gamma and beta (as done in DeepNet.cpp fuseBatchNormWithConv)
        const bool noBias = conv1.getParameter<bool>("NoBias");
        const Tensor<float>& bnScales
            = tensor_cast<float>(*(bn1.getScales()));
        const Tensor<float>& bnBiases
            = tensor_cast<float>(*(bn1.getBiases()));
        const Tensor<float>& bnMeans
            = tensor_cast<float>(*(bn1.getMeans()));
        const Tensor<float>& bnVariances
            = tensor_cast<float>(*(bn1.getVariances()));
        const double eps = bn1.getParameter<double>("Epsilon");

        float meanVariance = 0.0f;
        unsigned int count = 0;

        for (std::size_t output = 0; output < nbOutputs_conv1; ++output) {
            if (bnVariances(output) > 1.0e-12) {
                meanVariance += bnVariances(output);
                ++count;
            }
            else {
                std::cout << "    zero-variance " << conv1.getName()
                    << "[" << output << "]" << std::endl;
            }
        }

        meanVariance /= count;

        Tensor<float> gamma;
        gamma.resize({nbOutputs_conv1}, 0.0);
        Tensor<float> beta;
        beta.resize({nbOutputs_conv1}, 0.0);
        Tensor<float> bias_fusion;
        bias_fusion.resize({nbOutputs_conv1}, 0.0);

        //tensor for comparison with the reference
        Tensor<float> clipPerOutput;
        clipPerOutput.resize({nbOutputs_conv1}, 0.0);
        Tensor<float> scalePerOutput;
        scalePerOutput.resize({nbOutputs_conv1}, 0.0);
        //tensor rounded in biases and cliping values
        Tensor<float> clipPerOutputRound;
        clipPerOutputRound.resize({nbOutputs_conv1}, 0.0);

        Tensor<float> bias_fusion_rounded;
        bias_fusion_rounded.resize({nbOutputs_conv1}, 0.0);

        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            //Biases adjustments
            Tensor<float> bias;
            if (noBias)
                bias.resize({1}, 0.0);
            else
                conv1.getBias(output, bias);

            //factor for weights adjustments
            float factor = bnScales(output)
                    / std::sqrt(eps + ((bnVariances(output) > 1.0e-12)
                                ? bnVariances(output) : meanVariance));
            gamma(output) = factor;
            beta(output) = bnBiases(output) + (bias(0) - bnMeans(output)) * factor;

            //for weights range [-128,127]
            bias_fusion(output) = (beta(output)/gamma(output)) * ((float)range1/alpha1) * ((float)(range1+1)/2.0);
            clipPerOutput(output) = (alpha2/gamma(output)) * ((float)range1/alpha1) * ((float)(range1+1)/2.0);
            bias_fusion_rounded(output) = rintf((beta(output)/gamma(output)) * ((float)range1/alpha1) * ((float)(range1+1)/2.0));
            clipPerOutputRound(output) = rintf((alpha2/gamma(output)) * ((float)range1/alpha1) * ((float)(range1+1)/2.0));
            scalePerOutput(output) = (alpha1/(float)range1) * ((float)range2/alpha2) * gamma(output) *(2.0/(float)(range1+1));
                   
            //for weights range [-127, 127]
            //bias_fusion(output) = (beta(output)/gamma(output)) * ((float)range1/alpha1) * ((float)(range1-1)/2.0);
            //clipPerOutput(output) = (alpha2/gamma(output)) * ((float)range1/alpha1) * ((float)(range1-1)/2.0);
            //bias_fusion_rounded(output) = rintf((beta(output)/gamma(output)) * ((float)range1/alpha1) * ((float)(range1-1)/2.0));
            //clipPerOutputRound(output) = rintf((alpha2/gamma(output)) * ((float)range1/alpha1) * ((float)(range1-1)/2.0));
            //scalePerOutput(output) = (alpha1/(float)range1) * ((float)range2/alpha2) * gamma(output) *(2.0/(float)(range1-1));  
                   
        }  

        //not cliped, with bias 
        Tensor<float> conv1_add_bias;
        conv1_add_bias.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //cliped output 
        Tensor<float> conv1_clipped;
        conv1_clipped.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //scaled output 
        Tensor<float> conv1_scaled;
        conv1_scaled.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //round output 
        Tensor<float> conv1_rounded;
        conv1_rounded.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        for (unsigned int batch = 0; batch < batchSize; ++batch) {
            for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
                for (unsigned int oy = 0; oy < conv1_fused.getOutputsHeight(); ++oy) {
                    for (unsigned int ox = 0; ox < conv1_fused.getOutputsWidth(); ++ox) {
                        //Add bias term
                        float value = out_conv1_fused_prop(ox, oy, output, batch) 
                                        + bias_fusion(output);
                        conv1_add_bias(ox, oy, output, batch) = value;

                        //Clip the value with a clip factor par output
                        float clippedValue = (value < 0.0f) ? 0.0f 
                                            : (value < clipPerOutput(output)) ? value 
                                            : clipPerOutput(output);
                        conv1_clipped(ox, oy, output, batch) = clippedValue;
                        conv1_scaled(ox, oy, output, batch) = clippedValue*scalePerOutput(output);

                        //scale and round the result
                        conv1_rounded(ox, oy, output, batch) = rintf(clippedValue*scalePerOutput(output));
                    }
                }
            }
        }

        //not cliped, with bias 
        Tensor<float> conv1_add_bias_r;
        conv1_add_bias_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //cliped output 
        Tensor<float> conv1_clipped_r;
        conv1_clipped_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //scaled output 
        Tensor<float> conv1_scaled_r;
        conv1_scaled_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //round output 
        Tensor<float> conv1_rounded_r;
        conv1_rounded_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        for (unsigned int batch = 0; batch < batchSize; ++batch) {
            for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
                for (unsigned int oy = 0; oy < conv1_fused.getOutputsHeight(); ++oy) {
                    for (unsigned int ox = 0; ox < conv1_fused.getOutputsWidth(); ++ox) {
                        //Add bias term
                        float value = out_conv1_fused_prop(ox, oy, output, batch) 
                                        + bias_fusion_rounded(output);
                        conv1_add_bias_r(ox, oy, output, batch) = value;

                        //Clip the value with a clip factor par output
                        float clippedValue = (value < 0.0f) ? 0.0f 
                                            : (value < clipPerOutputRound(output)) ? value 
                                            : clipPerOutputRound(output);
                        conv1_clipped_r(ox, oy, output, batch) = clippedValue;
                        conv1_scaled_r(ox, oy, output, batch) = clippedValue*scalePerOutput(output);

                        //scale and round the result
                        conv1_rounded_r(ox, oy, output, batch) = rintf(clippedValue*scalePerOutput(output));
                    }
                }
            }
        }

        size_t dimsQ2 = quant_act_conv2.dimX()*quant_act_conv2.dimY()*quant_act_conv2.dimZ()*batchSize;
        Tensor<float> quant_conv2_unscaled;
        quant_conv2_unscaled.resize({quant_act_conv2.dimX(),quant_act_conv2.dimY(),quant_act_conv2.dimZ(),batchSize}, 0.0);
        for(unsigned int i = 0; i < dimsQ2; ++i)
        {
            quant_conv2_unscaled(i) = rintf(quant_act_conv2(i) * ((float)range2/alpha2));
        }

        double mse = 0.0, mse_r = 0.0;
        double max_error = 0.0;
        int faulty_pixel = 0;
        double faulty_pixel_percent = 0.0;

        //std::cout << " >>> number of 'pixels' after conv1 = " << (double) (batchSize*nbOutputs_conv1 * conv1_fused.getOutputsHeight() * conv1_fused.getOutputsWidth()) << std::endl;
        //4704
        for (unsigned int batch = 0; batch < batchSize; ++batch) {
            for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
                for (unsigned int oy = 0; oy < conv1_fused.getOutputsHeight(); ++oy) {
                    for (unsigned int ox = 0; ox < conv1_fused.getOutputsWidth(); ++ox) {
                        double error = conv1_rounded(ox, oy, output, batch) - quant_conv2_unscaled(ox, oy, output, batch);
                        double error_r = conv1_rounded_r(ox, oy, output, batch) - quant_conv2_unscaled(ox, oy, output, batch);

                        mse += std::pow(error, 2);
                        mse_r += std::pow(error_r, 2);

                        if(std::abs(error_r) > max_error)
                            max_error = std::abs(error_r);

                        if(std::abs(error_r) >= 1){
                            faulty_pixel++;
                        }
                    }
                }
            }
        }

        mse /= (double) batchSize* nbOutputs_conv1 * conv1_fused.getOutputsHeight() * conv1_fused.getOutputsWidth();
        mse_r /= (double) batchSize* nbOutputs_conv1 * conv1_fused.getOutputsHeight() * conv1_fused.getOutputsWidth();
        faulty_pixel_percent = ((double)faulty_pixel*100)/((double)(batchSize*nbOutputs_conv1 * conv1_fused.getOutputsHeight() * conv1_fused.getOutputsWidth()));

        v_mse.push_back(mse);
        v_mse_r.push_back(mse_r);
        v_mse_max.push_back(max_error);
        v_faulty_pixel.push_back(faulty_pixel);
        v_faulty_pixel_percent.push_back(faulty_pixel_percent);
        
    }

    std::ostringstream fileName_mse;
    fileName_mse << "bnFusion_approx_mse_" << range1 << "_" << range2 << ".dat";
    std::ostringstream fileName_mse_r;
    fileName_mse_r << "bnFusion_approx_mse_round_" << range1 << "_" << range2 << ".dat";
    std::ostringstream fileName_max_err;
    fileName_max_err << "bnFusion_approx_max_err_" << range1 << "_" << range2 << ".dat";
    std::ostringstream fileName_faulty_pixel;
    fileName_faulty_pixel << "bnFusion_approx_faulty_pixel_" << range1 << "_" << range2 << ".dat";
    std::ostringstream fileName_faulty_pixel_percent;
    fileName_faulty_pixel_percent << "bnFusion_approx_faulty_pixel_percent_" << range1 << "_" << range2 << ".dat";

    std::ofstream outFile_mse(fileName_mse.str());
    std::ofstream outFile_mse_r(fileName_mse_r.str());
    std::ofstream outFile_mse_max(fileName_max_err.str());
    std::ofstream outFile_faulty_pixel(fileName_faulty_pixel.str());
    std::ofstream outFile_faulty_pixel_percent(fileName_faulty_pixel_percent.str());

    for (std::size_t i = 0; i < v_mse.size(); ++i) {
        outFile_mse << v_mse[i] << "\n";
        outFile_mse_r << v_mse_r[i] << "\n";
        outFile_mse_max << v_mse_max[i] << "\n";
        outFile_faulty_pixel << v_faulty_pixel[i] << "\n";
        outFile_faulty_pixel_percent << v_faulty_pixel_percent[i] << "\n";
    }

    std::string fileName_img_mse = fileName_mse.str();
    fileName_img_mse.replace(fileName_img_mse.begin()+(fileName_img_mse.size()-4),fileName_img_mse.end(),".png");
    std::string fileName_img_mse_r = fileName_mse_r.str();
    fileName_img_mse_r.replace(fileName_img_mse_r.begin()+(fileName_img_mse_r.size()-4),fileName_img_mse_r.end(),".png");
    std::string fileName_img_max_err = fileName_max_err.str();
    fileName_img_max_err.replace(fileName_img_max_err.begin()+(fileName_img_max_err.size()-4),fileName_img_max_err.end(),".png");
    std::string fileName_img_faulty_pixel = fileName_faulty_pixel.str();
    fileName_img_faulty_pixel.replace(fileName_img_faulty_pixel.begin()+(fileName_img_faulty_pixel.size()-4),fileName_img_faulty_pixel.end(),".png");
    std::string fileName_img_faulty_pixel_percent = fileName_faulty_pixel_percent.str();
    fileName_img_faulty_pixel_percent.replace(fileName_img_faulty_pixel_percent.begin()+(fileName_img_faulty_pixel_percent.size()-4),fileName_img_faulty_pixel_percent.end(),".png");

    // plot results
    Gnuplot gnuplot_mse;
    gnuplot_mse.set("grid");   
    gnuplot_mse << "binwidth=0.01";
    gnuplot_mse << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_mse.setXlabel("MSE");
    gnuplot_mse.setYlabel("Number of images");
    gnuplot_mse.saveToFile(fileName_img_mse);
    gnuplot_mse.plot(fileName_mse.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));

    Gnuplot gnuplot_mse_r;
    gnuplot_mse_r.set("grid");   
    gnuplot_mse_r << "binwidth=0.01";
    gnuplot_mse_r << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_mse_r.setXlabel("MSE with rounding");
    gnuplot_mse_r.setYlabel("Number of images");
    gnuplot_mse_r.saveToFile(fileName_img_mse_r);
    gnuplot_mse_r.plot(fileName_mse_r.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));

    Gnuplot gnuplot_max_err;
    gnuplot_max_err.set("grid");   
    gnuplot_max_err << "binwidth=0.01";
    gnuplot_max_err << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_max_err.setXlabel("Max error");
    gnuplot_max_err.setYlabel("Number of images");
    gnuplot_max_err.saveToFile(fileName_img_max_err);
    gnuplot_max_err.plot(fileName_max_err.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));

    Gnuplot gnuplot_faulty_pixel;
    gnuplot_faulty_pixel.set("grid");   
    gnuplot_faulty_pixel << "binwidth=0.01";
    gnuplot_faulty_pixel << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_faulty_pixel.setXlabel("Number of faulty pixels");
    gnuplot_faulty_pixel.setYlabel("Number of images");
    gnuplot_faulty_pixel.saveToFile(fileName_img_faulty_pixel);
    gnuplot_faulty_pixel.plot(fileName_faulty_pixel.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));

    Gnuplot gnuplot_faulty_pixel_percent;
    gnuplot_faulty_pixel_percent.set("grid");   
    gnuplot_faulty_pixel_percent << "binwidth=0.01";
    gnuplot_faulty_pixel_percent << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_faulty_pixel_percent.setXlabel("Percentage of faulty pixels");
    gnuplot_faulty_pixel_percent.setYlabel("Number of images");
    gnuplot_faulty_pixel_percent.saveToFile(fileName_img_faulty_pixel_percent);
    gnuplot_faulty_pixel_percent.plot(fileName_faulty_pixel_percent.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));
    */
}

//read the entire test database, and compute MSE

TEST_DATASET(ConvCell_QuantizerSAT_BNFusion_Exact_LeNet_TestDatabase_Frame_CUDA_float,
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
             std::make_tuple(5U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 32U, 32U, 255,1.0,255,7.9986)
             )
{

    std::cout<<"BN fusion (exact) with CONV, SAT quantizer and trained LeNet parameters"<<std::endl;
    std::cout << kernelWidth << kernelHeight << subSampleX << subSampleY 
                << strideX << strideY << paddingX << paddingY 
                << channelsWidth << channelsHeight 
                << range1 << alpha1 << range2 << alpha2 << std::endl; 

    /*
    bool doQuant = true;

    CudaContext::setDevice(0);
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs_conv1 = 6;
    const unsigned int nbOutputs_conv2 = 16;
    const unsigned int nbChannels = 1;
          
    Network net;
    DeepNet dn(net);
    unsigned int batchSize = 1;

    StimuliProvider sp(getDatabase(), {channelsWidth, channelsHeight, 1}, batchSize);

    sp.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    sp.setCachePath();
    const unsigned int nbTest = getDatabase().getNbStimuli(Database::Test);
    std::cout << "Database test size = " << nbTest << std::endl;
    const unsigned int nbBatch = std::ceil(nbTest / (double)batchSize);

    std::vector<double> v_mse, v_mse_r, v_mse_max, v_faulty_pixel, v_faulty_pixel_percent;

    for (unsigned int b = 0; b < nbBatch; ++b) {
        
        if(b % 1000 == 0) std::cout << "b = " << b << std::endl;

        const unsigned int idx = b * batchSize;
        //sp.readBatch(Database::Test, idx);
        //to get inputs in the same order
        sp.readStimulusBatch(idx, Database::Test);

        Tensor<Float_T>& in = sp.getData();

        size_t dimsInput = in.dimX()*in.dimY()*in.dimZ()*batchSize;
        Tensor<Float_T> in_fused;
        in_fused.resize({in.dimX(),in.dimY(),in.dimZ(),in.dimB()});
        in_fused.fill(0.0);
        for(unsigned int i = 0; i < dimsInput; ++i)
        {
            in_fused(i) = in(i) * 255;
        }
    
        ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv1(dn, "conv1",
            std::vector<unsigned int>({kernelWidth, kernelHeight}),
            nbOutputs_conv1,
            std::vector<unsigned int>({subSampleX, subSampleY}),
            std::vector<unsigned int>({strideX, strideY}),
            std::vector<int>({(int)paddingX, (int)paddingY}),
            std::vector<unsigned int>({1U, 1U}),
            std::shared_ptr<Activation>());
        conv1.setParameter("NoBias", true);

        ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv1_fused(dn, "conv1_fused",
            std::vector<unsigned int>({kernelWidth, kernelHeight}),
            nbOutputs_conv1,
            std::vector<unsigned int>({subSampleX, subSampleY}),
            std::vector<unsigned int>({strideX, strideY}),
            std::vector<int>({(int)paddingX, (int)paddingY}),
            std::vector<unsigned int>({1U, 1U}),
            std::shared_ptr<Activation>());
        conv1_fused.setParameter("NoBias", true);

        //additional conv to compensate for weights transformation from [-1,1] to [-127,128], and back
        ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv1_fused_const(dn, "conv1_fused_const",
        std::vector<unsigned int>({kernelWidth, kernelHeight}),
        nbOutputs_conv1,
        std::vector<unsigned int>({subSampleX, subSampleY}),
        std::vector<unsigned int>({strideX, strideY}),
        std::vector<int>({(int)paddingX, (int)paddingY}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>());
        conv1_fused_const.setParameter("NoBias", true);

        BatchNormCell_Frame_CUDA<float> bn1(dn, "bn1", nbOutputs_conv1, std::shared_ptr<Activation>());

        ConvCell_QuantizerSAT_BNFusion_LeNet_Frame_CUDA_Test<float> conv2(dn, "conv2",
            std::vector<unsigned int>({kernelWidth, kernelHeight}),
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

        std::vector<float> v_w11 = {0.184185, 0.173881, 0.112021, 0.0815895, -0.173858, 
                                    -0.0106287, -0.112002, -0.343679, -0.312961, -0.0172239, 
                                    -0.171215, -0.291387, -0.217766, -0.177112, 0.37201, 
                                    -0.144616, -0.24736, -0.0290122, 0.000954552, 0.390555, 
                                    -0.226636, -0.0951204, 0.231468, 0.144098, 0.367307};

        Tensor<float> conv1_w1({kernelWidth, kernelHeight}, v_w11.begin(),v_w11.end());

        std::vector<float> v_w12 = {-0.177018, 0.139442, 0.288037, 0.158078, 0.0445046, 
                                    -0.385413, -0.191178, -5.84598e-07, 0.0907883, 0.11398, 
                                    0.0205704, -0.352544, -0.278075, -0.0861968, -0.0952327, 
                                    0.283583, -0.0489072, -0.09156, -0.126722, -0.179714, 
                                    0.034634, 0.0141254, -0.00935394, -0.071578, -0.11565};
        Tensor<float> conv1_w2({kernelWidth, kernelHeight}, v_w12.begin(),v_w12.end());

        std::vector<float> v_w13 = {-0.159753, 0.0313658, 0.145843, -0.228753, 0.153917, 
                                    0.0871841, 0.216748, -0.0137101, -0.62059, -0.337779, 
                                    -0.0262911, 0.179648, -0.0586273, -0.625386, -0.0737472, 
                                    -0.193735, 0.309977, 0.441459, 0.304995, 0.221394, 
                                    0.0382392, -0.158866, 0.486351, 0.351878, -0.0907166};
        Tensor<float> conv1_w3({kernelWidth, kernelHeight}, v_w13.begin(),v_w13.end());

        std::vector<float> v_w14 = {-0.0414309, -0.0289402, 0.0325286, -0.0799899, 0.0532336, 
                                    -0.209418, 0.0343616, -0.156739, -0.1015, -0.133251, 
                                    -0.178087, -0.0740313, -0.0604055, -0.0632791, -0.115115, 
                                    -0.0816227, -0.155578, -0.00729363, -0.167025, -0.322911, 
                                    -0.254545, -0.244077, -0.056426, -0.273549, -0.363163};
        Tensor<float> conv1_w4({kernelWidth, kernelHeight}, v_w14.begin(),v_w14.end());

        std::vector<float> v_w15 = {0.309687, -0.197089, 0.0191584, 3.49737e-05, 0.0382037, 
                                    0.0226852, 0.0331087, 0.122251, 0.0578961, -0.0379796, 
                                    -0.0278929, 0.387234, 0.387587, 0.105464, -0.186279, 
                                    0.0619652, 0.0902182, 0.0114893, -0.315561, -0.476871, 
                                    0.233503, -0.0460472, 0.0427327, -0.177359, -0.13211};
        Tensor<float> conv1_w5({kernelWidth, kernelHeight}, v_w15.begin(),v_w15.end());

        std::vector<float> v_w16 = {-0.205494, -0.252992, -0.149511, -0.141254, -0.220744, 
                                    -0.115654, -0.094326, -0.207501, -0.0156246, -0.2556, 
                                    -0.0641069, -0.108896, -0.0255057, -0.126213, -0.0662551, 
                                    -0.0450158, 0.0129077, 0.0074881, -0.0830555, -0.282294, 
                                    0.0423633, -0.141093, -0.247252, -0.151985, -0.224323};
        Tensor<float> conv1_w6({kernelWidth, kernelHeight}, v_w16.begin(),v_w16.end());

        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                Tensor<float> kernel({kernelWidth, kernelHeight});

                if(output==0) kernel=conv1_w1;
                if(output==1) kernel=conv1_w2;
                if(output==2) kernel=conv1_w3;
                if(output==3) kernel=conv1_w4;
                if(output==4) kernel=conv1_w5;
                if(output==5) kernel=conv1_w6;

                conv1.setWeight(output, channel, kernel);
            }

        }

        //set fake weights for conv2, doesn't matter; we need only quant input of conv2 for comparison
        for (unsigned int output = 0; output < nbOutputs_conv2; ++output) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                Tensor<float> kernel({kernelWidth, kernelHeight}, 1.0);
                conv2.setWeight(output, channel, kernel);
            }

        }

        std::vector<float> v_scale = {1.5143, 1.49043, 1.69947, 1.17242, 1.7597, 1.1505};
        std::vector<float> v_bias = {-0.0654494, -0.0264921, 0.494825, 0.0508658, 0.196104, 0.0267999};
        std::vector<float> v_mean = {-0.166211, -0.312195, 0.143546, -0.920134, 0.0963226, -0.955784};
        std::vector<float> v_variance = {0.697349, 0.323682, 0.430773, 1.60627, 0.465582, 1.65451};
    
        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            
            Tensor<float> scale({1},v_scale.at(output));
            Tensor<float> bias({1}, v_bias.at(output));
            Tensor<float> mean({1}, v_mean.at(output));
            Tensor<float> variance({1}, v_variance.at(output));

            bn1.setScale(output, scale);
            bn1.setBias(output, bias);
            bn1.setMean(output, mean);
            bn1.setVariance(output, variance);
        }

    
        conv1.propagate(true);
        bn1.propagate(true);
        conv2.propagate(true);

        conv1.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv1 = tensor_cast<float>(conv1.getOutputs());
        conv1.getOutputs().synchronizeHToD();

        bn1.getOutputs().synchronizeDToH();
        const Tensor<float>& out_bn1 = tensor_cast<float>(bn1.getOutputs());
        bn1.getOutputs().synchronizeHToD();
 
        quant2.getQuantizedActivations(0).synchronizeDToH();
        const Tensor<float>& quant_act_conv2 = tensor_cast<float>(quant2.getQuantizedActivations(0));
        quant2.getQuantizedActivations(0).synchronizeHToD();
   
        // ===> fuse BN by hands following SAT paper logic in S7.
        // a1 and alpha1 are from conv1; a2 and alpha2 are from conv2;
        // beta and gamma are from bn1
        // 0. create conv1_fused layer with inputs in range [0,255] and quantized weights from conv1 rescaled to [-127, 128]
        //set input and init
        conv1_fused.addInput(in_fused, out_diff);
        conv1_fused.initialize();

        conv1_fused_const.addInput(in_fused, out_diff);
        conv1_fused_const.initialize();
        //std::cout << "********************SET_INPUT_8-BITS_END********************\n\n" << std::endl; 

        //set weights for conv1_fused : quantized weights are in the quantizer of conv1
        quant1.getQuantizedWeights(0).synchronizeDToH();
        const Tensor<float>& quant_weights_conv1 = tensor_cast<float>(quant1.getQuantizedWeights(0));
        quant1.getQuantizedWeights(0).synchronizeHToD();

        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            for (unsigned int channel = 0; channel < nbChannels;
                ++channel) {

                Tensor<float> kernel_rescaled({kernelWidth,
                                    kernelHeight});
                
                for (unsigned int sx = 0; sx < kernelWidth; ++sx) {
                    for (unsigned int sy = 0; sy < kernelHeight; ++sy){
                        //range [-127,128]
                        //kernel_rescaled(sx, sy) = rintf(0.5*(quant_weights_conv1[output][channel](sx, sy)*range1+1));
                        //range [-128,127]
                        kernel_rescaled(sx, sy) = rintf(0.5*(quant_weights_conv1[output][channel](sx, sy)*range1-1));
                    }
                }
                conv1_fused.setWeight(output, channel, kernel_rescaled);
            }
        }

        //fill weights of conv1_fused_const with 0.5
        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            for (unsigned int channel = 0; channel < nbChannels;
                ++channel) {

                Tensor<float> kernel_const({kernelWidth,
                                    kernelHeight}, 0.5);
                conv1_fused_const.setWeight(output, channel, kernel_const);
            }
        }

        // 1. propagate
        conv1_fused.propagate(true);

        conv1_fused.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv1_fused_prop = tensor_cast<float>(conv1_fused.getOutputs());
        //std::cout << "[Conv1 Fusion][Output after propagate]" << std::endl;
        //std::cout << out_conv1_fused_prop << std::endl;
        conv1_fused.getOutputs().synchronizeHToD();

        conv1_fused_const.propagate(true);
        conv1_fused_const.getOutputs().synchronizeDToH();
        const Tensor<float>& out_conv1_fused_const_prop = tensor_cast<float>(conv1_fused_const.getOutputs());
        //std::cout << "[Conv1 Fusion Const][Output after propagate]" << std::endl;
        //std::cout << out_conv1_fused_const_prop << std::endl;
        conv1_fused_const.getOutputs().synchronizeHToD();

        // 2. get BN gamma and beta (as done in DeepNet.cpp fuseBatchNormWithConv)
        const bool noBias = conv1.getParameter<bool>("NoBias");
        const Tensor<float>& bnScales
            = tensor_cast<float>(*(bn1.getScales()));
        const Tensor<float>& bnBiases
            = tensor_cast<float>(*(bn1.getBiases()));
        const Tensor<float>& bnMeans
            = tensor_cast<float>(*(bn1.getMeans()));
        const Tensor<float>& bnVariances
            = tensor_cast<float>(*(bn1.getVariances()));
        const double eps = bn1.getParameter<double>("Epsilon");

        float meanVariance = 0.0f;
        unsigned int count = 0;

        for (std::size_t output = 0; output < nbOutputs_conv1; ++output) {
            if (bnVariances(output) > 1.0e-12) {
                meanVariance += bnVariances(output);
                ++count;
            }
            else {
                std::cout << "    zero-variance " << conv1.getName()
                    << "[" << output << "]" << std::endl;
            }
        }

        meanVariance /= count;

        Tensor<float> gamma;
        gamma.resize({nbOutputs_conv1}, 0.0);
        Tensor<float> beta;
        beta.resize({nbOutputs_conv1}, 0.0);
        Tensor<float> bias_fusion;
        bias_fusion.resize({nbOutputs_conv1}, 0.0);

        //tensor for comparison with the reference
        Tensor<float> clipPerOutput;
        clipPerOutput.resize({nbOutputs_conv1}, 0.0);
        Tensor<float> scalePerOutput;
        scalePerOutput.resize({nbOutputs_conv1}, 0.0);
        //tensor rounded in biases and cliping values
        Tensor<float> clipPerOutputRound;
        clipPerOutputRound.resize({nbOutputs_conv1}, 0.0);

        Tensor<float> bias_fusion_rounded;
        bias_fusion_rounded.resize({nbOutputs_conv1}, 0.0);

        for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
            //Biases adjustments
            Tensor<float> bias;
            if (noBias)
                bias.resize({1}, 0.0);
            else
                conv1.getBias(output, bias);

            //factor for weights adjustments
            float factor = bnScales(output)
                    / std::sqrt(eps + ((bnVariances(output) > 1.0e-12)
                                ? bnVariances(output) : meanVariance));
            gamma(output) = factor;
            beta(output) = bnBiases(output) + (bias(0) - bnMeans(output)) * factor;
            bias_fusion(output) = (beta(output)/gamma(output)) * ((float)range1/alpha1) * ((float)range1/2.0);
            clipPerOutput(output) = (alpha2/gamma(output)) * ((float)range1/alpha1) * ((float)range1/2.0);

            bias_fusion_rounded(output) = rintf((beta(output)/gamma(output)) * ((float)range1/alpha1) * ((float)range1/2.0));
            clipPerOutputRound(output) = rintf((alpha2/gamma(output)) * ((float)range1/alpha1) * ((float)range1/2.0));
            scalePerOutput(output) = (alpha1/(float)range1) * ((float)range2/alpha2) * gamma(output) *(2.0/(float)range1);
            
        }  

        //not cliped, with bias 
        Tensor<float> conv1_add_bias;
        conv1_add_bias.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //cliped output 
        Tensor<float> conv1_clipped;
        conv1_clipped.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //scaled output 
        Tensor<float> conv1_scaled;
        conv1_scaled.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //round output 
        Tensor<float> conv1_rounded;
        conv1_rounded.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        for (unsigned int batch = 0; batch < batchSize; ++batch) {
            for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
                for (unsigned int oy = 0; oy < conv1_fused.getOutputsHeight(); ++oy) {
                    for (unsigned int ox = 0; ox < conv1_fused.getOutputsWidth(); ++ox) {
                        //Add bias term and an additional contribution from weighs transformation
                        float value = out_conv1_fused_prop(ox, oy, output, batch)  
                                        //- rintf(out_conv1_fused_const_prop(ox, oy, output, batch))
                                        + rintf(out_conv1_fused_const_prop(ox, oy, output, batch))
                                        + bias_fusion(output);
                        conv1_add_bias(ox, oy, output, batch) = value;

                        //Clip the value with a clip factor par output
                        float clippedValue = (value < 0.0f) ? 0.0f 
                                            : (value < clipPerOutput(output)) ? value 
                                            : clipPerOutput(output);
                        conv1_clipped(ox, oy, output, batch) = clippedValue;
                        conv1_scaled(ox, oy, output, batch) = clippedValue*scalePerOutput(output);

                        //scale and round the result
                        conv1_rounded(ox, oy, output, batch) = rintf(clippedValue*scalePerOutput(output));
                    }
                }
            }
        }

        //not cliped, with bias 
        Tensor<float> conv1_add_bias_r;
        conv1_add_bias_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //cliped output 
        Tensor<float> conv1_clipped_r;
        conv1_clipped_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //scaled output 
        Tensor<float> conv1_scaled_r;
        conv1_scaled_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        //round output 
        Tensor<float> conv1_rounded_r;
        conv1_rounded_r.resize({conv1_fused.getOutputsWidth(),conv1_fused.getOutputsHeight(),nbOutputs_conv1,batchSize}, 0.0);

        for (unsigned int batch = 0; batch < batchSize; ++batch) {
            for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
                for (unsigned int oy = 0; oy < conv1_fused.getOutputsHeight(); ++oy) {
                    for (unsigned int ox = 0; ox < conv1_fused.getOutputsWidth(); ++ox) {
                        //Add bias term and an additional contribution from weighs transformation
                        float value = out_conv1_fused_prop(ox, oy, output, batch) 
                                        //- rintf(out_conv1_fused_const_prop(ox, oy, output, batch))
                                        + rintf(out_conv1_fused_const_prop(ox, oy, output, batch))
                                        + bias_fusion_rounded(output);
                        conv1_add_bias_r(ox, oy, output, batch) = value;

                        //Clip the value with a clip factor par output
                        float clippedValue = (value < 0.0f) ? 0.0f 
                                            : (value < clipPerOutputRound(output)) ? value 
                                            : clipPerOutputRound(output);
                        conv1_clipped_r(ox, oy, output, batch) = clippedValue;
                        conv1_scaled_r(ox, oy, output, batch) = clippedValue*scalePerOutput(output);

                        //scale and round the result
                        conv1_rounded_r(ox, oy, output, batch) = rintf(clippedValue*scalePerOutput(output));
                    }
                }
            }
        }

        size_t dimsQ2 = quant_act_conv2.dimX()*quant_act_conv2.dimY()*quant_act_conv2.dimZ()*batchSize;
        Tensor<float> quant_conv2_unscaled;
        quant_conv2_unscaled.resize({quant_act_conv2.dimX(),quant_act_conv2.dimY(),quant_act_conv2.dimZ(),batchSize}, 0.0);
        for(unsigned int i = 0; i < dimsQ2; ++i)
        {
            quant_conv2_unscaled(i) = rintf(quant_act_conv2(i) * ((float)range2/alpha2));
        }

        double mse = 0.0, mse_r = 0.0;
        double max_error = 0.0;
        int faulty_pixel = 0;
        double faulty_pixel_percent = 0.0;

        for (unsigned int batch = 0; batch < batchSize; ++batch) {
            for (unsigned int output = 0; output < nbOutputs_conv1; ++output) {
                for (unsigned int oy = 0; oy < conv1_fused.getOutputsHeight(); ++oy) {
                    for (unsigned int ox = 0; ox < conv1_fused.getOutputsWidth(); ++ox) {
                        double error = conv1_rounded(ox, oy, output, batch) - quant_conv2_unscaled(ox, oy, output, batch);
                        double error_r = conv1_rounded_r(ox, oy, output, batch) - quant_conv2_unscaled(ox, oy, output, batch);

                        mse += std::pow(error, 2);
                        mse_r += std::pow(error_r, 2);

                        if(std::abs(error_r) > max_error)
                            max_error = std::abs(error_r);

                        if(std::abs(error_r) >= 1){
                            faulty_pixel++;
                        }
                    }
                }
            }
        }
        mse /= (double) batchSize* nbOutputs_conv1 * conv1_fused.getOutputsHeight() * conv1_fused.getOutputsWidth();
        mse_r /= (double) batchSize* nbOutputs_conv1 * conv1_fused.getOutputsHeight() * conv1_fused.getOutputsWidth();
        faulty_pixel_percent = ((double)faulty_pixel*100)/((double)(batchSize*nbOutputs_conv1 * conv1_fused.getOutputsHeight() * conv1_fused.getOutputsWidth()));

        v_mse.push_back(mse);
        v_mse_r.push_back(mse_r);
        v_mse_max.push_back(max_error);
        v_faulty_pixel.push_back(faulty_pixel);
        v_faulty_pixel_percent.push_back(faulty_pixel_percent);
        
    }

    std::ostringstream fileName_mse;
    fileName_mse << "bnFusion_exact_mse_" << range1 << "_" << range2 << ".dat";
    std::ostringstream fileName_mse_r;
    fileName_mse_r << "bnFusion_exact_mse_round_" << range1 << "_" << range2 << ".dat";
    std::ostringstream fileName_max_err;
    fileName_max_err << "bnFusion_exact_max_err_" << range1 << "_" << range2 << ".dat";
    std::ostringstream fileName_faulty_pixel;
    fileName_faulty_pixel << "bnFusion_exact_faulty_pixel_" << range1 << "_" << range2 << ".dat";
    std::ostringstream fileName_faulty_pixel_percent;
    fileName_faulty_pixel_percent << "bnFusion_exact_faulty_pixel_percent_" << range1 << "_" << range2 << ".dat";

    std::ofstream outFile_mse(fileName_mse.str());
    std::ofstream outFile_mse_r(fileName_mse_r.str());
    std::ofstream outFile_mse_max(fileName_max_err.str());
    std::ofstream outFile_faulty_pixel(fileName_faulty_pixel.str());
    std::ofstream outFile_faulty_pixel_percent(fileName_faulty_pixel_percent.str());

    for (std::size_t i = 0; i < v_mse.size(); ++i) {
        outFile_mse << v_mse[i] << "\n";
        outFile_mse_r << v_mse_r[i] << "\n";
        outFile_mse_max << v_mse_max[i] << "\n";
        outFile_faulty_pixel << v_faulty_pixel[i] << "\n";
        outFile_faulty_pixel_percent << v_faulty_pixel_percent[i] << "\n";
    }

    std::string fileName_img_mse = fileName_mse.str();
    fileName_img_mse.replace(fileName_img_mse.begin()+(fileName_img_mse.size()-4),fileName_img_mse.end(),".png");
    std::string fileName_img_mse_r = fileName_mse_r.str();
    fileName_img_mse_r.replace(fileName_img_mse_r.begin()+(fileName_img_mse_r.size()-4),fileName_img_mse_r.end(),".png");
    std::string fileName_img_max_err = fileName_max_err.str();
    fileName_img_max_err.replace(fileName_img_max_err.begin()+(fileName_img_max_err.size()-4),fileName_img_max_err.end(),".png");
    std::string fileName_img_faulty_pixel = fileName_faulty_pixel.str();
    fileName_img_faulty_pixel.replace(fileName_img_faulty_pixel.begin()+(fileName_img_faulty_pixel.size()-4),fileName_img_faulty_pixel.end(),".png");
    std::string fileName_img_faulty_pixel_percent = fileName_faulty_pixel_percent.str();
    fileName_img_faulty_pixel_percent.replace(fileName_img_faulty_pixel_percent.begin()+(fileName_img_faulty_pixel_percent.size()-4),fileName_img_faulty_pixel_percent.end(),".png");

    // plot results
    Gnuplot gnuplot_mse;
    gnuplot_mse.set("grid");   
    gnuplot_mse << "binwidth=0.01";
    gnuplot_mse << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_mse.setXlabel("MSE");
    gnuplot_mse.setYlabel("Number of images");
    gnuplot_mse.saveToFile(fileName_img_mse);
    gnuplot_mse.plot(fileName_mse.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));

    Gnuplot gnuplot_mse_r;
    gnuplot_mse_r.set("grid");   
    gnuplot_mse_r << "binwidth=0.01";
    gnuplot_mse_r << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_mse_r.setXlabel("MSE with rounding");
    gnuplot_mse_r.setYlabel("Number of images");
    gnuplot_mse_r.saveToFile(fileName_img_mse_r);
    gnuplot_mse_r.plot(fileName_mse_r.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));

    Gnuplot gnuplot_max_err;
    gnuplot_max_err.set("grid");   
    gnuplot_max_err << "binwidth=0.01";
    gnuplot_max_err << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_max_err.setXlabel("Max error");
    gnuplot_max_err.setYlabel("Number of images");
    gnuplot_max_err.saveToFile(fileName_img_max_err);
    gnuplot_max_err.plot(fileName_max_err.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));

    Gnuplot gnuplot_faulty_pixel;
    gnuplot_faulty_pixel.set("grid");   
    gnuplot_faulty_pixel << "binwidth=0.01";
    gnuplot_faulty_pixel << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_faulty_pixel.setXlabel("Number of faulty pixels");
    gnuplot_faulty_pixel.setYlabel("Number of images");
    gnuplot_faulty_pixel.saveToFile(fileName_img_faulty_pixel);
    gnuplot_faulty_pixel.plot(fileName_faulty_pixel.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));

    Gnuplot gnuplot_faulty_pixel_percent;
    gnuplot_faulty_pixel_percent.set("grid");   
    gnuplot_faulty_pixel_percent << "binwidth=0.01";
    gnuplot_faulty_pixel_percent << "bin(x,width)=width*floor(x/width)+width/2.0";
    gnuplot_faulty_pixel_percent.setXlabel("Percentage of faulty pixels");
    gnuplot_faulty_pixel_percent.setYlabel("Number of images");
    gnuplot_faulty_pixel_percent.saveToFile(fileName_img_faulty_pixel_percent);
    gnuplot_faulty_pixel_percent.plot(fileName_faulty_pixel_percent.str(), std::string("using (bin($1,binwidth)):(1.0) smooth freq with boxes"));
    */
}




RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
