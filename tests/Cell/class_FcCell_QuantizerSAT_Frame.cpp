/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "N2D2.hpp"

#include "Database/MNIST_IDX_Database.hpp"
#include "Cell/FcCell_Frame.hpp"
#include "DeepNet.hpp"
#include "Xnet/Environment.hpp"
#include "Xnet/Network.hpp"
#include "third_party/half.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "utils/UnitTest.hpp"
#include "Cell/DropoutCell_Frame.hpp"
#include "Cell/SoftmaxCell_Frame.hpp"
#include "Quantizer/SATQuantizer_Frame.hpp"

using namespace N2D2;

template <class T>
class FcCell_Frame_Test : public FcCell_Frame<T> {
public:
    FcCell_Frame_Test(const DeepNet& deepNet, 
                           const std::string& name,
                           unsigned int nbOutputs,
                           const std::shared_ptr
                           <Activation>& activation)
        : Cell(deepNet, name, nbOutputs),
          FcCell(deepNet, name, nbOutputs),
          FcCell_Frame<T>(deepNet, name, nbOutputs, activation) {};

    friend class UnitTest_FcCell_QuantizerSAT_Frame_float_check_quantizer_SAT;
    friend class UnitTest_FcCell_QuantizerSAT_Frame_double_check_quantizer_SAT;
};

static MNIST_IDX_Database& getDatabase() {
    static MNIST_IDX_Database database(N2D2_DATA("mnist"));
    return database;
}

TEST_DATASET(FcCell_QuantizerSAT_Frame_float,
             check_quantizer_SAT,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight,
              size_t range,
              float alpha),
             std::make_tuple(2U, 3U, 3U, 15,10.0)
             )
{

    std::cout<<"FC layer check_quantizer_SAT"<<std::endl;

    const unsigned int nbChannels = 1;
          
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    Tensor<Float_T>& in = env.getData();
    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    //fill input image
    int counter = 0;
    for (unsigned int b = 0; b < in.dimB(); ++b) {
        for (unsigned int z = 0; z < in.dimZ(); ++z) {
            for (unsigned int y = 0; y < in.dimY(); ++y) {
                for (unsigned int x = 0; x < in.dimX(); ++x) {
                    //in(x, y, z, b) = 0.125f*counter;
                    in(x, y, z, b) = 1.0f;
                    counter++;
                    std::cout << "b, z, y, x = " << b << " , " << z << " , " << y << " , " << x << " , input = " << in(x, y, z, b) << std::endl;
                }
            }
        }
    }

    DropoutCell_Frame<Float_T> drop1(dn, "drop1", 1);
    drop1.setParameter<double>("Dropout", 0.0);
    FcCell_Frame_Test<float> fc1(dn, "fc1",
                               nbOutputs,
                               std::shared_ptr<Activation>());

    fc1.setParameter("NoBias", true);

    SATQuantizer_Frame<float> quant;
    quant.setRange(range);
    quant.setAlpha(alpha);
    quant.setQuantization(true);
    quant.setScaling(true);
    std::shared_ptr<Quantizer> quantizer = std::shared_ptr<Quantizer>(&quant, [](Quantizer *) {});

    SoftmaxCell_Frame<float> softmax1(dn, "softmax1", nbOutputs, true, 0);

    drop1.addInput(in,in);
    fc1.addInput(&drop1);
    softmax1.addInput(&fc1);
    drop1.initialize();
    fc1.setQuantizer(quantizer);
    fc1.initialize();
    softmax1.initialize();

    if(fc1.getQuantizer()){
        std::cout << "Added " <<  fc1.getQuantizer()->getType() <<
        " quantizer to " << fc1.getName() << std::endl;
    }

    //set weights to fc
    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    ASSERT_EQUALS(inputSize, channelsWidth * channelsHeight);
    ASSERT_EQUALS(outputSize, nbOutputs);

    float weight_tmp = 0.0f;

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            if(output==0) weight_tmp = 0.001;
            if(output==1) weight_tmp = 0.1;
            Tensor<float> weight({1}, weight_tmp);
            std::cout << "output = " << output << " , channel =  " << channel << " , weight = " << weight << std::endl;
            fc1.setWeight(output, channel, weight);
        }
    }

    drop1.propagate(false);
    fc1.propagate(false);
    softmax1.propagate(false);

    fc1.getOutputs().synchronizeDToH();
    const Tensor<float>& out = tensor_cast<float>(fc1.getOutputs());

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        std::cout << "output (quant) = " << out(output, 0) << std::endl;
    }

    fc1.getOutputs().synchronizeHToD();
    std::cout <<"end of propagate" << std::endl;
    softmax1.mDiffInputs.synchronizeDToH();
    softmax1.getOutputs().synchronizeDToH();
    const Tensor<float>& out_softmax1 = tensor_cast<float>(softmax1.getOutputs());
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
            std::cout << "softmax1.mDiffInputs(nout, batchPos) = " << softmax1.mDiffInputs(nout, batchPos) << std::endl;
        }
    }

    loss = softmax1.applyLoss();
    std::cout << "loss = " << loss << std::endl;
    softmax1.mDiffInputs.setValid();
    softmax1.mDiffInputs.synchronizeHToD();
    softmax1.getOutputs().synchronizeHToD();

    //backpropagate 
    softmax1.backPropagate();   
    fc1.backPropagate();
    drop1.backPropagate();

    fc1.update();

    Tensor<float> alphaEstimated = quant.getAlpha(0);
    alphaEstimated.synchronizeDToH();
    std::cout << "alphaEstimated = " << alphaEstimated << std::endl;

     for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<float> weight;
            fc1.getWeight(output, channel, weight);
            std::cout << "output = " << output << " , channel =  " << channel << " , weight = " << weight << std::endl;
        }
    }

}

// test in double 

TEST_DATASET(FcCell_QuantizerSAT_Frame_double,
             check_quantizer_SAT,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight,
              size_t range,
              double alpha),
             std::make_tuple(2U, 3U, 3U, 15,10.0)
             )
{

    std::cout<<"double :: FC layer check_quantizer_SAT"<<std::endl;

    const unsigned int nbChannels = 1;
          
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    Tensor<double> in;
    in.resize({channelsWidth, channelsHeight, 1});

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    //fill input image
    int counter = 0;
    for (unsigned int b = 0; b < in.dimB(); ++b) {
        for (unsigned int z = 0; z < in.dimZ(); ++z) {
            for (unsigned int y = 0; y < in.dimY(); ++y) {
                for (unsigned int x = 0; x < in.dimX(); ++x) {
                    //in(x, y, z, b) = 0.125*counter;
                    in(x, y, z, b) = 1.0;
                    counter++;
                    std::cout << "b, z, y, x = " << b << " , " << z << " , " << y << " , " << x << " , input = " << in(x, y, z, b) << std::endl;
                }
            }
        }
    }

    DropoutCell_Frame<double> drop1(dn, "drop1", 1);
    drop1.setParameter<double>("Dropout", 0.0);

    FcCell_Frame_Test<double> fc1(dn, "fc1",
                               nbOutputs,
                               std::shared_ptr<Activation>());

    fc1.setParameter("NoBias", true);

    SATQuantizer_Frame<double> quant;
    quant.setRange(range);
    quant.setAlpha(alpha);
    quant.setQuantization(true);
    quant.setScaling(true);
    std::shared_ptr<Quantizer> quantizer = std::shared_ptr<Quantizer>(&quant, [](Quantizer *) {});

    SoftmaxCell_Frame<double> softmax1(dn, "softmax1", nbOutputs, true, 0);

    drop1.addInput(in,in);
    fc1.addInput(&drop1);
    softmax1.addInput(&fc1);
    drop1.initialize();

    fc1.setQuantizer(quantizer);
    fc1.initialize();
    softmax1.initialize();

    if(fc1.getQuantizer()){
        std::cout << "Added " <<  fc1.getQuantizer()->getType() <<
        " quantizer to " << fc1.getName() << std::endl;
    }

    //set weights to fc
    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    ASSERT_EQUALS(inputSize, channelsWidth * channelsHeight);
    ASSERT_EQUALS(outputSize, nbOutputs);

    double weight_tmp = 0.0;

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            if(output==0) weight_tmp = 0.001;
            if(output==1) weight_tmp = 0.1;
            Tensor<double> weight({1}, weight_tmp);
            std::cout << "output = " << output << " , channel =  " << channel << " , weight = " << weight << std::endl;
            fc1.setWeight(output, channel, weight);
        }
    }

    drop1.propagate(false);
    fc1.propagate(false);
    softmax1.propagate(false);

    fc1.getOutputs().synchronizeDToH();
    const Tensor<double>& out = tensor_cast<double>(fc1.getOutputs());

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        std::cout << "output (quant) = " << out(output, 0) << std::endl;
    }

    fc1.getOutputs().synchronizeHToD();
    std::cout <<"end of propagate" << std::endl;

    softmax1.mDiffInputs.synchronizeDToH();
    softmax1.getOutputs().synchronizeDToH();
    const Tensor<double>& out_softmax1 = tensor_cast<double>(softmax1.getOutputs());
    double loss = 0.0;

    for(unsigned int nout = 0; nout < nbOutputs; ++nout){
        for (unsigned int batchPos = 0; batchPos < 1; ++batchPos){
            std::cout << "out_softmax1(nout, batchPos) = " << out_softmax1(nout, batchPos) << std::endl;
            if(nout==0) {
                softmax1.mDiffInputs(nout, batchPos) = 1.0;
            }
            if(nout==1) {
                softmax1.mDiffInputs(nout, batchPos) = 0.0;
            }
            std::cout << "softmax1.mDiffInputs(nout, batchPos) = " << softmax1.mDiffInputs(nout, batchPos) << std::endl;
        }
    }

    loss = softmax1.applyLoss();
    std::cout << "loss = " << loss << std::endl;
    softmax1.mDiffInputs.setValid();
    softmax1.mDiffInputs.synchronizeHToD();
    softmax1.getOutputs().synchronizeHToD();

    //backpropagate 
    softmax1.backPropagate();   
    fc1.backPropagate();
    drop1.backPropagate();

    fc1.update();

    Tensor<double> alphaEstimated = quant.getAlpha(0);
    alphaEstimated.synchronizeDToH();
    std::cout << "alphaEstimated = " << alphaEstimated << std::endl;

     for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<double> weight;
            fc1.getWeight(output, channel, weight);
            std::cout << "output = " << output << " , channel =  " << channel << " , weight = " << weight << std::endl;
        }
    }

}




RUN_TESTS()

