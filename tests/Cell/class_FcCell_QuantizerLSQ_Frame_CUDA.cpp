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

#ifdef CUDA

#include "N2D2.hpp"

#include "Database/MNIST_IDX_Database.hpp"
#include "Cell/FcCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "Xnet/Network.hpp"
#include "Xnet/Environment.hpp"
#include "third_party/half.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "utils/UnitTest.hpp"
#include "Cell/DropoutCell_Frame_CUDA.hpp"
#include "Cell/SoftmaxCell_Frame_CUDA.hpp"
#include "Quantizer/QAT/Cell/LSQ/LSQQuantizerCell_Frame_CUDA.hpp"
#include "Quantizer/QAT/Activation/LSQ/LSQQuantizerActivation_Frame_CUDA.hpp"
#include "Activation/LinearActivation_Frame_CUDA.hpp"
#include "Activation/LinearActivation_Frame.hpp"
#include "Cell/ActivationCell_Frame_CUDA.hpp"

using namespace N2D2;

template <class T>
class FcCell_Frame_Test_CUDA : public FcCell_Frame_CUDA<T> {
public:
    FcCell_Frame_Test_CUDA(const DeepNet& deepNet, 
                           const std::string& name,
                           unsigned int nbOutputs,
                           const std::shared_ptr
                           <Activation>& activation)
        : Cell(deepNet, name, nbOutputs),
          FcCell(deepNet, name, nbOutputs),
          FcCell_Frame_CUDA<T>(deepNet, name, nbOutputs, activation) {};

    friend class UnitTest_FcCell_QuantizerLSQ_Frame_CUDA_float_check_quantizer_LSQ;
};

static MNIST_IDX_Database& getDatabase() {
    static MNIST_IDX_Database database(N2D2_DATA("mnist"));
    return database;
}


TEST_DATASET(FcCell_QuantizerLSQ_Frame_CUDA_float,
             check_quantizer_LSQ,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight,
              size_t range),
             std::make_tuple(2U, 3U, 3U, 15)
             )
{

    std::cout<<"FC layer check with LSQ quantizer"<<std::endl;

    CudaContext::setDevice(0);
          
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1, 1});

    Tensor<Float_T> in;
    in.resize({channelsWidth, channelsHeight, 1, 1});

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    float stepEstimated1_pytorch = 0.1442825;

    //fill input image
    int counter = 0;
    for (unsigned int b = 0; b < in.dimB(); ++b) {
        for (unsigned int z = 0; z < in.dimZ(); ++z) {
            for (unsigned int y = 0; y < in.dimY(); ++y) {
                for (unsigned int x = 0; x < in.dimX(); ++x) {
                    in(x, y, z, b) = 0.125f*counter;
                    counter++;
                }
            }
        }
    }

    std::cout << "[Input]\n" << in << std::endl;

    const std::shared_ptr<Activation>& activation1 
            = std::make_shared<LinearActivation_Frame_CUDA<float> >();

    ActivationCell_Frame_CUDA<float> activation1_cell(  dn, 
                                                        "activation1", 
                                                        1, 
                                                        activation1);

    FcCell_Frame_Test_CUDA<float> fc1(dn, "fc1",
                               nbOutputs,
                               std::shared_ptr<Activation>());

    fc1.setParameter("NoBias", true);

    LSQQuantizerCell_Frame_CUDA<float> quant;
    quant.setRange(range);
    std::shared_ptr<QuantizerCell> quantizer = std::shared_ptr<QuantizerCell>(&quant, [](QuantizerCell *) {});

    SoftmaxCell_Frame_CUDA<float> softmax1(dn, "softmax1", nbOutputs, true, 0);

    Tensor<float> out_diff({channelsWidth, channelsHeight, 1, 1});

    activation1_cell.addInput(in,out_diff);
    fc1.addInput(&activation1_cell);
    softmax1.addInput(&fc1);

    fc1.setQuantizer(quantizer);

    activation1_cell.initialize();
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
            fc1.setWeight(output, channel, weight);
        }
    }

    for(unsigned int iter_index = 0; iter_index < 10000; ++iter_index){

            activation1_cell.propagate(false);
            fc1.propagate(false);
            softmax1.propagate(false);

            if(iter_index==9999){
                std::cout << "[Fc][FullPrecisionWeights before  ...]" << std::endl;
                for (unsigned int output = 0; output < outputSize; ++output) {
                    for (unsigned int channel = 0; channel < inputSize; ++channel) {
                        Tensor<float> weight;
                        fc1.getWeight(output, channel, weight);
                        std::cout << "[" << output << "][" << channel << "]: \n" 
                        << weight << std::endl;
                    }
                }

                quant.getStepSize().synchronizeDToH();
                CudaTensor<float> stepSize = quant.getStepSize();
                stepSize.synchronizeDToH();
                std::cout << "[Fc][StepSize before ... ]\n " << stepSize << std::endl;
                quant.getStepSize().synchronizeHToD();

                activation1_cell.getOutputs().synchronizeDToH();
                const Tensor<float>& out_act1 = tensor_cast<float>(activation1_cell.getOutputs());
                std::cout << "[Act1][Outputs]" << std::endl;
                std::cout << out_act1 << std::endl;
                activation1_cell.getOutputs().synchronizeHToD();

                fc1.getOutputs().synchronizeDToH();
                const Tensor<float>& out = tensor_cast<float>(fc1.getOutputs());
                for (unsigned int output = 0; output < out.dimZ(); ++output) {
                    std::cout << "[Fc][Outputs] = " << out(output, 0) << std::endl;
                }
                fc1.getOutputs().synchronizeHToD();
            }

            softmax1.mDiffInputs.synchronizeDToH();
            softmax1.getOutputs().synchronizeDToH();
            const CudaTensor<float>& out_softmax1 = cuda_tensor_cast<float>(softmax1.getOutputs());
            double loss = 0.0f;

            for(unsigned int nout = 0; nout < nbOutputs; ++nout){
                for (unsigned int batchPos = 0; batchPos < 1; ++batchPos){
                    if(iter_index==9999) std::cout << "out_softmax1(nout, batchPos) = " << out_softmax1(nout, batchPos) << std::endl;

                    if(nout==0) {
                        softmax1.mDiffInputs(nout, batchPos) = 1.0f;
                    }
                    if(nout==1) {
                        softmax1.mDiffInputs(nout, batchPos) = 0.0f;
                    }
                    if(iter_index==9999) std::cout << "softmax1.mDiffInputs(nout, batchPos) = " << softmax1.mDiffInputs(nout, batchPos) << std::endl;
                }
            }

            softmax1.mDiffInputs.synchronizeHToD();
            softmax1.getOutputs().synchronizeHToD();
            loss = softmax1.applyLoss();
            if(iter_index==9999) std::cout << "loss = " << loss << std::endl;

            //backpropagate 
            softmax1.backPropagate();   
            fc1.backPropagate();
            activation1_cell.backPropagate();

            fc1.update();
            activation1_cell.update(); 

            if(iter_index==9999){
                std::cout << "[Fc][FullPrecisionWeights after]" << std::endl;
                for (unsigned int output = 0; output < outputSize; ++output) {
                    for (unsigned int channel = 0; channel < inputSize; ++channel) {
                        Tensor<float> weight;
                        fc1.getWeight(output, channel, weight);
                        std::cout << "[" << output << "][" << channel << "]: \n" 
                        << weight << std::endl;
                    }
                }

                quant.getDiffFullPrecisionWeights(0).synchronizeDToH();
                CudaTensor<float> my_DiffFullPrecisionWeights = cuda_tensor_cast<float>(quant.getDiffFullPrecisionWeights(0));
                my_DiffFullPrecisionWeights.synchronizeDToH();
                quant.getDiffFullPrecisionWeights(0).synchronizeHToD();
                std::cout << "[Fc][DiffFullPrecisionWeights]\n" << my_DiffFullPrecisionWeights << std::endl;

                quant.getStepSize().synchronizeDToH();
                CudaTensor<float> stepSizeEst1 = quant.getStepSize();
                stepSizeEst1.synchronizeDToH();
                std::cout << "[Fc][StepSizeEst]\n " << stepSizeEst1 << std::endl;
                ASSERT_EQUALS_DELTA(stepSizeEst1(0), stepEstimated1_pytorch, 0.001);
                quant.getStepSize().synchronizeHToD();
            }

    }
}



RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
