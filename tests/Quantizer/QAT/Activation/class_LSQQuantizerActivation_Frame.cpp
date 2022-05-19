/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Inna KUCHER (inna.kucher@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include "N2D2.hpp"

#include "Quantizer/QAT/Activation/LSQ/LSQQuantizerActivation_Frame.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;


/************ Float ***************/
TEST_DATASET(   LSQQuantizerActivation_Frame_Float,
                activations_quantization_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t mRange, float stepSize, bool setOptInitStepSize),
                std::make_tuple(28U, 28U, 10U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 127, 0.05, false),
                std::make_tuple(28U, 28U, 4U, 1U, 31, 0.01, false),
                std::make_tuple(28U, 28U, 4U, 4U, 7, 0.01, false),
                std::make_tuple(28U, 28U, 4U, 16U, 1023, 0.05, false),
                std::make_tuple(28U, 28U, 16U, 8U, 2047, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 8191, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 16383, 0.2, false)
            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*8);
// initialize tensors
    Tensor<float> activations(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
    Tensor<float> activationsQ(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
// fill them with random values
    for (unsigned int index = 0; index < activations.size(); ++index) {
        activations(index) = Random::randUniform(-3.0, 3.0);
    }

    /*
    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (std::pow(2, (int) mRange) - 1) );
    */

    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (mRange) );

    for(unsigned int i = 0; i < activations.size(); ++ i)
    {
        activationsQ(i) = (activations(i)/stepSize);
        activationsQ(i) = activationsQ(i) <= actRange.first ? actRange.first 
                        : activationsQ(i) >= actRange.second ? actRange.second : activationsQ(i);
        activationsQ(i) = round(activationsQ(i)) * stepSize;
    }

    //mRange for LSQ = 2^{b-1}
    //int range = std::pow(2, (int) mRange - 1);

    LSQQuantizerActivation_Frame<float> quant;
    //quant.setBitPrecision(mRange);
    quant.setStepSizeValue(stepSize);
    quant.setOptInitStepSize(setOptInitStepSize);
    quant.setRange(mRange);
    //In-place operation
    quant.propagate(activations, false);
    for(unsigned int i = 0; i < activations.size(); ++i)
        ASSERT_EQUALS_DELTA(activationsQ(i), activations(i), 0.001);
}

TEST_DATASET(   LSQQuantizerActivation_Frame_Float,
                activations_quantization_backpropagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t mRange, float stepSize, bool setOptInitStepSize),
                std::make_tuple(28U, 28U, 10U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 127, 0.05, false),
                std::make_tuple(28U, 28U, 4U, 1U, 31, 0.01, false),
                std::make_tuple(28U, 28U, 4U, 4U, 7, 0.01, false),
                std::make_tuple(28U, 28U, 4U, 16U, 1023, 0.05, false),
                std::make_tuple(28U, 28U, 16U, 8U, 2047, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 8191, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 16383, 0.2, false)

            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*8);

    Tensor<float> fpActivations(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});

    Tensor<float> diffInput(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});

    Tensor<float> diffOutput(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});
    Tensor<float> diffOutputPred(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});

    Tensor<float> diffStepSize({1, 1, 1, 1});
    diffOutput.fill(0.0);
    diffOutputPred.fill(0.0);
    diffStepSize.fill(0.0);

    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (mRange) );

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<float> >();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        fpActivations(index) = Random::randUniform(-3.0, 3.0);
        diffInput(index) = Random::randUniform(-1.0, 1.0);
    }

    float gW = 1/std::sqrt(fpActivations.size() * actRange.second);

    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        float dQ = fpActivations(i)/stepSize;
        diffOutput(i) = diffInput(i)*(dQ <= (float)actRange.first ? 0.0f : (dQ >= (float)actRange.second ? 0.0f : 1.0f));
    }

    float stepSizeGrad = 0.0;
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {    
        float qData = fpActivations(i)/stepSize;
        qData = qData < (float) actRange.first ? (float) actRange.first
                        : (qData > (float) actRange.second ? (float) actRange.second
                        : rint(qData) - qData);
        qData *= diffInput(i);
        qData *= gW;
        stepSizeGrad += qData;
    }

    LSQQuantizerActivation_Frame<float> quant;
    //quant.setBitPrecision(mRange);
    quant.setStepSizeValue(stepSize);
    quant.setOptInitStepSize(setOptInitStepSize);
    quant.setRange(mRange);
    quant.setSolver(quantizerSolver);

    // LSQQuantizerActivation_Frame<T>::back_propagate( const BaseTensor& baseInput,
    //                                                            const BaseTensor& baseOutput,
    //                                                            const BaseTensor& baseDiffInput,
    //                                                            BaseTensor& baseDiffOutput)
    //==> baseOutput not use for the LSQ back_propagate 
    //
    quant.back_propagate(fpActivations, fpActivations, diffInput, diffOutputPred);

    for(unsigned int i = 0; i < diffOutput.size(); ++i) {
        ASSERT_EQUALS_DELTA(diffOutputPred(i), diffOutput(i), 0.001);
    }

    const Tensor<float>& diffStepSizePred = quant.getDiffStepSize();
        ASSERT_EQUALS_DELTA(diffStepSizePred(0,0,0,0), stepSizeGrad, 0.001);
}


/****************Double*****************/

TEST_DATASET(   LSQQuantizerActivation_Frame_Double,
                activations_quantization_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t mRange, double stepSize, bool setOptInitStepSize),
                std::make_tuple(28U, 28U, 10U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 127, 0.05, false),
                std::make_tuple(28U, 28U, 4U, 1U, 31, 0.01, false),
                std::make_tuple(28U, 28U, 4U, 4U, 7, 0.01, false),
                std::make_tuple(28U, 28U, 4U, 16U, 1023, 0.05, false),
                std::make_tuple(28U, 28U, 16U, 8U, 2047, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 8191, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 16383, 0.2, false)
            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*8);
// initialize tensors
    Tensor<double> activations(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
    Tensor<double> activationsQ(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
// fill them with random values
    for (unsigned int index = 0; index < activations.size(); ++index) {
        activations(index) = Random::randUniform(-3.0, 3.0);
    }

    /*
    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (std::pow(2, (int) mRange) - 1) );
    */

    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (mRange) );

    for(unsigned int i = 0; i < activations.size(); ++ i)
    {
        activationsQ(i) = (activations(i)/stepSize);
        activationsQ(i) = activationsQ(i) <= (double) actRange.first ? (double) actRange.first 
                        : activationsQ(i) >= (double) actRange.second ? (double) actRange.second : activationsQ(i);
        activationsQ(i) = round(activationsQ(i)) * stepSize;
    }

    //mRange for LSQ = 2^{b-1}
    //int range = std::pow(2, (int) mRange - 1);

    LSQQuantizerActivation_Frame<double> quant;
    //quant.setBitPrecision(mRange);
    quant.setStepSizeValue(stepSize);
    quant.setOptInitStepSize(setOptInitStepSize);
    quant.setRange(mRange);
    //In-place operation
    quant.propagate(activations, false);
    for(unsigned int i = 0; i < activations.size(); ++i)
        ASSERT_EQUALS_DELTA(activationsQ(i), activations(i), 0.001);
}

TEST_DATASET(   LSQQuantizerActivation_Frame_Double,
                activations_quantization_backpropagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t mRange, double stepSize, bool setOptInitStepSize),
                std::make_tuple(28U, 28U, 10U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, 0.2, false),

                std::make_tuple(28U, 28U, 1U, 4U, 127, 0.05, false),
                std::make_tuple(28U, 28U, 4U, 1U, 31, 0.01, false),
                std::make_tuple(28U, 28U, 4U, 4U, 7, 0.01, false),
                std::make_tuple(28U, 28U, 4U, 16U, 1023, 0.05, false),
                std::make_tuple(28U, 28U, 16U, 8U, 2047, 0.2, false),
                std::make_tuple(28U, 28U, 4U, 16U, 8191, 0.2, false),
                std::make_tuple(28U, 28U, 16U, 8U, 16383, 0.2, false)

            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*8);

    Tensor<double> fpActivations(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});

    Tensor<double> diffInput(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});

    Tensor<double> diffOutput(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});
    Tensor<double> diffOutputPred(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});

    Tensor<double> diffStepSize({1, 1, 1, 1});
    diffOutput.fill(double(0.0));
    diffOutputPred.fill(double(0.0));
    diffStepSize.fill(double(0.0));

    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (mRange) );

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<double> >();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        fpActivations(index) = Random::randUniform(-3.0, 3.0);
        diffInput(index) = Random::randUniform(-1.0, 1.0);
    }

    double gW = double(1.0)/std::sqrt(fpActivations.size() * actRange.second);

    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        double dQ = fpActivations(i)/stepSize;
        diffOutput(i) = diffInput(i)*(dQ <= (double)actRange.first ? double(0.0) : (dQ >= (double)actRange.second ? double(0.0) : double(1.0)));
    }

    double stepSizeGrad = 0.0;
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {    
        double qData = fpActivations(i)/stepSize;
        qData = qData < (double) actRange.first ? (double) actRange.first
                        : (qData > (double) actRange.second ? (double) actRange.second
                        : rint(qData) - qData);
        qData *= diffInput(i);
        qData *= gW;
        stepSizeGrad += qData;
    }

    LSQQuantizerActivation_Frame<double> quant;
    //quant.setBitPrecision(mRange);
    quant.setStepSizeValue(stepSize);
    quant.setOptInitStepSize(setOptInitStepSize);
    quant.setRange(mRange);
    quant.setSolver(quantizerSolver);

    // LSQQuantizerActivation_Frame<T>::back_propagate( const BaseTensor& baseInput,
    //                                                            const BaseTensor& baseOutput,
    //                                                            const BaseTensor& baseDiffInput,
    //                                                            BaseTensor& baseDiffOutput)
    //==> baseOutput not use for the LSQ back_propagate 
    //
    quant.back_propagate(fpActivations, fpActivations, diffInput, diffOutputPred);

    for(unsigned int i = 0; i < diffOutput.size(); ++i) {
        ASSERT_EQUALS_DELTA(diffOutputPred(i), diffOutput(i), 0.001);
    }

    const Tensor<double>& diffStepSizePred = quant.getDiffStepSize();
        ASSERT_EQUALS_DELTA(diffStepSizePred(0,0,0,0), stepSizeGrad, 0.001);
}

/****************Half Float*****************/

TEST_DATASET(   LSQQuantizerActivation_Frame_HalfFloat,
                activations_quantization_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t mRange, half_float::half stepSize, bool setOptInitStepSize),
                std::make_tuple(28U, 28U, 10U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 3, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 127, half_float::half(0.05), false),
                std::make_tuple(28U, 28U, 4U, 1U, 31, half_float::half(0.01), false),
                std::make_tuple(28U, 28U, 4U, 4U, 7, half_float::half(0.01), false),
                std::make_tuple(28U, 28U, 4U, 16U, 1023, half_float::half(0.05), false),
                std::make_tuple(28U, 28U, 16U, 8U, 2047, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 8191, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 16383, half_float::half(0.2), false)
            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*8);
// initialize tensors
    Tensor<half_float::half> activations(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
    Tensor<half_float::half> activationsQ(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
// fill them with random values
    for (unsigned int index = 0; index < activations.size(); ++index) {
        activations(index) = half_float::half_cast<half_float::half>(Random::randUniform(-3.0, 3.0));
    }

    /*
    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (std::pow(2, (int) mRange) - 1) );
    */

    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (mRange) );

    for(unsigned int i = 0; i < activations.size(); ++ i)
    {
        activationsQ(i) = (activations(i)/stepSize);
        activationsQ(i) = activationsQ(i) <= half_float::half_cast<half_float::half>(actRange.first) ? half_float::half_cast<half_float::half>(actRange.first) 
                        : activationsQ(i) >= half_float::half_cast<half_float::half>(actRange.second) ? half_float::half_cast<half_float::half>(actRange.second)
                        : activationsQ(i);
        activationsQ(i) = half_float::rint(activationsQ(i)) * stepSize;
    }

    //mRange for LSQ = 2^{b-1}
    //int range = std::pow(2, (int) mRange - 1);

    LSQQuantizerActivation_Frame<half_float::half> quant;
    //quant.setBitPrecision(mRange);
    quant.setStepSizeValue(half_float::half_cast<float>(stepSize));
    quant.setOptInitStepSize(setOptInitStepSize);
    quant.setRange(mRange);
    //In-place operation
    quant.propagate(activations, false);
    for(unsigned int i = 0; i < activations.size(); ++i)
        ASSERT_EQUALS_DELTA(activationsQ(i), activations(i), 0.001);
}

TEST_DATASET(   LSQQuantizerActivation_Frame_HalfFloat,
                activations_quantization_backpropagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t mRange, half_float::half stepSize, bool setOptInitStepSize),
                std::make_tuple(28U, 28U, 10U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 3, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 63, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 15, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 7, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 3, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2), false),

                std::make_tuple(28U, 28U, 1U, 4U, 127, half_float::half(0.05), false),
                std::make_tuple(28U, 28U, 4U, 1U, 31, half_float::half(0.01), false),
                std::make_tuple(28U, 28U, 4U, 4U, 7, half_float::half(0.01), false),
                std::make_tuple(28U, 28U, 4U, 16U, 1023, half_float::half(0.05), false),
                std::make_tuple(28U, 28U, 16U, 8U, 2047, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 4U, 16U, 8191, half_float::half(0.2), false),
                std::make_tuple(28U, 28U, 16U, 8U, 16383, half_float::half(0.2), false)

            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*8);

    Tensor<half_float::half> fpActivations(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});

    Tensor<half_float::half> diffInput(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});

    Tensor<half_float::half> diffOutput(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});
    Tensor<half_float::half> diffOutputPred(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});

    Tensor<half_float::half> diffStepSize({1, 1, 1, 1});
    diffOutput.fill(half_float::half(0.0));
    diffOutputPred.fill(half_float::half(0.0));
    diffStepSize.fill(half_float::half(0.0));

    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (mRange) );

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<half_float::half> >();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        fpActivations(index) = half_float::half_cast<half_float::half>(Random::randUniform(-3.0, 3.0));
        diffInput(index) = half_float::half_cast<half_float::half>(Random::randUniform(-1.0, 1.0));
    }

    half_float::half gW = half_float::half(1.0)/half_float::sqrt(half_float::half_cast<half_float::half>(fpActivations.size() * actRange.second));

    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        half_float::half dQ = fpActivations(i)/stepSize;
        diffOutput(i) = diffInput(i)*(dQ <= half_float::half_cast<half_float::half>(actRange.first) ? half_float::half(0.0) :
                                     (dQ >= half_float::half_cast<half_float::half>(actRange.second) ? half_float::half(0.0) :
                                     half_float::half(1.0)));
    }

    half_float::half stepSizeGrad = half_float::half(0.0);
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {    
        half_float::half qData = fpActivations(i)/stepSize;
        qData = (qData < half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                (qData > half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                (rint(qData) - qData);
        qData *= diffInput(i);
        stepSizeGrad += qData;
    }
    // if inside the for loop, rounding approximations increase to much
    stepSizeGrad *= gW;

    LSQQuantizerActivation_Frame<half_float::half> quant;
    //quant.setBitPrecision(mRange);
    quant.setStepSizeValue(half_float::half_cast<float>(stepSize));
    quant.setOptInitStepSize(setOptInitStepSize);
    quant.setRange(mRange);
    quant.setSolver(quantizerSolver);

    // LSQQuantizerActivation_Frame<T>::back_propagate( const BaseTensor& baseInput,
    //                                                            const BaseTensor& baseOutput,
    //                                                            const BaseTensor& baseDiffInput,
    //                                                            BaseTensor& baseDiffOutput)
    //==> baseOutput not use for the LSQ back_propagate 
    //
    quant.back_propagate(fpActivations, fpActivations, diffInput, diffOutputPred);

    for(unsigned int i = 0; i < diffOutput.size(); ++i) {
        ASSERT_EQUALS_DELTA(diffOutputPred(i), diffOutput(i), 0.001);
    }

    const Tensor<half_float::half>& diffStepSizePred = quant.getDiffStepSize();
        ASSERT_EQUALS_DELTA(diffStepSizePred(0,0,0,0), stepSizeGrad, 0.001);
}

RUN_TESTS()