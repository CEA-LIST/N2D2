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

#pragma omp parallel for if(activations.size()>16)
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

    std::pair<int, int> actRange = std::make_pair(0, (int)(mRange));

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<float> >();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        fpActivations(index) = Random::randUniform(-3.0, 3.0);
        diffInput(index) = Random::randUniform(-1.0, 1.0);
    }

    float gW = 1/std::sqrt(fpActivations.size() * actRange.second);

#pragma omp parallel for if(fpActivations.size()>16)
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        float dQ = fpActivations(i)/stepSize;
        diffOutput(i) = diffInput(i)*(dQ <= (float)actRange.first ? 0.0f : (dQ >= (float)actRange.second ? 0.0f : 1.0f));
    }

    float stepSizeGrad = 0.0f;
#pragma omp parallel for reduction(+:stepSizeGrad) if(fpActivations.size()>16)
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {    
        float qData = fpActivations(i)/stepSize;
        qData = qData <= (float) actRange.first ? (float) actRange.first
                        : (qData >= (float) actRange.second ? (float) actRange.second
                        : round(qData) - qData);
        stepSizeGrad += qData*diffInput(i);
    }
    stepSizeGrad *= gW;

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

    std::pair<int, int> actRange = std::make_pair(0, (int)(mRange));

#pragma omp parallel for if(activations.size() > 16)
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

    std::pair<int, int> actRange = std::make_pair(0, (int)(mRange));

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<double> >();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        fpActivations(index) = Random::randUniform(-3.0, 3.0);
        diffInput(index) = Random::randUniform(-1.0, 1.0);
    }

#pragma omp parallel for if(fpActivations.size() > 16)
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        double dQ = fpActivations(i)/stepSize;
        diffOutput(i) = diffInput(i)*(dQ <= (double)actRange.first ? double(0.0) : (dQ >= (double)actRange.second ? double(0.0) : double(1.0)));
    }

    double gW = double(1.0)/std::sqrt(fpActivations.size() * actRange.second);
    double stepSizeGrad = 0.0;

// #pragma omp parallel for reduction(+:stepSizeGrad) if(fpActivations.size())
//     for(unsigned int i = 0; i < fpActivations.size(); ++i)
//     {    
//         double qData = fpActivations(i)/stepSize;
//         qData = qData <= (double) actRange.first ? (double) actRange.first
//                         : (qData >= (double) actRange.second ? (double) actRange.second
//                         : round(qData) - qData);
//         stepSizeGrad += qData*diffInput(i);
//     }
//     stepSizeGrad *= gW;
unsigned int line = fpActivations.dimY()*fpActivations.dimZ();
for (unsigned int x = 0; x < fpActivations.dimX(); ++x) {
        double stepSizeGrad_loc = double(0.0);
#pragma omp parallel for schedule(static, 256) reduction(+:stepSizeGrad_loc)
        for(unsigned int i = 0; i < fpActivations.dimZ()*fpActivations.dimY()/4; ++i)
        {
            double qData_1 = fpActivations(line*x + 4*i)/(stepSize);
            qData_1 = (qData_1 <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qData_1 >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qData_1) - qData_1);
            double qData_2 = fpActivations(line*x + 4*i+1)/(stepSize);
            qData_2 = (qData_2 <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qData_2 >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qData_2) - qData_2);
            double qData_3 = fpActivations(line*x + 4*i+2)/(stepSize);
            qData_3 = (qData_3 <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qData_3 >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qData_3) - qData_3);
            double qData_4 = fpActivations(line*x + 4*i+3)/(stepSize);
            qData_4 = (qData_4 <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qData_4 >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qData_4) - qData_4);
            stepSizeGrad_loc += ((qData_1*diffInput(line*x + 4*i) + qData_2*diffInput(line*x + 4*i+1)) + (qData_3*diffInput(line*x + 4*i+2) + qData_4*diffInput(line*x + 4*i+3)));
        }
        for (unsigned int i=(line*(x+1) - line%4); i<line*(x+1); ++i) {
            double qDataDouble = fpActivations(i)/(stepSize);
            qDataDouble = (qDataDouble <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qDataDouble >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qDataDouble) - qDataDouble);
            stepSizeGrad_loc += (qDataDouble*diffInput(i));
        }
        stepSizeGrad += stepSizeGrad_loc;
    }
    // if inside the for loop, rounding approximations increase to much
    stepSizeGrad *= gW;

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

#pragma omp declare reduction(+ : half_float::half : omp_out = omp_in + omp_out) initializer(omp_priv=half_float::half(0.0))

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

#pragma omp parallel for if(activations.size() > 16)
    for(unsigned int i = 0; i < activations.size(); ++ i)
    {
        activationsQ(i) = (activations(i)/stepSize);
        activationsQ(i) = activationsQ(i) <= half_float::half_cast<half_float::half>(actRange.first) ? half_float::half_cast<half_float::half>(actRange.first) 
                        : activationsQ(i) >= half_float::half_cast<half_float::half>(actRange.second) ? half_float::half_cast<half_float::half>(actRange.second)
                        : activationsQ(i);
        activationsQ(i) = half_float::round(activationsQ(i)) * stepSize;
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

    /*-------------Half_float Precision Matrixes------------*/
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

    /*------------Init variables---------------*/
    std::pair<int, int> actRange = std::make_pair(0, (int)(mRange));

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<half_float::half> >();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        fpActivations(index) = half_float::half_cast<half_float::half>(Random::randUniform(-3.0, 3.0));
        diffInput(index) = half_float::half_cast<half_float::half>(Random::randUniform(-1.0, 1.0));
    }

/* 
 * The computation of the gradient scale factor may imply matrixes with more than 65504 features.
 * If the positive quantization level multiplied by the activation matrix size were to exceed this
 * value, it would be considered as infinit by the half_float standard.
 * 
 * Then gradScaleFactor = 1/sqrt(inf) would be considered equal to 0.
 * 
 * Thus the float/double precision type is used to compute the gradient scale factor before
 * converting it to half_float type.
 */
    half_float::half gW = half_float::half_cast<half_float::half>((1.0)/sqrt(fpActivations.size() * actRange.second));

    half_float::half stepSizeGrad = half_float::half(0.0);

    /*----------Backward Weights-----------*/
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        half_float::half dQ = fpActivations(i)/stepSize;
        diffOutput(i) = diffInput(i)*(dQ <= half_float::half_cast<half_float::half>(actRange.first) ? half_float::half(0.0) :
                                     (dQ >= half_float::half_cast<half_float::half>(actRange.second) ? half_float::half(0.0) :
                                     half_float::half(1.0)));
    }

    /*---------Backward Step Size-----------*/
    for (unsigned int i = (fpActivations.size()-fpActivations.size()%4); i < fpActivations.size(); ++i) {
        half_float::half qData = fpActivations(i)/stepSize;
        qData = (qData <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                (qData >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                (round(qData) - qData);
        stepSizeGrad += (qData*diffInput(i));
    }
#pragma omp parallel for schedule(static, 256) reduction(+:stepSizeGrad)
    for(unsigned int i = 0; i < fpActivations.size()/4; ++i)
    {
        half_float::half qData_1 = fpActivations(4*i)/stepSize;
        qData_1 = (qData_1 <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                (qData_1 >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                (round(qData_1) - qData_1);
        half_float::half qData_2 = fpActivations(4*i+1)/stepSize;
        qData_2 = (qData_2 <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                (qData_2 >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                (round(qData_2) - qData_2);
        half_float::half qData_3 = fpActivations(4*i+2)/stepSize;
        qData_3 = (qData_3 <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                (qData_3 >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                (round(qData_3) - qData_3);
        half_float::half qData_4 = fpActivations(4*i+3)/stepSize;
        qData_4 = (qData_4 <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                (qData_4 >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                (round(qData_4) - qData_4);
        stepSizeGrad += ((qData_1*diffInput(4*i)+qData_2*diffInput(4*i+1)) +(qData_3*diffInput(4*i+2)+qData_4*diffInput(4*i+3)));
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
// check how the computation diverges between double and half_float numbers
TEST_DATASET(   LSQQuantizerActivation_Frame_HalfFloat_DoubleComparition,
                activations_quantization_backpropagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t mRange, half_float::half stepSize),
                std::make_tuple(28U, 28U, 10U, 4U, 255, half_float::half(0.48131)),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.53485)),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.03313)),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.65432)),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.09827)),

                std::make_tuple(28U, 28U, 1U, 4U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 7, half_float::half(0.09014)),
                std::make_tuple(28U, 28U, 16U, 8U, 3, half_float::half(0.18681)),

                std::make_tuple(28U, 28U, 1U, 4U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2)),

                std::make_tuple(28U, 28U, 1U, 4U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 7, half_float::half(0.058754)),
                std::make_tuple(28U, 28U, 4U, 16U, 3, half_float::half(0.2345)),

                std::make_tuple(28U, 28U, 1U, 4U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2)),

                std::make_tuple(28U, 28U, 1U, 4U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 7, half_float::half(0.172127)),
                std::make_tuple(28U, 28U, 4U, 16U, 3, half_float::half(0.654)),

                std::make_tuple(28U, 28U, 1U, 4U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 255, half_float::half(0.2)),

                std::make_tuple(28U, 28U, 1U, 4U, 127, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 1U, 31, half_float::half(0.01)),
                std::make_tuple(28U, 28U, 4U, 4U, 7, half_float::half(0.01)),
                std::make_tuple(28U, 28U, 4U, 16U, 1023, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 16U, 8U, 2047, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 8191, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 16383, half_float::half(0.2))
            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*8);
    /*-------------Double Precision Matrices-------------*/
    Tensor<double> fpActivationsDouble(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});

    Tensor<double> diffInputDouble(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});

    Tensor<double> diffOutputDouble(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});
    Tensor<double> diffOutputPredDouble(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});
    
    Tensor<double> diffStepSizeDouble({1, 1, 1, 1});
    diffOutputDouble.fill(double(0.0));
    diffOutputPredDouble.fill(double(0.0));
    diffStepSizeDouble.fill(double(0.0));

    /*-------------Half_float Precision Matrices------------*/
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

    /*------------Init variables---------------*/
    std::pair<int, int> actRange = std::make_pair(0, (int)(mRange));

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<half_float::half> >();
    std::shared_ptr<Solver> quantizerSolverDouble = std::make_shared<SGDSolver_Frame<double>>();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        half_float::half a = half_float::half_cast<half_float::half>(Random::randUniform(-3.0, 3.0));
        fpActivations(index) = a;
        fpActivationsDouble(index) = half_float::half_cast<double>(a);
        half_float::half b = half_float::half_cast<half_float::half>(Random::randUniform(-1.0, 1.0));
        diffInput(index) = b;
        diffInputDouble(index) = half_float::half_cast<double>(b);
    }

/* 
 * The computation of the gradient scale factor may imply matrixes with more than 65504 features.
 * If the positive quantization level multiplied by the activation matrix size were to exceed this
 * value, it would be considered as infinit by the half_float standard for 1000 ~ 10000 features.
 * 
 * Then gradScaleFactor = 1/sqrt(inf) would be considered equal to 0.
 * 
 * Thus the float/double precision type is used to compute the gradient scale factor before
 * converting it to half_float type.
 */
    half_float::half gW = half_float::half_cast<half_float::half>((1.0)/sqrt(fpActivations.size() * actRange.second));
    double gWDouble = (1.0)/sqrt(fpActivations.size() * actRange.second);

    half_float::half stepSizeGrad = half_float::half(0.0);
    double stepSizeGradDouble = double(0.0);
    

    /*----------Backward Weights-----------*/
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        half_float::half dQ = fpActivations(i)/stepSize;
        diffOutput(i) = diffInput(i)*(dQ <= half_float::half_cast<half_float::half>(actRange.first) ? half_float::half(0.0) :
                                     (dQ >= half_float::half_cast<half_float::half>(actRange.second) ? half_float::half(0.0) :
                                     half_float::half(1.0)));
        
        double dQDouble = fpActivationsDouble(i)/half_float::half_cast<double>(stepSize);
        diffOutputDouble(i) = diffInputDouble(i)*(dQDouble <= (double)(actRange.first) ? (0.0) :
                                     (dQDouble >= (double)(actRange.second) ? (0.0) :
                                     (1.0)));
        if ((diffOutputDouble(i) > diffOutput(i)+0.001) || (diffOutputDouble(i)< diffOutput(i)-0.001)) {
            std::cout << "i= " << i << " -- mRange= " << mRange << " -- stepSize= " << stepSize << " -- dQ= " << dQ << " -- diffOuput(i)= " << 
            diffOutput(i) << " -- dQDouble= " << dQDouble << " -- diffOutputDouble(i)= " << diffOutputDouble(i) << 
            " -- fullPrecInput= " << fpActivations(i) << " -- fullPrecInputDouble= " << fpActivationsDouble(i) << std::endl;
        }
    }

    /*---------Backward Step Size-----------*/
    // unsigned int actSize = fpActivations.size();
    // for (unsigned int i=actSize-actSize%4; i<actSize; ++i) {

    // half_float::half qData = fpActivations(i)/stepSize;
    // qData = (qData <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
    //         (qData >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
    //         (round(qData) - qData);
    // stepSizeGrad += qData*diffInput(i);

    // double qDataDouble = fpActivationsDouble(i)/half_float::half_cast<double>(stepSize);
    // qDataDouble = (qDataDouble <= (double)(actRange.first)) ? (double)(actRange.first) :
    //         (qDataDouble >= (double)(actRange.second)) ? (double)(actRange.second) :
    //         (round(qDataDouble) - qDataDouble);
    // stepSizeGradDouble += (qDataDouble*diffInputDouble(i));
    // }

    unsigned int line = fpActivations.dimZ()*fpActivations.dimY();
    for (unsigned int x = 0; x < fpActivations.dimX(); ++x) {
        half_float::half stepSizeGrad_loc = half_float::half(0.0);
        double stepSizeGradDouble_loc = (double)0.0;
#pragma omp parallel for schedule(static, 256) reduction(+:stepSizeGrad_loc, stepSizeGradDouble_loc)
        for(unsigned int i = 0; i < fpActivations.dimZ()*fpActivations.dimY()/4; ++i)
        {
            half_float::half qData_1 = fpActivations(line*x + 4*i)/stepSize;
            qData_1 = (qData_1 <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                    (qData_1 >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                    (round(qData_1) - qData_1);
            half_float::half qData_2 = fpActivations(line*x + 4*i+1)/stepSize;
            qData_2 = (qData_2 <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                    (qData_2 >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                    (round(qData_2) - qData_2);
            half_float::half qData_3 = fpActivations(line*x + 4*i+2)/stepSize;
            qData_3 = (qData_3 <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                    (qData_3 >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                    (round(qData_3) - qData_3);
            half_float::half qData_4 = fpActivations(line*x + 4*i+3)/stepSize;
            qData_4 = (qData_4 <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                    (qData_4 >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                    (round(qData_4) - qData_4);
            stepSizeGrad_loc += ((qData_1*diffInput(line*x + 4*i) + qData_2*diffInput(line*x + 4*i+1))+(qData_3*diffInput(line*x + 4*i+2) + qData_4*diffInput(line*x + 4*i+3)));

            double qDataDouble_1 = fpActivationsDouble(line*x + 4*i)/half_float::half_cast<double>(stepSize);
            qDataDouble_1 = (qDataDouble_1 <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qDataDouble_1 >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qDataDouble_1) - qDataDouble_1);
            double qDataDouble_2 = fpActivationsDouble(line*x + 4*i+1)/half_float::half_cast<double>(stepSize);
            qDataDouble_2 = (qDataDouble_2 <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qDataDouble_2 >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qDataDouble_2) - qDataDouble_2);
            double qDataDouble_3 = fpActivationsDouble(line*x + 4*i+2)/half_float::half_cast<double>(stepSize);
            qDataDouble_3 = (qDataDouble_3 <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qDataDouble_3 >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qDataDouble_3) - qDataDouble_3);
            double qDataDouble_4 = fpActivationsDouble(line*x + 4*i+3)/half_float::half_cast<double>(stepSize);
            qDataDouble_4 = (qDataDouble_4 <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qDataDouble_4 >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qDataDouble_4) - qDataDouble_4);
            stepSizeGradDouble_loc += ((qDataDouble_1*diffInputDouble(line*x + 4*i) + qDataDouble_2*diffInputDouble(line*x + 4*i+1)) + (qDataDouble_3*diffInputDouble(line*x + 4*i+2) + qDataDouble_4*diffInputDouble(line*x + 4*i+3)));
        }
        for (unsigned int i=(line*(x + 1) - line%4); i<line*(x+1); ++i) {
            half_float::half qData = fpActivations(i)/stepSize;
            qData = (qData <= half_float::half_cast<half_float::half>(actRange.first)) ? half_float::half_cast<half_float::half>(actRange.first) :
                    (qData >= half_float::half_cast<half_float::half>(actRange.second)) ? half_float::half_cast<half_float::half>(actRange.second) :
                    (round(qData) - qData);
            stepSizeGrad_loc += qData*diffInput(i);

            double qDataDouble = fpActivationsDouble(i)/half_float::half_cast<double>(stepSize);
            qDataDouble = (qDataDouble <= (double)(actRange.first)) ? (double)(actRange.first) :
                    (qDataDouble >= (double)(actRange.second)) ? (double)(actRange.second) :
                    (round(qDataDouble) - qDataDouble);
            stepSizeGradDouble_loc += (qDataDouble*diffInputDouble(i));
        }
        stepSizeGrad += stepSizeGrad_loc;
        stepSizeGradDouble += stepSizeGradDouble_loc;
    }
    // if inside the for loop, rounding approximations increase to much
    stepSizeGrad *= gW;
    stepSizeGradDouble *= gWDouble;


    for(unsigned int i = 0; i < diffOutput.size(); ++i) {
        ASSERT_EQUALS_DELTA(diffOutput(i), diffOutputDouble(i), 0.001);
    }
    // if (stepSizeGrad - half_float::half_cast<half_float::half>(stepSizeGradDouble) >=0.01 || stepSizeGrad - half_float::half_cast<half_float::half>(stepSizeGradDouble) <=-0.01 )
    //     std::cout << "(" << outputH << ", " << outputW << ", " << nbOutputs << ", " << nbChannel << ", " << mRange << ", " << stepSize<< ")" << " -> stepSizeGrad= " << stepSizeGrad << std::endl;
    const double coeff = ((stepSizeGradDouble < 1)&&(stepSizeGradDouble > -1)) ? 0.05 :
                            (stepSizeGradDouble >=1) ? stepSizeGradDouble*0.05 : -stepSizeGradDouble*0.05;
    ASSERT_EQUALS_DELTA(stepSizeGrad, stepSizeGradDouble, coeff);
}

RUN_TESTS()
// On average over 38 iterations, step size gradient computed with half_float precision
// diverges from double precision one by 0.017 when computed with parallel programming
// paradigm and by 0.025 with sequential programming paradigm

// Half_float step size gradient computation for different input sizes
// 1
// static -- : Parallel : -2.58398 | Sequential : -2.58398 | Truth : -2.58609
// 10
// static -- : Parallel : 0.555664 | Sequential : 0.556152 | Truth : 0.55669
// 100
// static -- : Parallel : 1.3125 | Sequential : 1.30957 | Truth : 1.31843
// 1 000
// static -- : Parallel : -0.338623 | Sequential : -0.329102 | Truth : -0.34209
// 10 000
// static -- : Parallel : -0.150513 | Sequential : -0.260742 | Truth : -0.156353
// 100 000
// static :   Parallel : -0.199951 | Sequential : 0.240479 | Truth : -0.0398774
// 1 000 000
// static -- : Parallel : -0.0986938 | Sequential : -0.0389709 | Truth : -0.883042