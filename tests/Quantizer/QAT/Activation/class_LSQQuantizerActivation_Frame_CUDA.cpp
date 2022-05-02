/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Inna KUCHER (inna.kucher@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/
#ifdef CUDA

#include "N2D2.hpp"

#include "Quantizer/QAT/Activation/LSQ/LSQQuantizerActivation_Frame_CUDA.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(   LSQQuantizerActivation_Frame_CUDA_Float,
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

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*8);

    CudaTensor<float> activations(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
    Tensor<float> activationsQ(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});

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

    LSQQuantizerActivation_Frame_CUDA<float> quant;
    //quant.setBitPrecision(mRange);
    quant.setStepSizeValue(stepSize);
    quant.setOptInitStepSize(setOptInitStepSize);
    quant.setRange(mRange);
    //In-place operation
    activations.synchronizeHToD();
    quant.propagate(activations, false);
    activations.synchronizeDToH();
    for(unsigned int i = 0; i < activations.size(); ++i)
        ASSERT_EQUALS_DELTA(activationsQ(i), activations(i), 0.001);
}

TEST_DATASET(   LSQQuantizerActivation_Frame_CUDA_Float,
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

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*8);

    CudaTensor<float> fpActivations(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});

    CudaTensor<float> diffInput(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});

    Tensor<float> diffOutput(  { outputW, 
                                outputH, 
                                nbOutputs,
                                1});
    CudaTensor<float> diffOutputPred(  { outputW, 
                                    outputH, 
                                    nbOutputs,
                                    1});

    CudaTensor<float> diffStepSize({1, 1, 1, 1});
    diffOutput.fill(0.0);
    diffOutputPred.fill(0.0);
    diffStepSize.fill(0.0);

    std::pair<int, int> actRange = std::make_pair( 0, 
                                        (int) (mRange) );

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame_CUDA<float> >();
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

    LSQQuantizerActivation_Frame_CUDA<float> quant;
    //quant.setBitPrecision(mRange);
    quant.setStepSizeValue(stepSize);
    quant.setOptInitStepSize(setOptInitStepSize);
    quant.setRange(mRange);
    quant.setSolver(quantizerSolver);

    // LSQQuantizerActivation_Frame_CUDA<T>::back_propagate( const BaseTensor& baseInput,
    //                                                            const BaseTensor& baseOutput,
    //                                                            const BaseTensor& baseDiffInput,
    //                                                            BaseTensor& baseDiffOutput)
    //==> baseOutput not use for the LSQ back_propagate 
    //
    fpActivations.synchronizeHToD();
    diffInput.synchronizeHToD();
    diffOutputPred.synchronizeHToD();
    quant.back_propagate(fpActivations, fpActivations, diffInput, diffOutputPred);
    fpActivations.synchronizeDToH();
    diffInput.synchronizeDToH();
    diffOutputPred.synchronizeDToH();

    for(unsigned int i = 0; i < diffOutput.size(); ++i) {
        ASSERT_EQUALS_DELTA(diffOutputPred(i), diffOutput(i), 0.001);
    }

    const CudaTensor<float>& diffStepSizePred = quant.getDiffStepSize();
    diffStepSizePred.synchronizeDToH();
    ASSERT_EQUALS_DELTA(diffStepSizePred(0,0,0,0), stepSizeGrad, 0.001);
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif