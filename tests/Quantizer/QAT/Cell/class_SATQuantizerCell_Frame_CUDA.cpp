/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifdef CUDA

#include "N2D2.hpp"

#include "Quantizer/QAT/Cell/SAT/SATQuantizerCell_Frame_CUDA.hpp"
#include "Solver/SGDSolver.hpp"
#include "Solver/Solver.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"
#include "third_party/half.hpp"

using namespace N2D2;

TEST_DATASET(   SATQuantizerCell_Frame_CUDA_Float,
                weights_quantization_scaling_back_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t nbBits), 
                std::make_tuple(28U, 28U, 10U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 2),
                //std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 1),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 7),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 5),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 10),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 11),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 13),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 14)
                

            )
{

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*nbBits);

    // NOTE: tensor = diff quant tensor in test
    CudaTensor<float> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<float> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<float> weightsDiff({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });

    CudaTensor<float> biases({nbOutputs});
    CudaTensor<float> biasesQ({nbOutputs});
    CudaTensor<float> biasesDiff({nbOutputs});


    float max_W_abs = 0.0f;
    float range = std::pow(2,nbBits);

    for (unsigned int index = 0; index < weights.size(); ++index) {
        weights(index) = Random::randUniform(-1.0, 1.0);
        weightsQ(index) = std::tanh(weights(index));
        float abs_w = std::abs(weightsQ(index));
        if(abs_w > max_W_abs)
           max_W_abs = abs_w;
    }

    //cudaSDorefaQ_kernel
    /* 
        float q = 0.5f*( (x[i] / tanhMax) + 1.0f);
        q = (1./a)*rintf(q*a);
        y[i] = q*2.0f - 1.0f;
    */
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = 2.0f*((1.0f/range)*rint(0.5f*((weightsQ(i)/max_W_abs)+1.0f)*range))-1.0f;
    }

    float wSum = std::accumulate(weightsQ.begin(), weightsQ.end(), 0.0);
    float wMean = wSum / weightsQ.size();

    float wVariance = 0.0f;

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
    {
        wVariance += (weightsQ(i) - wMean)*(weightsQ(i) - wMean);
    }
    wVariance /= (weightsQ.size() - 1.0);

    float wSATscaling = std::sqrt(wVariance*weightsQ.dimB()*weightsQ.dimY()*weightsQ.dimX());
    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        weightsQ(i) /= wSATscaling;

    //cudaSSATGrad_kernel
    /*
        float revCoshData = 1/std::cosh(x[i]);
        float gradSAT = revCoshData*revCoshData * 1/(factors);
        output[i] = diff[i]*gradSAT;
    */
    for(unsigned int i = 0; i < weightsDiff.size(); ++ i){
        weightsDiff(i) = 
            weights(i)*((1/std::cosh(weights(i)))*(1/std::cosh(weights(i)))
                /(max_W_abs*wSATscaling));       
    }

    for (unsigned int index = 0; index < biases.size(); ++index) {
        biases(index) = Random::randUniform(-1.0, 1.0);
    }
    // For the moment, bias quantization is just identity
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesQ(index) = biases(index);
    }
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesDiff(index) = biases(index);
    }

    weights.synchronizeHToD();
    biases.synchronizeHToD();

    SATQuantizerCell_Frame_CUDA<float> quant;
    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.addBiases(biases, biases);
    quant.setRange(range);
    quant.setScaling(true);
    quant.setQuantization(true);
    quant.initialize();
    quant.propagate();

    CudaTensor<float> weightsEstimated = cuda_tensor_cast<float>(quant.getQuantizedWeights(0));
    weightsEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    CudaTensor<float> biasEstimated = cuda_tensor_cast<float>(quant.getQuantizedBiases());
    biasEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesQ.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesQ(i), biasEstimated(i), 0.001);

    quant.getQuantizedBiases().synchronizeDToH();

    quant.back_propagate();

    CudaTensor<float> weightsDiffEstimated = cuda_tensor_cast<float>(quant.getDiffFullPrecisionWeights(0));
    weightsDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);
    
    CudaTensor<float> biasDiffEstimated = cuda_tensor_cast<float>(quant.getDiffFullPrecisionBiases());
    biasDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesDiff(i), biasEstimated(i), 0.001);

}

TEST_DATASET(   SATQuantizerCell_Frame_CUDA_Float,
                weights_clamping_propagate_back_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t nbBits), 
                std::make_tuple(28U, 28U, 10U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 2),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 7),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 5),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 10),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 11),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 13),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 14)

            )
{

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*nbBits);

    // NOTE: tensor = diff quant tensor in test
    CudaTensor<float> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<float> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<float> weightsDiff({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });

    CudaTensor<float> biases({nbOutputs});
    CudaTensor<float> biasesQ({nbOutputs});
    CudaTensor<float> biasesDiff({nbOutputs});

    float max_W_abs = 0.0f;
    float range = std::pow(2,nbBits);

    for (unsigned int index = 0; index < weights.size(); ++index) {
        weights(index) = Random::randUniform(-1.0, 1.0);
        weightsQ(index) = std::tanh(weights(index));

        float abs_w = std::abs(weightsQ(index));

        if(abs_w > max_W_abs)
           max_W_abs = abs_w;
    }
    //cudaSDorefaQ_kernel
    /* 
        float q = 0.5f*( (x[i] / tanhMax) + 1.0f);
        y[i] = q*2.0f - 1.0f;
    */
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = 2.0f*(0.5f * ((weightsQ(i)/max_W_abs) + 1.0f )) - 1.0f;
    }

    float wSum = std::accumulate(weightsQ.begin(), weightsQ.end(), 0.0);
    float wMean = wSum / weightsQ.size();
    float wVariance = 0.0f;

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
    {
        wVariance += (weightsQ(i) - wMean)*(weightsQ(i) - wMean);
    }

    wVariance /= (weightsQ.size() - 1.0);

    float wSATscaling = std::sqrt(wVariance*weightsQ.dimB()*weightsQ.dimX()*weightsQ.dimY());
    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        weightsQ(i) /= wSATscaling;

    //cudaSSATGrad_kernel
    /*
        float revCoshData = 1/std::cosh(x[i]);
        float gradSAT = revCoshData*revCoshData * 1/(factors);
        output[i] = diff[i]*gradSAT;
    */
    for(unsigned int i = 0; i < weightsDiff.size(); ++ i){

        weightsDiff(i) = 
            weights(i)*((1/std::cosh(weights(i)))*(1/std::cosh(weights(i)))
                /(max_W_abs*wSATscaling));     
    }

    for (unsigned int index = 0; index < biases.size(); ++index) {
        biases(index) = Random::randUniform(-1.0, 1.0);
    }
    // For the moment, bias quantization is just identity
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesQ(index) = biases(index);
    }
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesDiff(index) = biases(index);
    }

    weights.synchronizeHToD();
    biases.synchronizeHToD();

    SATQuantizerCell_Frame_CUDA<float> quant;
    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.addBiases(biases, biases);
    quant.setRange(range);
    quant.setScaling(true);
    quant.setQuantization(false);
    quant.initialize();
    quant.propagate();

    CudaTensor<float> weightsEstimated = cuda_tensor_cast<float>(quant.getQuantizedWeights(0));
    weightsEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    CudaTensor<float> biasEstimated = cuda_tensor_cast<float>(quant.getQuantizedBiases());
    biasEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesQ.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesQ(i), biasEstimated(i), 0.001);

    quant.getQuantizedBiases().synchronizeDToH();

    quant.back_propagate();

    CudaTensor<float> weightsDiffEstimated = cuda_tensor_cast<float>(quant.getDiffFullPrecisionWeights(0));
    weightsDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);
    
    CudaTensor<float> biasDiffEstimated = cuda_tensor_cast<float>(quant.getDiffFullPrecisionBiases());
    biasDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesDiff(i), biasEstimated(i), 0.001);
}

TEST_DATASET(   SATQuantizerCell_Frame_CUDA_Float,
                weights_clamping_no_scaling_propagate_back,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t nbBits), 
                std::make_tuple(28U, 28U, 10U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 2),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 7),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 5),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 10),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 11),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 13),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 14)

            )
{

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*nbBits);

    // NOTE: tensor = diff quant tensor in test
    CudaTensor<float> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<float> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<float> weightsDiff({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });

    CudaTensor<float> biases({nbOutputs});
    CudaTensor<float> biasesQ({nbOutputs});
    CudaTensor<float> biasesDiff({nbOutputs});

    float max_W_abs = 0.0f;
    float range = std::pow(2,nbBits);

    for (unsigned int index = 0; index < weights.size(); ++index) {
        weights(index) = Random::randUniform(-1.0, 1.0);
        weightsQ(index) = std::tanh(weights(index));

        float abs_w = std::abs(weightsQ(index));

        if(abs_w > max_W_abs)
           max_W_abs = abs_w;
    }

    //cudaSDorefaQ_kernel
    /* 
        float q = 0.5f*( (x[i] / tanhMax) + 1.0f);
        y[i] = q*2.0f - 1.0f;
    */
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = 2.0f*(0.5f * ( (weightsQ(i)/max_W_abs) + 1.0f ) ) - 1.0f;
    }

    //cudaSSATGrad_kernel
    /*
        float revCoshData = 1/std::cosh(x[i]);
        float gradSAT = revCoshData*revCoshData * 1/(factors);
        output[i] = diff[i]*gradSAT;
    */
    for(unsigned int i = 0; i < weightsDiff.size(); ++ i){

        weightsDiff(i) = 
            weights(i)*((1/std::cosh(weights(i)))*(1/std::cosh(weights(i)))
                /(max_W_abs));     
    }


    for (unsigned int index = 0; index < biases.size(); ++index) {
        biases(index) = Random::randUniform(-1.0, 1.0);
    }
    // For the moment, bias quantization is just identity
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesQ(index) = biases(index);
    }
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesDiff(index) = biases(index);
    }


    weights.synchronizeHToD();
    biases.synchronizeHToD();

    SATQuantizerCell_Frame_CUDA<float> quant;
    quant.setQuantization(false);
    quant.setScaling(false);

    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.addBiases(biases, biases);
    quant.setRange(range);
    quant.initialize();

    quant.propagate();

    CudaTensor<float> weightsEstimated = cuda_tensor_cast<float>(quant.getQuantizedWeights(0));
    weightsEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    CudaTensor<float> biasEstimated = cuda_tensor_cast<float>(quant.getQuantizedBiases());
    biasEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesQ.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesQ(i), biasEstimated(i), 0.001);

    quant.getQuantizedBiases().synchronizeDToH();

    quant.back_propagate();

    CudaTensor<float> weightsDiffEstimated = cuda_tensor_cast<float>(quant.getDiffFullPrecisionWeights(0));
    weightsDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);
    
    CudaTensor<float> biasDiffEstimated = cuda_tensor_cast<float>(quant.getDiffFullPrecisionBiases());
    biasDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesDiff(i), biasEstimated(i), 0.001);
}

///double


TEST_DATASET(   SATQuantizerCell_Frame_CUDA_Double,
                weights_quantization_scaling_back_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t nbBits), 
                std::make_tuple(28U, 28U, 10U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 2),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 7),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 5),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 10),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 11),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 13),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 14)

            )
{

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*nbBits);

    // NOTE: tensor = diff quant tensor in test
    CudaTensor<double> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<double> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<double> weightsDiff({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });

    CudaTensor<double> biases({nbOutputs});
    CudaTensor<double> biasesQ({nbOutputs});
    CudaTensor<double> biasesDiff({nbOutputs});

    double max_W_abs = 0.0;
    float range = std::pow(2,nbBits);

    for (unsigned int index = 0; index < weights.size(); ++index) {
        weights(index) = Random::randUniform(-1.0, 1.0);
        weightsQ(index) = std::tanh(weights(index));
        double abs_w = std::abs(weightsQ(index));
        if(abs_w > max_W_abs)
           max_W_abs = abs_w;
    }

    //cudaSDorefaQ_kernel
    /* 
        float q = 0.5f*( (x[i] / tanhMax) + 1.0f);
        q = (1./a)*rintf(q*a);
        y[i] = q*2.0f - 1.0f;
    */
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = 2.0f*((1.0f/range)*rint(0.5f*((weightsQ(i)/max_W_abs)+1.0f)*range))-1.0f;
    }

    double wSum = std::accumulate(weightsQ.begin(), weightsQ.end(), 0.0);
    double wMean = wSum / weightsQ.size();
    double wVariance = 0.0;

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
    {
        wVariance += (weightsQ(i) - wMean)*(weightsQ(i) - wMean);
    }

    wVariance /= (weightsQ.size() - 1.0);

    double wSATscaling = std::sqrt(wVariance*weightsQ.dimB()*weightsQ.dimY()*weightsQ.dimX());
    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        weightsQ(i) /= wSATscaling;

    //cudaSSATGrad_kernel
    /*
        float revCoshData = 1/std::cosh(x[i]);
        float gradSAT = revCoshData*revCoshData * 1/(factors);
        output[i] = diff[i]*gradSAT;
    */
    for(unsigned int i = 0; i < weightsDiff.size(); ++ i){
        weightsDiff(i) = 
            weights(i)*((1/std::cosh(weights(i)))*(1/std::cosh(weights(i)))
                /(max_W_abs*wSATscaling));       
    }

    for (unsigned int index = 0; index < biases.size(); ++index) {
        biases(index) = Random::randUniform(-1.0, 1.0);
    }
    // For the moment, bias quantization is just identity
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesQ(index) = biases(index);
    }
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesDiff(index) = biases(index);
    }

    weights.synchronizeHToD();
    biases.synchronizeHToD();

    SATQuantizerCell_Frame_CUDA<double> quant;
    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.addBiases(biases, biases);
    quant.setRange(range);
    quant.setScaling(true);
    quant.setQuantization(true);
    quant.initialize();
    quant.propagate();

    CudaTensor<double> weightsEstimated = cuda_tensor_cast<double>(quant.getQuantizedWeights(0));
    weightsEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    CudaTensor<double> biasEstimated = cuda_tensor_cast<double>(quant.getQuantizedBiases());
    biasEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesQ.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesQ(i), biasEstimated(i), 0.001);

    quant.getQuantizedBiases().synchronizeDToH();

    quant.back_propagate();

    CudaTensor<double> weightsDiffEstimated = cuda_tensor_cast<double>(quant.getDiffFullPrecisionWeights(0));
    weightsDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);
    
    CudaTensor<double> biasDiffEstimated = cuda_tensor_cast<double>(quant.getDiffFullPrecisionBiases());
    biasDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesDiff(i), biasEstimated(i), 0.001);
}

TEST_DATASET(   SATQuantizerCell_Frame_CUDA_Double,
                weights_clamping_propagate_back_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t nbBits), 
                std::make_tuple(28U, 28U, 10U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 2),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 7),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 5),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 10),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 11),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 13),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 14)

            )
{

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*nbBits);

    // NOTE: tensor = diff quant tensor in test

    CudaTensor<double> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<double> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<double> weightsDiff({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });

    CudaTensor<double> biases({nbOutputs});
    CudaTensor<double> biasesQ({nbOutputs});
    CudaTensor<double> biasesDiff({nbOutputs});

    double max_W_abs = 0.0f;
    float range = std::pow(2,nbBits);

    for (unsigned int index = 0; index < weights.size(); ++index) {
        weights(index) = Random::randUniform(-1.0, 1.0);
        weightsQ(index) = std::tanh(weights(index));

        double abs_w = std::abs(weightsQ(index));

        if(abs_w > max_W_abs)
           max_W_abs = abs_w;
    }
    //cudaSDorefaQ_kernel
    /* 
        float q = 0.5f*( (x[i] / tanhMax) + 1.0f);
        y[i] = q*2.0f - 1.0f;
    */
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = 2.0f*(0.5f * ((weightsQ(i)/max_W_abs) + 1.0f )) - 1.0f;
    }

    double wSum = std::accumulate(weightsQ.begin(), weightsQ.end(), 0.0);
    double wMean = wSum / weightsQ.size();
    double wVariance = 0.0;

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
    {
        wVariance += (weightsQ(i) - wMean)*(weightsQ(i) - wMean);
    }

    wVariance /= (weightsQ.size() - 1.0);

    double wSATscaling = std::sqrt(wVariance*weightsQ.dimB()*weightsQ.dimX()*weightsQ.dimY());
    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        weightsQ(i) /= wSATscaling;

    //cudaSSATGrad_kernel
    /*
        float revCoshData = 1/std::cosh(x[i]);
        float gradSAT = revCoshData*revCoshData * 1/(factors);
        output[i] = diff[i]*gradSAT;
    */
    for(unsigned int i = 0; i < weightsDiff.size(); ++ i){

        weightsDiff(i) = 
            weights(i)*((1/std::cosh(weights(i)))*(1/std::cosh(weights(i)))
                /(max_W_abs*wSATscaling));     
    }

    for (unsigned int index = 0; index < biases.size(); ++index) {
        biases(index) = Random::randUniform(-1.0, 1.0);
    }
    // For the moment, bias quantization is just identity
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesQ(index) = biases(index);
    }
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesDiff(index) = biases(index);
    }

    weights.synchronizeHToD();
    biases.synchronizeHToD();

    SATQuantizerCell_Frame_CUDA<double> quant;
    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.addBiases(biases, biases);
    quant.setRange(range);
    quant.setScaling(true);
    quant.setQuantization(false);
    quant.initialize();
    quant.propagate();

    CudaTensor<double> weightsEstimated = cuda_tensor_cast<double>(quant.getQuantizedWeights(0));
    weightsEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    CudaTensor<double> biasEstimated = cuda_tensor_cast<double>(quant.getQuantizedBiases());
    biasEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesQ.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesQ(i), biasEstimated(i), 0.001);

    quant.getQuantizedBiases().synchronizeDToH();

    quant.back_propagate();

    CudaTensor<double> weightsDiffEstimated = cuda_tensor_cast<double>(quant.getDiffFullPrecisionWeights(0));
    weightsDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);
    
    CudaTensor<double> biasDiffEstimated = cuda_tensor_cast<double>(quant.getDiffFullPrecisionBiases());
    biasDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesDiff(i), biasEstimated(i), 0.001);
}

TEST_DATASET(   SATQuantizerCell_Frame_CUDA_Double,
                weights_clamping_no_scaling_propagate_back,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t nbBits), 
                std::make_tuple(28U, 28U, 10U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 2),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 7),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 5),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 10),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 11),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 13),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 14)

            )
{

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*nbBits);

    // NOTE: tensor = diff quant tensor in test

    CudaTensor<double> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<double> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<double> weightsDiff({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });

    CudaTensor<double> biases({nbOutputs});
    CudaTensor<double> biasesQ({nbOutputs});
    CudaTensor<double> biasesDiff({nbOutputs});

    double max_W_abs = 0.0f;
    float range = std::pow(2,nbBits);

    for (unsigned int index = 0; index < weights.size(); ++index) {
        weights(index) = Random::randUniform(-1.0, 1.0);
        weightsQ(index) = std::tanh(weights(index));

        double abs_w = std::abs(weightsQ(index));

        if(abs_w > max_W_abs)
           max_W_abs = abs_w;
    }

    //cudaSDorefaQ_kernel
    /* 
        float q = 0.5f*( (x[i] / tanhMax) + 1.0f);
        y[i] = q*2.0f - 1.0f;
    */
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = 2.0f*(0.5f * ( (weightsQ(i)/max_W_abs) + 1.0f ) ) - 1.0f;
    }

    //cudaSSATGrad_kernel
    /*
        float revCoshData = 1/std::cosh(x[i]);
        float gradSAT = revCoshData*revCoshData * 1/(factors);
        output[i] = diff[i]*gradSAT;
    */
    for(unsigned int i = 0; i < weightsDiff.size(); ++ i){

        weightsDiff(i) = 
            weights(i)*((1/std::cosh(weights(i)))*(1/std::cosh(weights(i)))
                /(max_W_abs));     
    }


    for (unsigned int index = 0; index < biases.size(); ++index) {
        biases(index) = Random::randUniform(-1.0, 1.0);
    }
    // For the moment, bias quantization is just identity
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesQ(index) = biases(index);
    }
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesDiff(index) = biases(index);
    }

    weights.synchronizeHToD();
    biases.synchronizeHToD();


    SATQuantizerCell_Frame_CUDA<double> quant;
    quant.setQuantization(false);
    quant.setScaling(false);

    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.addBiases(biases, biases);
    quant.setRange(range);
    quant.initialize();

    quant.propagate();

    CudaTensor<double> weightsEstimated = cuda_tensor_cast<double>(quant.getQuantizedWeights(0));
    weightsEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    CudaTensor<double> biasEstimated = cuda_tensor_cast<double>(quant.getQuantizedBiases());
    biasEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesQ.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesQ(i), biasEstimated(i), 0.001);

    quant.getQuantizedBiases().synchronizeDToH();

    quant.back_propagate();

    CudaTensor<double> weightsDiffEstimated = cuda_tensor_cast<double>(quant.getDiffFullPrecisionWeights(0));
    weightsDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);
    
    CudaTensor<double> biasDiffEstimated = cuda_tensor_cast<double>(quant.getDiffFullPrecisionBiases());
    biasDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesDiff(i), biasEstimated(i), 0.001);
}


//Half
/*
TEST_DATASET(   SATQuantizerCell_Frame_CUDA_Half,
                weights_quantization_scaling_back_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t nbBits), 
                std::make_tuple(28U, 28U, 10U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 3),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 6),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 4),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 2),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 8),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 7),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 5),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 3),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 10),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 11),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 13),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 14)
                
            )
{

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*nbBits);

    // NOTE: tensor = diff quant tensor in test

    CudaTensor<half_float::half> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<half_float::half> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    CudaTensor<half_float::half> weightsDiff({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });

    CudaTensor<half_float::half> biases({nbOutputs});
    CudaTensor<half_float::half> biasesQ({nbOutputs});
    CudaTensor<half_float::half> biasesDiff({nbOutputs});

    half_float::half max_W_abs = half_float::half(0.0);
    float range = std::pow(2,nbBits);

    for (unsigned int index = 0; index < weights.size(); ++index) {
        weights(index) = half_float::half(Random::randUniform(-1.0, 1.0));
        weightsQ(index) = half_float::half(std::tanh(weights(index)));
        half_float::half abs_w = half_float::half(std::abs(weightsQ(index)));
        if(abs_w > max_W_abs)
           max_W_abs = abs_w;
    }

    //cudaHDorefaQ_kernel
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = half_float::half(2.0f*((1.0f/range)*rint(0.5f*((weightsQ(i)/max_W_abs)+1.0f)*range))-1.0f);
    }

    half_float::half wSum = half_float::half(std::accumulate(weightsQ.begin(), weightsQ.end(), 0.0));
    half_float::half wMean = half_float::half(wSum / weightsQ.size());
    half_float::half wVariance = half_float::half(0.0);

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
    {
        wVariance += (weightsQ(i) - wMean)*(weightsQ(i) - wMean);
    }

    wVariance /= (weightsQ.size() - 1.0);

    half_float::half wSATscaling = half_float::half(std::sqrt(wVariance*weightsQ.dimB()*weightsQ.dimY()*weightsQ.dimX()));
    
    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        weightsQ(i) /= wSATscaling;
    

    //cudaSSATGrad_kernel
    for(unsigned int i = 0; i < weightsDiff.size(); ++ i){
        weightsDiff(i) = 
            weights(i)*((1/std::cosh(weights(i)))*(1/std::cosh(weights(i)))
                /(max_W_abs*wSATscaling));       
    }

    for (unsigned int index = 0; index < biases.size(); ++index) {
        biases(index) = Random::randUniform(-1.0, 1.0);
    }
    // For the moment, bias quantization is just identity
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesQ(index) = biases(index);
    }
    for (unsigned int index = 0; index < biases.size(); ++index) {
        biasesDiff(index) = biases(index);
    }

    weights.synchronizeHToD();
    biases.synchronizeHToD();

    SATQuantizerCell_Frame_CUDA<half_float::half> quant;
    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.addBiases(biases, biases);
    quant.setRange(range);
    quant.setScaling(true);
    quant.setQuantization(true);
    quant.initialize();
    quant.propagate();

    CudaTensor<half_float::half> weightsEstimated = cuda_tensor_cast<half_float::half>(quant.getQuantizedWeights(0));
    weightsEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.1);

    CudaTensor<half_float::half> biasEstimated = cuda_tensor_cast<half_float::half>(quant.getQuantizedBiases());
    biasEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesQ.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesQ(i), biasEstimated(i), 0.1);

    quant.getQuantizedBiases().synchronizeDToH();

    quant.back_propagate();

    CudaTensor<half_float::half> weightsDiffEstimated = cuda_tensor_cast<half_float::half>(quant.getDiffFullPrecisionWeights(0));
    weightsDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.1);
    
    CudaTensor<half_float::half> biasDiffEstimated = cuda_tensor_cast<half_float::half>(quant.getDiffFullPrecisionBiases());
    biasDiffEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < biasesDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(biasesDiff(i), biasEstimated(i), 0.1);

}
*/

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
