/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)
                    Inna KUCHER (inna.kucher@cea.fr)
                    Vincent TEMPLIER (vincent.templier@cea.fr)
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

#include "Quantizer/QAT/Cell/LSQ/LSQQuantizerCell_Frame_CUDA.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"
#include <math.h>

using namespace N2D2;


TEST_DATASET(   LSQQuantizerCell_Frame_CUDA_Float,
                weights_quant_propagate_backpropagate_update,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t mRange, float stepSizeW),
                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 255, 0.2),
                
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 255, 0.2),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 255, 0.2),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 255, 0.2),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 255, 0.2),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 63, 0.2),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 63, 0.2),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 15, 0.2),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 15, 0.2),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 127, 0.2),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3, 0.2),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 255, 0.2),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 255, 0.2),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 255, 0.2),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 255, 0.2),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 255, 0.2),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 63, 0.2),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 63, 0.2),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 15, 0.2),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 15, 0.2),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 127, 0.2),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 3, 0.2),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 255, 0.05),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 255, 0.05),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 255, 0.05),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 255, 0.05),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 255, 0.05),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 63, 0.05),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 63, 0.05),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 15, 0.05),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 15, 0.05),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 127, 0.05),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 3, 0.05),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 255, 0.01),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 255, 0.01),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 255, 0.01),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 255, 0.01),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 255, 0.01),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 127, 1.2),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 31, 1.1),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 127, 1.0),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 1023, 2.0),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 2047, 0.01),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8191, 0.01),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 16383, 0.01)
                

            )
{

    CudaContext::setDevice(0);
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*8);

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


    for (unsigned int index = 0; index < weights.size(); ++index) {
        weights(index) = Random::randUniform(-1.0, 1.0);
        weightsQ(index) = weights(index);
        weightsDiff(index) = weights(index);
    }

    std::pair<int, int> WeightsRange = std::make_pair((int) -((mRange + 1)/2), (int) ((mRange - 1)/2));

    //LSQ
    /*
        float q = data/s;
        q = q <= WeightsRange.first ? WeightsRange.first : q >= WeightsRange.second ? WeightsRange.second : q;
        q = round(q);
        return q*s;
    */
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = (weights(i)/stepSizeW);
        weightsQ(i) = weightsQ(i) <= WeightsRange.first ? WeightsRange.first 
                        : weightsQ(i) >= WeightsRange.second ? WeightsRange.second : weightsQ(i);
        weightsQ(i) = round(weightsQ(i)) * stepSizeW;
    }
    float gW = 1/std::sqrt(weightsQ.size() * WeightsRange.second);

    // dL/dx = dL/dq * dq/dx~ * dx~/dx
    // STE dq(x~)/dx~=1
    // dx~/dx = clamp(x, WeightsRange.first, WeightsRange.second)
    // => dL/dx = dL/dq * clamp(x, WeightsRange.first, WeightsRange.second)
    for(unsigned int i = 0; i < weights.size(); ++i)
    {
        float w = weights(i)/stepSizeW;
        weightsDiff(i) = weights(i)*(w <= (float)WeightsRange.first ? 0.0f : (w >= (float)WeightsRange.second ? 0.0f : 1.0f));
    }

    
    // LSQ grad
    /*
        data = data/s;
        if (data <= WeightsRange.first) return WeightsRange.first;
        if (data >= WeightsRange.first) return WeightsRange.second;
        return -data + round(data);
    */

    float stepWGrad = 0.0;
    for(unsigned int i = 0; i < weights.size(); ++i)
    { 
        float quantizedWeights = weights(i)/stepSizeW;
        quantizedWeights = quantizedWeights < (float) WeightsRange.first ? (float) WeightsRange.first
                        : (quantizedWeights > (float) WeightsRange.second ? (float) WeightsRange.second
                        : rint(quantizedWeights) - quantizedWeights);
        //quantizedWeights *= weightsDiff(i); -> weights(i); 
        //as qData *= diffQuantData_[i]; and diffQuantW = weights in the test
        quantizedWeights *= weights(i);
        quantizedWeights *= gW;
        stepWGrad += quantizedWeights;
    } 
    weights.synchronizeHToD();

    LSQQuantizerCell_Frame_CUDA<float> quant;
    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.setRange(mRange);
    quant.setOptInitStepSize(false);
    quant.setStepSizeValue(stepSizeW);
    quant.initialize();
    quant.propagate();

    CudaTensor<float> weightsEstimated = cuda_tensor_cast<float>(quant.getQuantizedWeights(0));
    weightsEstimated.synchronizeDToH();

    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    quant.back_propagate();

    CudaTensor<float> weightsDiffEstimated = cuda_tensor_cast<float>(quant.getDiffFullPrecisionWeights(0));
    weightsDiffEstimated.synchronizeDToH();

     for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);

    quant.update(1);
    quant.getStepSize().synchronizeDToH();
    CudaTensor<float> stepSizeEstimated = quant.getStepSize();
    stepSizeEstimated.synchronizeDToH();
    ASSERT_EQUALS_DELTA(stepSizeW+0.01*stepWGrad, stepSizeEstimated(0), 0.001);
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
