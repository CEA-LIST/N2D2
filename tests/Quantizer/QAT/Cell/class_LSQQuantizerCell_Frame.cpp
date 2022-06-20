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

#include "N2D2.hpp"

#include "Quantizer/QAT/Cell/LSQ/LSQQuantizerCell_Frame.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"
#include "third_party/half.hpp"
#include <math.h>

using namespace N2D2;


TEST_DATASET(   LSQQuantizerCell_Frame_Float,
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

    /*************Compute Answer*************/
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*8);

    // NOTE: tensor = diff quant tensor in test
    Tensor<float> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<float> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<float> weightsDiff({ kWidth,
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
#pragma omp parallel for
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = (weights(i)/stepSizeW);
        weightsQ(i) = weightsQ(i) <= WeightsRange.first ? WeightsRange.first 
                        : weightsQ(i) >= WeightsRange.second ? WeightsRange.second : weightsQ(i);
        weightsQ(i) = round(weightsQ(i)) * stepSizeW;
    }
    float gW = 1.0/std::sqrt(weightsQ.size() * WeightsRange.second);

    // dL/dx = dL/dq * dq/dx~ * dx~/dx
    // STE dq(x~)/dx~=1
    // dx~/dx = clamp(x, WeightsRange.first, WeightsRange.second)
    // => dL/dx = dL/dq * clamp(x, WeightsRange.first, WeightsRange.second)
#pragma omp parallel for
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
#pragma omp parallel for reduction(+:stepWGrad)
    for(unsigned int i = 0; i < weights.size(); ++i)
    { 
        float quantizedWeights = weights(i)/stepSizeW;
        quantizedWeights = quantizedWeights < (float) WeightsRange.first ? (float) WeightsRange.first
                        : (quantizedWeights > (float) WeightsRange.second ? (float) WeightsRange.second
                        : round(quantizedWeights) - quantizedWeights);
        //quantizedWeights *= weightsDiff(i); -> weights(i); 
        //as qData *= diffQuantData_[i]; and diffQuantW = weights in the test
        quantizedWeights *= weights(i);
        stepWGrad += quantizedWeights;
    }
    stepWGrad *= gW;

    /**************Estimate Answer**************/
    LSQQuantizerCell_Frame<float> quant;
    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.setRange(mRange);
    quant.setOptInitStepSize(false);
    quant.setStepSizeValue(stepSizeW);
    quant.initialize();
    quant.propagate();

    Tensor<float> weightsEstimated = tensor_cast<float>(quant.getQuantizedWeights(0));
    
    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    quant.back_propagate();

    Tensor<float> weightsDiffEstimated = tensor_cast<float>(quant.getDiffFullPrecisionWeights(0));

     for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);

    quant.update(1);
    quant.getStepSize();
    Tensor<float> diffStepSizeEstimated = quant.getDiffStepSize();
    ASSERT_EQUALS_DELTA(stepWGrad, diffStepSizeEstimated(0,0,0,0), 0.001);
}


/**************DOUBLE**************/
TEST_DATASET(   LSQQuantizerCell_Frame_Double,
                weights_quant_propagate_backpropagate_update,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t mRange, double stepSizeW),
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

    /*************Compute Answer*************/
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*8);

    // NOTE: tensor = diff quant tensor in test
    Tensor<double> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<double> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<double> weightsDiff({ kWidth,
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
        double q = data/s;
        q = q <= WeightsRange.first ? WeightsRange.first : q >= WeightsRange.second ? WeightsRange.second : q;
        q = round(q);
        return q*s;
    */
#pragma omp parallel for
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = (weights(i)/stepSizeW);
        weightsQ(i) = weightsQ(i) <= (double) WeightsRange.first ? (double) WeightsRange.first 
                        : weightsQ(i) >= (double) WeightsRange.second ? (double) WeightsRange.second : weightsQ(i);
        weightsQ(i) = round(weightsQ(i)) * stepSizeW;
    }
    double gW = 1.0/std::sqrt(weightsQ.size() * WeightsRange.second);

    // dL/dx = dL/dq * dq/dx~ * dx~/dx
    // STE dq(x~)/dx~=1
    // dx~/dx = clamp(x, WeightsRange.first, WeightsRange.second)
    // => dL/dx = dL/dq * clamp(x, WeightsRange.first, WeightsRange.second)
#pragma omp parallel for
    for(unsigned int i = 0; i < weights.size(); ++i)
    {
        double w = weights(i)/stepSizeW;
        weightsDiff(i) = weights(i)*(w <= (double)WeightsRange.first ? 0.0 : (w >= (double)WeightsRange.second ? 0.0 : 1.0));
    }

    
    // LSQ grad
    /*
        data = data/s;
        if (data <= WeightsRange.first) return WeightsRange.first;
        if (data >= WeightsRange.first) return WeightsRange.second;
        return -data + round(data);
    */

    double stepWGrad = double(0.0);
#pragma omp parallel for reduction(+:stepWGrad)
    for(unsigned int i = 0; i < weights.size(); ++i)
    { 
        double quantizedWeights = weights(i)/stepSizeW;
        quantizedWeights = quantizedWeights < (double) WeightsRange.first ? (double) WeightsRange.first
                        : (quantizedWeights > (double) WeightsRange.second ? (double) WeightsRange.second
                        : round(quantizedWeights) - quantizedWeights);
        //quantizedWeights *= weightsDiff(i); -> weights(i); 
        //as qData *= diffQuantData_[i]; and diffQuantW = weights in the test
        quantizedWeights *= weights(i);
        quantizedWeights *= gW;
        stepWGrad += quantizedWeights;
    }

    /**************Estimate Answer**************/
    LSQQuantizerCell_Frame<double> quant;
    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.setRange(mRange);
    quant.setOptInitStepSize(false);
    quant.setStepSizeValue(stepSizeW);
    quant.initialize();
    quant.propagate();

    Tensor<double> weightsEstimated = tensor_cast<double>(quant.getQuantizedWeights(0));
    
    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    quant.back_propagate();

    Tensor<double> weightsDiffEstimated = tensor_cast<double>(quant.getDiffFullPrecisionWeights(0));

     for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);

    quant.update(1);
    quant.getStepSize();
    Tensor<double> diffStepSizeEstimated = quant.getDiffStepSize();
    ASSERT_EQUALS_DELTA(stepWGrad, diffStepSizeEstimated(0,0,0,0), 0.001);
}


/******************half_float::half********************/
#pragma omp declare reduction(+ : half_float::half : omp_out = omp_in + omp_out) initializer(omp_priv=half_float::half(0.0))
TEST_DATASET(   LSQQuantizerCell_Frame_HalfFloat,
                weights_quant_propagate_backpropagate_update,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t mRange, half_float::half stepSizeW),
                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 255, half_float::half(0.2)),
                
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 255, half_float::half(0.2)),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 127, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3, half_float::half(0.2)),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 255, half_float::half(0.2)),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 127, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 3, half_float::half(0.2)),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 255, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 255, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 255, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 255, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 255, half_float::half(0.05)),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 63, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 63, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 15, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 15, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 127, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 3, half_float::half(0.05)),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 255, half_float::half(0.01)),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 255, half_float::half(0.01)),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 255, half_float::half(0.01)),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 255, half_float::half(0.01)),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 255, half_float::half(0.01)),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 127, half_float::half(1.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 31, half_float::half(1.1)),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 127, half_float::half(1.0)),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 1023, half_float::half(2.0)),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 2047, half_float::half(0.01)),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8191, half_float::half(0.01)),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 16383, half_float::half(0.01))

            )
{

    /*************Compute Answer*************/
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*8);

    // NOTE: tensor = diff quant tensor in test
    Tensor<half_float::half> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<half_float::half> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<half_float::half> weightsDiff({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });


    for (unsigned int index = 0; index < weights.size(); ++index) {
        weights(index) = half_float::half_cast<half_float::half>((Random::randUniform(-1.0, 1.0)));
        weightsQ(index) = weights(index);
        weightsDiff(index) = weights(index);
    }

    std::pair<int, int> WeightsRange = std::make_pair((int) -((mRange + 1)/2), (int) ((mRange - 1)/2));

    //LSQ
    /*
        half_float::half q = data/s;
        q = q <= WeightsRange.first ? WeightsRange.first : q >= WeightsRange.second ? WeightsRange.second : q;
        q = round(q);
        return q*s;
    */
   // Forward
#pragma omp parallel for
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = (weights(i)/stepSizeW);
        weightsQ(i) = (weightsQ(i) <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) :
                      (weightsQ(i) >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                      weightsQ(i);
        weightsQ(i) = round(weightsQ(i)) * stepSizeW;
    }
    half_float::half gW = half_float::half_cast<half_float::half>((1.0)/sqrt(weightsQ.size() * WeightsRange.second));

    // dL/dx = dL/dq * dq/dx~ * dx~/dx
    // STE dq(x~)/dx~=1
    // dx~/dx = clamp(x, WeightsRange.first, WeightsRange.second)
    // => dL/dx = dL/dq * clamp(x, WeightsRange.first, WeightsRange.second)
    // Backward weights
#pragma omp parallel for
    for(unsigned int i = 0; i < weights.size(); ++i)
    {
        half_float::half w = weights(i)/stepSizeW;
        weightsDiff(i) = weights(i)*((w <= (half_float::half_cast<half_float::half>(WeightsRange.first))) ? half_float::half(0.0) :
                                     (w >= (half_float::half_cast<half_float::half>(WeightsRange.second))) ? half_float::half(0.0) :
                                     half_float::half(1.0));
    }

    
    // LSQ grad
    /*
        data = data/s;
        if (data <= WeightsRange.first) return WeightsRange.first;
        if (data >= WeightsRange.first) return WeightsRange.second;
        return -data + round(data);
    */
   // Backward Step Size
    half_float::half stepSizeWGrad = half_float::half(0.0);
#pragma omp parallel for schedule(static, 256) reduction(+:stepSizeWGrad)
    for(unsigned int i = 0; i < weights.size()/4; ++i)
    { 
        half_float::half qWeights_1 = weights(4*i)/stepSizeW;    
        qWeights_1 = (qWeights_1 <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                    (qWeights_1 >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                    round(qWeights_1) - qWeights_1;
        half_float::half qWeights_2 = weights(4*i+1)/stepSizeW;    
        qWeights_2 = (qWeights_2 <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                    (qWeights_2 >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                    round(qWeights_2) - qWeights_2;
        half_float::half qWeights_3 = weights(4*i+2)/stepSizeW;    
        qWeights_3 = (qWeights_3 <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                    (qWeights_3 >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                    round(qWeights_3) - qWeights_3;
        half_float::half qWeights_4 = weights(4*i+3)/stepSizeW;    
        qWeights_4 = (qWeights_4 <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                    (qWeights_4 >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                    round(qWeights_4) - qWeights_4;
                    
        stepSizeWGrad += ((qWeights_1*weights(4*i) + qWeights_2*weights(4*i+1)) + (qWeights_3*weights(4*i+2) + qWeights_4*weights(4*i+3)));
    }
    for(unsigned int i= weights.size()-weights.size()%4; i<weights.size(); ++i) {
        half_float::half qWeights = weights(i)/stepSizeW;
        qWeights = (qWeights <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                    (qWeights >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                    (round(qWeights) - qWeights);
        //quantizedWeights *= weightsDiff(i); -> weights(i); 
        //as qData *= diffQuantData_[i]; and diffQuantW = weights in the test
        stepSizeWGrad += qWeights*weights(i);
    }
    // if inside the for loop, the rounding approximations increase to much
    stepSizeWGrad *= gW;

    /**************Estimate Answer**************/
    LSQQuantizerCell_Frame<half_float::half> quant;
    // NOTE: tensor = diff quant tensor in test
    quant.addWeights(weights, weights);
    quant.setRange(mRange);
    quant.setOptInitStepSize(false);
    // stepSizeValue is a parameter with a float value
    quant.setStepSizeValue(half_float::half_cast<float>(stepSizeW));
    quant.initialize();
    quant.propagate();

    Tensor<half_float::half> weightsEstimated = tensor_cast<half_float::half>(quant.getQuantizedWeights(0));
    
    for(unsigned int i = 0; i < weightsQ.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsQ(i), weightsEstimated(i), 0.001);

    quant.back_propagate();

    Tensor<half_float::half> weightsDiffEstimated = tensor_cast<half_float::half>(quant.getDiffFullPrecisionWeights(0));

     for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i), weightsDiffEstimated(i), 0.001);

    const Tensor<half_float::half>& diffStepSizePred = quant.getDiffStepSize();
    const double coeff = ((stepSizeWGrad < 8)&&(stepSizeWGrad > -8)) ? 0.001 :
                            (stepSizeWGrad >=8) ? stepSizeWGrad*0.01 : -stepSizeWGrad*0.01;
    if ((stepSizeWGrad > half_float::half_cast<half_float::half>(diffStepSizePred(0,0,0,0))+0.001) || (stepSizeWGrad < half_float::half_cast<half_float::half>(diffStepSizePred(0,0,0,0))-0.001))
        std::cout << "stepSize= " << stepSizeW << " --mRange= " << mRange << " -- s_half= " << stepSizeWGrad << " -- s_pred= " << diffStepSizePred(0,0,0,0) << std::endl;
    ASSERT_EQUALS_DELTA(diffStepSizePred(0,0,0,0), stepSizeWGrad, coeff);
}

TEST_DATASET(   LSQQuantizerCell_Frame_HalfFloat_DoubleComparition,
                weights_quant_propagate_backpropagate_update,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, size_t mRange, half_float::half stepSizeW),
                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 255, half_float::half(0.10612)),
                
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 255, half_float::half(0.06783)),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 255, half_float::half(0.06633)),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 255, half_float::half(0.16348)),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 255, half_float::half(0.07098)),

                std::make_tuple(28U, 28U, 1U, 4U, 1U, 1U, 63, half_float::half(0.00769)),
                std::make_tuple(28U, 28U, 4U, 1U, 1U, 1U, 63, half_float::half(0.18146)),
                std::make_tuple(28U, 28U, 4U, 4U, 1U, 1U, 15, half_float::half(0.16408)),
                std::make_tuple(28U, 28U, 4U, 16U, 1U, 1U, 15, half_float::half(0.24793)),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 127, half_float::half(0.11379)),
                std::make_tuple(28U, 28U, 16U, 8U, 1U, 1U, 3, half_float::half(0.23267)),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 255, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 255, half_float::half(0.2)),

                std::make_tuple(28U, 28U, 1U, 4U, 3U, 3U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 3U, 3U, 63, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 4U, 3U, 3U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 15, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 16U, 8U, 3U, 3U, 127, half_float::half(0.2)),
                std::make_tuple(28U, 28U, 4U, 16U, 3U, 3U, 3, half_float::half(0.34233)),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 255, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 255, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 255, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 255, half_float::half(0.05)),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 255, half_float::half(0.05)),

                std::make_tuple(28U, 28U, 1U, 4U, 5U, 5U, 63, half_float::half(0.02410)),
                std::make_tuple(28U, 28U, 4U, 1U, 5U, 5U, 63, half_float::half(0.06985)),
                std::make_tuple(28U, 28U, 4U, 4U, 5U, 5U, 15, half_float::half(0.07682)),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 15, half_float::half(0.0389)),
                std::make_tuple(28U, 28U, 16U, 8U, 5U, 5U, 127, half_float::half(0.05017)),
                std::make_tuple(28U, 28U, 4U, 16U, 5U, 5U, 3, half_float::half(0.00175)),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 255, half_float::half(0.00824)),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 255, half_float::half(0.01756)),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 255, half_float::half(0.01947)),
                //std::make_tuple(28U, 28U, 16U, 256U, 3U, 3U, 255, half_float::half(0.00991)), //to many parameters
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 255, half_float::half(0.00991)),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 255, half_float::half(0.01395)),

                std::make_tuple(28U, 28U, 1U, 4U, 7U, 7U, 127, half_float::half(1.2)),
                std::make_tuple(28U, 28U, 4U, 1U, 7U, 7U, 31, half_float::half(1.1)),
                std::make_tuple(28U, 28U, 4U, 4U, 7U, 7U, 127, half_float::half(1.0)),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 1023, half_float::half(1.83886)),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 2047, half_float::half(0.01556)),
                std::make_tuple(28U, 28U, 4U, 16U, 7U, 7U, 8191, half_float::half(0.00282)),
                std::make_tuple(28U, 28U, 16U, 8U, 7U, 7U, 16383, half_float::half(0.00766))
                

            )
{

    /*************Compute Answer*************/
    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*kHeight*kWidth*57);

    /*-----------float Precision Matrices------------*/
    Tensor<float> weightsfloat({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<float> weightsQfloat({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<float> weightsDifffloat({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });

    // NOTE: tensor = diff quant tensor in test
    Tensor<half_float::half> weights({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<half_float::half> weightsQ({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });
    Tensor<half_float::half> weightsDiff({ kWidth,
                                kHeight,
                                nbChannel,
                                nbOutputs                                
                                });


    for (unsigned int index = 0; index < weights.size(); ++index) {
        float a = Random::randUniform(-1.0, 1.0);
        weightsfloat(index) = a;
        weightsQfloat(index) = a;
        weightsDifffloat(index) = a;
        weights(index) = half_float::half_cast<half_float::half>(a);
        weightsQ(index) = weights(index);
        weightsDiff(index) = weights(index);
    }

    std::pair<int, int> WeightsRange = std::make_pair((int) -((mRange + 1)/2), (int) ((mRange - 1)/2));

    //LSQ
    /*
        half_float::half q = data/s;
        q = q <= WeightsRange.first ? WeightsRange.first : q >= WeightsRange.second ? WeightsRange.second : q;
        q = round(q);
        return q*s;
    */
   // Forward
#pragma omp parallel for
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = (weights(i)/stepSizeW);
        weightsQ(i) = (weightsQ(i) <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) :
                      (weightsQ(i) >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                      weightsQ(i);
        weightsQ(i) = round(weightsQ(i)) * stepSizeW;

        weightsQfloat(i) = (weightsfloat(i)/(float)stepSizeW);
        weightsQfloat(i) = (weightsQfloat(i) <= (float)(WeightsRange.first)) ? (float)(WeightsRange.first) :
                      (weightsQfloat(i) >= (float)(WeightsRange.second)) ? (float)(WeightsRange.second) :
                      weightsQfloat(i);
        weightsQfloat(i) = round(weightsQfloat(i)) * (float)stepSizeW;
    }
    half_float::half gW = half_float::half_cast<half_float::half>((1.0)/sqrt(weightsQ.size() * WeightsRange.second));
    float gWfloat = half_float::half_cast<float>(gW);

    // dL/dx = dL/dq * dq/dx~ * dx~/dx
    // STE dq(x~)/dx~=1
    // dx~/dx = clamp(x, WeightsRange.first, WeightsRange.second)
    // => dL/dx = dL/dq * clamp(x, WeightsRange.first, WeightsRange.second)
    // Backward weights
#pragma omp parallel for
    for(unsigned int i = 0; i < weights.size(); ++i)
    {
        half_float::half w = weights(i)/stepSizeW;
        weightsDiff(i) = weights(i)*((w <= (half_float::half_cast<half_float::half>(WeightsRange.first))) ? half_float::half(0.0) :
                                     (w >= (half_float::half_cast<half_float::half>(WeightsRange.second))) ? half_float::half(0.0) :
                                     half_float::half(1.0));
        
        float wfloat = weightsfloat(i)/(float)stepSizeW;
        weightsDifffloat(i) = weightsfloat(i)*((wfloat <= (float)(WeightsRange.first)) ? 0.0 :
                                     (w >= (float)(WeightsRange.second)) ? 0.0 :
                                     1.0);
    }

    
    // LSQ grad
    /*
        data = data/s;
        if (data <= WeightsRange.first) return WeightsRange.first;
        if (data >= WeightsRange.first) return WeightsRange.second;
        return -data + round(data);
    */
   // Backward Step Size
    unsigned int line = weights.dimY()*weights.dimZ();
    half_float::half stepSizeWGrad = half_float::half(0.0);
    float stepSizeWGradfloat = 0.0;

    for(unsigned int x = 0; x<weights.dimX(); ++x) {
        half_float::half stepSizeWGrad_loc = half_float::half(0.0);
        float stepSizeWGradfloat_loc = float(0.0);
#pragma omp parallel for schedule(dynamic, 64) reduction(+:stepSizeWGrad_loc, stepSizeWGradfloat_loc)
        for(unsigned int i = 0; i < line/4; ++i)
        {

            half_float::half qWeights_1 = weights(line*x + 4*i)/stepSizeW;    
            qWeights_1 = (qWeights_1 <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                        (qWeights_1 >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                        round(qWeights_1) - qWeights_1;
            half_float::half qWeights_2 = weights(line*x + 4*i+1)/stepSizeW;    
            qWeights_2 = (qWeights_2 <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                        (qWeights_2 >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                        round(qWeights_2) - qWeights_2;
            half_float::half qWeights_3 = weights(line*x + 4*i+2)/stepSizeW;    
            qWeights_3 = (qWeights_3 <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                        (qWeights_3 >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                        round(qWeights_3) - qWeights_3;
            half_float::half qWeights_4 = weights(line*x + 4*i+3)/stepSizeW;    
            qWeights_4 = (qWeights_4 <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                        (qWeights_4 >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                        round(qWeights_4) - qWeights_4;
                        
            stepSizeWGrad += ((qWeights_1*weights(line*x + 4*i) 
                                + qWeights_2*weights(line*x + 4*i+1)) 
                                + (qWeights_3*weights(line*x + 4*i+2) 
                                + qWeights_4*weights(line*x + 4*i+3)));


            float qWeightsfloat_1 = weightsfloat(line*x + 4*i)/(float)stepSizeW;
            qWeightsfloat_1 = (qWeightsfloat_1 <= (float)(WeightsRange.first)) ? (float)(WeightsRange.first) : 
                                (qWeightsfloat_1 >= (float)(WeightsRange.second)) ? (float)(WeightsRange.second) :
                                round(qWeightsfloat_1) - qWeightsfloat_1;
            float qWeightsfloat_2 = weightsfloat(line*x + 4*i+1)/(float)stepSizeW;
            qWeightsfloat_2 = (qWeightsfloat_2 <= (float)(WeightsRange.first)) ? (float)(WeightsRange.first) : 
                                (qWeightsfloat_2 >= (float)(WeightsRange.second)) ? (float)(WeightsRange.second) :
                                round(qWeightsfloat_2) - qWeightsfloat_2;
            float qWeightsfloat_3 = weightsfloat(line*x + 4*i+2)/(float)stepSizeW;
            qWeightsfloat_3 = (qWeightsfloat_3 <= (float)(WeightsRange.first)) ? (float)(WeightsRange.first) : 
                                (qWeightsfloat_3 >= (float)(WeightsRange.second)) ? (float)(WeightsRange.second) :
                                round(qWeightsfloat_3) - qWeightsfloat_3;
            float qWeightsfloat_4 = weightsfloat(line*x + 4*i+3)/(float)stepSizeW;
            qWeightsfloat_4 = (qWeightsfloat_4 <= (float)(WeightsRange.first)) ? (float)(WeightsRange.first) : 
                                (qWeightsfloat_4 >= (float)(WeightsRange.second)) ? (float)(WeightsRange.second) :
                                round(qWeightsfloat_4) - qWeightsfloat_4;
            stepSizeWGradfloat += (qWeightsfloat_1*weightsfloat(line*x + 4*i) 
                                    + qWeightsfloat_2*weightsfloat(line*x + 4*i+1) 
                                    + qWeightsfloat_3*weightsfloat(line*x + 4*i+2) 
                                    + qWeightsfloat_4*weightsfloat(line*x + 4*i+3));
        }
        for (unsigned int i=line*(x+1) - line%4; i<line*(x+1); ++i) {
            half_float::half qWeights = weights(i)/stepSizeW;    
            qWeights = (qWeights <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                        (qWeights >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                        round(qWeights) - qWeights;
            stepSizeWGrad += qWeights*weights(i);

            float qWeightsfloat = weightsfloat(i)/(float)stepSizeW;
            qWeightsfloat = (qWeightsfloat <= (float)(WeightsRange.first)) ? (float)(WeightsRange.first) : 
                            (qWeightsfloat >= (float)(WeightsRange.second)) ? (float)(WeightsRange.second) :
                            round(qWeightsfloat) - qWeightsfloat;
            stepSizeWGradfloat += qWeightsfloat*weightsfloat(i);
        }
        stepSizeWGrad+= stepSizeWGrad_loc;
        stepSizeWGradfloat += stepSizeWGradfloat_loc;
    }
    // if inside the for loop, the rounding approximations increase to much
    stepSizeWGrad *= gW;
    stepSizeWGradfloat *= gWfloat;

    for(unsigned int i = 0; i < weightsDiff.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i),  half_float::half_cast<half_float::half>(weightsDifffloat(i)), 0.001);
    
    // if (stepSizeWGrad - half_float::half_cast<half_float::half>(stepSizeWGradfloat) >=0.01)
    //     std::cout << "(" << outputH << ", " << outputW << ", " << nbOutputs << ", " << nbChannel << ", " << kHeight << " ," << kWidth << ", " << mRange << ", " << stepSizeW << ")" << " -> stepSizeWGrad= " << stepSizeWGrad << "\t" << weights.size()/((stepSizeWGradfloat-half_float::half_cast<float>(stepSizeWGrad))/(stepSizeWGradfloat+half_float::half_cast<float>(stepSizeWGrad))) << std::endl;
    const double coeff = ((stepSizeWGradfloat < 1)&&(stepSizeWGradfloat > -1)) ? 0.04 :
                            (stepSizeWGradfloat >=1) ? stepSizeWGradfloat*0.04 : -stepSizeWGradfloat*0.04;
    ASSERT_EQUALS_DELTA(stepSizeWGrad, half_float::half_cast<half_float::half>(stepSizeWGradfloat), coeff);
}

RUN_TESTS()