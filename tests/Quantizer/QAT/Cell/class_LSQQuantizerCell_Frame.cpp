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
    Tensor<float> stepSizeEstimated = quant.getStepSize();
    ASSERT_EQUALS_DELTA(stepSizeW+0.01*stepWGrad, stepSizeEstimated(0), 0.001);
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
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = (weights(i)/stepSizeW);
        weightsQ(i) = weightsQ(i) <= (double) WeightsRange.first ? (double) WeightsRange.first 
                        : weightsQ(i) >= (double) WeightsRange.second ? (double) WeightsRange.second : weightsQ(i);
        weightsQ(i) = round(weightsQ(i)) * stepSizeW;
    }
    double gW = 1/std::sqrt(weightsQ.size() * WeightsRange.second);

    // dL/dx = dL/dq * dq/dx~ * dx~/dx
    // STE dq(x~)/dx~=1
    // dx~/dx = clamp(x, WeightsRange.first, WeightsRange.second)
    // => dL/dx = dL/dq * clamp(x, WeightsRange.first, WeightsRange.second)
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
    for(unsigned int i = 0; i < weights.size(); ++i)
    { 
        double quantizedWeights = weights(i)/stepSizeW;
        quantizedWeights = quantizedWeights < (double) WeightsRange.first ? (double) WeightsRange.first
                        : (quantizedWeights > (double) WeightsRange.second ? (double) WeightsRange.second
                        : rint(quantizedWeights) - quantizedWeights);
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
    Tensor<double> stepSizeEstimated = quant.getStepSize();
    ASSERT_EQUALS_DELTA(stepSizeW+0.01*stepWGrad, stepSizeEstimated(0), 0.001);
}


/******************half_float::half********************/
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
        weights(index) = half_float::half_cast<half_float::half>(Random::randUniform(-1.0, 1.0));
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
    for(unsigned int i = 0; i < weightsQ.size(); ++ i)
    {
        weightsQ(i) = (weights(i)/stepSizeW);
        weightsQ(i) = (weightsQ(i) <= half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) :
                      (weightsQ(i) >= half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                      weightsQ(i);
        weightsQ(i) = rint(weightsQ(i)) * stepSizeW;
    }
    half_float::half gW = half_float::half(1.0)/half_float::sqrt(half_float::half_cast<half_float::half>(weightsQ.size() * WeightsRange.second));

    // dL/dx = dL/dq * dq/dx~ * dx~/dx
    // STE dq(x~)/dx~=1
    // dx~/dx = clamp(x, WeightsRange.first, WeightsRange.second)
    // => dL/dx = dL/dq * clamp(x, WeightsRange.first, WeightsRange.second)
    // Backward weights
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
    for(unsigned int i = 0; i < weights.size(); ++i)
    { 
        half_float::half quantizedWeights = weights(i)/stepSizeW;
        quantizedWeights = (quantizedWeights < half_float::half_cast<half_float::half>(WeightsRange.first)) ? half_float::half_cast<half_float::half>(WeightsRange.first) : 
                            (quantizedWeights > half_float::half_cast<half_float::half>(WeightsRange.second)) ? half_float::half_cast<half_float::half>(WeightsRange.second) :
                            rint(quantizedWeights) - quantizedWeights;
        //quantizedWeights *= weightsDiff(i); -> weights(i); 
        //as qData *= diffQuantData_[i]; and diffQuantW = weights in the test
        quantizedWeights *= weights(i);
        stepSizeWGrad += quantizedWeights;
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
    ASSERT_EQUALS_DELTA(diffStepSizePred(0,0,0,0), stepSizeWGrad, 0.001);
}

RUN_TESTS()