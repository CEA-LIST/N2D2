/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#include "Quantizer/QAT/Cell/Prune/PruneQuantizerCell_Frame.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"
#include "third_party/half.hpp"
#include <math.h>

using namespace N2D2;

TEST_DATASET(   PruneQuantizerCell_Frame_Float,
                weights_prune_static_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t kHeight, size_t kWidth, float threshold),
                std::make_tuple(28U, 28U, 6U, 1U, 3U, 3U, 0.2)               

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

    PruneQuantizerCell_Frame<float> quant;

    quant.addWeights(weights, weightsDiff);
    quant.setPruningMode(N2D2::PruneQuantizerCell::PruningMode::Static);
    quant.setThreshold(threshold);
    quant.initialize();
    quant.propagate();

    Tensor<float> weightsEstimated = tensor_cast<float>(quant.getQuantizedWeights(0));
    
    Tensor<unsigned int> masks = tensor_cast<unsigned int>(quant.getMasksWeights(0));
    
    for(unsigned int i = 0; i < weights.size(); ++i)
        ASSERT_EQUALS_DELTA(weights(i) * masks(i), weightsEstimated(i), 0.001);

    unsigned int count_zero = 0;
    for(unsigned int i = 0; i < masks.size(); ++i) {
        if (masks(i) == 0U) {
            ++count_zero;
        }
    }

    float zero_ratio = (float)count_zero / masks.size();
    ASSERT_EQUALS_DELTA(zero_ratio, threshold, 0.1);


    quant.back_propagate();

    Tensor<float> weightsDiffEstimated = tensor_cast<float>(quant.getDiffFullPrecisionWeights(0));

    for(unsigned int i = 0; i < weights.size(); ++i)
        ASSERT_EQUALS_DELTA(weightsDiff(i) * masks(i), weightsDiffEstimated(i), 0.001);
}

RUN_TESTS()