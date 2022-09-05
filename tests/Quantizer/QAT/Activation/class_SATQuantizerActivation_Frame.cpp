/**
 * (C) Copyright 2020 CEA LIST. All Rights Reserved.
 *  Contributor(s): David BRIAND (david.briand@cea.fr)
 *                  Vincent TEMPLIER (vincent.templier@cea.fr)
 * 
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 * 
 */

#include "N2D2.hpp"

#include "Quantizer/QAT/Activation/SAT/SATQuantizerActivation_Frame.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(   SATQuantizerActivation_Frame_Float,
                activations_quantization_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t nbBits, float alpha), 
                std::make_tuple(28U, 28U, 10U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 7, 1.2),
                std::make_tuple(28U, 28U, 4U, 1U, 5, 1.1),
                std::make_tuple(28U, 28U, 4U, 4U, 3, 1.0),
                std::make_tuple(28U, 28U, 4U, 16U, 10, 2.0),
                std::make_tuple(28U, 28U, 16U, 8U, 11, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 13, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 14, 1.5)

            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*nbBits);
    Tensor<float> activations(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
    Tensor<float> activationsQ(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});

    float range = std::pow(2,nbBits);

    for (unsigned int index = 0; index < activations.size(); ++index) {
        activations(index) = Random::randUniform(-3.0, 3.0);
    }

    // x~/alpha = 0.5*(|x|-|x-alpha|+alpha)
    // q = alpha * (1/a)*round(a*(x~/alpha))
    for(unsigned int i = 0; i < activations.size(); ++i)
    {
        //const float hardTanh = 0.5*(std::abs(activations(i)) -  std::abs(activations(i) - alpha) + alpha);
        const float hardTanh = (activations(i) < 0.0f) ? 0.0f : (activations(i) < alpha) ? activations(i) : alpha;
        activationsQ(i) = alpha*(1/range)*std::round( range * (hardTanh / alpha) );
    }

    SATQuantizerActivation_Frame<float> quant;

    quant.setRange(range);
    quant.setAlpha(alpha);
    //In-place operation
    quant.propagate(activations, false);
    for(unsigned int i = 0; i < activations.size(); ++i)
        ASSERT_EQUALS_DELTA(activationsQ(i), activations(i), 0.001);
}

TEST_DATASET(   SATQuantizerActivation_Frame_Double,
                activations_quantization_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t nbBits, float alpha), 
                std::make_tuple(28U, 28U, 10U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 7, 1.2),
                std::make_tuple(28U, 28U, 4U, 1U, 5, 1.1),
                std::make_tuple(28U, 28U, 4U, 4U, 3, 1.0),
                std::make_tuple(28U, 28U, 4U, 16U, 10, 2.0),
                std::make_tuple(28U, 28U, 16U, 8U, 11, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 13, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 14, 1.5)

            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*nbBits);
    Tensor<double> activations(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
    Tensor<double> activationsQ(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});

    double range = std::pow(2,nbBits);

    for (unsigned int index = 0; index < activations.size(); ++index) {
        activations(index) = Random::randUniform(-3.0, 3.0);
    }

    // x~/alpha = 0.5*(|x|-|x-alpha|+alpha)
    // q = alpha * (1/a)*round(a*(x~/alpha))
    for(unsigned int i = 0; i < activations.size(); ++i)
    {
        //const double hardTanh = 0.5*(std::abs(activations(i)) -  std::abs(activations(i) - alpha) + alpha);
        const double hardTanh = (activations(i) < 0.0) ? 0.0 : (activations(i) < alpha) ? activations(i) : alpha;
        activationsQ(i) = alpha*(1/range)*std::round( range * (hardTanh / alpha) );
    }

    SATQuantizerActivation_Frame<double> quant;
    quant.setRange(range);
    quant.setAlpha(alpha);

    //In-place operation
    quant.propagate(activations, false);
    for(unsigned int i = 0; i < activations.size(); ++i)
        ASSERT_EQUALS_DELTA(activationsQ(i), activations(i), 0.001);
}

/*
TEST_DATASET(   SATQuantizerActivation_Frame_Half,
                activations_quantization_propagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t nbBits, float alpha), 
                std::make_tuple(28U, 28U, 10U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 7, 1.2),
                std::make_tuple(28U, 28U, 4U, 1U, 5, 1.1),
                std::make_tuple(28U, 28U, 4U, 4U, 3, 1.0),
                std::make_tuple(28U, 28U, 4U, 16U, 10, 2.0),
                std::make_tuple(28U, 28U, 16U, 8U, 11, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 13, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 14, 1.5)

            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*nbBits);
    Tensor<half_float::half> activations(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});
    Tensor<half_float::half> activationsQ(  { outputW, 
                                      outputH, 
                                      nbOutputs,
                                      1});

    half_float::half range = half_float::half(std::pow(2,nbBits));

    for (unsigned int index = 0; index < activations.size(); ++index) {
        activations(index) = half_float::half(Random::randUniform(-3.0, 3.0));
    }

    // x~/alpha = 0.5*(|x|-|x-alpha|+alpha)
    // q = alpha * (1/a)*round(a*(x~/alpha))
    const half_float::half zero = half_float::half(0.0);
    const half_float::half one = half_float::half(1.0);
    const half_float::half alpha_half = half_float::half(alpha);

    for(unsigned int i = 0; i < activations.size(); ++i)
    {
        //const float hardTanh = 0.5*(std::abs(activations(i)) -  std::abs(activations(i) - alpha) + alpha);
        const half_float::half hardTanh = (activations(i) < zero) ? zero : (activations(i) < alpha_half) ? activations(i) : alpha_half;
        const half_float::half q = hardTanh / alpha_half;
        activationsQ(i) = ((one / range) * rint(range * q)) * q;
    }

    SATQuantizerActivation_Frame<half_float::half> quant;
    quant.setRange(range);
    quant.setAlpha(alpha);
    
    //In-place operation
    quant.propagate(activations, false);
    //Small delta for half precision...
    for(unsigned int i = 0; i < activations.size(); ++i) {
        ASSERT_EQUALS_DELTA(activationsQ(i), activations(i), 0.01);
    }
}
*/

TEST_DATASET(   SATQuantizerActivation_Frame_Float,
                activations_quantization_backpropagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t nbBits, float alpha), 
                std::make_tuple(28U, 28U, 10U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 7, 1.2),
                std::make_tuple(28U, 28U, 4U, 1U, 5, 1.1),
                std::make_tuple(28U, 28U, 4U, 4U, 3, 1.0),
                std::make_tuple(28U, 28U, 4U, 16U, 10, 2.0),
                std::make_tuple(28U, 28U, 16U, 8U, 11, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 13, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 14, 1.5)

            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*nbBits);
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

    Tensor<float> diffAlphas({1, 1, 1, 1});
    diffOutput.fill(0.0);
    diffOutputPred.fill(0.0);
    diffAlphas.fill(0.0);

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<float> >();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        fpActivations(index) = Random::randUniform(-3.0, 3.0);
        diffInput(index) = Random::randUniform(-1.0, 1.0);
    }
    
    float range = std::pow(2,nbBits);
    float abs_alpha = abs(alpha);

    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        float dQ = fpActivations(i) <= 0.0f ? 0.0f : (fpActivations(i) > abs_alpha ? 0.0f : 1.0f);
        diffOutput(i) = diffInput(i)*dQ;
    }

    float alphaGrad = 0.0;
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {    
        float hardTanh = (fpActivations(i) < 0.0f) ? 0.0f : (fpActivations(i) < abs_alpha) ? fpActivations(i) : abs_alpha;
        float qData = (1.0f/range)*rint(range*(hardTanh/abs_alpha));
        float dQalpha = fpActivations(i) >= abs_alpha ? 1.0f : (qData - hardTanh/abs_alpha);
        alphaGrad += diffInput(i)*dQalpha;    
    }


    SATQuantizerActivation_Frame<float> quant;

    quant.setRange(range);
    quant.setAlpha(alpha);
    quant.setSolver(quantizerSolver);

    // SATQuantizerActivation_Frame<T>::back_propagate( const BaseTensor& baseInput,
    //                                                            const BaseTensor& baseOutput,
    //                                                            const BaseTensor& baseDiffInput,
    //                                                            BaseTensor& baseDiffOutput)
    //==> baseOutput not use for the SAT back_propagate 
    //
    quant.back_propagate(fpActivations, fpActivations, diffInput, diffOutputPred);

    for(unsigned int i = 0; i < diffOutput.size(); ++i) {
        ASSERT_EQUALS_DELTA(diffOutputPred(i), diffOutput(i), 0.001);
    }

    const Tensor<float> diffAlphaPred = quant.getDiffAlpha();
    ASSERT_EQUALS_DELTA(diffAlphaPred(0,0,0,0), alphaGrad, 0.001);
}

TEST_DATASET(   SATQuantizerActivation_Frame_Double,
                activations_quantization_backpropagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t nbBits, float alpha), 
                std::make_tuple(28U, 28U, 10U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 7, 1.2),
                std::make_tuple(28U, 28U, 4U, 1U, 5, 1.1),
                std::make_tuple(28U, 28U, 4U, 4U, 3, 1.0),
                std::make_tuple(28U, 28U, 4U, 16U, 10, 2.0),
                std::make_tuple(28U, 28U, 16U, 8U, 11, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 13, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 14, 1.5)

            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*nbBits);
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

    Tensor<double> diffAlphas({1, 1, 1, 1});
    diffOutput.fill(0.0);
    diffOutputPred.fill(0.0);
    diffAlphas.fill(0.0);

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<double> >();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        fpActivations(index) = Random::randUniform(-3.0, 3.0);
        diffInput(index) = Random::randUniform(-1.0, 1.0);
    }
    
    double range = std::pow(2,nbBits);
    double abs_alpha = abs(alpha);

    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        double dQ = fpActivations(i) <= 0.0f ? 0.0f : (fpActivations(i) > abs_alpha ? 0.0f : 1.0f);
        diffOutput(i) = diffInput(i)*dQ;
    }

    double alphaGrad = 0.0;
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {    
        double hardTanh = (fpActivations(i) < 0.0f) ? 0.0f : (fpActivations(i) < abs_alpha) ? fpActivations(i) : abs_alpha;
        double qData = (1.0f/range)*rint(range*(hardTanh/abs_alpha));
        double dQalpha = fpActivations(i) >= abs_alpha ? 1.0f : (qData - hardTanh/abs_alpha);
        alphaGrad += diffInput(i)*dQalpha;    
    }


    SATQuantizerActivation_Frame<double> quant;

    quant.setRange(range);
    quant.setAlpha(alpha);
    quant.setSolver(quantizerSolver);

    // SATQuantizerActivation_Frame<T>::back_propagate( const BaseTensor& baseInput,
    //                                                            const BaseTensor& baseOutput,
    //                                                            const BaseTensor& baseDiffInput,
    //                                                            BaseTensor& baseDiffOutput)
    //==> baseOutput not use for the SAT back_propagate 
    //
    quant.back_propagate(fpActivations, fpActivations, diffInput, diffOutputPred);

    for(unsigned int i = 0; i < diffOutput.size(); ++i) {
        ASSERT_EQUALS_DELTA(diffOutputPred(i), diffOutput(i), 0.001);
    }

    const Tensor<double> diffAlphaPred = quant.getDiffAlpha();
    ASSERT_EQUALS_DELTA(diffAlphaPred(0,0,0,0), alphaGrad, 0.001);
}

/*
TEST_DATASET(   SATQuantizerActivation_Frame_Half,
                activations_quantization_backpropagate,
                (size_t outputH, size_t outputW, size_t nbOutputs, size_t nbChannel, size_t nbBits, float alpha), 
                std::make_tuple(28U, 28U, 10U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 6, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 4, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 4, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 3, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 2, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 1, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 1U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 4U, 8, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 8, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 8, 1.5),

                std::make_tuple(28U, 28U, 1U, 4U, 7, 1.2),
                std::make_tuple(28U, 28U, 4U, 1U, 5, 1.1),
                std::make_tuple(28U, 28U, 4U, 4U, 3, 1.0),
                std::make_tuple(28U, 28U, 4U, 16U, 10, 2.0),
                std::make_tuple(28U, 28U, 16U, 8U, 11, 1.5),
                std::make_tuple(28U, 28U, 4U, 16U, 13, 1.5),
                std::make_tuple(28U, 28U, 16U, 8U, 14, 1.5)

            )
{

    Random::mtSeed(outputH*outputW*nbOutputs*nbChannel*nbBits);
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
    half_float::half zero_half = half_float::half(0.0);
    half_float::half one_half = half_float::half(1.0);
    Tensor<half_float::half> diffAlphas({1, 1, 1, 1});
    diffOutput.fill(zero_half);
    diffOutputPred.fill(zero_half);
    diffAlphas.fill(zero_half);

    std::shared_ptr<Solver> quantizerSolver = std::make_shared<SGDSolver_Frame<half_float::half> >();
    for (unsigned int index = 0; index < fpActivations.size(); ++index) {
        fpActivations(index) = half_float::half(Random::randUniform(-3.0, 3.0));
        diffInput(index) = half_float::half(Random::randUniform(-1.0, 1.0));
    }
    
    half_float::half range = half_float::half(std::pow(2,nbBits));
    half_float::half abs_alpha = half_float::half(abs(alpha));

    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {
        half_float::half dQ = fpActivations(i) <= zero_half ? zero_half : (fpActivations(i) > abs_alpha ? zero_half : one_half);
        diffOutput(i) = diffInput(i)*dQ;
    }

    half_float::half alphaGrad = zero_half;
    for(unsigned int i = 0; i < fpActivations.size(); ++i)
    {    
        half_float::half hardTanh = (fpActivations(i) < zero_half) ? zero_half : (fpActivations(i) < abs_alpha) ? fpActivations(i) : abs_alpha;
        half_float::half qData = (one_half/range)*rint(range*(hardTanh/abs_alpha));
        half_float::half dQalpha = fpActivations(i) >= abs_alpha ? one_half : (qData - hardTanh/abs_alpha);
        alphaGrad += diffInput(i)*dQalpha;    
    }


    SATQuantizerActivation_Frame<half_float::half> quant;

    quant.setRange(range);
    quant.setAlpha(alpha);
    quant.setSolver(quantizerSolver);

    // SATQuantizerActivation_Frame<T>::back_propagate( const BaseTensor& baseInput,
    //                                                            const BaseTensor& baseOutput,
    //                                                            const BaseTensor& baseDiffInput,
    //                                                            BaseTensor& baseDiffOutput)
    //==> baseOutput not use for the SAT back_propagate 
    //
    quant.back_propagate(fpActivations, fpActivations, diffInput, diffOutputPred);

    for(unsigned int i = 0; i < diffOutput.size(); ++i) {
        ASSERT_EQUALS_DELTA(diffOutputPred(i), diffOutput(i), 0.1);
    }

    const Tensor<half_float::half> diffAlphaPred = quant.getDiffAlpha();
    ASSERT_EQUALS_DELTA(diffAlphaPred(0,0,0,0), alphaGrad, 0.1);
}
*/


RUN_TESTS()
