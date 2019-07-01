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

#include "N2D2.hpp"

#include "Cell/SoftmaxCell_Frame.hpp"
#include "DeepNet.hpp"
#include "Network.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

template <class T>
class SoftmaxCell_Frame_Test : public SoftmaxCell_Frame<T> {
public:
    SoftmaxCell_Frame_Test(const DeepNet& deepNet, const std::string& name, 
                           unsigned int nbOutputs)
        : Cell(deepNet, name, nbOutputs),
          SoftmaxCell(deepNet, name, nbOutputs),
          SoftmaxCell_Frame<T>(deepNet, name, nbOutputs) {};

    friend class UnitTest_SoftmaxCell_Frame_float_backPropagate;
    friend class UnitTest_SoftmaxCell_Frame_double_backPropagate;
    friend class UnitTest_SoftmaxCell_Frame_half_backPropagate;
};

////////////////////////////////////////////////////////////////////////////////
// float
////////////////////////////////////////////////////////////////////////////////
TEST_DATASET(SoftmaxCell_Frame_float,
             propagate,
             (unsigned int nbOutputs, unsigned int batchSize),
             std::make_tuple(1U, 1U),
             std::make_tuple(10U, 1U),
             std::make_tuple(100U, 1U),
             std::make_tuple(1000U, 1U),
             std::make_tuple(1U, 9U),
             std::make_tuple(10U, 9U),
             std::make_tuple(100U, 9U),
             std::make_tuple(1000U, 9U))
{
    Network net;
    DeepNet dn(net);

    SoftmaxCell_Frame<float> softmax1(dn, "softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor<float> inputs({1, 1, nbOutputs, batchSize});
    Tensor<float> diffOutputs;
    softmax1.addInput(inputs, diffOutputs);

    inputs.fill(0.0);
    softmax1.propagate();
    const Tensor<float>& outputs1 = tensor_cast<float>(softmax1.getOutputs());

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs1(o), 1.0 / (double)nbOutputs, 1.0e-6);
    }

    inputs.fill(1.0);
    softmax1.propagate();
    const Tensor<float>& outputs2 = tensor_cast<float>(softmax1.getOutputs());

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs2(o), 1.0 / (double)nbOutputs, 1.0e-6);
    }

    inputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs(0, batchPos) = 1.0;

    softmax1.propagate();
    const Tensor<float>& outputs3 = tensor_cast<float>(softmax1.getOutputs());

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        ASSERT_EQUALS_DELTA(outputs3(0, batchPos),
                            std::exp(1.0) / (std::exp(1.0) + (nbOutputs - 1)),
                            1.0e-6);

        for (unsigned int o = 1; o < nbOutputs; ++o) {
            ASSERT_EQUALS_DELTA(outputs3(o, batchPos),
                                1.0 / (std::exp(1.0) + (nbOutputs - 1)),
                                1.0e-6);
        }
    }
}

TEST_DATASET(SoftmaxCell_Frame_float,
             backPropagate,
             (unsigned int nbOutputs, unsigned int batchSize),
             std::make_tuple(1U, 1U),
             std::make_tuple(10U, 1U),
             std::make_tuple(100U, 1U),
             std::make_tuple(1000U, 1U),
             std::make_tuple(1U, 9U),
             std::make_tuple(10U, 9U),
             std::make_tuple(100U, 9U),
             std::make_tuple(1000U, 9U))
{
    Network net;
    DeepNet dn(net);
    
    SoftmaxCell_Frame_Test<float> softmax1(dn, "softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor<float> inputs({1, 1, nbOutputs, batchSize});
    Tensor<float> diffOutputs({1, 1, nbOutputs, batchSize});
    softmax1.addInput(inputs, diffOutputs);

    ASSERT_NOTHROW_ANY(softmax1.checkGradient(1.0e-3, 1.0e-2));
    diffOutputs.clearValid();

    inputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs(0, batchPos) = 1.0;

    softmax1.propagate();
    softmax1.mDiffInputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        softmax1.mDiffInputs(nbOutputs - 1, batchPos) = 1.0;

    softmax1.backPropagate();
    const Tensor<float>& outputs = tensor_cast<float>(softmax1.getOutputs());

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        for (unsigned int channel = 0; channel < nbOutputs; ++channel) {
            float gradient = 0.0;

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                gradient += ((output == channel) - outputs(channel, batchPos))
                            * outputs(output, batchPos)
                            * softmax1.mDiffInputs(output, batchPos);
            }

            ASSERT_EQUALS_DELTA(
                diffOutputs(channel, batchPos), gradient, 1.0e-6);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// double
////////////////////////////////////////////////////////////////////////////////
TEST_DATASET(SoftmaxCell_Frame_double,
             propagate,
             (unsigned int nbOutputs, unsigned int batchSize),
             std::make_tuple(1U, 1U),
             std::make_tuple(10U, 1U),
             std::make_tuple(100U, 1U),
             std::make_tuple(1000U, 1U),
             std::make_tuple(1U, 9U),
             std::make_tuple(10U, 9U),
             std::make_tuple(100U, 9U),
             std::make_tuple(1000U, 9U))
{
    Network net;
    DeepNet dn(net);
    
    SoftmaxCell_Frame<double> softmax1(dn, "softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor<double> inputs({1, 1, nbOutputs, batchSize});
    Tensor<double> diffOutputs;
    softmax1.addInput(inputs, diffOutputs);

    inputs.fill(0.0);
    softmax1.propagate();
    const Tensor<double>& outputs1 = tensor_cast<double>(softmax1.getOutputs());

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs1(o), 1.0 / (double)nbOutputs, 1.0e-6);
    }

    inputs.fill(1.0);
    softmax1.propagate();
    const Tensor<double>& outputs2 = tensor_cast<double>(softmax1.getOutputs());

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs2(o), 1.0 / (double)nbOutputs, 1.0e-6);
    }

    inputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs(0, batchPos) = 1.0;

    softmax1.propagate();
    const Tensor<double>& outputs3 = tensor_cast<double>(softmax1.getOutputs());

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        ASSERT_EQUALS_DELTA(outputs3(0, batchPos),
                            std::exp(1.0) / (std::exp(1.0) + (nbOutputs - 1)),
                            1.0e-6);

        for (unsigned int o = 1; o < nbOutputs; ++o) {
            ASSERT_EQUALS_DELTA(outputs3(o, batchPos),
                                1.0 / (std::exp(1.0) + (nbOutputs - 1)),
                                1.0e-6);
        }
    }
}

TEST_DATASET(SoftmaxCell_Frame_double,
             backPropagate,
             (unsigned int nbOutputs, unsigned int batchSize),
             std::make_tuple(1U, 1U),
             std::make_tuple(10U, 1U),
             std::make_tuple(100U, 1U),
             std::make_tuple(1000U, 1U),
             std::make_tuple(1U, 9U),
             std::make_tuple(10U, 9U),
             std::make_tuple(100U, 9U),
             std::make_tuple(1000U, 9U))
{
    Network net;
    DeepNet dn(net);
    
    SoftmaxCell_Frame_Test<double> softmax1(dn, "softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor<double> inputs({1, 1, nbOutputs, batchSize});
    Tensor<double> diffOutputs({1, 1, nbOutputs, batchSize});
    softmax1.addInput(inputs, diffOutputs);

    ASSERT_NOTHROW_ANY(softmax1.checkGradient(1.0e-3, 1.0e-3));
    diffOutputs.clearValid();

    inputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs(0, batchPos) = 1.0;

    softmax1.propagate();
    softmax1.mDiffInputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        softmax1.mDiffInputs(nbOutputs - 1, batchPos) = 1.0;

    softmax1.backPropagate();
    const Tensor<double>& outputs = tensor_cast<double>(softmax1.getOutputs());

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        for (unsigned int channel = 0; channel < nbOutputs; ++channel) {
            double gradient = 0.0;

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                gradient += ((output == channel) - outputs(channel, batchPos))
                            * outputs(output, batchPos)
                            * softmax1.mDiffInputs(output, batchPos);
            }

            ASSERT_EQUALS_DELTA(
                diffOutputs(channel, batchPos), gradient, 1.0e-6);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// half
////////////////////////////////////////////////////////////////////////////////
TEST_DATASET(SoftmaxCell_Frame_half,
             propagate,
             (unsigned int nbOutputs, unsigned int batchSize),
             std::make_tuple(1U, 1U),
             std::make_tuple(10U, 1U),
             std::make_tuple(100U, 1U),
             std::make_tuple(1000U, 1U),
             std::make_tuple(1U, 9U),
             std::make_tuple(10U, 9U),
             std::make_tuple(100U, 9U),
             std::make_tuple(1000U, 9U))
{
    Network net;
    DeepNet dn(net);
    
    SoftmaxCell_Frame<half_float::half> softmax1(dn, "softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor<half_float::half> inputs({1, 1, nbOutputs, batchSize});
    Tensor<half_float::half> diffOutputs;
    softmax1.addInput(inputs, diffOutputs);

    inputs.fill(half_float::half(0.0f));
    softmax1.propagate();
    const Tensor<half_float::half>& outputs1
        = tensor_cast<half_float::half>(softmax1.getOutputs());

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs1(o), 1.0f / (double)nbOutputs, 1.0e-4);
    }

    inputs.fill(half_float::half(1.0f));
    softmax1.propagate();
    const Tensor<half_float::half>& outputs2
        = tensor_cast<half_float::half>(softmax1.getOutputs());

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs2(o), 1.0f / (double)nbOutputs, 1.0e-4);
    }

    inputs.fill(half_float::half(0.0f));

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs(0, batchPos) = 1.0f;

    softmax1.propagate();
    const Tensor<half_float::half>& outputs3
        = tensor_cast<half_float::half>(softmax1.getOutputs());

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        ASSERT_EQUALS_DELTA(outputs3(0, batchPos),
                            std::exp(1.0f) / (std::exp(1.0f) + (nbOutputs - 1)),
                            1.0e-3);

        for (unsigned int o = 1; o < nbOutputs; ++o) {
            ASSERT_EQUALS_DELTA(outputs3(o, batchPos),
                                1.0f / (std::exp(1.0f) + (nbOutputs - 1)),
                                1.0e-3);
        }
    }
}

TEST_DATASET(SoftmaxCell_Frame_half,
             backPropagate,
             (unsigned int nbOutputs, unsigned int batchSize),
             std::make_tuple(1U, 1U),
             std::make_tuple(10U, 1U),
             std::make_tuple(100U, 1U),
             std::make_tuple(1000U, 1U),
             std::make_tuple(1U, 9U),
             std::make_tuple(10U, 9U),
             std::make_tuple(100U, 9U),
             std::make_tuple(1000U, 9U))
{
    Network net;
    DeepNet dn(net);
    
    SoftmaxCell_Frame_Test<half_float::half> softmax1(dn, "softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor<half_float::half> inputs({1, 1, nbOutputs, batchSize});
    Tensor<half_float::half> diffOutputs({1, 1, nbOutputs, batchSize});
    softmax1.addInput(inputs, diffOutputs);

    ASSERT_NOTHROW_ANY(softmax1.checkGradient(1.0, 1.0));
    diffOutputs.clearValid();

    inputs.fill(half_float::half(0.0f));

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs(0, batchPos) = 1.0f;

    softmax1.propagate();
    softmax1.mDiffInputs.fill(half_float::half(0.0f));

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        softmax1.mDiffInputs(nbOutputs - 1, batchPos) = 1.0;

    softmax1.backPropagate();
    const Tensor<half_float::half>& outputs
        = tensor_cast<half_float::half>(softmax1.getOutputs());

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        for (unsigned int channel = 0; channel < nbOutputs; ++channel) {
            half_float::half gradient(0.0f);

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                gradient += ((output == channel) - outputs(channel, batchPos))
                            * outputs(output, batchPos)
                            * softmax1.mDiffInputs(output, batchPos);
            }

            ASSERT_EQUALS_DELTA(
                diffOutputs(channel, batchPos), gradient, 1.0e-3);
        }
    }
}

RUN_TESTS()
