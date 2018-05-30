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
#include "utils/UnitTest.hpp"

using namespace N2D2;

class SoftmaxCell_Frame_Test : public SoftmaxCell_Frame {
public:
    SoftmaxCell_Frame_Test(const std::string& name, unsigned int nbOutputs)
        : Cell(name, nbOutputs),
          SoftmaxCell(name, nbOutputs),
          SoftmaxCell_Frame(name, nbOutputs) {};

    friend class UnitTest_SoftmaxCell_Frame_backPropagate;
};

TEST_DATASET(SoftmaxCell_Frame,
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
    SoftmaxCell_Frame softmax1("softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputs({1, 1, nbOutputs, batchSize});
    Tensor<Float_T> diffOutputs;
    softmax1.addInput(inputs, diffOutputs);

    inputs.fill(0.0);
    softmax1.propagate();
    const Tensor<Float_T>& outputs1 = softmax1.getOutputs();

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs1(o), 1.0 / (double)nbOutputs, 1.0e-6);
    }

    inputs.fill(1.0);
    softmax1.propagate();
    const Tensor<Float_T>& outputs2 = softmax1.getOutputs();

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs2(o), 1.0 / (double)nbOutputs, 1.0e-6);
    }

    inputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs[batchPos](0) = 1.0;

    softmax1.propagate();
    const Tensor<Float_T>& outputs3 = softmax1.getOutputs();

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        ASSERT_EQUALS_DELTA(outputs3[batchPos](0),
                            std::exp(1.0) / (std::exp(1.0) + (nbOutputs - 1)),
                            1.0e-6);

        for (unsigned int o = 1; o < nbOutputs; ++o) {
            ASSERT_EQUALS_DELTA(outputs3[batchPos](o),
                                1.0 / (std::exp(1.0) + (nbOutputs - 1)),
                                1.0e-6);
        }
    }
}

TEST_DATASET(SoftmaxCell_Frame,
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
    SoftmaxCell_Frame_Test softmax1("softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputs({1, 1, nbOutputs, batchSize});
    Tensor<Float_T> diffOutputs({1, 1, nbOutputs, batchSize});
    softmax1.addInput(inputs, diffOutputs);

    ASSERT_NOTHROW_ANY(softmax1.checkGradient(1.0e-3, 1.0e-3));

    inputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs[batchPos](0) = 1.0;

    softmax1.propagate();
    softmax1.mDiffInputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        softmax1.mDiffInputs[batchPos](nbOutputs - 1) = 1.0;

    softmax1.backPropagate();
    Tensor<Float_T>& outputs = softmax1.getOutputs();

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        for (unsigned int channel = 0; channel < nbOutputs; ++channel) {
            Float_T gradient = 0.0;

            for (unsigned int output = 0; output < nbOutputs; ++output) {
                gradient += ((output == channel) - outputs[batchPos](channel))
                            * outputs[batchPos](output)
                            * softmax1.mDiffInputs[batchPos](output);
            }

            ASSERT_EQUALS_DELTA(
                diffOutputs[batchPos](channel), gradient, 1.0e-6);
        }
    }
}

RUN_TESTS()
