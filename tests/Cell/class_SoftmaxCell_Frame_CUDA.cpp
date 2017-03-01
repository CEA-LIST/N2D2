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

#ifdef CUDA

#include "N2D2.hpp"

#include "Cell/SoftmaxCell_Frame_CUDA.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class SoftmaxCell_Frame_CUDA_Test : public SoftmaxCell_Frame_CUDA {
public:
    SoftmaxCell_Frame_CUDA_Test(const std::string& name, unsigned int nbOutputs)
        : Cell(name, nbOutputs),
          SoftmaxCell(name, nbOutputs),
          SoftmaxCell_Frame_CUDA(name, nbOutputs) {};

    friend class UnitTest_SoftmaxCell_Frame_CUDA_backPropagate;
};

TEST_DATASET(SoftmaxCell_Frame_CUDA,
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
    REQUIRED(UnitTest::CudaDeviceExists(3));

    SoftmaxCell_Frame_CUDA softmax1("softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputs(1, 1, nbOutputs, batchSize);
    Tensor4d<Float_T> diffOutputs;
    softmax1.addInput(inputs, diffOutputs);
    softmax1.initialize();

    inputs.fill(0.0);
    softmax1.propagate();
    const Tensor4d<Float_T>& outputs1 = softmax1.getOutputs();

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs1(o), 1.0 / (double)nbOutputs, 1.0e-6);
    }

    inputs.fill(1.0);
    softmax1.propagate();
    const Tensor4d<Float_T>& outputs2 = softmax1.getOutputs();

    for (unsigned int o = 0; o < nbOutputs; ++o) {
        ASSERT_EQUALS_DELTA(outputs2(o), 1.0 / (double)nbOutputs, 1.0e-6);
    }

    inputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs(0, batchPos) = 1.0;

    softmax1.propagate();
    const Tensor4d<Float_T>& outputs3 = softmax1.getOutputs();

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

TEST_DATASET(SoftmaxCell_Frame_CUDA,
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
    REQUIRED(UnitTest::CudaDeviceExists(3));

    SoftmaxCell_Frame_CUDA_Test softmax1("softmax1", nbOutputs);

    ASSERT_EQUALS(softmax1.getName(), "softmax1");
    ASSERT_EQUALS(softmax1.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputs(1, 1, nbOutputs, batchSize);
    Tensor4d<Float_T> diffOutputs(1, 1, nbOutputs, batchSize);
    softmax1.addInput(inputs, diffOutputs);
    softmax1.initialize();

    inputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        inputs(0, batchPos) = 1.0;

    softmax1.propagate();
    softmax1.mDiffInputs.fill(0.0);

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        softmax1.mDiffInputs(nbOutputs - 1, batchPos) = 1.0;

    softmax1.mDiffInputs.synchronizeHToD();

    softmax1.backPropagate();
    Tensor4d<Float_T>& outputs = softmax1.getOutputs();
    softmax1.mDiffOutputs.synchronizeDToH();

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        for (unsigned int channel = 0; channel < nbOutputs; ++channel) {
            Float_T gradient = 0.0;

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

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
