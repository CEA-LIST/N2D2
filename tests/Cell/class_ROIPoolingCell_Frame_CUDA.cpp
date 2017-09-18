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

#include "Database/MNIST_IDX_Database.hpp"
#include "Environment.hpp"
#include "Network.hpp"
#include "Cell/ConvCell_Frame_CUDA.hpp"
#include "Cell/ROIPoolingCell_Frame_CUDA.hpp"
#include "Transformation/ColorSpaceTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class ROIPoolingCell_Frame_CUDA_Test : public ROIPoolingCell_Frame_CUDA {
public:
    ROIPoolingCell_Frame_CUDA_Test(const std::string& name,
                              StimuliProvider& sp,
                              unsigned int outputsWidth,
                              unsigned int outputsHeight,
                              unsigned int nbOutputs,
                              ROIPooling pooling)
        : Cell(name, nbOutputs),
          ROIPoolingCell(name,
                         sp,
                         outputsWidth,
                         outputsHeight,
                         nbOutputs,
                         pooling),
          ROIPoolingCell_Frame_CUDA(name,
                               sp,
                               outputsWidth,
                               outputsHeight,
                               nbOutputs,
                               pooling) {};

    friend class UnitTest_ROIPoolingCell_Frame_CUDA_addInput__env;
    friend class UnitTest_ROIPoolingCell_Frame_CUDA_addInput;
    friend class UnitTest_ROIPoolingCell_Frame_CUDA_propagate_input_check;
};

TEST_DATASET(ROIPoolingCell_Frame_CUDA,
             ROIPoolingCell_Frame_CUDA,
             (unsigned int outputsWidth,
              unsigned int outputsHeight,
              unsigned int nbOutputs),
             std::make_tuple(7U, 7U, 10U),
             std::make_tuple(10U, 7U, 1U),
             std::make_tuple(7U, 10U, 2U))
{
    StimuliProvider sp(EmptyDatabase, 16, 16, 1, 5);
    ROIPoolingCell_Frame_CUDA pool1("pool1",
                               sp,
                               outputsWidth,
                               outputsHeight,
                               nbOutputs,
                               ROIPoolingCell::Max);

    ASSERT_EQUALS(pool1.getName(), "pool1");
    ASSERT_EQUALS(pool1.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(pool1.getOutputsHeight(), outputsHeight);
    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
}

TEST_DATASET(ROIPoolingCell_Frame_CUDA,
             addInput__env,
             (unsigned int outputsWidth,
              unsigned int outputsHeight,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(7U, 7U, 24U, 24U),
             std::make_tuple(10U, 7U, 24U, 24U),
             std::make_tuple(7U, 10U, 24U, 24U),
             std::make_tuple(7U, 7U, 32U, 24U),
             std::make_tuple(10U, 7U, 32U, 24U),
             std::make_tuple(7U, 10U, 32U, 24U),
             std::make_tuple(7U, 7U, 24U, 32U),
             std::make_tuple(10U, 7U, 24U, 32U),
             std::make_tuple(7U, 10U, 24U, 32U))
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    const unsigned int nbProposals = 2;

    Network net;
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    ROIPoolingCell_Frame_CUDA_Test pool1("pool1",
                                    env,
                                    outputsWidth,
                                    outputsHeight,
                                    1,
                                    ROIPoolingCell::Max);
    Tensor4d<Float_T> proposals(1, 1, 4, nbProposals);
    Tensor4d<Float_T> proposalsDiff;

    pool1.addInput(proposals, proposalsDiff);
    pool1.addInput(env);
    pool1.initialize();

    ASSERT_EQUALS(pool1.getNbChannels(), 4U + 1U);
    ASSERT_EQUALS(pool1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(pool1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(pool1.getNbOutputs(), 1);
    ASSERT_EQUALS(pool1.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(pool1.getOutputsHeight(), outputsHeight);

    // Internal state testing
    ASSERT_EQUALS(pool1.mInputs.size(), 2U);
    ASSERT_EQUALS(pool1.mInputs[0].size(), 8U);
    ASSERT_EQUALS(pool1.mInputs[1].size(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(pool1.mOutputs.size(),
                  nbProposals * 1 * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool1.mDiffInputs.size(),
                  nbProposals * 1 * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool1.mDiffOutputs.dataSize(), 0U);
}

TEST_DATASET(ROIPoolingCell_Frame_CUDA,
             addInput,
             (unsigned int outputsWidth,
              unsigned int outputsHeight,
              unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(7U, 7U, 10U, 64U, 64U),
             std::make_tuple(10U, 7U, 1U, 64U, 64U),
             std::make_tuple(7U, 10U, 2U, 64U, 64U),
             std::make_tuple(7U, 7U, 10U, 96U, 64U),
             std::make_tuple(10U, 7U, 1U, 96U, 64U),
             std::make_tuple(7U, 10U, 2U, 96U, 64U),
             std::make_tuple(7U, 7U, 10U, 64U, 96U),
             std::make_tuple(10U, 7U, 1U, 64U, 96U),
             std::make_tuple(7U, 10U, 2U, 64U, 96U))
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    const unsigned int nbProposals = 2;

    Network net;
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    ConvCell_Frame_CUDA conv1("conv1",
                              1,
                              1,
                              nbOutputs,
                              1,
                              1,
                              1,
                              1,
                              0,
                              0,
                              std::make_shared
                              <TanhActivation_Frame<Float_T> >());
    ROIPoolingCell_Frame_CUDA_Test pool2("pool2",
                                    env,
                                    outputsWidth,
                                    outputsHeight,
                                    nbOutputs,
                                    ROIPoolingCell::Max);

    CudaTensor4d<Float_T> proposals(1, 1, 4, nbProposals);
    CudaTensor4d<Float_T> proposalsDiff(1, 1, 4, nbProposals);

    proposals(0, 0) = 11;
    proposals(1, 0) = 27;
    proposals(2, 0) = 2 * outputsWidth;
    proposals(3, 0) = 2 * outputsHeight;
    proposals(0, 1) = 4;
    proposals(1, 1) = 1;
    proposals(2, 1) = 4 * outputsWidth;
    proposals(3, 1) = 3 * outputsHeight;
    proposals.synchronizeHToD();

    conv1.addInput(env);
    pool2.addInput(proposals, proposalsDiff);
    pool2.addInput(&conv1);
    conv1.initialize();
    pool2.initialize();

    ASSERT_EQUALS(pool2.getNbChannels(), 4U + nbOutputs);
    ASSERT_EQUALS(pool2.getChannelsWidth(), conv1.getOutputsWidth());
    ASSERT_EQUALS(pool2.getChannelsHeight(), conv1.getOutputsHeight());
    ASSERT_EQUALS(pool2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool2.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(pool2.getOutputsHeight(), outputsHeight);
    //pool2.checkGradient(1.0e-5, 1.0e-2);

    // Internal state testing
    ASSERT_EQUALS(pool2.mInputs.size(), 2U);
    ASSERT_EQUALS(pool2.mInputs[0].size(), 8U);
    ASSERT_EQUALS(pool2.mInputs[1].size(), nbOutputs
                    * conv1.getOutputsWidth() * conv1.getOutputsHeight());
    ASSERT_EQUALS(pool2.mOutputs.size(),
                  nbProposals * nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool2.mDiffInputs.size(),
                  nbProposals * nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool2.mDiffOutputs.size(), 2U);
    ASSERT_EQUALS(pool2.mDiffOutputs[0].size(), 8U);
    ASSERT_EQUALS(pool2.mDiffOutputs[1].size(), nbOutputs
                    * conv1.getOutputsWidth() * conv1.getOutputsHeight());
}

TEST_DATASET(ROIPoolingCell_Frame_CUDA,
             propagate_input_check,
             (unsigned int outputsWidth,
              unsigned int outputsHeight,
              unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(7U, 7U, 10U, 64U, 64U),
             std::make_tuple(10U, 7U, 1U, 64U, 64U),
             std::make_tuple(7U, 10U, 2U, 64U, 64U),
             std::make_tuple(7U, 7U, 10U, 96U, 64U),
             std::make_tuple(10U, 7U, 1U, 96U, 64U),
             std::make_tuple(7U, 10U, 2U, 96U, 64U),
             std::make_tuple(7U, 7U, 10U, 64U, 96U),
             std::make_tuple(10U, 7U, 1U, 64U, 96U),
             std::make_tuple(7U, 10U, 2U, 64U, 96U))
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    const unsigned int nbProposals = 2;

    Network net;
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    ROIPoolingCell_Frame_CUDA_Test pool1("pool1",
                                    env,
                                    outputsWidth,
                                    outputsHeight,
                                    nbOutputs,
                                    ROIPoolingCell::Max);

    CudaTensor4d<Float_T> proposals(1, 1, 4, nbProposals);
    CudaTensor4d<Float_T> proposalsDiff(1, 1, 4, nbProposals);
    CudaTensor4d<Float_T> inputs(channelsWidth, channelsHeight, nbOutputs, 1);
    CudaTensor4d<Float_T> inputsDiff(channelsWidth, channelsHeight, nbOutputs, 1);

    proposals(0, 0) = 11;
    proposals(1, 0) = 27;
    proposals(2, 0) = 2 * outputsWidth;
    proposals(3, 0) = 2 * outputsHeight;
    proposals(0, 1) = 4;
    proposals(1, 1) = 1;
    proposals(2, 1) = 4 * outputsWidth;
    proposals(3, 1) = 3 * outputsHeight;
    proposals.synchronizeHToD();

    for (unsigned int index = 0; index < inputs.size(); ++index)
        inputs(index) = Random::randUniform(-1.0, 1.0);

    inputs.synchronizeHToD();

    pool1.addInput(proposals, proposalsDiff);
    pool1.addInput(inputs, inputsDiff);
    pool1.initialize();

    pool1.propagate();

    const Tensor4d<Float_T>& out = pool1.getOutputs();

    for (unsigned int batch = 0; batch < nbProposals; ++batch) {
        const unsigned int poolWidth = Utils::round(proposals(2, batch)
                                                    / outputsWidth);
        const unsigned int poolHeight = Utils::round(proposals(3, batch)
                                                     / outputsHeight);

        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < pool1.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < pool1.getOutputsWidth(); ++ox) {
                    std::vector<Float_T> poolElem;

                    for (unsigned int y = 0; y < poolHeight; ++y) {
                        for (unsigned int x = 0; x < poolWidth; ++x) {
                            poolElem.push_back(inputs(
                                proposals(0, batch) + poolWidth * ox + x,
                                proposals(1, batch) + poolHeight * oy + y,
                                output, 0));
                        }
                    }

                    const Float_T poolValue
                        = *std::max_element(poolElem.begin(), poolElem.end());

                    ASSERT_EQUALS_DELTA(
                        out(ox, oy, output, batch), poolValue, 1e-12);
                }
            }
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
