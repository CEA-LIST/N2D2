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

#include "Cell/ConvCell_Frame.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "Environment.hpp"
#include "Network.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class ConvCell_Frame_Test : public ConvCell_Frame {
public:
    ConvCell_Frame_Test(const std::string& name,
                        unsigned int kernelWidth,
                        unsigned int kernelHeight,
                        unsigned int nbOutputs,
                        unsigned int subSampleX,
                        unsigned int subSampleY,
                        unsigned int strideX,
                        unsigned int strideY,
                        unsigned int paddingX,
                        unsigned int paddingY,
                        const std::shared_ptr<Activation<Float_T> >& activation)
        : Cell(name, nbOutputs),
          ConvCell(name,
                   kernelWidth,
                   kernelHeight,
                   nbOutputs,
                   subSampleX,
                   subSampleY,
                   strideX,
                   strideY,
                   paddingX,
                   paddingY),
          ConvCell_Frame(name,
                         kernelWidth,
                         kernelHeight,
                         nbOutputs,
                         subSampleX,
                         subSampleY,
                         strideX,
                         strideY,
                         paddingX,
                         paddingY,
                         activation) {};

    friend class UnitTest_ConvCell_Frame_addInput__env;
    friend class UnitTest_ConvCell_Frame_addInput;
    friend class UnitTest_ConvCell_Frame_propagate_input_check;
    friend class UnitTest_ConvCell_Frame_propagate_2_input_check;
    friend class UnitTest_ConvCell_Frame_setWeight;
};

TEST_DATASET(ConvCell_Frame,
             ConvCell_Frame,
             (unsigned int kernelWidth,
              unsigned int kernelHeight,
              unsigned int nbOutputs,
              unsigned int subSampleX,
              unsigned int subSampleY,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 1U, 0U, 0U),
             // 1
             std::make_tuple(2U, 5U, 2U, 1U, 1U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 3U, 2U, 2U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 4U, 1U, 3U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 5U, 1U, 1U, 2U, 2U, 0U, 0U),
             std::make_tuple(3U, 3U, 6U, 1U, 1U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 7U, 1U, 1U, 1U, 1U, 2U, 2U),
             std::make_tuple(3U, 3U, 8U, 1U, 1U, 1U, 1U, 1U, 3U),
             // 2
             std::make_tuple(2U, 5U, 1U, 2U, 2U, 1U, 1U, 0U, 0U),
             std::make_tuple(2U, 5U, 2U, 1U, 3U, 1U, 1U, 0U, 0U),
             std::make_tuple(2U, 5U, 3U, 1U, 1U, 2U, 2U, 0U, 0U),
             std::make_tuple(2U, 5U, 4U, 1U, 1U, 1U, 3U, 0U, 0U),
             std::make_tuple(2U, 5U, 5U, 1U, 1U, 1U, 1U, 2U, 2U),
             std::make_tuple(2U, 5U, 6U, 1U, 1U, 1U, 1U, 1U, 3U),
             // 3
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 2U, 2U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 1U, 1U, 2U, 2U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 1U, 1U, 1U, 3U),
             // 4
             std::make_tuple(2U, 5U, 10U, 1U, 1U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 2U, 2U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 1U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 1U, 1U, 3U, 2U, 2U),
             std::make_tuple(3U, 3U, 10U, 1U, 1U, 1U, 3U, 1U, 3U))
{
    ConvCell_Frame conv1("conv1",
                         kernelWidth,
                         kernelHeight,
                         nbOutputs,
                         subSampleX,
                         subSampleY,
                         strideX,
                         strideY,
                         paddingX,
                         paddingY,
                         std::make_shared<TanhActivation_Frame<Float_T> >());

    ASSERT_EQUALS(conv1.getName(), "conv1");
    ASSERT_EQUALS(conv1.getKernelWidth(), kernelWidth);
    ASSERT_EQUALS(conv1.getKernelHeight(), kernelHeight);
    ASSERT_EQUALS(conv1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(conv1.getStrideX(), strideX);
    ASSERT_EQUALS(conv1.getStrideY(), strideY);
}

TEST_DATASET(ConvCell_Frame,
             addInput__env,
             (unsigned int kernelWidth,
              unsigned int kernelHeight,
              unsigned int subSampleX,
              unsigned int subSampleY,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 2U, 2U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    const unsigned int nbOutputs = 10;

    Network net;
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    ConvCell_Frame_Test conv1("conv1",
                              kernelWidth,
                              kernelHeight,
                              nbOutputs,
                              subSampleX,
                              subSampleY,
                              strideX,
                              strideY,
                              paddingX,
                              paddingY,
                              std::make_shared
                              <TanhActivation_Frame<Float_T> >());
    conv1.setParameter("NoBias", true);
    conv1.addInput(env);
    conv1.initialize();

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        ASSERT_EQUALS(conv1.isConnection(0, output), true);
    }

    // const unsigned int oxSize = ((channelsWidth + 2*paddingX - kernelWidth +
    // strideX)/(double) strideX);
    // const unsigned int oySize = ((channelsHeight + 2*paddingY - kernelHeight
    // + strideY)/(double) strideY);
    const unsigned int outputsWidth = std::ceil(
        std::floor((channelsWidth + 2 * paddingX - kernelWidth + strideX)
                   / (double)strideX) / (double)subSampleX);
    const unsigned int outputsHeight = std::ceil(
        std::floor((channelsHeight + 2 * paddingY - kernelHeight + strideY)
                   / (double)strideY) / (double)subSampleY);

    ASSERT_EQUALS(conv1.getNbSharedSynapses(),
                  kernelWidth * kernelHeight * nbOutputs);
    ASSERT_EQUALS(conv1.getNbChannels(), 1U);
    ASSERT_EQUALS(conv1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(conv1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(conv1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(conv1.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(conv1.getOutputsHeight(), outputsHeight);
    // ASSERT_NOTHROW_ANY(conv1.checkGradient(1.0e-3, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(conv1.mInputs.dataSize(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(conv1.mOutputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(conv1.mDiffInputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(conv1.mDiffOutputs.dataSize(), 0U);
}

TEST_DATASET(ConvCell_Frame,
             addInput,
             (unsigned int kernelWidth,
              unsigned int kernelHeight,
              unsigned int subSampleX,
              unsigned int subSampleY,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 2U, 2U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    const unsigned int nbOutputs = 10;

    Network net;
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    ConvCell_Frame_Test conv1("conv1",
                              4,
                              4,
                              16,
                              1,
                              1,
                              2,
                              2,
                              0,
                              0,
                              std::make_shared
                              <TanhActivation_Frame<Float_T> >());
    ConvCell_Frame_Test conv2("conv2",
                              kernelWidth,
                              kernelHeight,
                              nbOutputs,
                              subSampleX,
                              subSampleY,
                              strideX,
                              strideY,
                              paddingX,
                              paddingY,
                              std::make_shared
                              <TanhActivation_Frame<Float_T> >());

    conv1.addInput(env);
    conv2.addInput(&conv1);
    conv1.initialize();
    conv2.initialize();

    for (unsigned int channel = 0; channel < 16; ++channel) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            ASSERT_EQUALS(conv2.isConnection(channel, output), true);
        }
    }

    const unsigned int outputsWidth = std::ceil(
        std::floor((conv1.getOutputsWidth() + 2 * paddingX - kernelWidth
                    + strideX) / (double)strideX) / (double)subSampleX);
    const unsigned int outputsHeight = std::ceil(
        std::floor((conv1.getOutputsHeight() + 2 * paddingY - kernelHeight
                    + strideY) / (double)strideY) / (double)subSampleY);

    ASSERT_EQUALS(conv2.getNbSharedSynapses(),
                  kernelWidth * kernelHeight * 16U * nbOutputs);
    ASSERT_EQUALS(conv2.getNbChannels(), 16U);
    ASSERT_EQUALS(conv2.getChannelsWidth(), conv1.getOutputsWidth());
    ASSERT_EQUALS(conv2.getChannelsHeight(), conv1.getOutputsHeight());
    ASSERT_EQUALS(conv2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(conv2.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(conv2.getOutputsHeight(), outputsHeight);
    // ASSERT_NOTHROW_ANY(conv2.checkGradient(1.0e-3, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(conv2.mInputs.dataSize(),
                  16U * conv1.getOutputsWidth() * conv1.getOutputsHeight());
    ASSERT_EQUALS(conv2.mOutputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(conv2.mDiffInputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(conv2.mDiffOutputs.dataSize(),
                  16U * conv1.getOutputsWidth() * conv1.getOutputsHeight());
}

TEST_DATASET(ConvCell_Frame,
             propagate_input_check,
             (unsigned int kernelWidth,
              unsigned int kernelHeight,
              unsigned int subSampleX,
              unsigned int subSampleY,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs = 5;

    Network net;

    ConvCell_Frame_Test conv1("conv1",
                              kernelWidth,
                              kernelHeight,
                              nbOutputs,
                              subSampleX,
                              subSampleY,
                              strideX,
                              strideY,
                              paddingX,
                              paddingY,
                              std::shared_ptr<Activation<Float_T> >());
    conv1.setParameter("NoBias", true);

    MNIST_IDX_Database database;
    database.load(N2D2_DATA("mnist"));

    Environment env(net, database, channelsWidth, channelsHeight, 1, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor4d<Float_T>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    conv1.addInput(env);
    conv1.initialize();

    ASSERT_EQUALS(conv1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(conv1.getNbChannels(), 1U);

    for (unsigned int output = 0; output < conv1.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < conv1.getNbChannels();
             ++channel) {
            for (unsigned int sx = 0; sx < conv1.getKernelWidth(); ++sx) {
                for (unsigned int sy = 0; sy < conv1.getKernelHeight(); ++sy)
                    conv1.setWeight(output,
                                    channel,
                                    sx,
                                    sy,
                                    1.0 + channel + conv1.getNbChannels()
                                                    * output);
            }
        }
    }

    conv1.propagate();

    const Tensor4d<Float_T>& out = conv1.getOutputs();

    for (unsigned int batch = 0; batch < 2; ++batch) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < conv1.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < conv1.getOutputsWidth(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        (int)paddingX - (int)(ox * strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        (int)paddingY - (int)(oy * strideY), 0);
                    const unsigned int sxMax = Utils::clamp<int>(
                        conv1.getChannelsWidth() + paddingX - ox * strideX,
                        0,
                        kernelWidth);
                    const unsigned int syMax = Utils::clamp<int>(
                        conv1.getChannelsHeight() + paddingY - oy * strideY,
                        0,
                        kernelHeight);

                    Float_T sum = 0.0;

                    for (unsigned int channel = 0;
                         channel < conv1.getNbChannels();
                         ++channel) {
                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                const unsigned int ix = (int)(ox * strideX + sx)
                                                        - (int)paddingX;
                                const unsigned int iy = (int)(oy * strideY + sy)
                                                        - (int)paddingY;

                                sum += in(ix, iy, channel, batch)
                                       * (1.0f + channel + conv1.getNbChannels()
                                                           * output);
                            }
                        }
                    }

                    ASSERT_EQUALS_DELTA(out(ox, oy, output, batch), sum, 1e-12);
                }
            }
        }
    }
}

TEST_DATASET(ConvCell_Frame,
             propagate_2_input_check,
             (unsigned int kernelWidth,
              unsigned int kernelHeight,
              unsigned int subSampleX,
              unsigned int subSampleY,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs = 5;

    Network net;

    ConvCell_Frame_Test conv1("conv1",
                              kernelWidth,
                              kernelHeight,
                              nbOutputs,
                              subSampleX,
                              subSampleY,
                              strideX,
                              strideY,
                              paddingX,
                              paddingY,
                              std::shared_ptr<Activation<Float_T> >());
    conv1.setParameter("NoBias", true);

    MNIST_IDX_Database database;
    database.load(N2D2_DATA("mnist"));

    Environment env(net, database, channelsWidth, channelsHeight, 1, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor4d<Float_T>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    conv1.addInput(env);
    conv1.addInput(env);
    conv1.initialize();

    ASSERT_EQUALS(conv1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(conv1.getNbChannels(), 2U);
    // ASSERT_NOTHROW_ANY(conv1.checkGradient(1.0e-3, 1.0e-3));

    for (unsigned int output = 0; output < conv1.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < conv1.getNbChannels();
             ++channel) {
            for (unsigned int sx = 0; sx < conv1.getKernelWidth(); ++sx) {
                for (unsigned int sy = 0; sy < conv1.getKernelHeight(); ++sy)
                    conv1.setWeight(output,
                                    channel,
                                    sx,
                                    sy,
                                    1.0 + channel + conv1.getNbChannels()
                                                    * output);
            }
        }
    }

    conv1.propagate();

    const Tensor4d<Float_T>& out = conv1.getOutputs();

    for (unsigned int batch = 0; batch < 2; ++batch) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < conv1.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < conv1.getOutputsWidth(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        (int)paddingX - (int)(ox * strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        (int)paddingY - (int)(oy * strideY), 0);
                    const unsigned int sxMax = Utils::clamp<int>(
                        conv1.getChannelsWidth() + paddingX - ox * strideX,
                        0,
                        kernelWidth);
                    const unsigned int syMax = Utils::clamp<int>(
                        conv1.getChannelsHeight() + paddingY - oy * strideY,
                        0,
                        kernelHeight);

                    Float_T sum = 0.0;

                    for (unsigned int channel = 0;
                         channel < conv1.getNbChannels();
                         ++channel) {
                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                const unsigned int ix = (int)(ox * strideX + sx)
                                                        - (int)paddingX;
                                const unsigned int iy = (int)(oy * strideY + sy)
                                                        - (int)paddingY;

                                sum += in(ix, iy, 0, batch)
                                       * (1.0f + channel + conv1.getNbChannels()
                                                           * output);
                            }
                        }
                    }

                    ASSERT_EQUALS_DELTA(out(ox, oy, output, batch), sum, 1e-4);
                }
            }
        }
    }
}

TEST_DATASET(ConvCell_Frame,
             setWeight,
             (unsigned int kernelWidth,
              unsigned int kernelHeight,
              unsigned int subSampleX,
              unsigned int subSampleY,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 2U, 2U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    const unsigned int nbOutputs = 10;

    Network net;
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    ConvCell_Frame_Test conv1("conv1",
                              kernelWidth,
                              kernelHeight,
                              nbOutputs,
                              subSampleX,
                              subSampleY,
                              strideX,
                              strideY,
                              paddingX,
                              paddingY,
                              std::make_shared
                              <TanhActivation_Frame<Float_T> >());
    conv1.setParameter("NoBias", true);
    conv1.addInput(env);
    conv1.initialize();

    for (unsigned int output = 0; output < conv1.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < conv1.getNbChannels();
             ++channel) {
            for (unsigned int sx = 0; sx < conv1.getKernelWidth(); ++sx) {
                for (unsigned int sy = 0; sy < conv1.getKernelHeight(); ++sy)
                    conv1.setWeight(
                        output, channel, sx, sy, output + channel + sx + sy);
            }
        }
    }

    for (unsigned int output = 0; output < conv1.getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < conv1.getNbChannels();
             ++channel) {
            for (unsigned int sx = 0; sx < conv1.getKernelWidth(); ++sx) {
                for (unsigned int sy = 0; sy < conv1.getKernelHeight(); ++sy) {
                    ASSERT_EQUALS(conv1.getWeight(output, channel, sx, sy),
                                  output + channel + sx + sy);
                }
            }
        }
    }
}

RUN_TESTS()
