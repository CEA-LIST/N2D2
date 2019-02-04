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

#include <tuple>
#include "N2D2.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "Environment.hpp"
#include "Network.hpp"
#include "Cell/PoolCell_Frame.hpp"
#include "third_party/half.hpp"
#include "Transformation/ColorSpaceTransformation.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

template <class T>
class PoolCell_Frame_Test : public PoolCell_Frame<T> {
public:
    PoolCell_Frame_Test(const std::string& name,
                        const std::vector<unsigned int>& poolDims,
                        unsigned int nbOutputs,
                        const std::vector<unsigned int>& strideDims,
                        const std::vector<unsigned int>& paddingDims,
                        PoolCell::Pooling pooling)
        : Cell(name, nbOutputs),
          PoolCell(name,
                   poolDims,
                   nbOutputs,
                   strideDims,
                   paddingDims,
                   pooling),
          PoolCell_Frame<T>(name,
                         poolDims,
                         nbOutputs,
                         strideDims,
                         paddingDims,
                         pooling) {};

    friend class UnitTest_PoolCell_Frame_addInput__env_float;
    friend class UnitTest_PoolCell_Frame_addInput_float;
    friend class UnitTest_PoolCell_Frame_propagate_input_check_float;
    friend class UnitTest_PoolCell_Frame_propagate_2_input_check_float;

    friend class UnitTest_PoolCell_Frame_addInput__env_half;
    friend class UnitTest_PoolCell_Frame_addInput_half;
    friend class UnitTest_PoolCell_Frame_propagate_input_check_half;
    friend class UnitTest_PoolCell_Frame_propagate_2_input_check_half;

    friend class UnitTest_PoolCell_Frame_addInput__env_double;
    friend class UnitTest_PoolCell_Frame_addInput_double;
    friend class UnitTest_PoolCell_Frame_propagate_input_check_double;
    friend class UnitTest_PoolCell_Frame_propagate_2_input_check_double;


};

static MNIST_IDX_Database database(N2D2_DATA("mnist"));

TEST_DATASET(PoolCell_Frame,
             PoolCell_Frame,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int nbOutputs,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 0U, 0U),
             // 1
             std::make_tuple(2U, 5U, 2U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 3U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 4U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 5U, 2U, 2U, 0U, 0U),
             std::make_tuple(3U, 3U, 6U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 7U, 1U, 1U, 2U, 2U),
             std::make_tuple(3U, 3U, 8U, 1U, 1U, 1U, 3U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 0U, 0U),
             std::make_tuple(2U, 5U, 2U, 1U, 1U, 0U, 0U),
             std::make_tuple(2U, 5U, 3U, 2U, 2U, 0U, 0U),
             std::make_tuple(2U, 5U, 4U, 1U, 3U, 0U, 0U),
             std::make_tuple(2U, 5U, 5U, 1U, 1U, 2U, 2U),
             std::make_tuple(2U, 5U, 6U, 1U, 1U, 1U, 3U),
             // 3
             std::make_tuple(3U, 3U, 10U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 1U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 2U, 2U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 1U, 2U, 2U),
             std::make_tuple(3U, 3U, 10U, 1U, 1U, 1U, 3U),
             // 4
             std::make_tuple(2U, 5U, 10U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 0U, 0U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 2U, 2U),
             std::make_tuple(3U, 3U, 10U, 1U, 3U, 1U, 3U))
{
    PoolCell_Frame<float> pool1("pool1",
                         std::vector<unsigned int>({poolWidth, poolHeight}),
                         nbOutputs,
                         std::vector<unsigned int>({strideX, strideY}),
                         std::vector<unsigned int>({paddingX, paddingY}),
                         PoolCell::Max);

    ASSERT_EQUALS(pool1.getName(), "pool1");
    ASSERT_EQUALS(pool1.getPoolWidth(), poolWidth);
    ASSERT_EQUALS(pool1.getPoolHeight(), poolHeight);
    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getStrideX(), strideX);
    ASSERT_EQUALS(pool1.getStrideY(), strideY);
}

TEST_DATASET(PoolCell_Frame,
             addInput__env_float,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    const unsigned int nbOutputs = 10;

    Network net;
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    PoolCell_Frame_Test<float> pool1("pool1",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);
    pool1.addInput(env);
    pool1.initialize();

    const unsigned int outputsWidth = (unsigned int)((channelsWidth
        + 2 * paddingX - poolWidth + strideX) / (double)strideX);
    const unsigned int outputsHeight = (unsigned int)((channelsHeight
        + 2 * paddingY - poolHeight + strideY) / (double)strideY);

    ASSERT_EQUALS(pool1.getNbChannels(), 1U);
    ASSERT_EQUALS(pool1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(pool1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(pool1.getOutputsHeight(), outputsHeight);

    // Internal state testing
    ASSERT_EQUALS(pool1.mInputs.dataSize(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(pool1.mOutputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool1.mDiffInputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool1.mDiffOutputs.dataSize(), 0U);
}

TEST_DATASET(PoolCell_Frame,
             addInput_float,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    const unsigned int nbOutputs = 10;

    Network net;
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    PoolCell_Frame_Test<float> pool1("pool1",
                              std::vector<unsigned int>({4, 4}),
                              16,
                              std::vector<unsigned int>({2, 2}),
                              std::vector<unsigned int>({0, 0}),
                              PoolCell::Max);
    PoolCell_Frame_Test<float> pool2("pool2",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);

    pool1.addInput(env);
    pool2.addInput(&pool1);
    pool1.initialize();
    pool2.initialize();

    const unsigned int outputsWidth = (unsigned int)((pool1.getOutputsWidth()
        + 2 * paddingX - poolWidth + strideX) / (double)strideX);
    const unsigned int outputsHeight = (unsigned int)((pool1.getOutputsHeight()
        + 2 * paddingY - poolHeight + strideY) / (double)strideY);

    ASSERT_EQUALS(pool2.getNbChannels(), 16U);
    ASSERT_EQUALS(pool2.getChannelsWidth(), pool1.getOutputsWidth());
    ASSERT_EQUALS(pool2.getChannelsHeight(), pool1.getOutputsHeight());
    ASSERT_EQUALS(pool2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool2.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(pool2.getOutputsHeight(), outputsHeight);
    //ASSERT_NOTHROW_ANY(pool1.checkGradient(1.0e-4, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(pool2.mInputs.dataSize(),
                  16U * pool1.getOutputsWidth() * pool1.getOutputsHeight());
    ASSERT_EQUALS(pool2.mOutputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool2.mDiffInputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool2.mDiffOutputs.dataSize(),
                  16U * pool1.getOutputsWidth() * pool1.getOutputsHeight());
}

TEST_DATASET(PoolCell_Frame,
             propagate_input_check_float,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs = 3;

    Network net;

    PoolCell_Frame_Test<float> pool1("pool1",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);


    Environment env(net, database, {channelsWidth, channelsHeight, 3}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.addTransformation(
        ColorSpaceTransformation(ColorSpaceTransformation::BGR));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<float>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 3U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    Tensor<bool> mapping;
    mapping << "1 0 0\n"
               "0 1 0\n"
               "0 0 1";

    pool1.addInput(env, 0, 0, 0, 0, mapping);
    pool1.initialize();

    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getNbChannels(), 3U);

    pool1.propagate();

    const Tensor<float>& out = tensor_cast<float>(pool1.getOutputs());

    for (unsigned int batch = 0; batch < 2; ++batch) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < pool1.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < pool1.getOutputsWidth(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        (int)paddingX - (int)(ox * strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        (int)paddingY - (int)(oy * strideY), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(pool1.getChannelsWidth() + paddingX
                                - ox * strideX,
                              0,
                              poolWidth);
                    const unsigned int syMax = Utils::clamp
                        <int>(pool1.getChannelsHeight() + paddingY
                                - oy * strideY,
                              0,
                              poolHeight);

                    const int ix = (int)(ox * strideX) - (int)paddingX;
                    const int iy = (int)(oy * strideY) - (int)paddingY;

                    // For each output, compute the pool value
                    float poolValue = 0.0;

                    for (unsigned int channel = 0;
                         channel < pool1.getNbChannels();
                         ++channel) {
                        if (channel != output)
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                if (in(ix + sx, iy + sy, channel, batch)
                                    > poolValue)
                                {
                                    poolValue = in(ix + sx,
                                                   iy + sy,
                                                   channel,
                                                   batch);
                                }
                            }
                        }
                    }

                    ASSERT_EQUALS_DELTA(
                        out(ox, oy, output, batch), poolValue, 1e-12);
                }
            }
        }
    }
}

TEST_DATASET(PoolCell_Frame,
             propagate_2_input_check_float,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs = 6;

    Network net;

    PoolCell_Frame_Test<float> pool1("pool1",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);


    Environment env(net, database, {channelsWidth, channelsHeight, 3}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.addTransformation(
        ColorSpaceTransformation(ColorSpaceTransformation::BGR));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<float>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 3U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    Tensor<bool> mapping1;
    mapping1 << "1 0 0 0 0 0\n"
                "0 1 0 0 0 0\n"
                "0 0 1 0 0 0";

    Tensor<bool> mapping2;
    mapping2 << "0 0 0 1 0 0\n"
                "0 0 0 0 1 0\n"
                "0 0 0 0 0 1";

    pool1.addInput(env, 0, 0, 0, 0, mapping1);
    pool1.addInput(env, 0, 0, 0, 0, mapping2);
    pool1.initialize();

    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getNbChannels(), 6U);

    pool1.propagate();

    const Tensor<float>& out = tensor_cast<float>(pool1.getOutputs());

    for (unsigned int batch = 0; batch < 2; ++batch) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < pool1.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < pool1.getOutputsWidth(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        (int)paddingX - (int)(ox * strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        (int)paddingY - (int)(oy * strideY), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(pool1.getChannelsWidth() + paddingX
                                - ox * strideX,
                              0,
                              poolWidth);
                    const unsigned int syMax = Utils::clamp
                        <int>(pool1.getChannelsHeight() + paddingY
                                - oy * strideY,
                              0,
                              poolHeight);

                    const int ix = (int)(ox * strideX) - (int)paddingX;
                    const int iy = (int)(oy * strideY) - (int)paddingY;

                    // For each output, compute the pool value
                    float poolValue = 0.0;

                    for (unsigned int channel = 0;
                         channel < pool1.getNbChannels();
                         ++channel) {
                        if (channel != output)
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                if (in(ix + sx, iy + sy, channel % 3, batch)
                                    > poolValue)
                                {
                                    poolValue = in(ix + sx,
                                                   iy + sy,
                                                   channel % 3,
                                                   batch);
                                }
                            }
                        }
                    }

                    ASSERT_EQUALS_DELTA(
                        out(ox, oy, output, batch), poolValue, 1e-12);
                }
            }
        }
    }
}

TEST_DATASET(PoolCell_Frame,
             addInput__env_half,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    const unsigned int nbOutputs = 10;

    Network net;
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    PoolCell_Frame_Test<half_float::half> pool1("pool1",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);
    pool1.addInput(env);
    pool1.initialize();

    const unsigned int outputsWidth = (unsigned int)((channelsWidth
        + 2 * paddingX - poolWidth + strideX) / (double)strideX);
    const unsigned int outputsHeight = (unsigned int)((channelsHeight
        + 2 * paddingY - poolHeight + strideY) / (double)strideY);

    ASSERT_EQUALS(pool1.getNbChannels(), 1U);
    ASSERT_EQUALS(pool1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(pool1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(pool1.getOutputsHeight(), outputsHeight);

    // Internal state testing
    ASSERT_EQUALS(pool1.mInputs.dataSize(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(pool1.mOutputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool1.mDiffInputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool1.mDiffOutputs.dataSize(), 0U);
}

TEST_DATASET(PoolCell_Frame,
             addInput_half,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    const unsigned int nbOutputs = 10;

    Network net;
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    PoolCell_Frame_Test<half_float::half> pool1("pool1",
                              std::vector<unsigned int>({4, 4}),
                              16,
                              std::vector<unsigned int>({2, 2}),
                              std::vector<unsigned int>({0, 0}),
                              PoolCell::Max);
    PoolCell_Frame_Test<half_float::half> pool2("pool2",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);

    pool1.addInput(env);
    pool2.addInput(&pool1);
    pool1.initialize();
    pool2.initialize();

    const unsigned int outputsWidth = (unsigned int)((pool1.getOutputsWidth()
        + 2 * paddingX - poolWidth + strideX) / (double)strideX);
    const unsigned int outputsHeight = (unsigned int)((pool1.getOutputsHeight()
        + 2 * paddingY - poolHeight + strideY) / (double)strideY);

    ASSERT_EQUALS(pool2.getNbChannels(), 16U);
    ASSERT_EQUALS(pool2.getChannelsWidth(), pool1.getOutputsWidth());
    ASSERT_EQUALS(pool2.getChannelsHeight(), pool1.getOutputsHeight());
    ASSERT_EQUALS(pool2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool2.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(pool2.getOutputsHeight(), outputsHeight);
    //ASSERT_NOTHROW_ANY(pool2.checkGradient(1.0e-3, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(pool2.mInputs.dataSize(),
                  16U * pool1.getOutputsWidth() * pool1.getOutputsHeight());
    ASSERT_EQUALS(pool2.mOutputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool2.mDiffInputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool2.mDiffOutputs.dataSize(),
                  16U * pool1.getOutputsWidth() * pool1.getOutputsHeight());
}

TEST_DATASET(PoolCell_Frame,
             propagate_input_check_half,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs = 3;

    Network net;

    PoolCell_Frame_Test<half_float::half> pool1("pool1",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);

    Environment env(net, database, {channelsWidth, channelsHeight, 3}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.addTransformation(
        ColorSpaceTransformation(ColorSpaceTransformation::BGR));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<float>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 3U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    Tensor<bool> mapping;
    mapping << "1 0 0\n"
               "0 1 0\n"
               "0 0 1";

    pool1.addInput(env, 0, 0, 0, 0, mapping);
    pool1.initialize();

    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getNbChannels(), 3U);

    pool1.propagate();

    const Tensor<float>& out = tensor_cast<float>(pool1.getOutputs());

    for (unsigned int batch = 0; batch < 2; ++batch) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < pool1.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < pool1.getOutputsWidth(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        (int)paddingX - (int)(ox * strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        (int)paddingY - (int)(oy * strideY), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(pool1.getChannelsWidth() + paddingX
                                - ox * strideX,
                              0,
                              poolWidth);
                    const unsigned int syMax = Utils::clamp
                        <int>(pool1.getChannelsHeight() + paddingY
                                - oy * strideY,
                              0,
                              poolHeight);

                    const int ix = (int)(ox * strideX) - (int)paddingX;
                    const int iy = (int)(oy * strideY) - (int)paddingY;

                    // For each output, compute the pool value
                    half_float::half poolValue = half_float::half(0.0);

                    for (unsigned int channel = 0;
                         channel < pool1.getNbChannels();
                         ++channel) {
                        if (channel != output)
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                if (in(ix + sx, iy + sy, channel, batch)
                                    > poolValue)
                                {
                                    poolValue = in(ix + sx,
                                                   iy + sy,
                                                   channel,
                                                   batch);
                                }
                            }
                        }
                    }

                    ASSERT_EQUALS_DELTA(
                        out(ox, oy, output, batch), poolValue, 1e-12);
                }
            }
        }
    }
}

TEST_DATASET(PoolCell_Frame,
             propagate_2_input_check_half,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs = 6;

    Network net;

    PoolCell_Frame_Test<half_float::half> pool1("pool1",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);

    Environment env(net, database, {channelsWidth, channelsHeight, 3}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.addTransformation(
        ColorSpaceTransformation(ColorSpaceTransformation::BGR));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<float>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 3U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    Tensor<bool> mapping1;
    mapping1 << "1 0 0 0 0 0\n"
                "0 1 0 0 0 0\n"
                "0 0 1 0 0 0";

    Tensor<bool> mapping2;
    mapping2 << "0 0 0 1 0 0\n"
                "0 0 0 0 1 0\n"
                "0 0 0 0 0 1";

    pool1.addInput(env, 0, 0, 0, 0, mapping1);
    pool1.addInput(env, 0, 0, 0, 0, mapping2);
    pool1.initialize();

    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getNbChannels(), 6U);

    pool1.propagate();

    const Tensor<half_float::half>& out = tensor_cast<half_float::half>(pool1.getOutputs());

    for (unsigned int batch = 0; batch < 2; ++batch) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < pool1.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < pool1.getOutputsWidth(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        (int)paddingX - (int)(ox * strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        (int)paddingY - (int)(oy * strideY), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(pool1.getChannelsWidth() + paddingX
                                - ox * strideX,
                              0,
                              poolWidth);
                    const unsigned int syMax = Utils::clamp
                        <int>(pool1.getChannelsHeight() + paddingY
                                - oy * strideY,
                              0,
                              poolHeight);

                    const int ix = (int)(ox * strideX) - (int)paddingX;
                    const int iy = (int)(oy * strideY) - (int)paddingY;

                    // For each output, compute the pool value
                    half_float::half poolValue = half_float::half(0.0);

                    for (unsigned int channel = 0;
                         channel < pool1.getNbChannels();
                         ++channel) {
                        if (channel != output)
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                if (in(ix + sx, iy + sy, channel % 3, batch)
                                    > poolValue)
                                {
                                    poolValue = in(ix + sx,
                                                   iy + sy,
                                                   channel % 3,
                                                   batch);
                                }
                            }
                        }
                    }

                    ASSERT_EQUALS_DELTA(
                        out(ox, oy, output, batch), poolValue, 1e-12);
                }
            }
        }
    }
}



TEST_DATASET(PoolCell_Frame,
             addInput__env_double,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    const unsigned int nbOutputs = 10;

    Network net;
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    PoolCell_Frame_Test<double> pool1("pool1",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);
    pool1.addInput(env);
    pool1.initialize();

    const unsigned int outputsWidth = (unsigned int)((channelsWidth
        + 2 * paddingX - poolWidth + strideX) / (double)strideX);
    const unsigned int outputsHeight = (unsigned int)((channelsHeight
        + 2 * paddingY - poolHeight + strideY) / (double)strideY);

    ASSERT_EQUALS(pool1.getNbChannels(), 1U);
    ASSERT_EQUALS(pool1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(pool1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(pool1.getOutputsHeight(), outputsHeight);

    // Internal state testing
    ASSERT_EQUALS(pool1.mInputs.dataSize(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(pool1.mOutputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool1.mDiffInputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool1.mDiffOutputs.dataSize(), 0U);
}

TEST_DATASET(PoolCell_Frame,
             addInput_double,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    const unsigned int nbOutputs = 10;

    Network net;
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    PoolCell_Frame_Test<double> pool1("pool1",
                              std::vector<unsigned int>({4, 4}),
                              16,
                              std::vector<unsigned int>({2, 2}),
                              std::vector<unsigned int>({0, 0}),
                              PoolCell::Max);
    PoolCell_Frame_Test<double> pool2("pool2",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);

    pool1.addInput(env);
    pool2.addInput(&pool1);
    pool1.initialize();
    pool2.initialize();

    const unsigned int outputsWidth = (unsigned int)((pool1.getOutputsWidth()
        + 2 * paddingX - poolWidth + strideX) / (double)strideX);
    const unsigned int outputsHeight = (unsigned int)((pool1.getOutputsHeight()
        + 2 * paddingY - poolHeight + strideY) / (double)strideY);

    ASSERT_EQUALS(pool2.getNbChannels(), 16U);
    ASSERT_EQUALS(pool2.getChannelsWidth(), pool1.getOutputsWidth());
    ASSERT_EQUALS(pool2.getChannelsHeight(), pool1.getOutputsHeight());
    ASSERT_EQUALS(pool2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool2.getOutputsWidth(), outputsWidth);
    ASSERT_EQUALS(pool2.getOutputsHeight(), outputsHeight);
    //ASSERT_NOTHROW_ANY(pool1.checkGradient(1.0e-4, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(pool2.mInputs.dataSize(),
                  16U * pool1.getOutputsWidth() * pool1.getOutputsHeight());
    ASSERT_EQUALS(pool2.mOutputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool2.mDiffInputs.size(),
                  nbOutputs * outputsWidth * outputsHeight);
    ASSERT_EQUALS(pool2.mDiffOutputs.dataSize(),
                  16U * pool1.getOutputsWidth() * pool1.getOutputsHeight());
}

TEST_DATASET(PoolCell_Frame,
             propagate_input_check_double,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs = 3;

    Network net;

    PoolCell_Frame_Test<double> pool1("pool1",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);

    Environment env(net, database, {channelsWidth, channelsHeight, 3}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.addTransformation(
        ColorSpaceTransformation(ColorSpaceTransformation::BGR));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<float>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 3U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    Tensor<bool> mapping;
    mapping << "1 0 0\n"
               "0 1 0\n"
               "0 0 1";

    pool1.addInput(env, 0, 0, 0, 0, mapping);
    pool1.initialize();

    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getNbChannels(), 3U);

    pool1.propagate();

    const Tensor<float>& out = tensor_cast<float>(pool1.getOutputs());

    for (unsigned int batch = 0; batch < 2; ++batch) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < pool1.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < pool1.getOutputsWidth(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        (int)paddingX - (int)(ox * strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        (int)paddingY - (int)(oy * strideY), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(pool1.getChannelsWidth() + paddingX
                                - ox * strideX,
                              0,
                              poolWidth);
                    const unsigned int syMax = Utils::clamp
                        <int>(pool1.getChannelsHeight() + paddingY
                                - oy * strideY,
                              0,
                              poolHeight);

                    const int ix = (int)(ox * strideX) - (int)paddingX;
                    const int iy = (int)(oy * strideY) - (int)paddingY;

                    // For each output, compute the pool value
                    double poolValue = double(0.0);

                    for (unsigned int channel = 0;
                         channel < pool1.getNbChannels();
                         ++channel) {
                        if (channel != output)
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                if (in(ix + sx, iy + sy, channel, batch)
                                    > poolValue)
                                {
                                    poolValue = in(ix + sx,
                                                   iy + sy,
                                                   channel,
                                                   batch);
                                }
                            }
                        }
                    }

                    ASSERT_EQUALS_DELTA(
                        out(ox, oy, output, batch), poolValue, 1e-12);
                }
            }
        }
    }
}

TEST_DATASET(PoolCell_Frame,
             propagate_2_input_check_double,
             (unsigned int poolWidth,
              unsigned int poolHeight,
              unsigned int strideX,
              unsigned int strideY,
              unsigned int paddingX,
              unsigned int paddingY,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             // 1
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 24U, 24U),
             // 2
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(2U, 5U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 32U),
             std::make_tuple(2U, 5U, 1U, 1U, 2U, 2U, 24U, 24U),
             std::make_tuple(2U, 5U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 3
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 0U, 0U, 32U, 24U),
             std::make_tuple(3U, 3U, 2U, 2U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 1U, 2U, 2U, 24U, 32U),
             std::make_tuple(3U, 3U, 1U, 1U, 1U, 3U, 32U, 24U),
             // 4
             std::make_tuple(2U, 5U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 0U, 0U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 2U, 2U, 24U, 24U),
             std::make_tuple(3U, 3U, 1U, 3U, 1U, 3U, 24U, 24U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs = 6;

    Network net;

    PoolCell_Frame_Test<double> pool1("pool1",
                              std::vector<unsigned int>({poolWidth, poolHeight}),
                              nbOutputs,
                              std::vector<unsigned int>({strideX, strideY}),
                              std::vector<unsigned int>({paddingX, paddingY}),
                              PoolCell::Max);


    Environment env(net, database, {channelsWidth, channelsHeight, 3}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.addTransformation(
        ColorSpaceTransformation(ColorSpaceTransformation::BGR));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<float>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 3U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    Tensor<bool> mapping1;
    mapping1 << "1 0 0 0 0 0\n"
                "0 1 0 0 0 0\n"
                "0 0 1 0 0 0";

    Tensor<bool> mapping2;
    mapping2 << "0 0 0 1 0 0\n"
                "0 0 0 0 1 0\n"
                "0 0 0 0 0 1";

    pool1.addInput(env, 0, 0, 0, 0, mapping1);
    pool1.addInput(env, 0, 0, 0, 0, mapping2);
    pool1.initialize();

    ASSERT_EQUALS(pool1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(pool1.getNbChannels(), 6U);

    pool1.propagate();

    const Tensor<double>& out = tensor_cast<double>(pool1.getOutputs());

    for (unsigned int batch = 0; batch < 2; ++batch) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < pool1.getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < pool1.getOutputsWidth(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        (int)paddingX - (int)(ox * strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        (int)paddingY - (int)(oy * strideY), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(pool1.getChannelsWidth() + paddingX
                                - ox * strideX,
                              0,
                              poolWidth);
                    const unsigned int syMax = Utils::clamp
                        <int>(pool1.getChannelsHeight() + paddingY
                                - oy * strideY,
                              0,
                              poolHeight);

                    const int ix = (int)(ox * strideX) - (int)paddingX;
                    const int iy = (int)(oy * strideY) - (int)paddingY;

                    // For each output, compute the pool value
                    double poolValue = double(0.0);

                    for (unsigned int channel = 0;
                         channel < pool1.getNbChannels();
                         ++channel) {
                        if (channel != output)
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                if (in(ix + sx, iy + sy, channel % 3, batch)
                                    > poolValue)
                                {
                                    poolValue = in(ix + sx,
                                                   iy + sy,
                                                   channel % 3,
                                                   batch);
                                }
                            }
                        }
                    }

                    ASSERT_EQUALS_DELTA(
                        out(ox, oy, output, batch), poolValue, 1e-12);
                }
            }
        }
    }
}


RUN_TESTS()
