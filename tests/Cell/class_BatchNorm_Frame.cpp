/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Cell/BatchNormCell_Frame.hpp"
#include "Cell/ConvCell_Frame.hpp"
#include "Environment.hpp"
#include "Network.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class BatchNormCell_Frame_Test : public BatchNormCell_Frame {
public:
    BatchNormCell_Frame_Test(const std::string& name,
                             unsigned int nbOutputs,
                             const std::shared_ptr
                             <Activation<Float_T> >& activation)
        : Cell(name, nbOutputs),
          BatchNormCell(name, nbOutputs),
          BatchNormCell_Frame(name, nbOutputs, activation) {};

    friend class UnitTest_BatchNormCell_Frame_setScales;
    friend class UnitTest_BatchNormCell_Frame_addInput__env;
    friend class UnitTest_BatchNormCell_Frame_addInput;
};

TEST(BatchNormCell_Frame, setScales)
{
    Network net;
    Environment env(net, EmptyDatabase, 10, 10);

    BatchNormCell_Frame_Test bn1(
        "bn1", 1, std::shared_ptr<Activation<Float_T> >());
    BatchNormCell_Frame_Test bn2(
        "bn2", 1, std::shared_ptr<Activation<Float_T> >());

    bn1.addInput(env);
    bn2.addInput(env);

    bn2.setScales(bn1.getScales());
    bn1.initialize();
    bn2.initialize();

    ASSERT_EQUALS(bn1.getScale(0, 0, 0), 1.0);
    ASSERT_EQUALS(bn2.getScale(0, 0, 0), 1.0);

    bn1.setScale(0, 0, 0, 2.0);

    ASSERT_EQUALS(bn1.getScale(0, 0, 0), 2.0);
    ASSERT_EQUALS(bn2.getScale(0, 0, 0), 2.0);
}

TEST_DATASET(BatchNormCell_Frame,
             addInput__env,
             (unsigned int channelsWidth, unsigned int channelsHeight),
             std::make_tuple(24U, 24U),
             std::make_tuple(24U, 32U),
             std::make_tuple(32U, 24U))
{
    Network net;
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    BatchNormCell_Frame_Test bn1(
        "bn1", 1, std::shared_ptr<Activation<Float_T> >());
    bn1.addInput(env);
    bn1.initialize();

    ASSERT_EQUALS(bn1.getNbChannels(), 1U);
    ASSERT_EQUALS(bn1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(bn1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(bn1.getNbOutputs(), 1U);
    ASSERT_EQUALS(bn1.getOutputsWidth(), channelsWidth);
    ASSERT_EQUALS(bn1.getOutputsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(bn1.checkGradient(1.0e-3, 1.0e-2));

    // Internal state testing
    ASSERT_EQUALS(bn1.mInputs.dataSize(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(bn1.mOutputs.size(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(bn1.mDiffInputs.size(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(bn1.mDiffOutputs.dataSize(), 0U);
}

TEST_DATASET(BatchNormCell_Frame,
             addInput,
             (unsigned int channelsWidth, unsigned int channelsHeight),
             std::make_tuple(24U, 24U),
             std::make_tuple(24U, 32U),
             std::make_tuple(32U, 24U))
{
    const unsigned int nbOutputs = 16;

    Network net;
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    ConvCell_Frame conv1("conv1",
                         3,
                         3,
                         nbOutputs,
                         1,
                         1,
                         1,
                         1,
                         0,
                         0,
                         std::make_shared<TanhActivation_Frame<Float_T> >());

    BatchNormCell_Frame_Test bn1(
        "bn1", nbOutputs, std::shared_ptr<Activation<Float_T> >());

    conv1.addInput(env);
    bn1.addInput(&conv1);
    conv1.initialize();
    bn1.initialize();

    ASSERT_EQUALS(bn1.getNbChannels(), nbOutputs);
    ASSERT_EQUALS(bn1.getChannelsWidth(), conv1.getOutputsWidth());
    ASSERT_EQUALS(bn1.getChannelsHeight(), conv1.getOutputsHeight());
    ASSERT_EQUALS(bn1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(bn1.getOutputsWidth(), conv1.getOutputsWidth());
    ASSERT_EQUALS(bn1.getOutputsHeight(), conv1.getOutputsHeight());
    // ASSERT_NOTHROW_ANY(bn1.checkGradient(1.0e-3, 1.0e-2));

    // Internal state testing
    ASSERT_EQUALS(bn1.mInputs.dataSize(),
                  nbOutputs * conv1.getOutputsWidth()
                  * conv1.getOutputsHeight());
    ASSERT_EQUALS(bn1.mOutputs.size(),
                  nbOutputs * conv1.getOutputsWidth()
                  * conv1.getOutputsHeight());
    ASSERT_EQUALS(bn1.mDiffInputs.size(),
                  nbOutputs * conv1.getOutputsWidth()
                  * conv1.getOutputsHeight());
    ASSERT_EQUALS(bn1.mDiffOutputs.dataSize(),
                  nbOutputs * conv1.getOutputsWidth()
                  * conv1.getOutputsHeight());
}

RUN_TESTS()
