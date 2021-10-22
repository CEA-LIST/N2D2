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

#include "Xnet/Environment.hpp"
#include "Activation/RectifierActivation_Frame.hpp"
#include "Cell/BatchNormCell_Frame.hpp"
#include "Cell/ConvCell_Frame.hpp"
#include "Database/DIR_Database.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "DeepNet.hpp"
#include "Xnet/Network.hpp"
#include "Cell/FcCell_Frame.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(DeepNet, DeepNet)
{
    Network net;
    DeepNet deepNet(net);

    ASSERT_EQUALS((bool)deepNet.getDatabase(), false);
    ASSERT_EQUALS((bool)deepNet.getStimuliProvider(), false);
    ASSERT_EQUALS(deepNet.getLayers().size(), 1U);
    ASSERT_EQUALS(deepNet.getLayers()[0].size(), 1U);
    ASSERT_EQUALS(deepNet.getLayers()[0][0], "env");
    ASSERT_EQUALS(deepNet.getLayer(0).size(), 1U);
    ASSERT_EQUALS(deepNet.getLayer(0)[0], "env");
    ASSERT_EQUALS(deepNet.getTargets().size(), 0U);
    ASSERT_THROW_ANY(deepNet.getTarget()->getDefaultTarget());
}

TEST(DeepNet, addCell)
{
    Network net;
    DeepNet deepNet(net);

    std::shared_ptr<ConvCell> convCell(new ConvCell_Frame<Float_T>(deepNet, "conv",
                                        std::vector<unsigned int>{5, 5}, 10));
    deepNet.addCell(convCell, std::vector<std::shared_ptr<Cell> >(1));

    ASSERT_EQUALS(deepNet.getLayers().size(), 2U);
    ASSERT_EQUALS(deepNet.getLayers()[0].size(), 1U);
    ASSERT_EQUALS(deepNet.getLayers()[1].size(), 1U);
    ASSERT_EQUALS(deepNet.getLayers()[1][0], "conv");

    ASSERT_EQUALS(deepNet.getParentCells("conv").size(), 1U);
    ASSERT_EQUALS((bool)deepNet.getParentCells("conv")[0], false);
}

TEST(DeepNet, addCell_bis)
{
    Network net;
    DeepNet deepNet(net);

    std::shared_ptr<ConvCell> convCell(new ConvCell_Frame<Float_T>(deepNet, "conv",
                                        std::vector<unsigned int>{5, 5}, 10));
    std::shared_ptr<FcCell> fcCell(new FcCell_Frame<Float_T>(deepNet, "fc", 10));
    deepNet.addCell(convCell, std::vector<std::shared_ptr<Cell> >(1));
    deepNet.addCell(fcCell, std::vector<std::shared_ptr<Cell> >(1));

    ASSERT_EQUALS(deepNet.getLayers().size(), 2U);
    ASSERT_EQUALS(deepNet.getLayers()[0].size(), 1U);
    ASSERT_EQUALS(deepNet.getLayers()[1].size(), 2U);
    ASSERT_EQUALS(deepNet.getLayers()[1][0], "conv");
    ASSERT_EQUALS(deepNet.getLayers()[1][1], "fc");

    ASSERT_EQUALS(deepNet.getParentCells("conv").size(), 1U);
    ASSERT_EQUALS((bool)deepNet.getParentCells("conv")[0], false);
    ASSERT_EQUALS(deepNet.getParentCells("fc").size(), 1U);
    ASSERT_EQUALS((bool)deepNet.getParentCells("fc")[0], false);
}

TEST(DeepNet, addCell_ter)
{
    Network net;
    DeepNet deepNet(net);

    std::shared_ptr<ConvCell> convCell(new ConvCell_Frame<Float_T>(deepNet, "conv",
                                        std::vector<unsigned int>{5, 5}, 10));
    std::shared_ptr<FcCell> fcCell(new FcCell_Frame<Float_T>(deepNet, "fc", 10));
    deepNet.addCell(convCell, std::vector<std::shared_ptr<Cell> >(1));
    deepNet.addCell(fcCell, std::vector<std::shared_ptr<Cell> >(1, convCell));

    ASSERT_EQUALS(deepNet.getLayers().size(), 3U);
    ASSERT_EQUALS(deepNet.getLayers()[0].size(), 1U);
    ASSERT_EQUALS(deepNet.getLayers()[1].size(), 1U);
    ASSERT_EQUALS(deepNet.getLayers()[1][0], "conv");
    ASSERT_EQUALS(deepNet.getLayers()[2].size(), 1U);
    ASSERT_EQUALS(deepNet.getLayers()[2][0], "fc");

    ASSERT_EQUALS(deepNet.getParentCells("conv").size(), 1U);
    ASSERT_EQUALS((bool)deepNet.getParentCells("conv")[0], false);
    ASSERT_EQUALS(deepNet.getParentCells("fc").size(), 1U);
    ASSERT_EQUALS(deepNet.getParentCells("fc")[0], convCell);
}

TEST(DeepNet, setDatabase)
{
    Network net;
    DeepNet deepNet(net);

    ASSERT_EQUALS((bool)deepNet.getDatabase(), false);

    std::shared_ptr<DIR_Database> database(new DIR_Database);
    deepNet.setDatabase(database);

    ASSERT_EQUALS(deepNet.getDatabase(), database);
}

TEST(DeepNet, setEnvironment)
{
    Network net;
    DeepNet deepNet(net);

    ASSERT_EQUALS((bool)deepNet.getStimuliProvider(), false);

    std::shared_ptr<DIR_Database> database(new DIR_Database);
    std::shared_ptr<Environment> env(new Environment(net, *database, {10, 1, 1}));
    deepNet.setStimuliProvider(env);

    ASSERT_EQUALS(deepNet.getStimuliProvider(), env);
}

TEST(DeepNet, fuseBatchNorm)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    const unsigned int nbOutputs = 5;
    const unsigned int channelsWidth = 24;
    const unsigned int channelsHeight = 24;

    Network net;
    DeepNet deepNet(net);

    std::shared_ptr<ConvCell_Frame<double> > conv1(
        new ConvCell_Frame<double>(deepNet, "conv1",
        std::vector<unsigned int>({3, 3}),
        nbOutputs,
        std::vector<unsigned int>({1, 1}),
        std::vector<unsigned int>({1, 1}),
        std::vector<int>({(int)0, (int)0}),
        std::vector<unsigned int>({1U, 1U}),
        std::shared_ptr<Activation>()));

    std::shared_ptr<BatchNormCell_Frame<double> > bn1(
        new BatchNormCell_Frame<double>(deepNet, "bn1",
        nbOutputs,
        std::make_shared<RectifierActivation_Frame<double> >()));

    MNIST_IDX_Database database;
    database.load(N2D2_DATA("mnist"));

    Environment env(net, database, {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    deepNet.addCell(conv1, std::vector<std::shared_ptr<Cell> >(1));
    deepNet.addCell(bn1, std::vector<std::shared_ptr<Cell> >(1, conv1));

    conv1->addInput(env);
    bn1->addInput(conv1.get());

    conv1->initialize();
    bn1->initialize();


    for (unsigned int output = 0; output < nbOutputs; ++output) {
        Tensor<double> scale({1}, Random::randNormal(1.0, 0.5));
        Tensor<double> bias({1}, Random::randUniform(-0.5, 0.5));
        Tensor<double> mean({1}, Random::randUniform(-0.5, 0.5));
        Tensor<double> variance({1}, Random::randUniform(0.0, 0.15));

        bn1->setScale(output, scale);
        bn1->setBias(output, bias);
        bn1->setMean(output, mean);
        bn1->setVariance(output, variance);
    }

    ASSERT_EQUALS(deepNet.getLayers().size(), 3U);

    // Outputs before fuse
    conv1->propagate(true);
    bn1->propagate(true);
    const Tensor<double>& outputsRef = tensor_cast<double>(bn1->getOutputs());

    conv1->logFreeParametersDistrib("class_DeepNet_conv1_ref.log");

    // Fuse!
    deepNet.fuseBatchNorm();

    ASSERT_EQUALS(deepNet.getLayers().size(), 2U);
    ASSERT_EQUALS(conv1->getParameter<bool>("NoBias"), false);
    ASSERT_EQUALS(conv1->getActivation()->getType(), RectifierActivation::Type);

    // Outputs after fuse
    conv1->propagate(true);
    const Tensor<double>& outputsFuse = tensor_cast<double>(conv1->getOutputs());

    conv1->logFreeParametersDistrib("class_DeepNet_conv1_fuse.log");

    for (unsigned int batch = 0; batch < 2; ++batch) {
        for (unsigned int output = 0; output < nbOutputs; ++output) {
            for (unsigned int oy = 0; oy < conv1->getOutputsHeight(); ++oy) {
                for (unsigned int ox = 0; ox < conv1->getOutputsWidth(); ++ox) {
                    ASSERT_EQUALS_DELTA(outputsRef(ox, oy, output, batch),
                                        outputsFuse(ox, oy, output, batch),
                                        1e-9);
                }
            }
        }
    }
}

RUN_TESTS()
