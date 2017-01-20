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
#include "Database/DIR_Database.hpp"
#include "DeepNet.hpp"
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
    ASSERT_EQUALS(deepNet.getSignalsDiscretization(), 0U);
    ASSERT_EQUALS(deepNet.getFreeParametersDiscretization(), 0U);
    ASSERT_THROW_ANY(deepNet.getTarget()->getDefaultTarget());
}

TEST(DeepNet, addCell)
{
    Network net;
    DeepNet deepNet(net);

    std::shared_ptr<ConvCell> convCell(new ConvCell_Frame("conv", 5, 5, 10));
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

    std::shared_ptr<ConvCell> convCell(new ConvCell_Frame("conv", 5, 5, 10));
    std::shared_ptr<FcCell> fcCell(new FcCell_Frame("fc", 10));
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

    std::shared_ptr<ConvCell> convCell(new ConvCell_Frame("conv", 5, 5, 10));
    std::shared_ptr<FcCell> fcCell(new FcCell_Frame("fc", 10));
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
    std::shared_ptr<Environment> env(new Environment(net, *database, 10));
    deepNet.setStimuliProvider(env);

    ASSERT_EQUALS(deepNet.getStimuliProvider(), env);
}

RUN_TESTS()
