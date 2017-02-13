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

#include "Cell/FcCell_Frame.hpp"
#include "StimuliProvider.hpp"
#include "Target/Target.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(Target, Target)
{
    std::shared_ptr<FcCell> fcCell(new FcCell_Frame("fc", 10));
    std::shared_ptr
        <StimuliProvider> sp(new StimuliProvider(EmptyDatabase, 32, 32));
    Target target("TEST", fcCell, sp);

    ASSERT_EQUALS(target.getParameter<bool>("DataAsTarget"), false);
    ASSERT_EQUALS(target.getName(), "TEST");
    ASSERT_EQUALS(target.getCell(), fcCell);
    ASSERT_EQUALS(target.getStimuliProvider(), sp);
    ASSERT_EQUALS(target.getNbTargets(), 10U);
    ASSERT_EQUALS(target.getTargetTopN(), 1U);
    ASSERT_EQUALS(target.getTargetValue(), 1.0);
    ASSERT_EQUALS(target.getDefaultValue(), 0.0);
}

TEST_DATASET(Target,
             Target_bis,
             (unsigned int nbOutputs,
              unsigned int nbTargets,
              double targetValue,
              double defaultValue,
              unsigned int targetTopN),
             std::make_tuple(1U, 2U, 1.0, 0.0, 1U),
             std::make_tuple(2U, 2U, 0.0, 1.0, 2U),
             std::make_tuple(3U, 3U, 0.5, -0.5, 3U))
{
    std::shared_ptr<FcCell> fcCell(new FcCell_Frame("fc", nbOutputs));
    std::shared_ptr
        <StimuliProvider> sp(new StimuliProvider(EmptyDatabase, 32, 32));
    Target target("TEST", fcCell, sp, targetValue, defaultValue, targetTopN);

    ASSERT_EQUALS(target.getParameter<bool>("DataAsTarget"), false);
    ASSERT_EQUALS(target.getName(), "TEST");
    ASSERT_EQUALS(target.getCell(), fcCell);
    ASSERT_EQUALS(target.getStimuliProvider(), sp);
    ASSERT_EQUALS(target.getNbTargets(), nbTargets);
    ASSERT_EQUALS(target.getTargetTopN(), targetTopN);
    ASSERT_EQUALS(target.getTargetValue(), targetValue);
    ASSERT_EQUALS(target.getDefaultValue(), defaultValue);
}

TEST(Target, Target_ter)
{
    UnitTest::FileWriteContent("Target_LabelsMapping.in",
                               "label0 1\n"
                               "label1 3\n"
                               "label2 2\n"
                               "label3 0");

    Database database;
    database.addStimulus("s1", "label0");
    database.addStimulus("s2", "label1");
    database.addStimulus("s3", "label2");
    database.addStimulus("s4", "label3");

    std::shared_ptr<FcCell> fcCell(new FcCell_Frame("fc", 4));
    std::shared_ptr<StimuliProvider> sp(new StimuliProvider(database, 32, 32));

    Target target("TEST", fcCell, sp);
    target.labelsMapping("Target_LabelsMapping.in");

    ASSERT_EQUALS(target.getLabelTarget(0), 1);
    ASSERT_EQUALS(target.getLabelTarget(1), 3);
    ASSERT_EQUALS(target.getLabelTarget(2), 2);
    ASSERT_EQUALS(target.getLabelTarget(3), 0);
    ASSERT_THROW_ANY(target.getDefaultTarget()); // no default target

    ASSERT_EQUALS(target.getTargetLabels(0).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(0)[0], 3);
    ASSERT_EQUALS(target.getTargetLabels(1).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(1)[0], 0);
    ASSERT_EQUALS(target.getTargetLabels(2).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(2)[0], 2);
    ASSERT_EQUALS(target.getTargetLabels(3).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(3)[0], 1);

    ASSERT_EQUALS(target.getTargetLabelsName().size(), 4U);
    ASSERT_EQUALS(target.getTargetLabelsName()[0], "label3");
    ASSERT_EQUALS(target.getTargetLabelsName()[1], "label0");
    ASSERT_EQUALS(target.getTargetLabelsName()[2], "label2");
    ASSERT_EQUALS(target.getTargetLabelsName()[3], "label1");
}

TEST(Target, Target_ter2)
{
    UnitTest::FileWriteContent("Target_LabelsMapping.in",
                               "label0 1\n"
                               "label1 3\n"
                               "default 2\n"
                               "label3 0");

    Database database;
    database.addStimulus("s1", "label0");
    database.addStimulus("s2", "label1");
    database.addStimulus("s3", "label2");
    database.addStimulus("s4", "label3");

    std::shared_ptr<FcCell> fcCell(new FcCell_Frame("fc", 4));
    std::shared_ptr<StimuliProvider> sp(new StimuliProvider(database, 32, 32));

    Target target("TEST", fcCell, sp);
    target.labelsMapping("Target_LabelsMapping.in");

    ASSERT_EQUALS(target.getLabelTarget(0), 1);
    ASSERT_EQUALS(target.getLabelTarget(1), 3);
    ASSERT_EQUALS(target.getLabelTarget(2), 2);
    ASSERT_EQUALS(target.getLabelTarget(3), 0);
    ASSERT_EQUALS(target.getDefaultTarget(), 2);

    ASSERT_EQUALS(target.getTargetLabels(0).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(0)[0], 3);
    ASSERT_EQUALS(target.getTargetLabels(1).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(1)[0], 0);
    ASSERT_EQUALS(target.getTargetLabels(2).size(), 0U);
    ASSERT_EQUALS(target.getTargetLabels(3).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(3)[0], 1);

    ASSERT_EQUALS(target.getTargetLabelsName().size(), 4U);
    ASSERT_EQUALS(target.getTargetLabelsName()[0], "label3");
    ASSERT_EQUALS(target.getTargetLabelsName()[1], "label0");
    ASSERT_EQUALS(target.getTargetLabelsName()[2], "default");
    ASSERT_EQUALS(target.getTargetLabelsName()[3], "label1");
}

TEST(Target, Target_ter3)
{
    UnitTest::FileWriteContent("Target_LabelsMapping.in",
                               "label0 1\n"
                               "label1 3\n"
                               "default 2\n"
                               "* 0");

    Database database;
    database.addStimulus("s1", "label0");
    database.addStimulus("s2", "label1");
    database.addStimulus("s3", "label2");
    database.addStimulus("s4", "label3");
    database.addStimulus("s5", -1);
    database.addStimulus("s6", -1);

    std::shared_ptr<FcCell> fcCell(new FcCell_Frame("fc", 4));
    std::shared_ptr<StimuliProvider> sp(new StimuliProvider(database, 32, 32));

    Target target("TEST", fcCell, sp);
    target.labelsMapping("Target_LabelsMapping.in");

    ASSERT_EQUALS(target.getLabelTarget(0), 1);
    ASSERT_EQUALS(target.getLabelTarget(1), 3);
    ASSERT_EQUALS(target.getLabelTarget(2), 2);
    ASSERT_EQUALS(target.getLabelTarget(3), 2);
    ASSERT_EQUALS(target.getDefaultTarget(), 2);

    ASSERT_EQUALS(target.getTargetLabels(0).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(0)[0], -1);
    ASSERT_EQUALS(target.getTargetLabels(1).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(1)[0], 0);
    ASSERT_EQUALS(target.getTargetLabels(2).size(), 0U);
    ASSERT_EQUALS(target.getTargetLabels(3).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(3)[0], 1);

    ASSERT_EQUALS(target.getTargetLabelsName().size(), 4U);
    ASSERT_EQUALS(target.getTargetLabelsName()[0], "*");
    ASSERT_EQUALS(target.getTargetLabelsName()[1], "label0");
    ASSERT_EQUALS(target.getTargetLabelsName()[2], "default");
    ASSERT_EQUALS(target.getTargetLabelsName()[3], "label1");
}

TEST(Target, setLabelTarget)
{
    Database database;
    database.addStimulus("s1", "label0");
    database.addStimulus("s2", "label1");
    database.addStimulus("s3", "label2");
    database.addStimulus("s4", "label3");
    database.addStimulus("s5", -1);
    database.addStimulus("s6", -1);

    std::shared_ptr<FcCell> fcCell(new FcCell_Frame("fc", 4));
    std::shared_ptr<StimuliProvider> sp(new StimuliProvider(database, 32, 32));

    Target target("TEST", fcCell, sp);
    target.setLabelTarget(0, 1);
    target.setLabelTarget(1, 3);
    target.setDefaultTarget(2);
    target.setLabelTarget(-1, 0);

    ASSERT_EQUALS(target.getLabelTarget(0), 1);
    ASSERT_EQUALS(target.getLabelTarget(1), 3);
    ASSERT_EQUALS(target.getLabelTarget(2), 2);
    ASSERT_EQUALS(target.getLabelTarget(3), 2);
    ASSERT_EQUALS(target.getDefaultTarget(), 2);

    ASSERT_EQUALS(target.getTargetLabels(0).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(0)[0], -1);
    ASSERT_EQUALS(target.getTargetLabels(1).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(1)[0], 0);
    ASSERT_EQUALS(target.getTargetLabels(2).size(), 0U);
    ASSERT_EQUALS(target.getTargetLabels(3).size(), 1U);
    ASSERT_EQUALS(target.getTargetLabels(3)[0], 1);

    ASSERT_EQUALS(target.getTargetLabelsName().size(), 4U);
    ASSERT_EQUALS(target.getTargetLabelsName()[0], "*");
    ASSERT_EQUALS(target.getTargetLabelsName()[1], "label0");
    ASSERT_EQUALS(target.getTargetLabelsName()[2], "default");
    ASSERT_EQUALS(target.getTargetLabelsName()[3], "label1");
}

RUN_TESTS()
