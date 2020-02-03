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

#include "Database/CelebA_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(CelebA_Database, load)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA/Anno")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA/Eval")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA/Img")));

    Random::mtSeed(0);

    CelebA_Database db(false, true);
    db.load(N2D2_DATA("CelebA/Img"), N2D2_DATA("CelebA/Anno"));

    const unsigned int nbStimuli = 202599;
    const unsigned int nbIdentities = 10177;

    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli);
    ASSERT_EQUALS(db.getNbROIs(), 0U);
    ASSERT_EQUALS(db.getNbLabels(), nbIdentities);

    ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 162770U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 19867U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Test), 19962U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);

    ASSERT_EQUALS(db.getLabelID("1"), 0);
    ASSERT_EQUALS(db.getLabelID("2"), 1);
    ASSERT_EQUALS(db.getLabelID("3"), 2);
    ASSERT_EQUALS(db.getLabelID(std::to_string(nbIdentities)), nbIdentities - 1);

    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(0)), "000001.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(0), db.getLabelID("2880"));
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(1)), "000002.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(1), db.getLabelID("2937"));
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(2)), "000003.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(2), db.getLabelID("8692"));
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(nbStimuli - 1)), "202599.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(nbStimuli - 1), db.getLabelID("10101"));
}

TEST(CelebA_Database, load_inTheWhild)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA/Anno")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA/Eval")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA/Img")));

    Random::mtSeed(0);

    CelebA_Database db(true, true);
    db.load(N2D2_DATA("CelebA/Img"), N2D2_DATA("CelebA/Anno"));

    const unsigned int nbStimuli = 202599;
    const unsigned int nbIdentities = 10177;

    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli);
    ASSERT_EQUALS(db.getNbROIs(), nbStimuli);
    ASSERT_EQUALS(db.getNbLabels(), nbIdentities);

    ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 162770U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 19867U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Test), 19962U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);

    ASSERT_EQUALS(db.getLabelID("1"), 0);
    ASSERT_EQUALS(db.getLabelID("2"), 1);
    ASSERT_EQUALS(db.getLabelID("3"), 2);
    ASSERT_EQUALS(db.getLabelID(std::to_string(nbIdentities)), nbIdentities - 1);

    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(0)), "000001.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(0), -1);
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(1)), "000002.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(1), -1);
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(2)), "000003.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(2), -1);
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(nbStimuli - 1)), "202599.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(nbStimuli - 1), -1);

    const std::vector<std::shared_ptr<ROI> > roi0 = db.getStimulusROIs(0);
    ASSERT_EQUALS(roi0.size(), 1);
    ASSERT_EQUALS(roi0[0]->getLabel(), db.getLabelID("2880"));
    ASSERT_EQUALS(roi0[0]->getBoundingRect(), cv::Rect(95, 71, 226, 313));

    const std::vector<std::shared_ptr<ROI> > roi1 = db.getStimulusROIs(1);
    ASSERT_EQUALS(roi1.size(), 1);
    ASSERT_EQUALS(roi1[0]->getLabel(), db.getLabelID("2937"));
    ASSERT_EQUALS(roi1[0]->getBoundingRect(), cv::Rect(72, 94, 221, 306));

    const std::vector<std::shared_ptr<ROI> > roi2 = db.getStimulusROIs(2);
    ASSERT_EQUALS(roi2.size(), 1);
    ASSERT_EQUALS(roi2[0]->getLabel(), db.getLabelID("8692"));
    ASSERT_EQUALS(roi2[0]->getBoundingRect(), cv::Rect(216, 59, 91, 126));

    const std::vector<std::shared_ptr<ROI> > roiE = db.getStimulusROIs(nbStimuli - 1);
    ASSERT_EQUALS(roiE.size(), 1);
    ASSERT_EQUALS(roiE[0]->getLabel(), db.getLabelID("10101"));
    ASSERT_EQUALS(roiE[0]->getBoundingRect(), cv::Rect(101, 101, 179, 248));
}

RUN_TESTS()
