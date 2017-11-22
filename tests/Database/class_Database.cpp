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

#include "Database/Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class Database_Test : public Database {
public:
    Database_Test(unsigned int nbStimuli, unsigned int nbLabels)
        : mNbStimuli(nbStimuli), mNbLabels(nbLabels)
    {
    }

    void load(const std::string& /*dataPath*/,
              const std::string& /*labelPath*/ = "",
              bool /*extractROIs*/ = false)
    {
        for (unsigned int i = 0; i < mNbStimuli; ++i) {
            std::stringstream label;
            label << "label_" << (i % mNbLabels);

            std::stringstream name;
            name << "stimulus_" << i;

            mStimuli.push_back(Stimulus(name.str(), labelID(label.str())));
            mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
        }
    }

    void dumpStimuli(const std::string& fileName) const
    {
        std::ofstream data(fileName.c_str());

        if (!data.good())
            throw std::runtime_error("Could not create data file: " + fileName);

        for (std::vector<Stimulus>::const_iterator it = mStimuli.begin(),
                                                   itBegin = mStimuli.begin(),
                                                   itEnd = mStimuli.end();
             it != itEnd;
             ++it) {
            data << (it - itBegin) << " " << (*it).name << " " << (*it).label
                 << "\n";
        }
    }

    void dumpLabels(const std::string& fileName) const
    {
        std::ofstream data(fileName.c_str());

        if (!data.good())
            throw std::runtime_error("Could not create data file: " + fileName);

        for (std::vector<std::string>::const_iterator it = mLabelsName.begin(),
                                                      itBegin
                                                      = mLabelsName.begin(),
                                                      itEnd = mLabelsName.end();
             it != itEnd;
             ++it) {
            data << (it - itBegin) << " " << (*it) << "\n";
        }
    }

    friend class UnitTest_Database_getLabelsStimuliSetIndexes;

private:
    unsigned int mNbStimuli;
    unsigned int mNbLabels;
};

TEST(Database, load)
{
    Database_Test db(10, 2);
    db.load("");

    ASSERT_EQUALS(db.getNbStimuli(), 10U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 0U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 0U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Test), 0U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 10U);
    ASSERT_EQUALS(db.getNbLabels(), 2U);
}

TEST(Database, load_empty_database)
{
    std::shared_ptr<Database> db;
    db = std::make_shared<Database>();
    db->load("", "tests_data/ilsvrc2012_labels.dat");
    ASSERT_EQUALS(db->getNbLabels(), 1000U);
    ASSERT_EQUALS(db->getLabelName(0), "tench, Tinca tinca");

}

TEST_DATASET(Database,
             partitionStimuli,
             (unsigned int nbStimuli, Database::StimuliSet stimuliSet),
             std::make_tuple(0, Database::Learn),
             std::make_tuple(50, Database::Learn),
             std::make_tuple(100, Database::Learn),
             std::make_tuple(0, Database::Validation),
             std::make_tuple(50, Database::Validation),
             std::make_tuple(100, Database::Validation),
             std::make_tuple(0, Database::Test),
             std::make_tuple(50, Database::Test),
             std::make_tuple(100, Database::Test),
             std::make_tuple(0, Database::Unpartitioned),
             std::make_tuple(50, Database::Unpartitioned),
             std::make_tuple(100, Database::Unpartitioned))
{
    Random::mtSeed(0);

    Database_Test db(100, 5);
    db.load("");
    db.partitionStimuli(nbStimuli, stimuliSet);

    ASSERT_EQUALS(db.getNbStimuli(), 100U);

    if (stimuliSet == Database::Unpartitioned) {
        ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 0U);
        ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 0U);
        ASSERT_EQUALS(db.getNbStimuli(Database::Test), 0U);
        ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 100U);
    } else {
        ASSERT_EQUALS(db.getNbStimuli(stimuliSet), nbStimuli);
        ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned),
                      100U - nbStimuli);

        if (stimuliSet == Database::Learn) {
            ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 0U);
            ASSERT_EQUALS(db.getNbStimuli(Database::Test), 0U);
        } else if (stimuliSet == Database::Validation) {
            ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 0U);
            ASSERT_EQUALS(db.getNbStimuli(Database::Test), 0U);
        } else {
            ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 0U);
            ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 0U);
        }
    }
}

TEST_DATASET(Database,
             partitionStimuli__throw,
             (unsigned int nbStimuli, Database::StimuliSet stimuliSet),
             std::make_tuple(101, Database::Learn),
             std::make_tuple(1000, Database::Learn),
             std::make_tuple(101, Database::Validation),
             std::make_tuple(1000, Database::Validation),
             std::make_tuple(101, Database::Test),
             std::make_tuple(1000, Database::Test))
{
    Random::mtSeed(0);

    Database_Test db(100, 5);
    db.load("");

    ASSERT_THROW(db.partitionStimuli(nbStimuli, stimuliSet),
                 std::runtime_error);
}

TEST_DATASET(Database,
             getLabelsStimuliSetIndexes,
             (unsigned int nbStimuli, unsigned int nbLabels),
             std::make_tuple(5, 1),
             std::make_tuple(5, 2),
             std::make_tuple(5, 3),
             std::make_tuple(5, 4),
             std::make_tuple(5, 5),
             std::make_tuple(10, 1),
             std::make_tuple(10, 2),
             std::make_tuple(10, 3),
             std::make_tuple(10, 4),
             std::make_tuple(10, 5))
{
    Database_Test db(nbStimuli, nbLabels);
    db.load("");

    std::ostringstream fileName;
    fileName << "Database_getLabelsStimuliSetIndexes(NS" << nbStimuli << "_NL"
             << nbLabels << ")[stimuli].dat";
    db.dumpStimuli("Database/" + fileName.str());

    fileName.str(std::string());
    fileName << "Database_getLabelsStimuliSetIndexes(NS" << nbStimuli << "_NL"
             << nbLabels << ")[labels].dat";
    db.dumpLabels("Database/" + fileName.str());

    std::vector<std::vector<unsigned int> > indexes
        = db.getLabelsStimuliSetIndexes(Database::Unpartitioned);

    ASSERT_EQUALS(indexes.size(), nbLabels);

    for (int label = 0; label < (int)nbLabels; ++label) {
        ASSERT_EQUALS(indexes[label].size(),
                      nbStimuli / nbLabels
                      + (label < (int)(nbStimuli % nbLabels)));
    }
}

TEST_DATASET(Database,
             partitionStimuliPerLabel,
             (unsigned int nbStimuliPerLabel, Database::StimuliSet stimuliSet),
             std::make_tuple(0, Database::Learn),
             std::make_tuple(10, Database::Learn),
             std::make_tuple(20, Database::Learn),
             std::make_tuple(0, Database::Validation),
             std::make_tuple(10, Database::Validation),
             std::make_tuple(20, Database::Validation),
             std::make_tuple(0, Database::Test),
             std::make_tuple(10, Database::Test),
             std::make_tuple(20, Database::Test),
             std::make_tuple(0, Database::Unpartitioned),
             std::make_tuple(10, Database::Unpartitioned),
             std::make_tuple(20, Database::Unpartitioned))
{
    Random::mtSeed(0);

    Database_Test db(100, 5);
    db.load("");

    db.partitionStimuliPerLabel(nbStimuliPerLabel, stimuliSet);

    ASSERT_EQUALS(db.getNbStimuli(), 100U);

    if (stimuliSet == Database::Unpartitioned) {
        ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 0U);
        ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 0U);
        ASSERT_EQUALS(db.getNbStimuli(Database::Test), 0U);
        ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 100U);
    } else {
        ASSERT_EQUALS(db.getNbStimuli(stimuliSet), 5 * nbStimuliPerLabel);
        ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned),
                      5 * (20U - nbStimuliPerLabel));

        if (stimuliSet == Database::Learn) {
            ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 0U);
            ASSERT_EQUALS(db.getNbStimuli(Database::Test), 0U);
        } else if (stimuliSet == Database::Validation) {
            ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 0U);
            ASSERT_EQUALS(db.getNbStimuli(Database::Test), 0U);
        } else {
            ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 0U);
            ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 0U);
        }
    }
}

TEST_DATASET(Database,
             partitionStimuli__learn_validation_test,
             (double learn, double validation, double test),
             std::make_tuple(0.5, 0.0, 0.0),
             std::make_tuple(1.0, 0.0, 0.0),
             std::make_tuple(0.0, 0.5, 0.0),
             std::make_tuple(0.0, 1.0, 0.0),
             std::make_tuple(0.0, 0.0, 0.5),
             std::make_tuple(0.0, 0.0, 1.0),
             std::make_tuple(0.5, 0.5, 0.0),
             std::make_tuple(0.7, 0.2, 0.0),
             std::make_tuple(0.0, 0.5, 0.5),
             std::make_tuple(0.0, 0.7, 0.2),
             std::make_tuple(0.5, 0.0, 0.5),
             std::make_tuple(0.2, 0.0, 0.7),
             std::make_tuple(0.5, 0.3, 0.2),
             std::make_tuple(0.2, 0.3, 0.2))
{
    Random::mtSeed(0);

    Database_Test db(100, 10);
    db.load("");

    db.partitionStimuli(learn, validation, test);

    ASSERT_EQUALS(db.getNbStimuli(), 100U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn),
                  (unsigned int)Utils::round(learn * 100));
    ASSERT_EQUALS(db.getNbStimuli(Database::Validation),
                  (unsigned int)Utils::round(validation * 100));
    ASSERT_EQUALS(db.getNbStimuli(Database::Test),
                  (unsigned int)Utils::round(test * 100));
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned),
                  (unsigned int)Utils::round((1.0 - learn - validation - test)
                                             * 100));
}

TEST_DATASET(Database,
             partitionStimuliPerLabel__learn_validation_test,
             (double learn, double validation, double test),
             std::make_tuple(0.5, 0.0, 0.0),
             std::make_tuple(1.0, 0.0, 0.0),
             std::make_tuple(0.0, 0.5, 0.0),
             std::make_tuple(0.0, 1.0, 0.0),
             std::make_tuple(0.0, 0.0, 0.5),
             std::make_tuple(0.0, 0.0, 1.0),
             std::make_tuple(0.5, 0.5, 0.0),
             std::make_tuple(0.7, 0.2, 0.0),
             std::make_tuple(0.0, 0.5, 0.5),
             std::make_tuple(0.0, 0.7, 0.2),
             std::make_tuple(0.5, 0.0, 0.5),
             std::make_tuple(0.2, 0.0, 0.7),
             std::make_tuple(0.5, 0.3, 0.2),
             std::make_tuple(0.2, 0.3, 0.2))
{
    Random::mtSeed(0);

    Database_Test db(100, 10);
    db.load("");

    db.partitionStimuliPerLabel(learn, validation, test);

    ASSERT_EQUALS(db.getNbStimuli(), 100U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn),
                  (unsigned int)Utils::round(learn * 100));
    ASSERT_EQUALS(db.getNbStimuli(Database::Validation),
                  (unsigned int)Utils::round(validation * 100));
    ASSERT_EQUALS(db.getNbStimuli(Database::Test),
                  (unsigned int)Utils::round(test * 100));
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned),
                  (unsigned int)Utils::round((1.0 - learn - validation - test)
                                             * 100));
}

TEST_DATASET(Database,
             getLabelName,
             (unsigned int nbStimuli, unsigned int nbLabels),
             std::make_tuple(5, 1),
             std::make_tuple(5, 2),
             std::make_tuple(5, 3),
             std::make_tuple(5, 4),
             std::make_tuple(5, 5),
             std::make_tuple(10, 1),
             std::make_tuple(10, 2),
             std::make_tuple(10, 3),
             std::make_tuple(10, 4),
             std::make_tuple(10, 5))
{
    Database_Test db(nbStimuli, nbLabels);
    db.load("");

    for (int label = 0; label < (int)nbLabels; ++label) {
        std::stringstream labelStr;
        labelStr << "label_" << label;

        ASSERT_EQUALS(db.getLabelName(label), labelStr.str());
    }
}

TEST_DATASET(Database,
             getLabelID,
             (unsigned int nbStimuli, unsigned int nbLabels),
             std::make_tuple(5, 1),
             std::make_tuple(5, 2),
             std::make_tuple(5, 3),
             std::make_tuple(5, 4),
             std::make_tuple(5, 5),
             std::make_tuple(10, 1),
             std::make_tuple(10, 2),
             std::make_tuple(10, 3),
             std::make_tuple(10, 4),
             std::make_tuple(10, 5))
{
    Database_Test db(nbStimuli, nbLabels);
    db.load("");

    for (int label = 0; label < (int)nbLabels; ++label) {
        std::stringstream labelStr;
        labelStr << "label_" << label;

        ASSERT_EQUALS(db.getLabelID(labelStr.str()), label);
    }
}

TEST_DATASET(Database,
             removeLabel,
             (unsigned int nbStimuli, unsigned int nbLabels),
             std::make_tuple(5, 1),
             std::make_tuple(5, 2),
             std::make_tuple(5, 3),
             std::make_tuple(5, 4),
             std::make_tuple(5, 5),
             std::make_tuple(10, 1),
             std::make_tuple(10, 2),
             std::make_tuple(10, 3),
             std::make_tuple(10, 4),
             std::make_tuple(10, 5))
{
    Database_Test db(nbStimuli, nbLabels);
    db.load("");

    ASSERT_EQUALS(db.getLabelName(0), "label_0");
    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), nbStimuli);
    ASSERT_EQUALS(db.getNbLabels(), nbLabels);

    db.removeLabel(0);

    if (nbLabels > 1) {
        ASSERT_EQUALS(db.getLabelName(0), "label_1");
    }

    ASSERT_EQUALS(db.getNbStimuli(),
                  nbStimuli - std::ceil(nbStimuli / (double)nbLabels));
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned),
                  nbStimuli - std::ceil(nbStimuli / (double)nbLabels));
    ASSERT_EQUALS(db.getNbLabels(), nbLabels - 1);
}


RUN_TESTS()
