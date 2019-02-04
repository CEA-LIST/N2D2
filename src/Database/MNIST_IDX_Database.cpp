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

#include "Database/MNIST_IDX_Database.hpp"

N2D2::MNIST_IDX_Database::MNIST_IDX_Database(double validation)
    : IDX_Database(), mValidation(validation)
{
    // ctor
}
N2D2::MNIST_IDX_Database::MNIST_IDX_Database(const std::string& dataPath,
                                             const std::string& labelPath,
                                             bool extractROIs,
                                             double validation)
    : MNIST_IDX_Database(validation)
{
    load(dataPath, labelPath, extractROIs);
}

void N2D2::MNIST_IDX_Database::load(const std::string& dataPath,
                                    const std::string& labelPath,
                                    bool /*extractROIs*/)
{
    const std::string labelPathDef = (labelPath.empty()) ? dataPath : labelPath;

    // Create labels in order, so that the output number will match the digit
    assert(getNbLabels() == 0);
    labelID("0");
    labelID("1");
    labelID("2");
    labelID("3");
    labelID("4");
    labelID("5");
    labelID("6");
    labelID("7");
    labelID("8");
    labelID("9");
    assert(getNbLabels() == 10);

    // Learn and validation stimuli
    IDX_Database::load(dataPath + "/train-images-idx3-ubyte",
                       labelPathDef + "/train-labels-idx1-ubyte");
    partitionStimuli(1.0 - mValidation, mValidation, 0.0);

    // Test stimuli
    IDX_Database::load(dataPath + "/t10k-images-idx3-ubyte",
                       labelPathDef + "/t10k-labels-idx1-ubyte");
    partitionStimuli(0.0, 0.0, 1.0);

    assert(getNbLabels() == 10);


    for (unsigned int i=0; i<getNbLabels(); ++i) {
        std::vector<unsigned int> stimuliIds =
            getLabelStimuliSetIndexes(i, Database::Learn);
        mStimuliPerLabelTrain.insert(
                    std::make_pair(i, stimuliIds));
    }

    for (unsigned int i=0; i<getNbLabels(); ++i) {
        std::vector<unsigned int> stimuliIds =
            getLabelStimuliSetIndexes(i, Database::Test);
        mStimuliPerLabelTest.insert(
                    std::make_pair(i, stimuliIds));
    }
}

std::vector<unsigned int>
N2D2::MNIST_IDX_Database::loadRelationSample(Database::StimuliSet set)
{
    unsigned int numberVariables = 3;
    //unsigned int maxNumber = 10;
    unsigned int maxNumber = 2;
    std::vector<unsigned int> sample;
    std::vector<unsigned int> stimulusVariables;
    for (unsigned int k=0; k<numberVariables; k++){
        if (k == numberVariables-1){
            unsigned int sum = 0;
            for (unsigned int p=0; p<numberVariables-1; p++){
                sum += stimulusVariables[p];
            }
            stimulusVariables.push_back(sum % maxNumber);
        }
        else {
            stimulusVariables.push_back(Random::randUniform(0, maxNumber-1));
        }

        std::cout << "Var " << k << ": " << stimulusVariables.back()
        << " " << getLabelName(stimulusVariables.back()) << std::endl;

        if (set == Database::Learn) {
            sample.push_back(
                mStimuliPerLabelTrain.at(stimulusVariables.back())
                [Random::randUniform(0, mStimuliPerLabelTrain.at(stimulusVariables.back()).size()-1)]);
                //             [0]);

        }
        else if (set == Database::Test) {
            sample.push_back(
                mStimuliPerLabelTest.at(stimulusVariables.back())
                [Random::randUniform(0, mStimuliPerLabelTest.at(stimulusVariables.back()).size()-1)]);
        }
        else {
            throw std::runtime_error(
                "MNIST_IDX_Database::loadRelationSample: unknown stimuli set");
        }
    }
    return sample;

}

