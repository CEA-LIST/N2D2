/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Vincent TEMPLIER (vincent.templier@cea.fr)

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

#include "Adversarial.hpp"


void N2D2::attackLauncher(std::shared_ptr<N2D2::DeepNet>& deepNet, const std::string attStr, const bool targeted) 
{
    /// Degradation rate
    const float eps = 0.3f;

    if (attStr == "PGD") {
        /// Number of attacks on each image
        const unsigned int nbIter = 40;
        /// Gradient step
        const float gradStep = eps/(float)nbIter;
        const bool random_start = true;
        deepNet->PGD_attack(eps, nbIter, gradStep, targeted, random_start);
    }
    else if (attStr == "GN") {
        deepNet->GN_attack(eps);
    } 
    else if (attStr == "Vanilla") {
        deepNet->Vanilla_attack();
    } 
    else if (attStr == "FGSM") {
        deepNet->FGSM_attack(eps, targeted);
    } 
    else {
        std::cout << "\nUnknown attack !" << std::endl;
        std::exit(0);
    }
}

void N2D2::singleTestAdv(std::shared_ptr<N2D2::DeepNet>& deepNet, std::string dirName) {
    const std::shared_ptr<Database>& database = deepNet->getDatabase();
    const std::shared_ptr<StimuliProvider>& sp = deepNet->getStimuliProvider();
    const std::string attackName = sp->getAttack();

    if (deepNet->getTargets().size() != 1) {
        std::stringstream msgStr;
        msgStr << "Adversarial attack can only be performed with only one target\n"
               << "Please change it in the ini file\n";

        throw std::runtime_error(msgStr.str());
    }

    /// Run targeted attacks or not
    const bool targeted = false;
    /// Database used for attacks
    Database::StimuliSet set = Database::Test;

    // Selecting the images
    sp->readRandomBatch(set);
    sp->synchronize();

    Tensor<Float_T>& dataInput = sp->getData();
    Tensor<Float_T> dataInputCopy = dataInput.clone();

    Tensor<int>& labels = sp->getLabelsData();
    Tensor<int> labelsCopy = labels.clone();

    // Target design
    // Target is the following class to the initial class
    // Can be changed if the user wishes
    if (targeted) {
        Tensor<int>& labels = sp->getLabelsData();
        for (unsigned int i = 0; i < labels.size(); ++i)
            labels(i) = (labels(i) + 1) % ((int)database->getNbLabels());
        labels.synchronizeHToD();
    }

    attackLauncher(deepNet, attackName, targeted);

    std::cout << attackName << " attack" << std::endl;
    if (targeted) std::cout << "Targeted mode" << std::endl;
    else std::cout << "Untargeted mode" << std::endl;

    sp->synchronize();
    deepNet->test(set);

    deepNet->getTarget()->getEstimatedLabels().synchronizeDToH();
    deepNet->getTarget()->getEstimatedLabelsValue().synchronizeDToH();

    std::vector<char> successes;
    successes.resize(dataInput.dimB(), 0);

    for (unsigned int i = 0; i < successes.size(); ++i) {
        if (targeted) {
            if (deepNet->getTarget()->getEstimatedLabels()[i](0) == labels[i](0)) {
                successes[i] = 1;
            }
        } else {
            if (deepNet->getTarget()->getEstimatedLabels()[i](0) != labels[i](0)) {
                successes[i] = 1;
            }
        }
    }

    for (unsigned int i = 0; i < sp->getBatchSize(); ++i) {
        std::cout << "BatchPos " << i << ": ";
        if (successes[i] == 1) {
            std::cout << "\033[0;32m" << "Successful attack";
        } else {
            std::cout << "\033[0;31m" << "Failed attack";
        }
        std::cout << " (label: " << labelsCopy[i](0) 
                    << ", estimated: " << deepNet->getTarget()->getEstimatedLabels()[i](0)
                    << " with " << deepNet->getTarget()->getEstimatedLabelsValue()[i](0)*100 << "%"
                    << ")";

        std::cout << "\033[0m" << std::endl;
    }


    Tensor<Float_T> Noise = Tensor<Float_T>({dataInput.dimX(),dataInput.dimY(),dataInput.dimZ(), dataInput.dimB()});
    for (unsigned int i = 0; i < Noise.size(); ++i)
        Noise(i) = dataInput(i) - dataInputCopy(i);

    std::ostringstream subfolderName;
    subfolderName << dirName << "/" << attackName << "/Solo";
    Utils::createDirectories(subfolderName.str());

    // Log results in the adversarial subfolder
    for (unsigned int i = 0; i < dataInput.dimB(); ++i) {
        std::ostringstream resultFolderName;
        resultFolderName << subfolderName.str() << "/batchPos_" << i;
        Utils::createDirectories(resultFolderName.str());

        std::ostringstream fileNameOriginal, fileNameModified, fileNameNoise;

        fileNameOriginal << resultFolderName.str() << "/original.dat";
        fileNameModified << resultFolderName.str() << "/modified.dat";
        fileNameNoise << resultFolderName.str() << "/noise.dat";

        StimuliProvider::logData(fileNameOriginal.str(), dataInputCopy[i]);
        StimuliProvider::logData(fileNameModified.str(), dataInput[i]);
        StimuliProvider::logData(fileNameNoise.str(), Noise[i]);
    }

    //deepNet->logOutputs(dirName + "/" + attackName + "/outputs_adv");
}

void N2D2::multiTestAdv(std::shared_ptr<N2D2::DeepNet>& deepNet, std::string dirName) {
    const std::shared_ptr<Database>& database = deepNet->getDatabase();
    const std::shared_ptr<StimuliProvider>& sp = deepNet->getStimuliProvider();
    const std::string attackName = sp->getAttack();

    /// Run targeted attacks or not
    const bool targeted = false;
    /// Number of images used 
    const unsigned int nbImages = 2000;
    /// Database used for attacks
    Database::StimuliSet set = Database::Test;

    /// Number of successful attacks, whatever the class
    unsigned int counterSuccess = 0;
    /// Number of successful attacks due to network errors
    unsigned int counterError = 0;

    std::vector<unsigned int> labelSuccesses;
    labelSuccesses.resize(database->getNbLabels(), 0);

    std::vector<unsigned int> labelTotal;
    labelTotal.resize(database->getNbLabels(), 0);

    std::cout << "Starting the adversarial test..." << std::endl;

    unsigned int imgCounter = 0;
    while (imgCounter < nbImages) {

        // Selecting the images
        sp->readBatch(set, imgCounter);
        sp->synchronize();

        Tensor<Float_T>& dataInput = sp->getData();
        Tensor<Float_T> dataInputCopy = dataInput.clone();

        Tensor<int>& labels = sp->getLabelsData();
        Tensor<int> labelsCopy = labels.clone();

        deepNet->test(set);
        deepNet->getTarget()->getEstimatedLabels().synchronizeDToH();

        for (unsigned int i = 0; i < sp->getBatchSize(); ++i) {
            if (deepNet->getTarget()->getEstimatedLabels()[i](0) != labels[i](0)) {
                ++counterError;
            }
        }
        
        // Target design
        // Target is the following class to the initial class
        // Can be changed if the user wishes
        if (targeted) {
            Tensor<int>& labels = sp->getLabelsData();
            for (unsigned int i = 0; i < labels.size(); ++i)
                labels(i) = (labels(i) + 1) % ((int)database->getNbLabels());
            labels.synchronizeHToD();
        }

        attackLauncher(deepNet, attackName, targeted);

        sp->synchronize();
        deepNet->test(set);

        deepNet->getTarget()->getEstimatedLabels().synchronizeDToH();
        deepNet->getTarget()->getEstimatedLabelsValue().synchronizeDToH();

        for (unsigned int i = 0; i < sp->getBatchSize(); ++i) {
            if (targeted) {
                if (deepNet->getTarget()->getEstimatedLabels()[i](0) == labels[i](0)) {
                    labelSuccesses[labelsCopy[i](0)] += 1;
                    ++counterSuccess;
                }
            } else {
                if (deepNet->getTarget()->getEstimatedLabels()[i](0) != labels[i](0)) {
                    labelSuccesses[labelsCopy[i](0)] += 1;
                    ++counterSuccess;
                }
            }
            labelTotal[labelsCopy[i](0)] += 1;
        }

        imgCounter += sp->getBatchSize();
        std::cout << std::flush << "\rTreating " << imgCounter << "/" << nbImages;

    }

    // Analysis of the results
    std::cout << std::endl << "Analysis of the results..." << std::endl;

    std::ostringstream subfolderName;
    subfolderName << dirName << "/" << attackName << "/Multi";
    Utils::createDirectories(subfolderName.str());

    std::cout << "Successful attacks: " << ((float)counterSuccess / imgCounter)*100 << "%" << std::endl;
    std::cout << "including network errors: " << ((float)counterError / imgCounter)*100 << "%" << std::endl;

    for (unsigned int i = 0; i < database->getNbLabels(); ++i) {
        if (labelTotal[i] > 0) {
            std::cout << "  - successful attacks on class " << i << ": " << ((float)labelSuccesses[i] / labelTotal[i])*100 << "% " 
                    << "(" << labelSuccesses[i] << "/" << labelTotal[i] << ")" << std::endl;
        } else {
            std::cout << "  - no attack made on class " << i << std::endl;
        }
    }

}