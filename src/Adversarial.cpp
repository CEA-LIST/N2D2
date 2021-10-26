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
#include "StimuliProvider.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "utils/Utils.hpp"

N2D2::Adversarial::Adversarial(const N2D2::Adversarial::Attack_T attackName) 
    : // Variables
      mName(attackName),
      mEps(0.1f),
      mNbIterations(10U),
      mRandomStart(false),
      mTargeted(false)

{
    // ctor
}

void N2D2::Adversarial::attackLauncher(std::shared_ptr<N2D2::DeepNet>& deepNet) 
{
    switch (mName) {
    case PGD:
        {
            /// Gradient step
            const float gradStep = mEps/(float)mNbIterations;
            PGD_attack(deepNet, mEps, mNbIterations, 
                       gradStep, mTargeted, mRandomStart);
            break;
        }

    case GN:
        GN_attack(deepNet, mEps);
        break;

    case Vanilla:
        Vanilla_attack();
        break;

    case FGSM:
        FGSM_attack(deepNet, mEps, mTargeted);
        break;
    
    case None:
    default:
        throw std::runtime_error("Unknown adversarial attack");
    }
}

void N2D2::Adversarial::singleTestAdv(std::shared_ptr<N2D2::DeepNet>& deepNet, std::string dirName) {
    const std::shared_ptr<Database>& database = deepNet->getDatabase();
    const std::shared_ptr<StimuliProvider>& sp = deepNet->getStimuliProvider();
    const std::string attackName = Utils::toString(mName);

    if (deepNet->getTargets().size() != 1) {
        std::stringstream msgStr;
        msgStr << "Adversarial attack can only be performed with only one target\n"
               << "Please change it in the ini file\n";

        throw std::runtime_error(msgStr.str());
    }

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
    if (mTargeted) {
        Tensor<int>& labels = sp->getLabelsData();
        for (unsigned int i = 0; i < labels.size(); ++i)
            labels(i) = (labels(i) + 1) % ((int)database->getNbLabels());
        labels.synchronizeHToD();
    }

    attackLauncher(deepNet);

    std::cout << attackName << " attack" << std::endl;
    if (mTargeted) std::cout << "Targeted mode" << std::endl;
    else std::cout << "Untargeted mode" << std::endl;

    sp->synchronize();
    deepNet->test(set);

    deepNet->getTarget()->getEstimatedLabels().synchronizeDToH();
    deepNet->getTarget()->getEstimatedLabelsValue().synchronizeDToH();

    std::vector<char> successes;
    successes.resize(dataInput.dimB(), 0);

    for (unsigned int i = 0; i < successes.size(); ++i) {
        if (mTargeted) {
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

void N2D2::Adversarial::multiTestAdv(std::shared_ptr<N2D2::DeepNet>& deepNet, std::string dirName) {
    const std::shared_ptr<Database>& database = deepNet->getDatabase();
    const std::shared_ptr<StimuliProvider>& sp = deepNet->getStimuliProvider();
    const std::string attackName = Utils::toString(mName);

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
        if (mTargeted) {
            Tensor<int>& labels = sp->getLabelsData();
            for (unsigned int i = 0; i < labels.size(); ++i)
                labels(i) = (labels(i) + 1) % ((int)database->getNbLabels());
            labels.synchronizeHToD();
        }

        attackLauncher(deepNet);

        sp->synchronize();
        deepNet->test(set);

        deepNet->getTarget()->getEstimatedLabels().synchronizeDToH();
        deepNet->getTarget()->getEstimatedLabelsValue().synchronizeDToH();

        for (unsigned int i = 0; i < sp->getBatchSize(); ++i) {
            if (mTargeted) {
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

// ----------------------------------------------------------------------------
// --------------------------- Adversarial attacks ----------------------------
// ----------------------------------------------------------------------------

void N2D2::Vanilla_attack()
{
    // Nothing
}

void N2D2::GN_attack(std::shared_ptr<DeepNet>& deepNet, 
                     const float eps)
{
    Tensor<Float_T>& dataInput = deepNet->getStimuliProvider()->getData();

    // Adding gaussian noise (mean = 0, standard deviation = 1)
    for (unsigned int i = 0; i < dataInput.size(); ++i) {
        dataInput(i) += eps * Random::randNormal(0.0, 1.0);
        dataInput(i) = std::max(0.0f, std::min(dataInput(i), 1.0f));
    }
}

void N2D2::FGSM_attack(std::shared_ptr<DeepNet>& deepNet, 
                       const float eps, 
                       const bool targeted)
{
    Tensor<Float_T>& dataInput = deepNet->getStimuliProvider()->getData();
    int _targeted = targeted ? 1 : (-1);

    // first layer after env cell
    // used to retrieve diffOutputs from this layer
    std::shared_ptr<Cell_Frame_Top> firstLayer 
        = std::dynamic_pointer_cast<Cell_Frame_Top>(deepNet->getCell(deepNet->getLayers()[1][0]));

    // Requires to call Database::Learn to access all the information
    // provided by Target::provideTargets
    deepNet->propagate(Database::Learn, true, NULL);
    deepNet->backPropagate(NULL);

    // Saving gradients in gradInputs
    firstLayer->getDiffOutputs().synchronizeDToH();
    Tensor<Float_T> gradInputs = tensor_cast<Float_T>(firstLayer->getDiffOutputs());

    for (unsigned int i = 0; i < dataInput.size(); ++i) {
        gradInputs(i) = (Float_T(0) < gradInputs(i)) - (gradInputs(i) < Float_T(0));
        dataInput(i) -= _targeted * eps * gradInputs(i);
        dataInput(i) = std::max(0.0f, std::min(dataInput(i), 1.0f));
    }
}

void N2D2::FFGSM_attack(std::shared_ptr<DeepNet>& deepNet, 
                        const float eps, 
                        const float alpha,
                        const bool targeted)
{
    Tensor<Float_T>& dataInput = deepNet->getStimuliProvider()->getData();
    const Tensor<Float_T> dataInputCopy = dataInput.clone();
    int _targeted = targeted ? 1 : (-1);

    // first layer after env cell
    // used to retrieve diffOutputs from this layer
    std::shared_ptr<Cell_Frame_Top> firstLayer 
        = std::dynamic_pointer_cast<Cell_Frame_Top>(deepNet->getCell(deepNet->getLayers()[1][0]));
    
    for (unsigned int i = 0; i < dataInput.size(); ++i) {
        dataInput(i) += eps * Random::randUniform(-1.0, 1.0);
        dataInput(i) = std::max(0.0f, std::min(dataInput(i), 1.0f));
    }

    // Requires to call Database::Learn to access all the information
    // provided by Target::provideTargets
    deepNet->propagate(Database::Learn, true, NULL);
    deepNet->backPropagate(NULL);

    // Saving gradients in gradInputs
    firstLayer->getDiffOutputs().synchronizeDToH();
    Tensor<Float_T> gradInputs = tensor_cast<Float_T>(firstLayer->getDiffOutputs());

    for (unsigned int i = 0; i < dataInput.size(); ++i) {
        gradInputs(i) = (Float_T(0) < gradInputs(i)) - (gradInputs(i) < Float_T(0));    // sign method
        dataInput(i) -= _targeted * alpha * gradInputs(i);
        dataInput(i) = std::max(dataInputCopy(i) - eps, std::min(dataInput(i), dataInputCopy(i) + eps));
        dataInput(i) = std::max(0.0f, std::min(dataInput(i), 1.0f));                    // clamping
    }
}

void N2D2::PGD_attack(std::shared_ptr<DeepNet>& deepNet,
                      const float eps, 
                      const unsigned int nbIter, 
                      const float alpha,
                      const bool targeted,
                      const bool random_start)
{
    const std::shared_ptr<StimuliProvider>& sp = deepNet->getStimuliProvider();

    Tensor<Float_T>& dataInput = sp->getData();
    const Tensor<Float_T> dataInputCopy = dataInput.clone();
    const Tensor<int>& labels = sp->getLabelsData();
    // weird behaviour (should be this line but it only works with the opposite)
    // int _targeted = targeted ? 1 : (-1);
    int _targeted = targeted ? (-1) : 1;
    unsigned int nbAttempts = 0;

    // first layer after env cell
    // used to retrieve diffOutputs from this layer
    std::shared_ptr<Cell_Frame_Top> firstLayer 
        = std::dynamic_pointer_cast<Cell_Frame_Top>(deepNet->getCell(deepNet->getLayers()[1][0]));

    std::vector<int> successes;
    successes.resize(dataInput.dimB(), -1);
    unsigned int counterSuccess = 0;

    if (random_start) {
        for (unsigned int i = 0; i < dataInput.size(); ++i) {
            dataInput(i) += eps * Random::randUniform(-1.0, 1.0);
            dataInput(i) = std::max(0.0f, std::min(dataInput(i), 1.0f));
        }
    }

    while (counterSuccess < successes.size() 
            && nbAttempts < nbIter) {
        
        sp->synchronize();
        
        // Requires to call Database::Learn to access all the information
        // provided by Target::provideTargets
        deepNet->propagate(Database::Learn, true, NULL);
        deepNet->backPropagate(NULL);

        // Saving signed gradients in gradInputs
        firstLayer->getDiffOutputs().synchronizeDToH();
        Tensor<Float_T> gradInputs = tensor_cast<Float_T>(firstLayer->getDiffOutputs());
        deepNet->getTarget()->getEstimatedLabels().synchronizeDToH();
        Tensor<int> top1Indexes = tensor_cast<int>(deepNet->getTarget()->getEstimatedLabels());

        for (unsigned int i = 0; i < dataInput.dimB(); ++i) {
            if (successes[i] == -1) {
                bool success = false;
                if (targeted) {
                    if (top1Indexes[i](0) == labels[i](0)) {
                        success = true;
                    }
                } else {
                    if (top1Indexes[i](0) != labels[i](0)) {
                        success = true;
                    }
                }

                if (!success) {
                    if (nbAttempts != nbIter-1) {
                        for (unsigned int j = 0; j < dataInput[i].size(); ++j) {
                            gradInputs[i](j) = (Float_T(0) < gradInputs[i](j)) - (gradInputs[i](j) < Float_T(0));   // signed gradiants
                            dataInput[i](j) -= _targeted * alpha * gradInputs[i](j);
                            dataInput[i](j) = std::max(dataInputCopy[i](j) - eps, std::min(dataInput[i](j), dataInputCopy[i](j) + eps));
                            dataInput[i](j) = std::max(0.0f, std::min(dataInput[i](j), 1.0f));                      // clamping
                        }
                    }
                } else {
                    successes[i] = nbAttempts + 1;
                    ++counterSuccess;
                }
            }
        }
        ++nbAttempts;
    }
}
