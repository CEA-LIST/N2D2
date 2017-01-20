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

#include "Database/CKP_Database.hpp"

const char* N2D2::CKP_Database::Emotion[] = {
    /* 0 */ "neutral",
    /* 1 */ "anger",
    /* 2 */ "contempt",
    /* 3 */ "disgust",
    /* 4 */ "fear",
    /* 5 */ "happy",
    /* 6 */ "sadness",
    /* 7 */ "surprise"};

N2D2::CKP_Database::CKP_Database(double learn,
                                 double validation,
                                 unsigned int nbNeutral)
    : mLearn(learn), mValidation(validation), mNbNeutral(nbNeutral)
{
    // ctor
}

void N2D2::CKP_Database::load(const std::string& dataPath,
                              const std::string& /*labelPath*/,
                              bool /*extractROIs*/)
{
    loadEmotionStimuli(dataPath);
    partitionStimuliPerLabel(
        mLearn, mValidation, 1.0 - mLearn - mValidation, true);
    partitionStimuli(0.0, 0.0, 1.0);
}

void N2D2::CKP_Database::loadEmotionStimuli(const std::string& dirPath)
{
    DIR* pDir = opendir(dirPath.c_str());

    if (pDir == NULL)
        throw std::runtime_error("CKP_Database::loadEmotionLabel(): couldn't "
                                 "open database directory: " + dirPath);

    struct dirent* pFile;
    struct stat fileStat;
    std::vector<std::string> subDirs;
    std::string emotionFile;
    std::vector<std::string> files;

    while ((pFile = readdir(pDir))) {
        const std::string fileName(pFile->d_name);
        const std::string filePath(dirPath + "/" + fileName);

        // Ignore file in case of stat failure
        if (stat(filePath.c_str(), &fileStat) < 0)
            continue;
        // Exclude current and parent directories
        if (!strcmp(pFile->d_name, ".") || !strcmp(pFile->d_name, ".."))
            continue;

        if (S_ISDIR(fileStat.st_mode))
            subDirs.push_back(filePath);
        else {
            const std::string fileExtension = Utils::fileExtension(fileName);

            if (fileExtension == "txt") {
                if (emotionFile.empty())
                    emotionFile = filePath;
                else
                    throw std::runtime_error("CKP_Database::loadEmotionStimuli("
                                             "): an emotion code already exist "
                                             "in: " + dirPath);
            } else if (fileExtension == "png")
                files.push_back(filePath);
            else {
                std::cout << Utils::cnotice << "Notice: file " << fileName
                          << " does not appear to be a valid stimulus,"
                             " ignoring." << Utils::cdef << std::endl;
            }
        }
    }

    closedir(pDir);

    if (!emotionFile.empty()) {
        const int emotionCode = loadEmotionLabel(emotionFile);
        const int neutralLabel = labelID(Emotion[0]);
        const int label = labelID(Emotion[emotionCode]);

        // Load stimuli contained in this directory
        std::sort(files.begin(), files.end());

        for (unsigned int i = 0, size = files.size(); i < size; ++i) {
            mStimuli.push_back(
                Stimulus(files[i], (i < mNbNeutral) ? neutralLabel : label));
            mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
        }
    }

    // Recursively load stimuli contained in the subdirectories
    std::sort(subDirs.begin(), subDirs.end());

    for (std::vector<std::string>::const_iterator it = subDirs.begin(),
                                                  itEnd = subDirs.end();
         it != itEnd;
         ++it)
        loadEmotionStimuli(*it);
}

unsigned int N2D2::CKP_Database::loadEmotionLabel(const std::string
                                                  & fileName) const
{
    std::ifstream labelFile(fileName.c_str());

    if (!labelFile.good())
        throw std::runtime_error(
            "CKP_Database::loadEmotionLabel(): could not open label file: "
            + fileName);

    double label;

    if (!(labelFile >> label))
        throw std::runtime_error(
            "CKP_Database::loadEmotionLabel(): unreadable label file: "
            + fileName);

    if (labelFile.get() != '\n' || labelFile.get() != EOF)
        throw std::runtime_error("CKP_Database::loadEmotionLabel(): extra data "
                                 "at end of line in label file: " + fileName);

    if (label != (double)((unsigned int)label))
        throw std::runtime_error(
            "CKP_Database::loadEmotionLabel(): wrong label type in label file: "
            + fileName);

    return (unsigned int)label;
}
