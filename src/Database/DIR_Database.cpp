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

#include "Database/DIR_Database.hpp"

N2D2::DIR_Database::DIR_Database(bool loadDataInMemory)
    : Database(loadDataInMemory)
{
    // ctor
}

void N2D2::DIR_Database::setValidExtensions(
    const std::vector<std::string>& validExtensions)
{
    if (!validExtensions.empty())
        std::cout << "Valid extensions are: " << validExtensions << std::endl;

    mValidExtensions = validExtensions;
}

void N2D2::DIR_Database::load(const std::string& dataPath,
                              const std::string& /*labelPath*/,
                              bool /*extractROIs*/)
{
    loadDir(dataPath, 1, "", 1);
}

void N2D2::DIR_Database::loadDir(const std::string& dirPath,
                                 int depth,
                                 const std::string& labelName,
                                 int labelDepth)
{
    DIR* pDir = opendir(dirPath.c_str());

    if (pDir == NULL)
        throw std::runtime_error("Couldn't open database directory: "
                                 + dirPath);

    struct dirent* pFile;
    struct stat fileStat;
    std::vector<std::string> subDirs;
    std::vector<std::string> files;

    std::cout << "Loading directory database \"" << dirPath << "\""
              << std::endl;

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
            // Exclude files with wrong extension
            std::string fileExtension = Utils::fileExtension(fileName);
            std::transform(fileExtension.begin(),
                           fileExtension.end(),
                           fileExtension.begin(),
                           ::tolower);

            if (mValidExtensions.empty() || std::find(mValidExtensions.begin(),
                                                      mValidExtensions.end(),
                                                      fileExtension)
                                            != mValidExtensions.end()) {
                if (!Registrar<DataFile>::exists(fileExtension)) {
                    std::cout << Utils::cnotice << "Notice: file " << fileName
                              << " does not appear to be a valid stimulus,"
                                 " ignoring." << Utils::cdef << std::endl;
                    continue;
                }

                files.push_back(filePath);
            }
        }
    }

    closedir(pDir);

    if (!files.empty()) {
        // Load stimuli contained in this directory
        std::sort(files.begin(), files.end());

        const int dirLabelID = (labelDepth >= 0) ? labelID(labelName) : -1;

        for (std::vector<std::string>::const_iterator it = files.begin(),
                                                      itEnd = files.end();
             it != itEnd;
             ++it) {
            mStimuli.push_back(Stimulus(*it, dirLabelID));
            mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
        }
    }

    if (depth != 0) {
        // Recursively load stimuli contained in the subdirectories
        std::sort(subDirs.begin(), subDirs.end());

        for (std::vector<std::string>::const_iterator it = subDirs.begin(),
                                                      itEnd = subDirs.end();
             it != itEnd;
             ++it) {
            if (labelDepth > 0)
                loadDir(*it,
                        depth - 1,
                        labelName + "/" + Utils::baseName(*it),
                        labelDepth - 1);
            else
                loadDir(*it, depth - 1, labelName, labelDepth);
        }
    }

    std::cout << "Found " << mStimuli.size() << " stimuli" << std::endl;
}

void N2D2::DIR_Database::loadFile(const std::string& fileName)
{
    loadFile(fileName, -1);
}

void N2D2::DIR_Database::loadFile(const std::string& fileName,
                                  const std::string& labelName)
{
    loadFile(fileName, labelID(labelName));
}

void N2D2::DIR_Database::loadFile(const std::string& fileName, int label)
{
    if (!std::ifstream(fileName.c_str()).good())
        throw std::runtime_error(
            "DIR_Database::loadFile(): File does not exist: " + fileName);

    mStimuli.push_back(Stimulus(fileName, label));
    mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
}
