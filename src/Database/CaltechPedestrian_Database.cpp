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

#include "Database/CaltechPedestrian_Database.hpp"

N2D2::CaltechPedestrian_Database::CaltechPedestrian_Database(double validation,
                                                             bool singleLabel,
                                                             bool incAmbiguous)
    : mValidation(validation),
      mSingleLabel(singleLabel),
      mIncAmbiguous(incAmbiguous)
{
    // ctor
}

void N2D2::CaltechPedestrian_Database::load(const std::string& dataPath,
                                            const std::string& labelPath,
                                            bool /*extractROIs*/)
{
    const std::string labelPathDef = (labelPath.empty()) ? dataPath : labelPath;

    std::cout << "Loading database [                    ]   0%  " << std::flush;
    loadSet(dataPath + "/set00", labelPathDef + "/set00");
    std::cout << "\rLoading database [=>                  ]   9%  "
              << std::flush;
    loadSet(dataPath + "/set01", labelPathDef + "/set01");
    std::cout << "\rLoading database [===>                ]  18%  "
              << std::flush;
    loadSet(dataPath + "/set02", labelPathDef + "/set02");
    std::cout << "\rLoading database [====>               ]  27%  "
              << std::flush;
    loadSet(dataPath + "/set03", labelPathDef + "/set03");
    std::cout << "\rLoading database [======>             ]  36%  "
              << std::flush;
    loadSet(dataPath + "/set04", labelPathDef + "/set04");
    std::cout << "\rLoading database [========>           ]  45%  "
              << std::flush;
    loadSet(dataPath + "/set05", labelPathDef + "/set05");
    std::cout << "\rLoading database [==========>         ]  55%  "
              << std::flush;
    partitionStimuli(1.0 - mValidation, mValidation, 0.0);

    loadSet(dataPath + "/set06", labelPathDef + "/set06");
    std::cout << "\rLoading database [============>       ]  64%  "
              << std::flush;
    loadSet(dataPath + "/set07", labelPathDef + "/set07");
    std::cout << "\rLoading database [==============>     ]  73%  "
              << std::flush;
    loadSet(dataPath + "/set08", labelPathDef + "/set08");
    std::cout << "\rLoading database [===============>    ]  82%  "
              << std::flush;
    loadSet(dataPath + "/set09", labelPathDef + "/set09");
    std::cout << "\rLoading database [=================>  ]  91%  "
              << std::flush;
    loadSet(dataPath + "/set10", labelPathDef + "/set10");
    std::cout << "\rLoading database [====================] 100%  "
              << std::endl;
    partitionStimuli(0.0, 0.0, 1);
}

void N2D2::CaltechPedestrian_Database::loadSet(const std::string& dataPath,
                                               const std::string& labelPath)
{
    DIR* pDir = opendir(dataPath.c_str());

    if (pDir == NULL)
        throw std::runtime_error("CaltechPedestrian_Database::loadSet(): "
                                 "couldn't open database directory: "
                                 + dataPath);

    struct dirent* pFile;
    struct stat fileStat;
    std::vector<std::string> subDirs;
    std::vector<std::string> files;

    while ((pFile = readdir(pDir))) {
        const std::string fileName(pFile->d_name);
        const std::string filePath(dataPath + "/" + fileName);

        // Ignore file in case of stat failure
        if (stat(filePath.c_str(), &fileStat) < 0)
            continue;
        // Exclude current and parent directories
        if (!strcmp(pFile->d_name, ".") || !strcmp(pFile->d_name, ".."))
            continue;

        if (S_ISDIR(fileStat.st_mode))
            subDirs.push_back(fileName);
        else {
            if (Utils::fileExtension(fileName) == "jpg")
                files.push_back(fileName);
            else {
                std::cout << Utils::cnotice << "Notice: file " << fileName
                          << " does not appear to be a valid stimulus,"
                             " ignoring." << Utils::cdef << std::endl;
            }
        }
    }

    closedir(pDir);

    if (!files.empty()) {
        // Load stimuli contained in this directory
        std::sort(files.begin(), files.end());

        for (std::vector<std::string>::const_iterator it = files.begin(),
                                                      itEnd = files.end();
             it != itEnd;
             ++it) {
            mStimuli.push_back(Stimulus(dataPath + "/" + (*it), -1));
            mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);

            loadBackStimulusROIs(labelPath + "/" + Utils::fileBaseName(*it)
                                 + ".txt");
        }
    }

    // Recursively load stimuli contained in the subdirectories
    std::sort(subDirs.begin(), subDirs.end());

    for (std::vector<std::string>::const_iterator it = subDirs.begin(),
                                                  itEnd = subDirs.end();
         it != itEnd;
         ++it)
        loadSet(dataPath + "/" + (*it), labelPath + "/" + (*it));
}

void N2D2::CaltechPedestrian_Database::loadBackStimulusROIs(const std::string
                                                            & fileName)
{
    std::ifstream dataRoi(fileName.c_str());

    if (!dataRoi.good())
        throw std::runtime_error("CaltechPedestrian_Database::"
                                 "loadBackStimulusROIs(): could not open ROI "
                                 "data file: " + fileName);

    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    std::string line;

    while (std::getline(dataRoi, line)) {
        // Remove optional comments
        line.erase(std::find(line.begin(), line.end(), '%'), line.end());
        // Left trim & right trim (right trim necessary for extra "!value.eof()"
        // check later)
        line.erase(
            line.begin(),
            std::find_if(line.begin(),
                         line.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
        line.erase(std::find_if(line.rbegin(),
                                line.rend(),
                                std::not1(std::ptr_fun<int, int>(std::isspace)))
                       .base(),
                   line.end());

        if (line.empty())
            continue;

        std::stringstream values(line);
        std::string label;

        if (!(values >> label))
            throw std::runtime_error("Unreadable line in data file: "
                                     + fileName);

        // Not a person or people
        if (label == "person-fa")
            continue;

        if (label == "people" && mSingleLabel)
            label = "person";

        int bb_l; // values are sometimes negative in the annotation files
        unsigned int bb_t, bb_w, bb_h;
        bool occ;
        int bbv_l; // values are sometimes negative in the annotation files
        unsigned int bbv_t, bbv_w, bbv_h;
        bool ign;
        unsigned int ang;

        if (!(values >> bb_l)
            || !(Utils::signChecked<unsigned int>(values) >> bb_t)
            || !(Utils::signChecked<unsigned int>(values) >> bb_w)
            || !(Utils::signChecked<unsigned int>(values) >> bb_h)
            || !(values >> occ) || !(values >> bbv_l)
            || !(Utils::signChecked<unsigned int>(values) >> bbv_t)
            || !(Utils::signChecked<unsigned int>(values) >> bbv_w)
            || !(Utils::signChecked<unsigned int>(values) >> bbv_h)
            || !(values >> ign)
            || !(Utils::signChecked<unsigned int>(values) >> ang)) {
            throw std::runtime_error("Unreadable value in data file: "
                                     + fileName);
        }

        if (!values.eof())
            throw std::runtime_error("Extra data at end of line in data file: "
                                     + fileName);

        // Check BB
        if (bb_l < 0) {
            // std::cout << Utils::cwarning << "BB left border < 0 for image: "
            // << fileName << Utils::cdef << std::endl;
            bb_w -= -bb_l;
            bb_l = 0;
        }

        if (bb_l + bb_w > 640) {
            // std::cout << Utils::cwarning << "BB right border > 640 for image:
            // " << fileName << Utils::cdef << std::endl;
            bb_w = 640 - bb_l;
        }

        if (bb_t + bb_h > 480) {
            // std::cout << Utils::cwarning << "BB bottom border > 480 for
            // image: " << fileName << Utils::cdef << std::endl;
            bb_h = 480 - bb_t;
        }

        // Check BBV
        if (bbv_l < 0) {
            // std::cout << Utils::cwarning << "BBV left border < 0 for image: "
            // << fileName << Utils::cdef << std::endl;
            bbv_w -= -bbv_l;
            bbv_l = 0;
        }

        if (bbv_l + bbv_w > 640) {
            // std::cout << Utils::cwarning << "BBV right border > 640 for
            // image: " << fileName << Utils::cdef << std::endl;
            bbv_w = 640 - bbv_l;
        }

        if (bbv_t + bbv_h > 480) {
            // std::cout << Utils::cwarning << "BBV bottom border > 480 for
            // image: " << fileName << Utils::cdef << std::endl;
            bbv_h = 480 - bbv_t;
        }

        if (label == "person?") {
            if (mIncAmbiguous)
                label = "person";
            else
                ign = true;
        }

        const int labelId = (!ign) ? labelID(label) : -1;

        if (occ) {
            mStimuli.back().ROIs.push_back(new RectangularROI<int>(
                labelId,
                RectangularROI<int>::Point_T(bbv_l, bbv_t),
                bbv_w,
                bbv_h));
        } else {
            mStimuli.back().ROIs.push_back(new RectangularROI<int>(
                labelId, RectangularROI<int>::Point_T(bb_l, bb_t), bb_w, bb_h));
        }
    }
}
