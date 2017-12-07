/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "Database/IMDBWIKI_Database.hpp"

const std::locale
N2D2::IMDBWIKI_Database::csvIMDBLocale(std::locale(),
                                       new N2D2::Utils::streamIgnore(";"));

N2D2::IMDBWIKI_Database::IMDBWIKI_Database(
    bool WikiSet, bool IMDBSet, bool CropFrame, double learn, double validation)
    : DIR_Database(),
      mWiki(WikiSet),
      mIMDB(IMDBSet),
      mCrop(CropFrame),
      mLearn(learn),
      mValidation(validation),
      mNbCorruptedFrames(0)
{
    // ctor
}

void N2D2::IMDBWIKI_Database::load(const std::string& dataPath,
                                   const std::string& labelPath,
                                   bool /*extractROIs*/)
{

    if (mIMDB) {
        if (!mCrop)
            loadStimuli(dataPath + "/imdb", labelPath + "/imdb.csv");
        else
            loadStimuli(dataPath + "/imdb_crop", labelPath + "/imdb.csv");
    }

    if (mWiki) {
        if (!mCrop)
            loadStimuli(dataPath + "/wiki", labelPath + "/wiki.csv");
        else
            loadStimuli(dataPath + "/wiki_crop", labelPath + "/wiki.csv");
    }

    partitionStimuli(mLearn, mValidation, 1.0 - mLearn - mValidation);
}

void N2D2::IMDBWIKI_Database::loadStimuli(const std::string& dirPath,
                                          const std::string& labelPath)
{
    const unsigned int nbMsgMax = 100;

    /**
    Loading the "label_file".csv used to get :
        -stimuli path
        -stimuli name
        -stimuli x0,y0,x1,y1 of the ROI
        -stimuli date of birth
        -stimuli date of shooting
        -stimuli gender
        -stimuli face score gives by the IMDB laboratory
        -stimuli second face score
    Each of these parameters are accessible throught facesParam
    **/
    const std::vector<FaceParameters> facesParam
        = loadFaceParameters(labelPath);

    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    unsigned int nbMsg = 0;

    for (unsigned int face = 0; face < facesParam.size(); ++face) {
        const double age = facesParam[face].photo_taken - facesParam[face].dob;

        if ((facesParam[face].x0 == 1 && facesParam[face].y0 == 1)
            || (facesParam[face].x1 == 1 && facesParam[face].y1 == 1))
        {
            ++nbMsg;
            ++mNbCorruptedFrames;

            if (nbMsgMax > 0 && nbMsg < nbMsgMax) {
                std::cout << Utils::cwarning << "Wrong label location for frame "
                          << dirPath + "/" + facesParam[face].full_path
                          << Utils::cdef
                          << " (corrupted frame(s): " << mNbCorruptedFrames
                          << ")" << std::endl;
            }
        }
        else {
            std::ostringstream labelStr;
            if (facesParam[face].gender != 0.0
                && facesParam[face].gender != 1.0)
            {
                ++nbMsg;

                if (nbMsgMax > 0 && nbMsg < nbMsgMax) {
                    std::cout << Utils::cnotice
                              << "Gender type not defined for frame "
                              << dirPath + "/" + facesParam[face].full_path
                              << Utils::cdef << std::endl;
                }

                labelStr << "?";
            }
            else
                labelStr << ((facesParam[face].gender == 0.0) ? "F" : "M");

            labelStr << "-";

            if (age < 0.0 || age > 99.0) {
                ++nbMsg;

                if (nbMsgMax > 0 && nbMsg < nbMsgMax) {
                    std::cout << Utils::cnotice
                              << "Age out of bound (negative or greater than 100 "
                                 "years old) for frame "
                              << dirPath + "/" + facesParam[face].full_path
                              << Utils::cdef << std::endl;
                }

                labelStr << "?";
            }
            else
                labelStr << age;

            if (!mCrop) {
                mStimuli.push_back(
                    Stimulus(dirPath + "/" + facesParam[face].full_path, -1));
                mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
                mStimuli.back().ROIs.push_back(new RectangularROI<int>(
                    labelID(labelStr.str()),
                    RectangularROI
                    <int>::Point_T(facesParam[face].x0, facesParam[face].y0),
                    facesParam[face].x1 - facesParam[face].x0,
                    facesParam[face].y1 - facesParam[face].y0));
            } else
                loadFile(dirPath + "/" + facesParam[face].full_path,
                         labelID(labelStr.str()));
        }

        if (nbMsgMax > 0 && nbMsg == nbMsgMax) {
            std::cout << Utils::cnotice << "Already " << nbMsg << " messages, "
                "ignoring further notices..." << Utils::cdef << std::endl;

            ++nbMsg;
        }
    }

    if (nbMsgMax > 0 && nbMsg > nbMsgMax + 1) {
        std::cout << Utils::cwarning << (nbMsg - nbMsgMax - 1)
            << " messages (warning and/or notices) were silenced"
            << Utils::cdef << std::endl;
    }
}

std::vector<N2D2::IMDBWIKI_Database::FaceParameters>
N2D2::IMDBWIKI_Database::loadFaceParameters(const std::string& path) const
{
    std::ifstream file(path.c_str());

    if (!file.good())
        throw std::runtime_error(
            "IMDBWIKI_Database::loadSetFaceParameters(): could not open "
            + path);

    std::string line;
    std::vector<FaceParameters> data;

    while (std::getline(file, line)) {
        // Left trim & right trim
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

        if (line[0] == '#') {
            std::cout << Utils::cnotice << "Ignoring commented line: \"" << line
                      << "\" in file: " << path << Utils::cdef << std::endl;
            continue;
        }

        // Replace HTML special chars
        line = Utils::searchAndReplace(line, "&amp;", "&");

        std::stringstream values(line);
        values.imbue(csvIMDBLocale);

        FaceParameters fp;

        if (!(values >> fp.full_path) || values.get() != ';'
            || (values.peek() != ';'
                && !(values >> fp.name)) // handle the case of missing name
            || !(values >> fp.x0) || !(values >> fp.y0) || !(values >> fp.x1)
            || !(values >> fp.y1) || !(values >> fp.dob)
            || !(values >> fp.photo_taken) || !(values >> fp.gender)) {
            throw std::runtime_error("IMDBWIKI_Database::loadSetFaceParameters("
                                     "): unreadable value in line \"" + line
                                     + "\" for file: " + path);
        } else
            data.push_back(fp);
    }

    std::cout << "IMDBWIKI_DATABASE: Found: " << data.size()
              << " data labelled." << std::endl;
    return data;
}
