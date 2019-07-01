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

#include "Database/Actitracker_Database.hpp"

N2D2::Actitracker_Database::Actitracker_Database(double learn,
                                                 double validation,
                                                 bool useUnlabeledForTest)
    : Database(true),
      mWindowSize(this, "WindowSize", 90),
      mOverlapping(this, "Overlapping", 0.5),
      mLearn(learn),
      mValidation(validation),
      mUseUnlabeledForTest(useUnlabeledForTest)
{
    // ctor
    mCsvLocale = std::locale(std::locale(),
                             new N2D2::Utils::streamIgnore(",; \t"));
}

void N2D2::Actitracker_Database::load(const std::string& dataPath,
                                      const std::string& /*labelPath*/,
                                      bool /*extractROIs*/)
{
    loadRaw(dataPath + "/WISDM_at_v2.0_raw.txt");
    partitionStimuli(mLearn, mValidation,
        (mUseUnlabeledForTest) ? 0.0 : (1.0 - mLearn - mValidation));

    if (mUseUnlabeledForTest) {
        loadRaw(dataPath + "/WISDM_at_v2.0_unlabeled_raw.txt");
        partitionStimuli(0.0, 0.0, 1.0);
    }
}

void N2D2::Actitracker_Database::loadRaw(const std::string& fileName)
{
    // Set activities labels in alphabetical order
    mLabelsName = {"Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"};
    
    const std::size_t nbActivities = mLabelsName.size();
    const unsigned int nbMsgMax = 100;

    // 1. Read data
    std::ifstream data(fileName.c_str());

    if (!data.good()) {
        throw std::runtime_error("Actitracker_Database::load():"
            " could not open data file: " + fileName);
    }

    std::string line;
    std::vector<RawData> rawData;
    unsigned int nbMsg = 0;

    while (std::getline(data, line)) {
        if (line.empty())
            continue;

        std::stringstream values(line);
        values.imbue(mCsvLocale);

        RawData rawDataRow;

        if (!(values >> rawDataRow.user)) {
            throw std::runtime_error("Unreadable user on line \"" + line
                + "\" in data file: " + fileName);
        }

        if (!(values >> rawDataRow.activity)) {
            ++nbMsg;

            // Files contain missing data like "579,,;"
            if (nbMsgMax > 0 && nbMsg < nbMsgMax) {
                std::cout << Utils::cwarning << "Unreadable activity on line \""
                    << line << "\" in data file: " << fileName << Utils::cdef
                    << std::endl;
            }

            continue;
        }

        if (!(values >> rawDataRow.timestamp)) {
            throw std::runtime_error("Unreadable timestamp on line \"" + line
                + "\" in data file: " + fileName);
        }

        if (!(values >> rawDataRow.xAcceleration)) {
            throw std::runtime_error("Unreadable x-acceleration on line \""
                + line + "\" in data file: " + fileName);
        }

        if (!(values >> rawDataRow.yAcceleration)) {
            throw std::runtime_error("Unreadable y-acceleration on line \""
                + line + "\" in data file: " + fileName);
        }

        if (!(values >> rawDataRow.zAcceleration)) {
            throw std::runtime_error("Unreadable z-acceleration on line \""
                + line + "\" in data file: " + fileName);
        }

        if (values.get() != ';' || values.get() != EOF) {
            throw std::runtime_error("Extra data at end of line \""
                + line + "\" in data file: " + fileName);
        }

        rawData.push_back(rawDataRow);
    }

    if (nbMsgMax > 0 && nbMsg > nbMsgMax + 1) {
        std::cout << Utils::cwarning << (nbMsg - nbMsgMax - 1)
            << " further messages (warning and/or notices) were silenced"
            << Utils::cdef << std::endl;
    }

    // 2. Generate stimuli
    const unsigned int overlap = Utils::round(mWindowSize * mOverlapping);
    const unsigned int nbSegments = (rawData.size() - mWindowSize) / overlap;

    mStimuli.reserve(mStimuli.size() + nbSegments);
    mStimuliData.reserve(mStimuliData.size() + nbSegments);

    for (unsigned int s = 0; s < nbSegments; ++s) {
        const unsigned int start = s * overlap;

        Tensor<float> data({3, mWindowSize});

        std::map<int, unsigned int> labels; // label -> nb_times_in_window
        int maxLabel = 0; // Most present label in the window
        unsigned int maxLabelCount = 0;

        for (unsigned int i = 0; i < mWindowSize; ++i) {
            data(0, i) = rawData[start + i].xAcceleration;
            data(1, i) = rawData[start + i].yAcceleration;
            data(2, i) = rawData[start + i].zAcceleration;

            const int label = labelID(rawData[start + i].activity);
            std::map<int, unsigned int>::iterator it;
            std::tie(it, std::ignore) = labels.insert(std::make_pair(label, 0));
            ++(*it).second;

            if ((*it).second > maxLabelCount) {
                maxLabel = label;
                maxLabelCount = (*it).second;
            }
        }

        std::ostringstream nameStr;
        nameStr << Utils::baseName(fileName) << "[" << start << ":"
            << (start + mWindowSize) << "]";
        mStimuli.push_back(Stimulus(nameStr.str(), maxLabel));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);

        /**
         * When converting a N2D2::Tensor to a cv::Mat, the 'N2D2::Tensor<T>::operator cv::Mat()'
         * method directly passes the pointer of the data in the Tensor to the created cv::Mat.
         * The returned cv::Mat is thus a shalow copy of the data in the Tensor that will 
         * become invalid at the end of the method as the Tensor data will be deleted. 
         * Make a clone to avoid this.
         * 
         * TODO: Modifiy 'N2D2::Tensor<T>::operator cv::Mat()', not intuitive and source of errors.
         */
        cv::Mat data_mat = data;
        mStimuliData.push_back(data_mat.clone());
    }

    // The call to labelID may add new unexpected activies in mLabelsName
    if(nbActivities != mLabelsName.size()) {
        throw std::runtime_error("Unexpected activities. "
                                 "We got " + std::to_string(mLabelsName.size()) + 
                                 " activities while we expected " + std::to_string(nbActivities) + 
                                 " activities");
    }
}
