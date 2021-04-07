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

#include "Database/CSV_Database.hpp"

N2D2::CSV_Database::CSV_Database()
    : Database(true)
{
    // ctor
    mCsvLocale = std::locale(std::locale(),
                             new N2D2::Utils::streamIgnore(",; \t"));
}

void N2D2::CSV_Database::load(const std::string& dataPath,
                              const std::string& /*labelPath*/,
                              bool /*extractROIs*/)
{
    loadRaw(dataPath);
}

void N2D2::CSV_Database::loadRaw(const std::string& fileName,
                                 int labelColumn,
                                 int nbHeaderLine)
{
    // 1. Read data
    std::ifstream data(fileName.c_str());

    if (!data.good()) {
        throw std::runtime_error("CSV_Database::load():"
            " could not open data file: " + fileName);
    }

    std::string line;
    size_t nbColumns = 0;
    int labelCol = labelColumn;
    size_t numLine = 0;

    while (std::getline(data, line)) {
        ++numLine;

        // Skip header lines
        if ((int)numLine <= nbHeaderLine)
            continue;

        // Skip empty lines
        if (line.empty())
            continue;

        if (labelCol < 0) {
            std::stringstream values(line);
            values.imbue(mCsvLocale);

            std::string column;
            while (values >> column)
                ++nbColumns;

            labelCol = nbColumns + labelColumn;
        }

        std::stringstream values(line);
        values.imbue(mCsvLocale);

        std::string labelName;
        Tensor<float> rawData;

        for (int col = 0; values.good() && values.peek() != EOF; ++col) {
            if (col == labelCol) {
                if (!(values >> Utils::quoted(labelName))) {
                    throw std::runtime_error("Unreadable label on line \""
                        + line + "\" in data file: " + fileName);
                }
            }
            else if (values.peek() != EOF) {
                float value;

                if (!(values >> value)) {
                    throw std::runtime_error("Unreadable value on line \""
                        + line + "\" in data file: " + fileName);
                }

                rawData.push_back(value);
            }
        }

        std::ostringstream nameStr;
        nameStr << Utils::baseName(fileName) << "[" << numLine << "]";

        mStimuli.push_back(Stimulus(nameStr.str(), labelID(labelName)));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);

        cv::Mat data_mat = rawData;
        mStimuliData.push_back(data_mat.clone());
    }
}
