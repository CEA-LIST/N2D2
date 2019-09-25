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

#include "LabelFile/CsvLabelFile.hpp"
#include "ROI/RectangularROI.hpp"
#include "utils/Utils.hpp"

N2D2::Registrar<N2D2::LabelFile> N2D2::CsvLabelFile::mRegistrar(
    {"csv", "dat", "txt"}, N2D2::CsvLabelFile::create);

N2D2::CsvLabelFile::CsvLabelFile():
    mNoImageSize(false)
{
    // ctor
    setReadDelimiters();
}

void N2D2::CsvLabelFile::setReadDelimiters(const std::string& delimiters)
{
    mCsvLocale = (delimiters.empty())
                     ? std::locale(std::locale(),
                                   new N2D2::Utils::streamIgnore(",; \t"))
                     : std::locale(std::locale(),
                                   new N2D2::Utils::streamIgnore(delimiters));
}

std::map<std::string, std::vector<N2D2::ROI*> >
N2D2::CsvLabelFile::read(const std::string& fileName,
                         std::function<int(const std::string&)> labelID)
{
    std::ifstream dataRoi(fileName.c_str());

    if (!dataRoi.good())
        throw std::runtime_error(
            "Database::loadROIs(): could not open ROI data file: " + fileName);

    std::string line;
    std::map<std::string, std::vector<ROI*> > ROIs;

    while (std::getline(dataRoi, line)) {
        // Remove optional comments
        line.erase(std::find(line.begin(), line.end(), '#'), line.end());
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
        values.imbue(mCsvLocale);

        std::string name;

        if (!(values >> name))
            throw std::runtime_error("Unreadable line in data file: "
                                     + fileName);

        // There is a ROI
        if (!mNoImageSize) {
            unsigned int width, height;

            if (!(Utils::signChecked<unsigned int>(values) >> width)
                || !(Utils::signChecked<unsigned int>(values) >> height)) {
                std::cout << Utils::cwarning
                          << "Warning: unreadable image size value on line \""
                          << line << "\" in data file: " << fileName
                          << Utils::cdef << std::endl;
                continue;
            }
        }

        // x2 and y2 are assumed to be exclusive
        double x1, y1, x2, y2;
        std::string label;

        if (!(values >> x1)
            || !(values >> y1)
            || !(values >> x2)
            || !(values >> y2)
            || !(values >> label)) {
            throw std::runtime_error("Unreadable value in data file: "
                                     + fileName);
        }

        if (x1 < 0 || x2 < 0 || y1 < 0 || y2 < 0) {
            std::cout << Utils::cwarning
                      << "Warning: negative coordinates on line \""
                      << line << "\" in data file: " << fileName
                      << Utils::cdef << std::endl;
        }

        if (!values.eof())
            throw std::runtime_error("Extra data at end of line in data file: "
                                     + fileName);

        std::map<std::string, std::vector<ROI*> >::iterator it;
        std::tie(it, std::ignore)
            = ROIs.insert(std::make_pair(name, std::vector<ROI*>()));

        (*it).second.push_back(
            new RectangularROI<int>(labelID(label),
                                    RectangularROI<int>::Point_T(x1, y1),
                                    RectangularROI<int>::Point_T(x2, y2)));
    }

    return ROIs;
}
