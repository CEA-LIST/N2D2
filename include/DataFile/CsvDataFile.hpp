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

#ifndef N2D2_CSVDATAFILE_H
#define N2D2_CSVDATAFILE_H

#include <iomanip>

#include "DataFile/DataFile.hpp"
#include "containers/Tensor2d.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class CsvDataFile : public DataFile {
public:
    static std::shared_ptr<CsvDataFile> create()
    {
        return std::make_shared<CsvDataFile>();
    }

    CsvDataFile();
    void setReadDelimiters(const std::string& delimiters = "");
    virtual cv::Mat read(const std::string& fileName);
    virtual void write(const std::string& fileName, const cv::Mat& data);
    template <class T> Tensor2d<T> read(const std::string& fileName);
    template <class T>
    void write(const std::string& fileName,
               const Tensor2d<T>& data,
               char delimiter = ';');
    virtual ~CsvDataFile() {};

protected:
    std::locale mCsvLocale;

private:
    static Registrar<DataFile> mRegistrar;
};
}

template <class T>
N2D2::Tensor2d<T> N2D2::CsvDataFile::read(const std::string& fileName)
{
    std::ifstream dataFile(fileName.c_str());

    if (!dataFile.good())
        throw std::runtime_error("Could not open data file: " + fileName);

    std::string line;
    unsigned int nbRows = 0;
    unsigned int nbCols = 0;
    Tensor2d<T> data;

    while (std::getline(dataFile, line)) {
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

        unsigned int nbValues = 0;
        T value;

        while (values >> value) {
            ++nbValues;
            data.push_back(value);
        }

        if (!values.eof())
            throw std::runtime_error("Extra data at end of line in data file: "
                                     + fileName);

        if (nbCols == 0)
            nbCols = nbValues;
        else if (nbValues != nbCols)
            throw std::runtime_error("Wrong number of columns in data file: "
                                     + fileName);

        ++nbRows;
    }

    assert(data.data().size() == nbCols * nbRows);

    data.resize(nbCols, nbRows);
    return data;
}

template <class T>
void N2D2::CsvDataFile::write(const std::string& fileName,
                              const Tensor2d<T>& data,
                              char delimiter)
{
    std::ofstream dataFile(fileName.c_str());
    dataFile.precision(std::numeric_limits<T>::digits10 + 1);

    if (!dataFile.good())
        throw std::runtime_error("Could not create data file: " + fileName);

    for (unsigned int i = 0; i < data.dimY(); ++i) {
        for (unsigned int j = 0; j < data.dimX(); ++j) {
            if (j > 0)
                dataFile << delimiter;

            dataFile << data(j, i);
        }

        dataFile << "\n";
    }

    if (!dataFile.good())
        throw std::runtime_error("Error writing data file: " + fileName);
}

#endif // N2D2_CSVDATAFILE_H
