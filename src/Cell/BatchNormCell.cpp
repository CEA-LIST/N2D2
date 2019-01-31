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

#include "Cell/BatchNormCell.hpp"
#include "utils/Utils.hpp"

const char* N2D2::BatchNormCell::Type = "BatchNorm";

N2D2::BatchNormCell::BatchNormCell(const std::string& name,
                                   unsigned int nbOutputs)
    : Cell(name, nbOutputs), mEpsilon(this, "Epsilon", 0.0)
{
    // ctor
}

void N2D2::BatchNormCell::exportFreeParameters(const std::string
                                               & fileName) const
{
    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string scalesFile = fileBase + "_scales" + fileExt;
    const std::string biasesFile = fileBase + "_biases" + fileExt;
    const std::string meansFile = fileBase + "_means" + fileExt;
    const std::string variancesFile = fileBase + "_variances" + fileExt;

    std::ofstream scales(scalesFile.c_str());

    if (!scales.good())
        throw std::runtime_error("Could not create parameter file: "
                                 + scalesFile);

    std::ofstream biases(biasesFile.c_str());

    if (!biases.good())
        throw std::runtime_error("Could not create parameter file: "
                                 + biasesFile);

    std::ofstream means(meansFile.c_str());

    if (!means.good())
        throw std::runtime_error("Could not create parameter file: "
                                 + meansFile);

    std::ofstream variances(variancesFile.c_str());

    if (!variances.good())
        throw std::runtime_error("Could not create parameter file: "
                                 + variancesFile);

    std::shared_ptr<BaseTensor> baseScales = getScales();

    for (size_t index = 0; index < baseScales->size(); ++index) {
        Tensor<Float_T> scale;      getScale(index, scale);
        Tensor<Float_T> bias;       getBias(index, bias);
        Tensor<Float_T> mean;       getMean(index, mean);
        Tensor<Float_T> variance;   getVariance(index, variance);

        scales << scale(0) << "\n";
        biases << bias(0) << "\n";
        means << mean(0) << "\n";
        variances << variance(0) << "\n";
    }
}

void N2D2::BatchNormCell::importFreeParameters(const std::string& fileName,
                                               bool ignoreNotExists)
{
    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const bool singleFile = (std::ifstream(fileName.c_str()).good());
    const std::string scalesFile = (singleFile) ? fileName
        : fileBase + "_scales" + fileExt;
    const std::string biasesFile = (singleFile) ? fileName
        : fileBase + "_biases" + fileExt;
    const std::string meansFile = (singleFile) ? fileName
        : fileBase + "_means" + fileExt;
    const std::string variancesFile = (singleFile) ? fileName
        : fileBase + "_variances" + fileExt;

    std::ifstream biases(biasesFile.c_str());

    if (!biases.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << biasesFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + biasesFile);
    }

    std::ifstream scales_;
    std::ifstream means_;
    std::ifstream variances_;

    if (!singleFile) {
        scales_.open(scalesFile.c_str());

        if (!scales_.good()) {
            // Scales is optional: using default value of 1 if not found
            if (ignoreNotExists) {
                std::cout << Utils::cnotice
                          << "Notice: Could not open synaptic file: "
                          << scalesFile << Utils::cdef << std::endl;
            } else
                throw std::runtime_error("Could not open synaptic file: "
                                         + scalesFile);
        }

        means_.open(meansFile.c_str());

        if (!means_.good())
            throw std::runtime_error("Could not open synaptic file: "
                                     + meansFile);

        variances_.open(variancesFile.c_str());

        if (!variances_.good())
            throw std::runtime_error("Could not open synaptic file: "
                                     + variancesFile);
    }

    std::ifstream& scales = (!singleFile) ? scales_ : biases;
    std::ifstream& means = (!singleFile) ? means_ : biases;
    std::ifstream& variances = (!singleFile) ? variances_ : biases;
    
    means.precision(12);
    variances.precision(12);
    biases.precision(12);

    std::shared_ptr<BaseTensor> baseScales = getScales();
    std::vector<size_t> dim = {(size_t) 1};
    for (size_t index = 0; index < baseScales->size(); ++index) {
        double w;
        Tensor<Float_T> scale(dim, 1.0);
        Tensor<Float_T> bias(dim, 0.0);
        Tensor<Float_T> mean(dim, 0.0);
        Tensor<Float_T> variance(dim, 0.0);

        if (!(scales >> w))
        {
            if(!ignoreNotExists)
                throw std::runtime_error(
                    "Error while reading scale in parameter file: "
                    + scalesFile);
        }
        else
            scale(0) = w;

        if (!(biases >> w))
            throw std::runtime_error(
                "Error while reading bias in parameter file: "
                + biasesFile);

        bias(0) = w;

        if (!(means >> w))
            throw std::runtime_error(
                "Error while reading mean in parameter file: "
                + meansFile);

        mean(0) = w;

        if (!(variances >> w))
            throw std::runtime_error(
                "Error while reading variance in parameter file: "
                + variancesFile);

        if (w < 0.0)
            throw std::runtime_error(
                "Negative variance in parameter file: "
                + variancesFile);

        variance(0) = w;

        setScale(index, scale);
        setBias(index, bias);
        setMean(index, mean);
        setVariance(index, variance);
    }

    // Discard trailing whitespaces
    while (std::isspace(biases.peek()))
        biases.ignore();

    if (biases.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Parameter file size larger than expected: "
                                 + biasesFile);

    if (!singleFile) {
        // Discard trailing whitespaces
        while (std::isspace(scales.peek()))
            scales.ignore();

        if (scales.get() != std::fstream::traits_type::eof())
            throw std::runtime_error("Synaptic file size larger than expected: "
                                     + scalesFile);

        // Discard trailing whitespaces
        while (std::isspace(means.peek()))
            means.ignore();

        if (means.get() != std::fstream::traits_type::eof())
            throw std::runtime_error("Synaptic file size larger than expected: "
                                     + meansFile);

        // Discard trailing whitespaces
        while (std::isspace(variances.peek()))
            variances.ignore();

        if (variances.get() != std::fstream::traits_type::eof())
            throw std::runtime_error("Synaptic file size larger than expected: "
                                     + variancesFile);
    }
}

void N2D2::BatchNormCell::getStats(Stats& stats) const
{
    stats.nbNodes += getOutputsSize();
}

void N2D2::BatchNormCell::setOutputsDims()
{
    mOutputsDims = mInputsDims;
}
