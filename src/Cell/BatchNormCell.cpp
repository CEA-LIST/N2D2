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
    std::ofstream syn(fileName.c_str());

    if (!syn.good())
        throw std::runtime_error("Could not create parameter file: "
                                 + fileName);

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int oy = 0; oy < 1; ++oy) {
            for (unsigned int ox = 0; ox < 1; ++ox) {
                syn << getScale(output, ox, oy) << " "
                    << getBias(output, ox, oy) << " " << getMean(output, ox, oy)
                    << " " << getVariance(output, ox, oy) << "\n";
            }
        }
    }
}

void N2D2::BatchNormCell::importFreeParameters(const std::string& fileName,
                                               bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str());

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open parameter file: " << fileName
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open parameter file: "
                                     + fileName);
    }

    double w;

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int oy = 0; oy < 1; ++oy) {
            for (unsigned int ox = 0; ox < 1; ++ox) {
                if (!(syn >> w))
                    throw std::runtime_error(
                        "Error while reading scale in parameter file: "
                        + fileName);

                setScale(output, ox, oy, w);

                if (!(syn >> w))
                    throw std::runtime_error(
                        "Error while reading bias in parameter file: "
                        + fileName);

                setBias(output, ox, oy, w);

                if (!(syn >> w))
                    throw std::runtime_error(
                        "Error while reading mean in parameter file: "
                        + fileName);

                setMean(output, ox, oy, w);

                if (!(syn >> w))
                    throw std::runtime_error(
                        "Error while reading variance in parameter file: "
                        + fileName);

                if (w < 0.0)
                    throw std::runtime_error(
                        "Negative variance in parameter file: " + fileName);

                setVariance(output, ox, oy, w);
            }
        }
    }

    // Discard trailing whitespaces
    while (std::isspace(syn.peek()))
        syn.ignore();

    if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Parameter file size larger than expected: "
                                 + fileName);
}

void N2D2::BatchNormCell::getStats(Stats& stats) const
{
    stats.nbNodes += getNbOutputs() * getOutputsWidth() * getOutputsHeight();
}

void N2D2::BatchNormCell::setOutputsSize()
{
    mOutputsWidth = mChannelsWidth;
    mOutputsHeight = mChannelsHeight;
}
