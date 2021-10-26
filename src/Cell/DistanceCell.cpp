/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#include <algorithm>
#include <limits>
#include <string>

#include "DeepNet.hpp"
#include "Cell/DistanceCell.hpp"
#include "utils/Registrar.hpp"
#include "Filler/Filler.hpp"

const char* N2D2::DistanceCell::Type = "Distance";

N2D2::RegistryMap_T& N2D2::DistanceCell::registry() {
    static RegistryMap_T rMap;
    return rMap;
}

N2D2::DistanceCell::DistanceCell(const DeepNet& deepNet, const std::string& name,
                               unsigned int nbOutputs, double margin, double centercoef)
    : Cell(deepNet, name, nbOutputs),
      mMargin(std::move(margin)),
      mCenterCoef(std::move(centercoef)),
      mEndIT(this, "EndRampIT", 0)
{
}

const char* N2D2::DistanceCell::getType() const {
    return Type;
}

void N2D2::DistanceCell::getStats(Stats& /*stats*/) const {
}

void N2D2::DistanceCell::exportFreeParameters(const std::string& fileName) const
{
    const std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string weightsFile = fileBase + "_means" + fileExt;

    std::ofstream weights(weightsFile.c_str());

    if (!weights.good())
        throw std::runtime_error("Could not create synaptic file: "
                                 + weightsFile);

    const unsigned int channelsSize = getInputsSize();

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < channelsSize; ++channel) {
            Tensor<Float_T> weight;
            getWeight(output, channel, weight);
            weights << weight(0) << " ";
        }
        weights << "\n";
    }
}

void N2D2::DistanceCell::importFreeParameters(const std::string& fileName,
                                        bool ignoreNotExists)
{
    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const bool singleFile = (std::ifstream(fileName.c_str()).good());
    const std::string weightsFile = (singleFile) ? fileName
        : fileBase + "_means" + fileExt;

    std::ifstream weights(weightsFile.c_str());

    if (!weights.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << weightsFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + weightsFile);
    }

    Tensor<double> weight({1});

    const unsigned int channelsSize = getInputsSize();
    std::cout << "Import means : check it is feature size (512):  " << channelsSize << std::endl;
    
    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < channelsSize; ++channel) {

            if (!(weights >> weight(0)))
                throw std::runtime_error("Error while reading synaptic file: "
                                        + fileName);

            setWeight(output, channel, weight);
        }
    }
    
    // Discard trailing whitespaces
    while (std::isspace(weights.peek()))
        weights.ignore();

    if (weights.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Synaptic file size larger than expected: "
                                 + weightsFile);
}

void N2D2::DistanceCell::setOutputsDims() {
    mOutputsDims[0] = mInputsDims[0];
    mOutputsDims[1] = mInputsDims[1];
}
