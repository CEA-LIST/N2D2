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

#include "Scaling.hpp"
#include "Activation/Activation.hpp"

N2D2::Activation::Activation()
    : mScaling()
{
    // ctor
}

void N2D2::Activation::propagate(const Cell& cell,
                                 BaseTensor& inOut,
                                 bool inference)
{
    propagate(cell, inOut, inOut, inference);
}

void N2D2::Activation::backPropagate(const Cell& cell,
                                     const BaseTensor& output,
                                     BaseTensor& diffInOut)
{
    backPropagate(cell, output, output, diffInOut, diffInOut);
}

const N2D2::Scaling& N2D2::Activation::getActivationScaling() const {
    return mScaling;
}

void N2D2::Activation::setActivationScaling(Scaling scaling) {
    mScaling = std::move(scaling);
}
void N2D2::Activation::exportParameters(const std::string& dirName,
                                        const std::string& cellName) const
{
    if (mQuantizer)
        mQuantizer->exportParameters(dirName, cellName);
}

void N2D2::Activation::importParameters(const std::string& dirName,
                            const std::string& cellName,
                            const bool ignoreNotExists)
{
    if (mQuantizer)
        mQuantizer->importParameters(dirName, cellName, ignoreNotExists);
}

void N2D2::Activation::save(const std::string& dirName) const
{
    Utils::createDirectories(dirName);

    // Save parameters
    saveParameters(dirName + "/" + getType() + ".cfg");

    // Save internal state
    const std::string fileName = dirName + "/" + getType() + ".state";

    std::ofstream state(fileName.c_str(), std::fstream::binary);

    if (!state.good())
        throw std::runtime_error("Could not create state file (.STATE): "
                                 + fileName);

    const std::string logFileName = dirName + "/" + getType() + ".log";

    std::ofstream log(logFileName.c_str());

    if (!log.good())
        throw std::runtime_error("Could not create log file (.LOG): "
                                 + logFileName);

    saveInternal(state, log);

    if (!state.good())
        throw std::runtime_error("Error writing state file: " + fileName);

    if (!log.good())
        throw std::runtime_error("Error writing log file: " + logFileName);
}

void N2D2::Activation::load(const std::string& dirName)
{
    // Load parameters
    loadParameters(dirName + "/" + getType() + ".cfg");

    // Load internal state
    const std::string fileName = dirName + "/" + getType() + ".state";

    std::ifstream state(fileName.c_str(), std::fstream::binary);

    if (!state.good())
        throw std::runtime_error("Could not open state file (.STATE): "
                                 + fileName);

    loadInternal(state);

    if (state.eof()) {
        throw std::runtime_error(
            "End-of-file reached prematurely in state file (.STATE): "
            + fileName);
    }
    else if (!state.good()) {
        throw std::runtime_error("Error while reading state file (.STATE): "
                                 + fileName);
    }
    else if (state.get() != std::fstream::traits_type::eof()) {
        throw std::runtime_error(
            "State file (.STATE) size larger than expected: "
            + fileName);
    }
}
