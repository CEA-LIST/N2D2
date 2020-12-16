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
    : mQuantizationLevels(this, "QuantizationLevels", 0U),
      mQuantizationDelay(this, "QuantizationDelay", 1000000U),
      mMovingAverage(this, "MovingAverage", EMA),
      mMA_Window(this, "MA_Window", 10000U),
      mEMA_Alpha(this, "EMA_Alpha", 0.0),
      mLog2RoundingRate(this, "Log2RoundingRate", 0.0),
      mLog2RoundingPower(this, "Log2RoundingPower", 1.0),
      mNbSteps(0),
      mPreQuantizeScaling(1.0),
      mScaling()
{
    // ctor
}


const N2D2::Scaling& N2D2::Activation::getActivationScaling() const {
    return mScaling;
}

void N2D2::Activation::setActivationScaling(Scaling scaling) {
    mScaling = scaling;
}

void N2D2::Activation::setPreQuantizeScaling(double scaling) {
    mPreQuantizeScaling = (scaling > 0.0) ? scaling : 1.0;
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

    state.write(reinterpret_cast<const char*>(&mNbSteps), sizeof(mNbSteps));
    state.write(reinterpret_cast<const char*>(&mPreQuantizeScaling),
                                              sizeof(mPreQuantizeScaling));

    const std::string logFileName = dirName + "/" + getType() + ".log";

    std::ofstream log(logFileName.c_str());

    if (!log.good())
        throw std::runtime_error("Could not create log file (.LOG): "
                                 + logFileName);

    log << "nbSteps: " << mNbSteps << "\n"
        << "preQuantizeScaling: " << mPreQuantizeScaling << std::endl;

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

    state.read(reinterpret_cast<char*>(&mNbSteps), sizeof(mNbSteps));
    state.read(reinterpret_cast<char*>(&mPreQuantizeScaling),
                                       sizeof(mPreQuantizeScaling));

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
