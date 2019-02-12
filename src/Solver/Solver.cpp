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

#include "Solver/Solver.hpp"
#include "utils/Utils.hpp"

unsigned long long int N2D2::Solver::mMaxSteps = 0;
unsigned long long int N2D2::Solver::mLogSteps = 0;
double N2D2::Solver::mGlobalLearningRate = 0.0;

void N2D2::Solver::save(const std::string& dirName) const
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

void N2D2::Solver::load(const std::string& dirName)
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
