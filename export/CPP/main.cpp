/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifndef STIMULI_DIRECTORY
#define STIMULI_DIRECTORY "stimuli"
#endif

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


#include "cpp_utils.hpp"
#include "env.hpp"
#include "Network.hpp"

template<typename Input_T>
void readStimulus(const N2D2::Network& network, const std::string& stimulusPath, 
                  std::vector<Input_T>& inputBuffer, 
                  std::vector<std::int32_t>& expectedOutputBuffer)
{
    envRead(stimulusPath, inputBuffer.size(),
            network.inputHeight(), network.inputWidth(),
            (DATA_T*) inputBuffer.data(), //TODO
            expectedOutputBuffer.size(), expectedOutputBuffer.data());
}

template<typename Input_T>
std::size_t processInput(const N2D2::Network& network, std::vector<Input_T>& inputBuffer, 
                            std::vector<std::int32_t>& expectedOutputBuffer,
                            std::vector<std::int32_t>& predictedOutputBuffer) 
{
    network.propagate(inputBuffer.data(), predictedOutputBuffer.data());

    std::size_t nbValidPredictions = 0;
    assert(expectedOutputBuffer.size() == predictedOutputBuffer.size());
    for(std::size_t i = 0; i < expectedOutputBuffer.size(); i++) {
        if(predictedOutputBuffer[i] == expectedOutputBuffer[i]) {
            nbValidPredictions++;
        }
    }

    return nbValidPredictions;
}


int main(int argc, char* argv[]) try {
    std::string stimulus;

    for(int iarg = 1; iarg < argc; iarg++) {
        const std::string arg = argv[iarg];
        if(arg == "-stimulus" && iarg + 1 < argc) {
            stimulus = argv[iarg + 1];
            iarg++;
        }
        else if(arg == "-h" || arg == "-help") {
            std::cout << argv[0] << " [-stimulus stimulus]" << std::endl;
            std::exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument '" + arg + "' "
                                     "or missing parameter(s) for the argument.\n"
                                     "Try '" + std::string(argv[0]) + "' -h for more information.");
        }
    }

    const N2D2::Network network{};

#if ENV_DATA_UNSIGNED
    std::vector<UDATA_T> inputBuffer(network.inputSize());
#else
    std::vector<DATA_T> inputBuffer(network.inputSize());
#endif

    std::vector<std::int32_t> expectedOutputBuffer(network.outputHeight()*network.outputWidth());
    std::vector<std::int32_t> predictedOutputBuffer(network.outputHeight()*network.outputWidth());

    double successRate;
    if(!stimulus.empty()) {
        readStimulus(network, stimulus, inputBuffer, expectedOutputBuffer);
        const std::size_t nbValidPredictions = processInput(network, inputBuffer, 
                                                            expectedOutputBuffer, 
                                                            predictedOutputBuffer);
        
        successRate = 1.0*nbValidPredictions/expectedOutputBuffer.size()*100;
        std::cout << std::fixed << std::setprecision(2) 
                  << 1.0*nbValidPredictions/expectedOutputBuffer.size() << "/1" << std::endl;
    }
    else {
        const std::vector<std::string> stimuliFiles = getFilesList(STIMULI_DIRECTORY);

        double validPredictionsRatio = 0;
        for(auto it = stimuliFiles.begin(); it != stimuliFiles.end(); ++it) {
            const std::string& file = *it;

            readStimulus(network, file, inputBuffer, expectedOutputBuffer);
            const std::size_t nbValidPredictions = processInput(network, inputBuffer, 
                                                                expectedOutputBuffer, 
                                                                predictedOutputBuffer);
            validPredictionsRatio += 1.0*nbValidPredictions/expectedOutputBuffer.size();

            std::cout << std::fixed << std::setprecision(2) 
                      << validPredictionsRatio << "/" << (std::distance(stimuliFiles.begin(), it) + 1) 
                      << " (" << 100.0*validPredictionsRatio/
                                 (std::distance(stimuliFiles.begin(), it) + 1) << "%)"
                      << std::endl;
        }

        successRate = validPredictionsRatio/stimuliFiles.size()*100;
        std::cout << "\n\nScore: " << std::fixed << std::setprecision(2) << successRate << "%" << std::endl;
    }

#ifdef OUTPUTFILE
    std::ofstream success_result("success_rate.txt");
    if (!success_result.good()) {
        throw std::runtime_error("Could not create file:  success_rate.txt");
    }

    success_result << std::fixed << std::setprecision(2) << successRate;
#endif
}
catch(const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
}
