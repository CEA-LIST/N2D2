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
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpp_utils.hpp"
#include "env.hpp"
#include "Network.hpp"

template<typename Input_T, typename Output_T>
void readStimulus(const N2D2::Network& network, const std::string& stimulusPath, 
                  std::vector<Input_T>& inputBuffer, 
                  std::vector<Output_T>& expectedOutputBuffer)
{
    envRead(stimulusPath, inputBuffer.size(),
            network.inputHeight(), network.inputWidth(),
            (DATA_T*) inputBuffer.data(), //TODO
            expectedOutputBuffer.size(), expectedOutputBuffer.data());
}

template<typename Input_T, typename Output_T>
double processInput(const N2D2::Network& network, std::vector<Input_T>& inputBuffer, 
                            std::vector<Output_T>& expectedOutputBuffer,
                            std::vector<Output_T>& predictedOutputBuffer) 
{
    network.propagate(inputBuffer.data(), predictedOutputBuffer.data());

    std::size_t nbPredictions = 0;
    std::size_t nbValidPredictions = 0;

    assert(expectedOutputBuffer.size() == predictedOutputBuffer.size());
    for(std::size_t i = 0; i < expectedOutputBuffer.size(); i++) {
        if (expectedOutputBuffer[i] >= 0) {
            ++nbPredictions;

            if(predictedOutputBuffer[i] == expectedOutputBuffer[i]) {
                ++nbValidPredictions;
            }
        }
    }

    return (nbPredictions > 0)
        ? nbValidPredictions / (double)nbPredictions : 0.0;
}


int main(int argc, char* argv[]) {
    std::string stimulus;

    for(int iarg = 1; iarg < argc; iarg++) {
        const std::string arg = argv[iarg];
        if(arg == "-stimulus" && iarg + 1 < argc) {
            stimulus = argv[iarg + 1];
            iarg++;
        }
        else if(arg == "-h" || arg == "-help") {
            printf("%s [-stimulus stimulus]\n", argv[0]);
            std::exit(0);
        }
        else {
            N2D2_THROW_OR_ABORT(std::runtime_error,
                "Unknown argument '" + arg + "' "
                "or missing parameter(s) for the argument.\n"
                "Try '" + std::string(argv[0]) + "' -h for more information.");
        }
    }

#ifdef _OPENMP
    omp_set_num_threads(8);
#endif

    const N2D2::Network network{};

#if ENV_DATA_UNSIGNED
    std::vector<UDATA_T> inputBuffer(network.inputSize());
#else
    std::vector<DATA_T> inputBuffer(network.inputSize());
#endif

    std::vector<Target_T> expectedOutputBuffer(OUTPUTS_SIZE[0]);
    std::vector<Target_T> predictedOutputBuffer(OUTPUTS_SIZE[0]);

    double successRate;
    if(!stimulus.empty()) {
        readStimulus(network, stimulus, inputBuffer, expectedOutputBuffer);
        const double success = processInput(network, inputBuffer, 
                                                            expectedOutputBuffer, 
                                                            predictedOutputBuffer);
        
        successRate = success*100;
        printf("%02f/1\n", success);
    }
    else {
        const std::vector<std::string> stimuliFiles = getFilesList(STIMULI_DIRECTORY);

        double success = 0;
        for(auto it = stimuliFiles.begin(); it != stimuliFiles.end(); ++it) {
            const std::string& file = *it;

            readStimulus(network, file, inputBuffer, expectedOutputBuffer);
            const double nbValidRatio = processInput(network, inputBuffer, 
                                                                expectedOutputBuffer, 
                                                                predictedOutputBuffer);
            success += nbValidRatio;

            printf("%02f/%d (%02f%%)\n", success,
                (int)(std::distance(stimuliFiles.begin(), it) + 1),
                100.0*success/
                                 (std::distance(stimuliFiles.begin(), it) + 1));
        }

        successRate = success/stimuliFiles.size()*100;
        printf("\n\nScore: %02f%%\n", successRate);
    }

#ifdef OUTPUTFILE
    FILE *f = fopen("success_rate.txt", "w");
    if (f == NULL) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Could not create file:  success_rate.txt");
    }
    fprintf(f, "%f", successRate);
    fclose(f);
#endif
}

