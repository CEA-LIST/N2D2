
/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND  (david.briand@cea.fr)

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

#include "Export/CPP_TensorRT/CPP_TensorRT_StimuliProvider.hpp"
#include "Export/CellExport.hpp"

void N2D2::CPP_TensorRT_StimuliProvider::generateCalibFiles(StimuliProvider& sp,
                                                            const std::string& dirName,
                                                            Database::StimuliSet set,
                                                            DeepNet* /*deepNet*/)
{
    
    Utils::createDirectories(dirName);
    const unsigned int envSizeX = sp.getSizeX();
    const unsigned int envSizeY = sp.getSizeY();
    const unsigned int nbChannels = sp.getNbChannels();
    const unsigned int nbStimuli = sp.getDatabase().getNbStimuli(set);
    std::cout << "-> Exporting calibrations files to \"" << dirName  << "\" folder" << std::endl;
    std::cout << "--> Stimuli calibrations dimensions: {" << nbChannels 
            << ", " << envSizeY << ", " << envSizeX << "} Numbers of calibrations files: "  
            << nbStimuli << std::endl;

    /*
        if (CellExport::mPrecision > 0) {
            StimuliData stimuliData(dirName + "_stats", sp);
            stimuliData.generate(sp.getDatabase().getStimuliSetMask(set));
            stimuliData.logValueRange();

            const StimuliData::Value& globalValue =
       stimuliData.getGlobalValue();
            const double normValue = std::max(std::abs(globalValue.minVal),
       std::abs(globalValue.maxVal));

            if (normValue != 1.0) {
                std::cout << Utils::cwarning << "Integer stimuli export with
       range != 1 (" << normValue << ")"
                    << Utils::cdef << std::endl;
            }

            wMax/= normValue;
        }
    */

#ifdef CUDA
    int dev = 0;
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    unsigned int progress = 0, progressPrev = 0;

#pragma omp parallel for schedule(dynamic)
    for (unsigned int i = 0; i < nbStimuli; ++i) {
#ifdef CUDA
        CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

        std::stringstream stimuliName;
        stimuliName << dirName << "/batch_calibration" << i << ".batch";

        std::ofstream calibStimuli(stimuliName.str().c_str(),
                                 std::fstream::binary);

        if (!calibStimuli.good()) {
#pragma omp critical(generateCalibFiles)
            throw std::runtime_error("Could not create stimuli binary file: "
                                     + stimuliName.str());
        }

        const uint32_t sizeX = envSizeX;
        const uint32_t sizeY = envSizeY;
        const uint32_t sizeZ = nbChannels;
        const uint32_t sizeN = 1U;
        
        calibStimuli.write(reinterpret_cast<const char*>(&sizeN),
                            sizeof(sizeN));
        calibStimuli.write(reinterpret_cast<const char*>(&sizeZ),
                            sizeof(sizeZ));
        calibStimuli.write(reinterpret_cast<const char*>(&sizeY),
                            sizeof(sizeY));
        calibStimuli.write(reinterpret_cast<const char*>(&sizeX),
                            sizeof(sizeX));

        StimuliProvider provider = sp.cloneParameters();
        provider.readStimulus(set, i);

        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            const Tensor<Float_T> frame = provider.getDataChannel(channel);

            for (unsigned int y = 0; y < envSizeY; ++y) {
                for (unsigned int x = 0; x < envSizeX; ++x) {

                    const float value = (float)frame(x, y);
                    calibStimuli.write(reinterpret_cast<const char*>(&value),
                                        sizeof(value));

                }
            }
        }

        if (!calibStimuli.good()) {
#pragma omp critical(generateCalibFiles)
            throw std::runtime_error("Error writing stimuli binary file: "
                                     + stimuliName.str());
        }

        calibStimuli.close();
        // Progress bar
        progress = (unsigned int)(20.0 * (i + 1) / (double)nbStimuli);

        if (progress > progressPrev) {
#pragma omp critical(generateCalibFiles)
            if (progress > progressPrev) {
                std::cout << std::string(progress - progressPrev, '.')
                        << std::flush;
                progressPrev = progress;
            }
        }
    }
}
