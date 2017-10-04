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

#include "Export/StimuliProviderExport.hpp"

void N2D2::StimuliProviderExport::generate(StimuliProvider& sp,
                                           const std::string& dirName,
                                           const std::string& type,
                                           Database::StimuliSet set,
                                           DeepNet* deepNet)
{
    if (Registrar<StimuliProviderExport>::exists(type)) {
        Registrar<StimuliProviderExport>::create(type)(sp,
                                                       dirName,
                                                       set,
                                                       deepNet);
    }
    else {
        // Default generator
        StimuliProviderExport::generate(sp,
                                        dirName,
                                        set,
                                        deepNet);
    }
}

void N2D2::StimuliProviderExport::generate(StimuliProvider& sp,
                                           const std::string& dirName,
                                           Database::StimuliSet set,
                                           DeepNet* deepNet)
{
    double wMax = (double)(std::pow(2,
                                    ((DeepNetExport::mEnvDataUnsigned)
                                         ? CellExport::mPrecision
                                         : (CellExport::mPrecision - 1))) - 1);

    Utils::createDirectories(dirName);

    const unsigned int envSizeX = sp.getSizeX();
    const unsigned int envSizeY = sp.getSizeY();
    const unsigned int nbChannels = sp.getNbChannels();
    const unsigned int size = sp.getDatabase().getNbStimuli(set);
    const unsigned int zeroPad = (size > 0) ? std::ceil(std::log10(size)) : 0;
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

    std::ofstream stimuliList((dirName + ".list").c_str());

    if (!stimuliList.good()) {
        throw std::runtime_error("Could not create stimuli list file: "
                                 + dirName + ".list");
    }

    for (unsigned int i = 0; i < size; ++i) {
        std::stringstream stimuliName;
        stimuliName << dirName << "/env" << std::setfill('0')
                    << std::setw(zeroPad) << i << ".pgm";
        stimuliList << Utils::baseName(dirName) << "/"
                    << Utils::baseName(stimuliName.str()) << "\n";

        std::ofstream envStimuli(stimuliName.str().c_str(),
                                 std::fstream::binary);

        if (!envStimuli.good())
            throw std::runtime_error("Could not create stimuli binary file: "
                                     + stimuliName.str());

        const unsigned int maxValue
            = (CellExport::mPrecision > 0 && CellExport::mPrecision <= 8)
                  ? 255
                  : 65535;

        envStimuli << "P5\n" << envSizeX << " " << envSizeY << "\n" << maxValue
                   << "\n";

        sp.readStimulus(set, i);

        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            const Tensor2d<Float_T> frame = sp.getData(channel);

            for (unsigned int y = 0; y < envSizeY; ++y) {
                for (unsigned int x = 0; x < envSizeX; ++x) {
                    if (CellExport::mPrecision == CellExport::Float64) {
                        const double value = (double)frame(x, y);
                        envStimuli.write(reinterpret_cast<const char*>(&value),
                                         sizeof(value));
                    } else if (CellExport::mPrecision == CellExport::Float32
                               || CellExport::mPrecision
                                  == CellExport::Float16) {
                        const float value = (float)frame(x, y);
                        envStimuli.write(reinterpret_cast<const char*>(&value),
                                         sizeof(value));
                    } else if (CellExport::mPrecision <= 8) {
                        const int8_t value = (int8_t)(wMax * frame(x, y));
                        envStimuli.write(reinterpret_cast<const char*>(&value),
                                         sizeof(value));
                    } else if (CellExport::mPrecision <= 16) {
                        const int16_t value = (int16_t)(wMax * frame(x, y));
                        envStimuli.write(reinterpret_cast<const char*>(&value),
                                         sizeof(value));
                    } else if (CellExport::mPrecision <= 32) {
                        const int32_t value = (int32_t)(wMax * frame(x, y));
                        envStimuli.write(reinterpret_cast<const char*>(&value),
                                         sizeof(value));
                    } else {
                        const int64_t value = (int64_t)(wMax * frame(x, y));
                        envStimuli.write(reinterpret_cast<const char*>(&value),
                                         sizeof(value));
                    }
                }
            }
        }

        const Tensor2d<int> label = sp.getLabelsData(0);
        const std::vector<std::shared_ptr<Target> > outputTargets
                                                    =  deepNet->getTargets();
        const unsigned int netNbTarget = outputTargets.size();

        for(unsigned int t = 0; t < netNbTarget; ++t) {
            for (unsigned int y = 0; y < label.dimY(); ++y) {
                for (unsigned int x = 0; x < label.dimX(); ++x) {
                    const uint32_t outputTarget
                        = (deepNet != NULL)
                              ? deepNet->getTarget(t)->getLabelTarget(label(x, y))
                              : label(x, y);
                    envStimuli.write(reinterpret_cast<const char*>(&outputTarget),
                                     sizeof(outputTarget));
                }
            }
        }
        if (!envStimuli.good())
            throw std::runtime_error("Error writing stimuli binary file: "
                                     + stimuliName.str());

        envStimuli.close();
    }
}
