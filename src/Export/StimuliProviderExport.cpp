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
                                           int nbStimuliMax,
                                           bool normalize,
                                           DeepNet* deepNet)
{
    if (Registrar<StimuliProviderExport>::exists(type)) {
        Registrar<StimuliProviderExport>::create(type)(sp,
                                                       dirName,
                                                       set,
                                                       nbStimuliMax,
                                                       normalize,
                                                       deepNet);
    }
    else {
        // Default generator
        StimuliProviderExport::generate(sp,
                                        dirName,
                                        set,
                                        nbStimuliMax,
                                        normalize,
                                        deepNet);
    }
}

void N2D2::StimuliProviderExport::generate(StimuliProvider& sp,
                                           const std::string& dirName,
                                           Database::StimuliSet set,
                                           int nbStimuliMax,
                                           bool normalize,
                                           DeepNet* deepNet)
{
    Utils::createDirectories(dirName);

    double scaling;
    bool unsignedData;
    std::tie(scaling, unsignedData) = getScaling(sp, dirName, set, normalize);

    const unsigned int envSizeX = sp.getSizeX();
    const unsigned int envSizeY = sp.getSizeY();
    const unsigned int nbChannels = sp.getNbChannels();
    const unsigned int size = (nbStimuliMax >= 0)
        ? std::min(sp.getDatabase().getNbStimuli(set),
                   (unsigned int)nbStimuliMax)
        : sp.getDatabase().getNbStimuli(set);
    const unsigned int zeroPad = (size > 0) ? std::ceil(std::log10(size)) : 0;

    std::ofstream stimuliList((dirName + ".list").c_str());

    if (!stimuliList.good()) {
        throw std::runtime_error("Could not create stimuli list file: "
                                 + dirName + ".list");
    }

    std::cout << "Exporting " << set << " dataset to \"" << dirName << "\""
        << std::flush;

    unsigned int progress = 0, progressPrev = 0;

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
                    }
                    else if (CellExport::mPrecision == CellExport::Float32
                               || CellExport::mPrecision
                                  == CellExport::Float16) {
                        const float value = (float)frame(x, y);
                        envStimuli.write(reinterpret_cast<const char*>(&value),
                                         sizeof(value));
                    }
                    else if (CellExport::mPrecision <= 8) {
                        const long long int approxValue
                            = CellExport::getIntApprox(scaling * frame(x, y));

                        if (unsignedData) {
                            const uint8_t value
                                = (uint8_t)Utils::clamp<long long int>(
                                approxValue,
                                std::numeric_limits<uint8_t>::min(),
                                std::numeric_limits<uint8_t>::max());
                            envStimuli.write(reinterpret_cast<const char*>
                                             (&value),
                                             sizeof(value));
                        }
                        else {
                            const int8_t value
                                = (int8_t)Utils::clamp<long long int>(
                                approxValue,
                                std::numeric_limits<int8_t>::min(),
                                std::numeric_limits<int8_t>::max());
                            envStimuli.write(reinterpret_cast<const char*>
                                             (&value),
                                             sizeof(value));
                        }
                    }
                    else if (CellExport::mPrecision <= 16) {
                        const long long int approxValue
                            = CellExport::getIntApprox(scaling * frame(x, y));

                        if (unsignedData) {
                            const uint16_t value
                                = (uint16_t)Utils::clamp<long long int>(
                                approxValue,
                                std::numeric_limits<uint16_t>::min(),
                                std::numeric_limits<uint16_t>::max());
                            envStimuli.write(reinterpret_cast<const char*>
                                             (&value),
                                             sizeof(value));
                        }
                        else {
                            const int16_t value
                                = (int16_t)Utils::clamp<long long int>(
                                approxValue,
                                std::numeric_limits<int16_t>::min(),
                                std::numeric_limits<int16_t>::max());
                            envStimuli.write(reinterpret_cast<const char*>
                                             (&value),
                                             sizeof(value));
                        }
                    }
                    else if (CellExport::mPrecision <= 32) {
                        const long long int approxValue
                            = CellExport::getIntApprox(scaling * frame(x, y));

                        if (unsignedData) {
                            const uint32_t value
                                = (uint32_t)Utils::clamp<long long int>(
                                approxValue,
                                std::numeric_limits<uint32_t>::min(),
                                std::numeric_limits<uint32_t>::max());
                            envStimuli.write(reinterpret_cast<const char*>
                                             (&value),
                                             sizeof(value));
                        }
                        else {
                            const int32_t value
                                = (int32_t)Utils::clamp<long long int>(
                                approxValue,
                                std::numeric_limits<int32_t>::min(),
                                std::numeric_limits<int32_t>::max());
                            envStimuli.write(reinterpret_cast<const char*>
                                             (&value),
                                             sizeof(value));
                        }
                    }
                    else {
                        const long long int approxValue
                            = CellExport::getIntApprox(scaling * frame(x, y));

                        if (unsignedData) {
                            const uint64_t value = approxValue;
                            envStimuli.write(reinterpret_cast<const char*>
                                             (&value),
                                             sizeof(value));
                        }
                        else {
                            const int64_t value = approxValue;
                            envStimuli.write(reinterpret_cast<const char*>
                                             (&value),
                                             sizeof(value));
                        }
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

        // Progress bar
        progress = (unsigned int)(20.0 * (i + 1) / (double)size);

        if (progress > progressPrev) {
            std::cout << std::string(progress - progressPrev, '.')
                      << std::flush;
            progressPrev = progress;
        }
    }

    std::cout << std::endl;
}

N2D2::StimuliData N2D2::StimuliProviderExport::getStimuliData(
    StimuliProvider& sp,
    const std::string& dirName,
    Database::StimuliSet set)
{
    StimuliData stimuliData(dirName + "_stats", sp);
    stimuliData.generate(sp.getDatabase().getStimuliSetMask(set));
    stimuliData.logValueRange();
    return stimuliData;
}

double N2D2::StimuliProviderExport::getStimuliRange(
    StimuliProvider& sp,
    const std::string& dirName,
    Database::StimuliSet set)
{
    const StimuliData::Value& globalValue
        = getStimuliData(sp, dirName, set).getGlobalValue();
    return std::max(std::abs(globalValue.minVal),
                    std::abs(globalValue.maxVal));
}

std::pair<double, bool> N2D2::StimuliProviderExport::getScaling(
    StimuliProvider& sp,
    const std::string& dirName,
    Database::StimuliSet set,
    bool normalize)
{
    if (CellExport::mPrecision < 0 && !normalize)
        return std::make_pair(1.0, false);

    const StimuliData::Value& globalValue
        = getStimuliData(sp, dirName, set).getGlobalValue();
    const double dataRange = std::max(std::abs(globalValue.minVal),
                                      std::abs(globalValue.maxVal));
    const bool unsignedData = (globalValue.minVal >= 0
                               && DeepNetExport::mUnsignedData);

    double scalingValue = 1.0;

    if (CellExport::mPrecision > 0) {
        DeepNetExport::mEnvDataUnsigned = unsignedData;

        const unsigned int nbBits = ((unsignedData) ? CellExport::mPrecision
                                                : (CellExport::mPrecision - 1));
        scalingValue = (double)(std::pow(2, nbBits) - 1);
    }

    if (normalize) {
        if (dataRange != 1.0) {
            scalingValue /= dataRange;

            std::cout << Utils::cnotice << "Stimuli export with"
                " range != 1 (" << dataRange << "). Data will be normalized."
                << Utils::cdef << std::endl;
        }
    }
    else if (CellExport::mPrecision > 0) {
        if (dataRange > 1.0) {
            std::cout << Utils::cwarning << "Integer stimuli export with"
                " range > 1 (" << dataRange << "). Data will be truncated,"
                " possible data loss." << Utils::cdef << std::endl;
        }
        else if (dataRange < 1.0) {
            std::cout << Utils::cnotice << "Integer stimuli export with"
                " range < 1 (" << dataRange << "). The full "
                << (int)CellExport::mPrecision << " bits data range will not be"
                " used." << Utils::cdef << std::endl;
        }
    }

    return std::make_pair(scalingValue, unsignedData);
}
