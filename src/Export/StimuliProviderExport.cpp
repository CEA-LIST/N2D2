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
#include "DeepNet.hpp"
#include "StimuliProvider.hpp"
#include "Export/CellExport.hpp"
#include "Export/DeepNetExport.hpp"

void N2D2::StimuliProviderExport::generate(StimuliProvider& sp,
                                           const std::string& dirName,
                                           const std::string& type,
                                           Database::StimuliSet set,
                                           int nbStimuliMax,
                                           bool normalize,
                                           DeepNet* deepNet,
                                           ExportFormat exportFormat)
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
                                        deepNet,
                                        exportFormat);
    }
}

void N2D2::StimuliProviderExport::generate(StimuliProvider& sp,
                                           const std::string& dirName,
                                           Database::StimuliSet set,
                                           int nbStimuliMax,
                                           bool normalize,
                                           DeepNet* deepNet,
                                           ExportFormat exportFormat)
{
    Utils::createDirectories(dirName);

    double scaling;
    bool unsignedData;
    std::tie(scaling, unsignedData) = getScaling(sp, dirName, set, normalize);

    generate(sp, dirName, set, scaling, unsignedData, nbStimuliMax, deepNet, exportFormat);
}

void N2D2::StimuliProviderExport::generate(StimuliProvider& sp,
                                           const std::string& dirName,
                                           Database::StimuliSet set,
                                           double scaling,
                                           bool unsignedData,
                                           int nbStimuliMax,
                                           DeepNet* deepNet,
                                           ExportFormat exportFormat)
{
    assert(sp.getBatchSize() == 1);

    Utils::createDirectories(dirName);

    // Truncate is the natural approx. method for the input, as it is generally
    // already originating from INT8 images.
    // Truncate is the appropriate method when using the QuantizationLevels
    // option in StimuliProvider
    const CellExport::IntApprox approxMethod = CellExport::Truncate;

    const unsigned int envSizeX = sp.getSizeX();
    const unsigned int envSizeY = sp.getSizeY();
    const unsigned int nbChannels = sp.getNbChannels();
    const unsigned int size = (nbStimuliMax >= 0)
        ? std::min(sp.getDatabase().getNbStimuli(set),
                   (unsigned int)nbStimuliMax)
        : sp.getDatabase().getNbStimuli(set);
    const unsigned int zeroPad = (size > 0) ? std::ceil(std::log10(size)) : 0;

    const std::string stimuliListName = dirName + "/../stimuli.list";
    std::ofstream stimuliList(stimuliListName);
    if (!stimuliList.good()) {
        throw std::runtime_error("Could not create stimuli list file: " + stimuliListName + ".");
    }

    std::cout << "Exporting " << set << " dataset to \"" << dirName << "\""
        << std::flush;

    unsigned int progress = 0, progressPrev = 0;

    for (unsigned int i = 0; i < size; ++i) {
        std::stringstream stimuliName;
        stimuliName << dirName << "/env" << std::setfill('0')
                    << std::setw(zeroPad) << i;

        if (nbChannels > 1)
            stimuliName << ".ppm";
        else
            stimuliName << ".pgm";

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

        if (nbChannels > 1)
            envStimuli << "P6\n";
        else
            envStimuli << "P5\n";

        envStimuli << envSizeX << " " << envSizeY << "\n" << maxValue << "\n";

        sp.readStimulus(set, i);

        if(exportFormat == CHW) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                const Tensor<Float_T> frame = sp.getData(channel);

                for (unsigned int y = 0; y < envSizeY; ++y) {
                    for (unsigned int x = 0; x < envSizeX; ++x) {
                        writeStimulusValue(frame(x, y), unsignedData, 
                                           scaling, approxMethod, envStimuli);
                    }
                }
            }
        }
        else {
            assert(exportFormat == HWC);
            for (unsigned int y = 0; y < envSizeY; ++y) {
                for (unsigned int x = 0; x < envSizeX; ++x) {
                    for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                        writeStimulusValue(sp.getData()(x, y, channel, 0), unsignedData, 
                                           scaling, approxMethod, envStimuli);
                    }
                }
            }
        }

        const Tensor<int> label = sp.getLabelsData(0);
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

template<typename T>
T N2D2::StimuliProviderExport::getScaledData(Float_T data, double scaling, 
                                             CellExport::IntApprox approxMethod) 
{
    const long long int approxValue = CellExport::getIntApprox(scaling * data, approxMethod);
    const T value = (T) Utils::clamp<long long int>(approxValue,
                                                    std::numeric_limits<T>::min(),
                                                    std::numeric_limits<T>::max());
    return value;
}

namespace N2D2 {
template<>
float StimuliProviderExport::getScaledData(Float_T data, double, CellExport::IntApprox) {
    return (float) data;
}

template<>
double StimuliProviderExport::getScaledData(Float_T data, double, CellExport::IntApprox) {
    return (double) data;
}
}

void N2D2::StimuliProviderExport::writeStimulusValue(Float_T value, bool unsignedData, double scaling, 
                                                     CellExport::IntApprox approxMethod, 
                                                     std::ofstream& envStimuli,
                                                     bool asBinary) 
{
    if (CellExport::mPrecision == CellExport::Float64) {
        const double val = getScaledData<double>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else if (CellExport::mPrecision == CellExport::Float32 || 
             CellExport::mPrecision == CellExport::Float16) 
    {
        const float val = getScaledData<float>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else if (CellExport::mPrecision <= 8 && unsignedData) {
        const uint8_t val = getScaledData<uint8_t>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else if (CellExport::mPrecision <= 8 && !unsignedData) {
        const int8_t val = getScaledData<int8_t>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else if (CellExport::mPrecision <= 16 && unsignedData) {
        const uint16_t val = getScaledData<uint16_t>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else if (CellExport::mPrecision <= 16 && !unsignedData) {
        const int16_t val = getScaledData<int16_t>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else if (CellExport::mPrecision <= 32 && unsignedData) {
        const uint32_t val = getScaledData<uint32_t>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else if (CellExport::mPrecision <= 32 && !unsignedData) {
        const int32_t val = getScaledData<int32_t>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else if (CellExport::mPrecision <= 64 && unsignedData) {
        const uint64_t val = getScaledData<uint64_t>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else if (CellExport::mPrecision <= 64 && !unsignedData) {
        const int64_t val = getScaledData<int64_t>(value, scaling, approxMethod);
        writeToStream(val, envStimuli, asBinary);
    }
    else {
        throw std::runtime_error("Unsupported precision.");
    }

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
    StimuliData::Value globalValue
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

    StimuliData::Value globalValue
        = getStimuliData(sp, dirName, set).getGlobalValue();
    double dataRange = std::max(std::abs(globalValue.minVal),
                                std::abs(globalValue.maxVal));

    if (dataRange == 0.0) {
        std::cout << Utils::cwarning << "No data (range is 0.0), use a"
            " default data range of 1.0." << Utils::cdef << std::endl;
        dataRange = 1.0;
    }

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
