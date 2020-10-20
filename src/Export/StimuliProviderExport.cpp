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

void N2D2::StimuliProviderExport::generate(const DeepNet& deepNet, StimuliProvider& sp,
                                           const std::string& dirName, const std::string& type,
                                           Database::StimuliSet set,
                                           bool unsignedData, CellExport::Precision precision,
                                           int nbStimuliMax,
                                           ExportFormat exportFormat)
{
    if (Registrar<StimuliProviderExport>::exists(type)) {
        Registrar<StimuliProviderExport>::create(type)(deepNet, sp, dirName, set, unsignedData, 
                                                       precision, nbStimuliMax);
    }
    else {
        // Default generator
        StimuliProviderExport::generate(deepNet, sp, dirName, set, unsignedData,
                                        precision, nbStimuliMax, exportFormat);
    }
}

void N2D2::StimuliProviderExport::generate(const DeepNet& deepNet, StimuliProvider& sp,
                                           const std::string& dirName,
                                           Database::StimuliSet set,
                                           bool unsignedData, CellExport::Precision precision,
                                           int nbStimuliMax,
                                           ExportFormat exportFormat)
{
    Utils::createDirectories(dirName);


    const std::size_t envSizeX = sp.getSizeX();
    const std::size_t envSizeY = sp.getSizeY();
    const std::size_t nbChannels = sp.getNbChannels();
    const std::size_t size = nbStimuliMax >= 0?std::min(sp.getDatabase().getNbStimuli(set),
                                                        static_cast<unsigned int>(nbStimuliMax)):
                                               sp.getDatabase().getNbStimuli(set);
    const std::size_t zeroPad = size > 0?std::ceil(std::log10(size)):0;

    const std::string stimuliListName = dirName + "/../stimuli.list";
    std::ofstream stimuliList(stimuliListName);
    if (!stimuliList.good()) {
        throw std::runtime_error("Could not create stimuli list file: " + stimuliListName + ".");
    }

    std::cout << "Exporting " << set << " dataset to \"" << dirName << "\"" << std::flush;


    std::size_t progress = 0, progressPrev = 0;
    for (std::size_t i = 0; i < size; ++i) {
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

        if (!envStimuli.good()) {
            throw std::runtime_error("Could not create stimuli binary file: " + stimuliName.str());
        }


        if (nbChannels > 1) {
            envStimuli << "P6\n";
        }
        else {
            envStimuli << "P5\n";
        }

        const std::size_t maxValue = (precision > 0 && precision <= 8)?255:65535;
        envStimuli << envSizeX << " " << envSizeY << "\n" << maxValue << "\n";

        sp.readStimulusBatch(set, i);

        // TODO Optimize loop to avoid checking 'precision' and 'unsignedData' constantly.
        if(exportFormat == CHW) {
            for (std::size_t channel = 0; channel < nbChannels; channel++) {
                for (std::size_t y = 0; y < envSizeY; y++) {
                    for (std::size_t x = 0; x < envSizeX; x++) {
                        writeStimulusValue(sp.getData()(x, y, channel, 0), 
                                           unsignedData, precision, envStimuli);
                    }
                }
            }
        }
        else {
            assert(exportFormat == HWC);
            for (std::size_t y = 0; y < envSizeY; y++) {
                for (std::size_t x = 0; x < envSizeX; x++) {
                    for (std::size_t channel = 0; channel < nbChannels; channel++) {
                        writeStimulusValue(sp.getData()(x, y, channel, 0), 
                                           unsignedData, precision, envStimuli);
                    }
                }
            }
        }

        const std::vector<std::shared_ptr<Target> > outputTargets
            = deepNet.getTargets();
        const std::size_t nbTargets = outputTargets.size();

        for(std::size_t t = 0; t < nbTargets; ++t) {
            const std::shared_ptr<Target>& target = outputTargets[t];
            target->provideTargets(set);

            const Tensor<int>& targetData = target->getTargets();

            for (std::size_t y = 0; y < targetData.dimY(); ++y) {
                for (std::size_t x = 0; x < targetData.dimX(); ++x) {
                    const int32_t outputTarget = targetData(x, y, 0, 0);

                    envStimuli.write(reinterpret_cast<const char*>(&outputTarget),
                                     sizeof(outputTarget));
                }
            }
        }
        if (!envStimuli.good()) {
            throw std::runtime_error("Error writing stimuli binary file: " + stimuliName.str());
        }

        envStimuli.close();

        // Progress bar
        progress = (std::size_t)(20.0 * (i + 1) / (double)size);

        if (progress > progressPrev) {
            std::cout << std::string(progress - progressPrev, '.') << std::flush;
            progressPrev = progress;
        }
    }

    std::cout << std::endl;
}

void N2D2::StimuliProviderExport::writeStimulusValue(Float_T value, bool unsignedData, 
                                                     CellExport::Precision precision,
                                                     std::ofstream& envStimuli,
                                                     bool asBinary) 
{
    if (precision == CellExport::Float64) {
        writeToStream(static_cast<double>(value), envStimuli, asBinary);
    }
    else if (precision == CellExport::Float32 || precision == CellExport::Float16) {
        writeToStream(static_cast<float>(value), envStimuli, asBinary);
    }
    else if (precision <= 8 && unsignedData) {
        writeToStream(Utils::saturate_cast<std::uint8_t>(CellExport::getIntFreeParameter(value)), 
                      envStimuli, asBinary);
    }
    else if (precision <= 8 && !unsignedData) {
        writeToStream(Utils::saturate_cast<std::int8_t>(CellExport::getIntFreeParameter(value)), 
                      envStimuli, asBinary);
    }
    else if (precision <= 16 && unsignedData) {
        writeToStream(Utils::saturate_cast<std::uint16_t>(CellExport::getIntFreeParameter(value)), 
                      envStimuli, asBinary);
    }
    else if (precision <= 16 && !unsignedData) {
        writeToStream(Utils::saturate_cast<std::int16_t>(CellExport::getIntFreeParameter(value)), 
                      envStimuli, asBinary);
    }
    else if (precision <= 32 && unsignedData) {
        writeToStream(Utils::saturate_cast<std::uint32_t>(CellExport::getIntFreeParameter(value)), 
                      envStimuli, asBinary);
    }
    else if (precision <= 32 && !unsignedData) {
        writeToStream(Utils::saturate_cast<std::int32_t>(CellExport::getIntFreeParameter(value)), 
                      envStimuli, asBinary);
    }
    else if (precision <= 64 && unsignedData) {
        writeToStream(Utils::saturate_cast<std::uint64_t>(CellExport::getIntFreeParameter(value)), 
                      envStimuli, asBinary);
    }
    else if (precision <= 64 && !unsignedData) {
        writeToStream(Utils::saturate_cast<std::int64_t>(CellExport::getIntFreeParameter(value)), 
                      envStimuli, asBinary);
    }
    else {
        throw std::runtime_error("Unsupported precision.");
    }

}

N2D2::StimuliData N2D2::StimuliProviderExport::getStimuliData(StimuliProvider& sp,
                                                              const std::string& dirName,
                                                              Database::StimuliSet set)
{
    StimuliData stimuliData(dirName + "_stats", sp);
    stimuliData.generate(sp.getDatabase().getStimuliSetMask(set), true);
    stimuliData.logValueRange();

    return stimuliData;
}

double N2D2::StimuliProviderExport::stimuliRange(StimuliProvider& sp,
                                                 const std::string& dirName,
                                                 Database::StimuliSet set)
{
    StimuliData::Value globalValue = getStimuliData(sp, dirName, set).getGlobalValue();
    return std::max(std::abs(globalValue.minVal), std::abs(globalValue.maxVal));
}

bool N2D2::StimuliProviderExport::unsignedStimuli(StimuliProvider& sp,
                                                  const std::string& dirName,
                                                  Database::StimuliSet set)
{
    if (CellExport::mPrecision < 0) {
        return false;
    }

    StimuliData::Value globalValue = getStimuliData(sp, dirName, set).getGlobalValue();
    return globalValue.minVal >= 0 && DeepNetExport::mUnsignedData;
}
