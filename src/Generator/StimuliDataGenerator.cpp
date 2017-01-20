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

#include "Generator/StimuliDataGenerator.hpp"

std::shared_ptr<N2D2::StimuliData> N2D2::StimuliDataGenerator::generate(
    StimuliProvider& sp, IniParser& iniConfig, const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const Database::StimuliSetMask applyTo
        = iniConfig.getProperty
          <Database::StimuliSetMask>("ApplyTo", Database::All);
    const bool logSizeRange = iniConfig.getProperty
                              <bool>("LogSizeRange", false);
    const bool logValueRange = iniConfig.getProperty
                               <bool>("LogValueRange", false);

    std::shared_ptr<StimuliData> stimuliData(new StimuliData(section, sp));
    stimuliData->setParameters(iniConfig.getSection(section, true));
    stimuliData->generate(applyTo);
    stimuliData->displayData();

    if (logSizeRange)
        stimuliData->logSizeRange();

    if (logValueRange)
        stimuliData->logValueRange();

    const StimuliData::Value& globalValue = stimuliData->getGlobalValue();
    iniConfig.setProperty("_GlobalValue.minVal", globalValue.minVal);
    iniConfig.setProperty("_GlobalValue.maxVal", globalValue.maxVal);
    iniConfig.setProperty("_GlobalValue.mean", globalValue.mean);
    iniConfig.setProperty("_GlobalValue.stdDev", globalValue.stdDev);

    if (stimuliData->getParameter<bool>("MeanData"))
        iniConfig.setProperty("_MeanData", section + "/meanData.bin");

    return stimuliData;
}
