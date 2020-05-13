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

#include "Generator/CEnvironmentGenerator.hpp"

std::shared_ptr<N2D2::CEnvironment> N2D2::CEnvironmentGenerator::generate(
    Database& database, IniParser& iniConfig, const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");
    const bool cudaSpike = iniConfig.getProperty
                                  <bool>("SpikeCUDA", false);
    std::vector<size_t> size;

    if (iniConfig.isProperty("Size"))
        size = iniConfig.getProperty<std::vector<size_t> >("Size");
    else {
        size.push_back(iniConfig.getProperty<size_t>("SizeX"));
        size.push_back(iniConfig.getProperty<size_t>("SizeY"));

        if (iniConfig.isProperty("SizeD"))
            size.push_back(iniConfig.getProperty<size_t>("SizeD"));

        size.push_back(iniConfig.getProperty<size_t>("NbChannels", 1U));
    }

    const unsigned int batchSize = iniConfig.getProperty
                                   <unsigned int>("BatchSize", 1U);
    const bool compositeStimuli = iniConfig.getProperty
                                  <bool>("CompositeStimuli", false);
    const std::string cachePath = iniConfig.getProperty
                                  <std::string>("CachePath", "");
    const bool readStream = iniConfig.getProperty
                                  <bool>("ReadStream", false);

    if (cudaSpike) {
#ifndef CUDA
        std::cout << Utils::cwarning << "Warning: to use CUDA spike generation "
        << "N2D2 must be compiled with CUDA enabled.\n";
        std::cout << "*** Using standard cenv instead "
            << Utils::cdef << std::endl;
#else

        std::shared_ptr<CEnvironment_CUDA> cEnv(new CEnvironment_CUDA(
            database, size, batchSize, compositeStimuli));


        cEnv->setCachePath(cachePath);

        if (readStream){
            iniConfig.setProperty("_EpochSize", 1);
        }
        else {
            iniConfig.setProperty("_EpochSize", database.getNbStimuli(Database::Learn));
        }

        const std::string configSection = iniConfig.getProperty
                                          <std::string>("ConfigSection", "");

        if (!configSection.empty())
            cEnv->setParameters(iniConfig.getSection(configSection));

        generateSubSections(cEnv, iniConfig, section);
        return cEnv;
#endif
    }

    std::shared_ptr<CEnvironment> cEnv(new CEnvironment(
        database, size, batchSize, compositeStimuli));
    cEnv->setCachePath(cachePath);

    if (readStream){
        iniConfig.setProperty("_EpochSize", 1);
    }
    else {
        iniConfig.setProperty("_EpochSize", database.getNbStimuli(Database::Learn));
    }

    const std::string configSection = iniConfig.getProperty
                                      <std::string>("ConfigSection", "");

    if (!configSection.empty())
        cEnv->setParameters(iniConfig.getSection(configSection));

    generateSubSections(cEnv, iniConfig, section);
    return cEnv;
}
