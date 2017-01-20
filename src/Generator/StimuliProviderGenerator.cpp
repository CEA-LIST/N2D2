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

#include "Generator/StimuliProviderGenerator.hpp"

std::shared_ptr<N2D2::StimuliProvider> N2D2::StimuliProviderGenerator::generate(
    Database& database, IniParser& iniConfig, const std::string& section)
{
    if (!iniConfig.currentSection(section))
        throw std::runtime_error("Missing [" + section + "] section.");

    const unsigned int sizeX = iniConfig.getProperty<unsigned int>("SizeX");
    const unsigned int sizeY = iniConfig.getProperty<unsigned int>("SizeY");
    const unsigned int nbChannels = iniConfig.getProperty
                                    <unsigned int>("NbChannels", 1U);
    const unsigned int batchSize = iniConfig.getProperty
                                   <unsigned int>("BatchSize", 1U);
    const bool compositeStimuli = iniConfig.getProperty
                                  <bool>("CompositeStimuli", false);
    const std::string cachePath = iniConfig.getProperty
                                  <std::string>("CachePath", "");

    std::shared_ptr<StimuliProvider> sp(new StimuliProvider(
        database, sizeX, sizeY, nbChannels, batchSize, compositeStimuli));
    sp->setCachePath(cachePath);

    iniConfig.setProperty("_EpochSize", database.getNbStimuli(Database::Learn));

    const std::string configSection = iniConfig.getProperty
                                      <std::string>("ConfigSection", "");

    if (!configSection.empty())
        sp->setParameters(iniConfig.getSection(configSection));

    generateSubSections(sp, iniConfig, section);
    return sp;
}

void
N2D2::StimuliProviderGenerator::generateSubSections(const std::shared_ptr
                                                    <N2D2::StimuliProvider>& sp,
                                                    IniParser& iniConfig,
                                                    const std::string& section)
{
    const std::vector<std::string> subSections
        = iniConfig.getSections(section + ".*");

    for (std::vector<std::string>::const_iterator it = subSections.begin(),
                                                  itEnd = subSections.end();
         it != itEnd;
         ++it) {
        if (!iniConfig.currentSection(*it, false))
            throw std::runtime_error("Missing [" + (*it) + "] section.");

        if (Utils::match(section + ".StimuliData*", *it)) {
            std::shared_ptr<StimuliData> stimuliData
                = StimuliDataGenerator::generate(*sp, iniConfig, *it);
        } else if (Utils::match(section + ".*Transformation*", *it)) {
            const Database::StimuliSetMask applyTo
                = iniConfig.getProperty
                  <Database::StimuliSetMask>("ApplyTo", Database::All);

            const bool isChannel = iniConfig.isProperty("Channel");
            const unsigned int channel = (isChannel) ? iniConfig.getProperty
                                                       <unsigned int>("Channel")
                                                     : 0;

            std::shared_ptr<Transformation> transformation
                = TransformationGenerator::generate(iniConfig, *it);

            if (Utils::match(section + ".Transformation*", *it))
                sp->addTransformation(transformation, applyTo);
            else if (Utils::match(section + ".OnTheFlyTransformation*", *it))
                sp->addOnTheFlyTransformation(transformation, applyTo);
            else if (Utils::match(section + ".ChannelTransformation*", *it)) {
                if (isChannel)
                    sp->addChannelTransformation(
                        channel, transformation, applyTo);
                else
                    sp->addChannelTransformation(transformation, applyTo);
            } else if (Utils::match(section + ".ChannelOnTheFlyTransformation*",
                                    *it)) {
                if (isChannel)
                    sp->addChannelOnTheFlyTransformation(
                        channel, transformation, applyTo);
                else
                    sp->addChannelOnTheFlyTransformation(transformation,
                                                         applyTo);
            } else if (Utils::match(section + ".ChannelsTransformation*", *it))
                sp->addChannelsTransformation(transformation, applyTo);
            else if (Utils::match(section + ".ChannelsOnTheFlyTransformation*",
                                  *it))
                sp->addChannelsOnTheFlyTransformation(transformation, applyTo);
            else
                throw std::runtime_error("Unknown StimuliProvider type: ["
                                         + (*it) + "].");
        }
    }
}
