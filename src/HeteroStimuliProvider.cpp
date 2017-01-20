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

#include "HeteroStimuliProvider.hpp"

N2D2::HeteroStimuliProvider::HeteroStimuliProvider(const StimuliProvider& item)
{
    // ctor
    mItems.push_back(std::make_shared<StimuliProvider>(item));
}

N2D2::HeteroStimuliProvider::HeteroStimuliProvider(const std::shared_ptr
                                                   <StimuliProvider>& item)
{
    // ctor
    mItems.push_back(item);
}

void
N2D2::HeteroStimuliProvider::addTransformation(const CompositeTransformation
                                               & transformation,
                                               Database::StimuliSetMask setMask)
{
    std::for_each(
        mItems.begin(),
        mItems.end(),
        std::bind(static_cast
                  <void (StimuliProvider::*)(const CompositeTransformation&,
                                             Database::StimuliSetMask)>(
                      &StimuliProvider::addTransformation),
                  std::placeholders::_1,
                  transformation,
                  setMask));
}

void N2D2::HeteroStimuliProvider::addChannelTransformation(
    const CompositeTransformation& transformation,
    Database::StimuliSetMask setMask)
{
    std::for_each(
        mItems.begin(),
        mItems.end(),
        std::bind(static_cast
                  <void (StimuliProvider::*)(const CompositeTransformation&,
                                             Database::StimuliSetMask)>(
                      &StimuliProvider::addChannelTransformation),
                  std::placeholders::_1,
                  transformation,
                  setMask));
}

void N2D2::HeteroStimuliProvider::addChannelsTransformation(
    const CompositeTransformation& transformation,
    Database::StimuliSetMask setMask)
{
    std::for_each(mItems.begin(),
                  mItems.end(),
                  std::bind(&StimuliProvider::addChannelsTransformation,
                            std::placeholders::_1,
                            transformation,
                            setMask));
}

void N2D2::HeteroStimuliProvider::readRandomBatch(Database::StimuliSet set)
{
    const unsigned int batchSize = mItems[0]->getBatchSize();

    for (unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        readRandomStimulus(set, batchPos);
}

N2D2::Database::StimulusID
N2D2::HeteroStimuliProvider::readRandomStimulus(Database::StimuliSet set,
                                                unsigned int batchPos)
{
    Database& database = mItems[0]->getDatabase();
    const unsigned int index
        = Random::randUniform(0, database.getNbStimuli(set) - 1);

    return readStimulus(set, index, batchPos);
}

void N2D2::HeteroStimuliProvider::readStimulus(Database::StimulusID id,
                                               Database::StimuliSet set,
                                               unsigned int batchPos)
{
    std::for_each(mItems.begin(),
                  mItems.end(),
                  std::bind(static_cast
                            <void (StimuliProvider::*)(Database::StimulusID,
                                                       Database::StimuliSet,
                                                       unsigned int)>(
                                &StimuliProvider::readStimulus),
                            std::placeholders::_1,
                            id,
                            set,
                            batchPos));
}

N2D2::Database::StimulusID N2D2::HeteroStimuliProvider::readStimulus(
    Database::StimuliSet set, unsigned int index, unsigned int batchPos)
{
    Database& database = mItems[0]->getDatabase();

    std::for_each(
        mItems.begin(),
        mItems.end(),
        std::bind(
            static_cast
            <Database::StimulusID (StimuliProvider::*)(Database::StimuliSet,
                                                       unsigned int,
                                                       unsigned int)>(
                &StimuliProvider::readStimulus),
            std::placeholders::_1,
            set,
            index,
            batchPos));

    return database.getStimulusID(set, index);
}

void N2D2::HeteroStimuliProvider::saveParameters(const std::string
                                                 & fileName) const
{
    mItems.front()->saveParameters(fileName);
}

void N2D2::HeteroStimuliProvider::loadParameters(const std::string& fileName,
                                                 bool ignoreNotExists,
                                                 bool ignoreUnknown)
{
    if (!std::ifstream(fileName.c_str()).good()) {
        if (ignoreNotExists)
            std::cout << "Notice: Could not open configuration file: "
                      << fileName << std::endl;
        else
            throw std::runtime_error("Could not open configuration file: "
                                     + fileName);
    } else {
        // ignoreNotExists is set to false in the following calls, because the
        // file is supposed to exist and be readable.
        // Otherwise, something is really wrong and it's probably better to
        // throw an exception!
        std::for_each(mItems.begin(),
                      mItems.end(),
                      std::bind(&StimuliProvider::loadParameters,
                                std::placeholders::_1,
                                fileName,
                                false,
                                ignoreUnknown));
    }
}

void N2D2::HeteroStimuliProvider::push_back(const StimuliProvider& item)
{
    if (!mItems.empty()) {
        if (&(item.getDatabase()) != &(mItems[0]->getDatabase()))
            throw std::runtime_error("HeteroStimuliProvider::push_back(): "
                                     "items must share the same database");

        if (item.getBatchSize() != mItems[0]->getBatchSize())
            throw std::runtime_error("HeteroStimuliProvider::push_back(): "
                                     "items must have the same batch size");
    }

    mItems.push_back(std::make_shared<StimuliProvider>(item));
}

unsigned int N2D2::HeteroStimuliProvider::size() const
{
    return mItems.size();
}

std::shared_ptr<N2D2::StimuliProvider> N2D2::HeteroStimuliProvider::
operator[](unsigned int k)
{
    return mItems.at(k);
}

const std::shared_ptr<N2D2::StimuliProvider> N2D2::HeteroStimuliProvider::
operator[](unsigned int k) const
{
    return mItems.at(k);
}
