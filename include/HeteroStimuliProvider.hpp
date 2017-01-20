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

#ifndef N2D2_HETEROSTIMULIPROVIDER_H
#define N2D2_HETEROSTIMULIPROVIDER_H

#include <iterator>
#include <vector>

#include "StimuliProvider.hpp"

namespace N2D2 {
/**
 * This class is a collection of one or several StimuliProvider that must share:
 * - Same database
 * - Same batch size
 * (this constrains are ensured in the HeteroStimuliProvider::push_back()
 *method)
 * This, in order to ensure that the same stimuli batch are generated for all
 *the StimuliProvider in the collection, e.g. when
 * calling HeteroStimuliProvider::readRandomBatch().
 * The first item must be provided at construction, which ensures that the
 *collection cannot be empty.
 *
 * However, the transformations applied to the stimuli can be different:
 * - Different channel transformations
 * - Different stimuli size after transformations
 * - Different number of channels
 *
 * The main purpose of this class is to allow multiscale inputs, in a similar
 *fashion to the old "map" mechanism in N2D2, but
 * with a much higher flexibility and genericity.
*/
class HeteroStimuliProvider {
public:
    /// Create an HeteroStimuliProvider from a COPY of a StimuliProvider or an
    /// other derived object.
    /// Since it is a COPY, changes on the object passed on argument will have
    /// no effect!
    HeteroStimuliProvider(const StimuliProvider& item);
    /// Create an HeteroStimuliProvider from a pointer to a StimuliProvider or
    /// an other derived object
    HeteroStimuliProvider(const std::shared_ptr<StimuliProvider>& item);
    void addTransformation(const CompositeTransformation& transformation,
                           Database::StimuliSetMask setMask = Database::All);
    void addChannelTransformation(const CompositeTransformation& transformation,
                                  Database::StimuliSetMask setMask
                                  = Database::All);
    void
    addChannelsTransformation(const CompositeTransformation& transformation,
                              Database::StimuliSetMask setMask = Database::All);
    void readRandomBatch(Database::StimuliSet set);
    Database::StimulusID readRandomStimulus(Database::StimuliSet set,
                                            unsigned int batchPos = 0);
    void readStimulus(Database::StimulusID id,
                      Database::StimuliSet set,
                      unsigned int batchPos = 0);
    Database::StimulusID readStimulus(Database::StimuliSet set,
                                      unsigned int index,
                                      unsigned int batchPos = 0);
    template <class VT> void setParameter(const std::string& name, VT value);
    void saveParameters(const std::string& fileName) const;
    void loadParameters(const std::string& fileName,
                        bool ignoreNotExists = false,
                        bool ignoreUnknown = false);
    void push_back(const StimuliProvider& item);
    unsigned int size() const;
    std::shared_ptr<StimuliProvider> operator[](unsigned int k);
    const std::shared_ptr<StimuliProvider> operator[](unsigned int k) const;
    virtual ~HeteroStimuliProvider() {};

protected:
    std::vector<std::shared_ptr<StimuliProvider> > mItems;
};
}

template <class VT>
void N2D2::HeteroStimuliProvider::setParameter(const std::string& name,
                                               VT value)
{
    for (std::vector<std::shared_ptr<StimuliProvider> >::iterator it
         = mItems.begin(),
         itEnd = mItems.end();
         it != itEnd;
         ++it)
        (*it)->setParameter<VT>(name, value);
}

#endif // N2D2_HETEROSTIMULIPROVIDER_H
