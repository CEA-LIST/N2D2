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

#include "HeteroEnvironment.hpp"

N2D2::HeteroEnvironment::HeteroEnvironment(const Environment& item)
    : HeteroStimuliProvider(item)
{
    // ctor
}

N2D2::HeteroEnvironment::HeteroEnvironment(const std::shared_ptr
                                           <Environment>& item)
    : HeteroStimuliProvider(item)
{
    // ctor
}

void N2D2::HeteroEnvironment::addMap(double scale)
{
    std::vector<double>::iterator it
        = std::find(mScales.begin(), mScales.end(), scale);

    if (it == mScales.end()) {
        std::shared_ptr<Environment> baseEnv = std::static_pointer_cast
            <Environment>(mItems[0]);
        const unsigned int width = scale * baseEnv->getSizeX();
        const unsigned int height = scale * baseEnv->getSizeY();

        if (scale == 1.0) {
            // By convention, the first item in the collection is at scale 1.0
            // If the scale 1.0 does not exist, add the rescale transformation
            mItems[0]->addTransformation(RescaleTransformation(width, height));
        } else {
            push_back(Environment(baseEnv->getNetwork(),
                                  baseEnv->getDatabase(),
                                  width,
                                  height,
                                  1,
                                  baseEnv->getBatchSize(),
                                  baseEnv->isCompositeStimuli()));
            mItems.back()->addTransformation(
                RescaleTransformation(width, height));
        }

        mScales.push_back(scale);
    }
}

void N2D2::HeteroEnvironment::addPyramidMaps(double scaleDown)
{
    // By convention, the first item in the collection is at scale 1.0
    const std::shared_ptr<Environment> baseEnv = std::static_pointer_cast
        <Environment>(mItems[0]);
    const unsigned int width = baseEnv->getSizeX();
    const unsigned int height = baseEnv->getSizeY();
    double scale = 1.0;

    do {
        addMap(scale);
        scale *= scaleDown;
    } while (scale * width >= 2.0 && scale * height >= 2.0);
}

void N2D2::HeteroEnvironment::propagate(Time_T start, Time_T end)
{
    for (std::vector<std::shared_ptr<StimuliProvider> >::const_iterator it
         = mItems.begin(),
         itEnd = mItems.end();
         it != itEnd;
         ++it) {
        std::static_pointer_cast<Environment>(*it)->propagate(start, end);
    }
}

const std::vector<N2D2::NodeEnv*> N2D2::HeteroEnvironment::getNodes() const
{
    std::vector<NodeEnv*> nodes;

    for (std::vector<std::shared_ptr<StimuliProvider> >::const_iterator it
         = mItems.begin(),
         itEnd = mItems.end();
         it != itEnd;
         ++it) {
        const std::vector<NodeEnv*> itemNodes = std::static_pointer_cast
                                                <Environment>(*it)->getNodes();
        nodes.insert(nodes.end(), itemNodes.begin(), itemNodes.end());
    }

    return nodes;
}

unsigned int N2D2::HeteroEnvironment::getNbNodes() const
{
    unsigned int nbNodes = 0;

    for (std::vector<std::shared_ptr<StimuliProvider> >::const_iterator it
         = mItems.begin(),
         itEnd = mItems.end();
         it != itEnd;
         ++it) {
        nbNodes += std::static_pointer_cast<Environment>(*it)->getNbNodes();
    }

    return nbNodes;
}

std::shared_ptr<N2D2::Environment> N2D2::HeteroEnvironment::
operator[](unsigned int k)
{
    return std::static_pointer_cast<Environment>(mItems.at(k));
}

const std::shared_ptr<N2D2::Environment> N2D2::HeteroEnvironment::
operator[](unsigned int k) const
{
    return std::static_pointer_cast<Environment>(mItems.at(k));
}
