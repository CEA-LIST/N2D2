/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_NODEOUT_H
#define N2D2_NODEOUT_H

#include "Cell_Spike.hpp"
#include "Node.hpp"

namespace N2D2 {
class NodeOut : public Node {
public:
    NodeOut(Network& net,
            Cell_Spike& cell,
            unsigned int channel,
            double scale,
            double orientation,
            unsigned int x,
            unsigned int y = 0);
    void addLink(Node* origin);
    inline void
    incomingSpike(Node* origin, Time_T timestamp, EventType_T type = 0);
    Cell_Spike& getCell() const
    {
        return mCell;
    };
    unsigned int getChannel() const
    {
        return mChannel;
    };
    virtual ~NodeOut() {};

private:
    // Internal variables
    Cell_Spike& mCell;
    const unsigned int mChannel;
};
}

void N2D2::NodeOut::incomingSpike(Node* /*origin*/,
                                  Time_T timestamp,
                                  EventType_T type)
{
    mNet.newEvent(this, NULL, timestamp, type);
}

#endif // N2D2_NODEOUT_H
