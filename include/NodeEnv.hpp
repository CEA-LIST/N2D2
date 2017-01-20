/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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

#ifndef N2D2_NODEENV_H
#define N2D2_NODEENV_H

#include "Node.hpp"

namespace N2D2 {
/**
 * Input node, receives external stimuli. It is not a real neuron as it has no
 * input or integration circuit.
*/
class NodeEnv : public Node {
public:
    NodeEnv(Network& net,
            double scale,
            double orientation,
            unsigned int x,
            unsigned int y = 0);
    inline void
    incomingSpike(Node* origin, Time_T timestamp, EventType_T type = 0);
    virtual ~NodeEnv() {};
};
}

void N2D2::NodeEnv::incomingSpike(Node* /*origin*/,
                                  Time_T timestamp,
                                  EventType_T type)
{
    mNet.newEvent(this, NULL, timestamp, type);
}

#endif // N2D2_NODEENV_H
