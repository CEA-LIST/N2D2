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

#ifndef N2D2_NODESYNC_H
#define N2D2_NODESYNC_H

#include "Node.hpp"

namespace N2D2 {
class Xcell;

/**
 * This is a very special node type, which is used to emulate pipelining and
 * synchronization between the layers of the network, in
 * the currently frame-based proposed architecture.
*/
class NodeSync : public Node {
public:
    NodeSync(Network& net, Xcell& cell);
    void addLink(Node* origin);
    void incomingSpike(Node* origin, Time_T timestamp, EventType_T type = 0);
    Node* getParent() const
    {
        return mLink;
    };
    virtual ~NodeSync() {};

private:
    // Internal variables
    Xcell& mCell;
    Node* mLink;
};
}

#endif // N2D2_NODESYNC_H
