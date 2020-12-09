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

#ifndef N2D2_SPIKEEVENT_H
#define N2D2_SPIKEEVENT_H

#include "Network.hpp"
#include "Node.hpp"

namespace N2D2 {
/**
 * Each event generated in the network is an instance of this class. It contains
 * two main methods:
 * - release(): commit the event (= make it append right now).
 * - discard(): cancel the event, so that it will never be released in the
 * future.
*/
class SpikeEvent {
public:
    inline SpikeEvent(Node* origin,
                      Node* destination,
                      Time_T timestamp,
                      EventType_T type);
    inline void initialize(Node* origin,
                           Node* destination,
                           Time_T timestamp,
                           EventType_T type);
    inline Time_T release();
    void discard()
    {
        mDiscarded = true;
    };
    bool isDiscarded() const
    {
        return mDiscarded;
    };
    Time_T getTimestamp() const
    {
        return mTimestamp;
    };
    EventType_T getType() const
    {
        return mType;
    };
    // We really want this function to be inlined for better performances
    inline bool operator<(const SpikeEvent& event) const;
    virtual ~SpikeEvent() {};

private:
    // Internal variables
    Node* mOrigin;
    Node* mDestination;
    Time_T mTimestamp;
    EventType_T mType;
    bool mDiscarded;
};
}

N2D2::SpikeEvent::SpikeEvent(Node* origin,
                             Node* destination,
                             Time_T timestamp,
                             EventType_T type)
    : mOrigin(origin),
      mDestination(destination),
      mTimestamp(timestamp),
      mType(type),
      mDiscarded(false)
{
    // ctor
}

void N2D2::SpikeEvent::initialize(Node* origin,
                                  Node* destination,
                                  Time_T timestamp,
                                  EventType_T type)
{
    mOrigin = origin;
    mDestination = destination;
    mTimestamp = timestamp;
    mType = type;
    mDiscarded = false;
}

N2D2::Time_T N2D2::SpikeEvent::release()
{
    if (mDestination == NULL)
        mOrigin->emitSpike(mTimestamp, mType);
    else
        mDestination->incomingSpike(mOrigin, mTimestamp, mType);

    return mTimestamp;
}

bool N2D2::SpikeEvent::operator<(const SpikeEvent& event) const
{
    return (mTimestamp > event.mTimestamp
            || (mTimestamp == event.mTimestamp && mDestination == NULL
                && event.mDestination != NULL));
}

#endif // N2D2_SPIKEEVENT_H
