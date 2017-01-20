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

#include "Node.hpp"

unsigned int N2D2::Node::mIdCnt = 1;

N2D2::Node::Node(Network& net)
    : NetworkObserver(net),
      mActivityRecording(false),
      mId(mIdCnt++),
      mLastActivationTime(0),
      mScale(1.0),
      mOrientation(0.0),
      mLayer(0),
      mArea(0, 0, 0, 0)
{
    // ctor
}

void N2D2::Node::addBranch(Node* branch)
{
    mBranches.push_back(branch);
}

void N2D2::Node::removeBranch(Node* branch)
{
    mBranches.erase(std::remove(mBranches.begin(), mBranches.end(), branch),
                    mBranches.end());
}

void N2D2::Node::emitSpike(Time_T timestamp, EventType_T type)
{
    if (mActivityRecording)
        mNet.recordSpike(mId, timestamp, type);

    mLastActivationTime = timestamp;

    std::for_each(mBranches.begin(),
                  mBranches.end(),
                  std::bind(&Node::propagateSpike,
                            std::placeholders::_1,
                            this,
                            timestamp,
                            type));
}

unsigned int
N2D2::Node::getActivity(Time_T start, Time_T stop, EventType_T type) const
{
    if (!mActivityRecording)
        throw std::runtime_error("Activity not recorded for this node.");

    const NodeEvents_T& record = mNet.getSpikeRecording(mId);
    unsigned int activity = 0;

    for (NodeEvents_T::const_iterator it = record.begin(), itEnd = record.end();
         it != itEnd;
         ++it) {
        if ((*it).second == type && (start == 0 || (*it).first >= start)
            && (stop == 0 || (*it).first < stop))
            ++activity;
    }

    return activity;
}

std::pair<N2D2::Time_T, bool> N2D2::Node::getFirstActivationTime(
    Time_T start, Time_T stop, EventType_T type) const
{
    if (!mActivityRecording)
        throw std::runtime_error("Activity not recorded for this node.");

    const NodeEvents_T& record = mNet.getSpikeRecording(mId);

    for (NodeEvents_T::const_iterator it = record.begin(), itEnd = record.end();
         it != itEnd;
         ++it) {
        if ((*it).second == type && (start == 0 || (*it).first >= start)
            && (stop == 0 || (*it).first < stop))
            return std::make_pair((*it).first, true);
    }

    return std::make_pair(0, false);
}
