/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#ifndef N2D2_AER_DATABASE_H
#define N2D2_AER_DATABASE_H

#include "Database.hpp"
#include "Network.hpp"

namespace N2D2 {

struct AerReadEvent {
    unsigned int x;
    unsigned int y;
    unsigned int channel;
    Time_T time;

    AerReadEvent(unsigned int x,
                unsigned int y,
                unsigned int channel,
                Time_T time);
};


class AER_Database : public Database {
public:

    AER_Database(bool loadDataInMemory=true);
    //virtual void load(const std::string& /*dataPath*/,
    //                  const std::string& labelPath,
    //                  bool /*extractROIs*/);
    //virtual cv::Mat getStimulusData(StimulusID id);
    virtual std::vector<AerReadEvent> loadAerStimulusData(StimuliSet set,
                                          StimulusID id,
                                          Time_T start,
                                          Time_T stop,
                                          unsigned int repetitions=1,
                                          unsigned int partialStimulus=0)=0;

    virtual ~AER_Database(){};

protected:

};
}


#endif // N2D2_AER_DATABASE_H
