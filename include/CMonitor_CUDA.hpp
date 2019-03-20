/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes Thiele (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
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

#ifndef N2D2_CMONITOR_CUDA_H
#define N2D2_CMONITOR_CUDA_H


#include "CMonitor.hpp"
#include "CEnvironment_CUDA.hpp"
#include "CMonitor_CUDA_kernels.hpp"
#include "Cell/Cell_CSpike_CUDA.hpp"




namespace N2D2 {
/**
 * The CMonitor_CUDA class provides tools to monitor the activity of the network.
*/
class CMonitor_CUDA: public CMonitor {
public:
    CMonitor_CUDA();
    virtual void initialize(unsigned int nbTimesteps,
                            unsigned int nbClasses=0);
    virtual bool tick(Time_T timestamp);
    virtual void update(Time_T start, Time_T stop);


protected:

    unsigned int mDeviceMaxThreads;
    unsigned int mDeviceWarpSize;

};
}




#endif // N2D2_CMONITOR_CUDA_H
