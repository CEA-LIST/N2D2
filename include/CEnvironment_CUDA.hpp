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

#ifndef N2D2_CENVIRONMENT_CUDA_H
#define N2D2_CENVIRONMENT_CUDA_H


#include "CEnvironment.hpp"
#include "CEnvironment_CUDA_kernels.hpp"


namespace N2D2 {
class CEnvironment_CUDA : public CEnvironment {
public:
    CEnvironment_CUDA(Database& database,
                     const std::vector<size_t>& size,
                     unsigned int batchSize = 1,
                     unsigned int nbSubStimuli = 1,
                     bool compositeStimuli = false);
    virtual void tick(Time_T timestamp, Time_T start, Time_T stop);
    virtual void reset(Time_T timestamp);
    virtual const Tensor<char>& getTickData(unsigned int subIdx) const
    {
        //TODO: There is a problem with that
        std::cout << "CEnv_CUDA getTickData const" << std::endl;
        exit(0);
        return mTickData[subIdx];
    };
    virtual Tensor<char>& getTickData(unsigned int subIdx)
    {
        mTickData[subIdx].synchronizeDToH();
        return mTickData[subIdx];
    };
    virtual Interface<char>& getTickData()
    {
        for (unsigned int k=0; k<mTickData.size(); k++){
            mTickData[k].synchronizeDToH();
        }
        return mTickData;
    };
    virtual Tensor<char>& getTickOutputs()
    {
        mTickOutputs.synchronizeDToH();
        return mTickOutputs;
    };
    virtual ~CEnvironment_CUDA();

protected:
    CudaInterface<Time_T> mNextEventTime;
    CudaInterface<char> mNextEventType;
    std::vector<curandState*> mCurandStates;

};
}

#endif // N2D2_CENVIRONMENT_CUDA_H
