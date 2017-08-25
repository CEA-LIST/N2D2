/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_CUDNN_INFERENCE_H
#define N2D2_CUDNN_INFERENCE_H

#include "cpp_utils.hpp"
#include "../dnn/include/network.hpp"

/**
 * n2d2_cudnn_inference
*/
class n2d2_cudnn_inference {
public:

     /**
     * Abstract n2d2_cudnn_inference constructor
     *
     * @param batchSize     Size of the maximum batch
     * @param devID         Device ID to address
    */
    n2d2_cudnn_inference(unsigned int batchSize = 1,
                            unsigned int devID = 0,
                            bool profiling = false);

    /**
     * Initialize
     *
     * @param batchSize     Size of the maximum batch
     * @param devID         Device ID to address
    */
    void initialize(unsigned int batchSize = 1,
                    unsigned int devID = 0,
                    bool profiling = false);
    void execute(DATA_T* input_data);
    void estimated(uint32_t* output_data, unsigned int target = 0);
    void getProfiling(unsigned int nbIter = 1);
    /// Setting up the batchSize.
    void setBatchSize(unsigned int batchSize = 1);
    /// Setting up the device ID.
    void setDeviceID(unsigned int devID = 0);

    /// Returns the batch size
    unsigned int getBatchSize() const
    {
        return mBatchSize;
    };

    /// Returns the device ID
    unsigned int getDeviceID() const
    {
        return mDeviceID;
    };

    unsigned int inputDimX();
    unsigned int inputDimY();
    unsigned int inputDimZ();

    std::vector<unsigned int> outputDimX();
    std::vector<unsigned int> outputDimY();
    std::vector<unsigned int> outputDimZ();
    std::vector<unsigned int> outputTarget();

    /// Destructor
    ~n2d2_cudnn_inference() { free_memory(); };

protected:
    // Batch size
    unsigned int mBatchSize;
    // Device ID
    unsigned int mDeviceID;
    // Number of network targets
    unsigned int mNbTargets;
    // Profiling flag
    bool mProfiling;

};

#endif // N2D2_CUDNN_INFERENCE_H


