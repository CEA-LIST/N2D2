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

#ifndef N2D2_TENSORRT_INFERENCE_H
#define N2D2_TENSORRT_INFERENCE_H
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <iterator>
#include <vector>
#include <string>
#include <algorithm> // std::sort
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <iomanip>
#include <stdexcept>
#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h> // (u)intx_t typedef

#ifdef WRAPPER_PYTHON

#ifndef BOOST_NATIVE
#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;
#else
#include <boost/numpy.hpp>
namespace np = boost::numpy;
#endif

#include <boost/python.hpp>
#include <boost/scoped_array.hpp>
namespace p = boost::python;

#endif

/**
 * n2d2_tensorRT_inference
*/
class n2d2_tensorRT_inference {
public:
    n2d2_tensorRT_inference();

     /**
     * Abstract n2d2_tensorRT_inference constructor
     *
     * @param batchSize     Size of the maximum batch
     * @param devID         Device ID to address
     * @param nbIter        Number of tensorRT build iterations
    */
    n2d2_tensorRT_inference(unsigned int batchSize,
                            unsigned int devID = 0,
                            unsigned int nbIter = 1,
                            int bitPrecision = -32,
                            bool profiling = false,
                            std::string inputEngineFile = "",
                            std::string outputEngineFile = "",
                            bool useINT8 = false);

    /**
     * Initialize
     *
     * @param batchSize     Size of the maximum batch
     * @param devID         Device ID to address
     * @param nbIter        Number of tensorRT build iterations
    */
    void initialize(unsigned int batchSize = 1,
                    unsigned int devID = 0,
                    unsigned int nbIter = 1,
                    int bitPrecision = -32,
                    bool profiling = false,
                    std::string inputEngineFile = "",
                    std::string outputEngineFile = "",
                    bool useINT8 = false);
    void execute(float* input_data);
    void executeGPU(float** inout_buffer);
    void estimated(uint32_t* output_data,
                   unsigned int target = 0,
                   float threshold = 0.0,
                   bool useGPU = false);

    void overlay(unsigned char* overlay_data,
                 unsigned int batchSize = 1,
                   unsigned int target = 0,
                   float alpha = 0.5);

    void* getDevPtr(unsigned int target);
    void getProfiling(unsigned int nbIter = 1);
    /// Setting up the batchSize.
    void setBatchSize(unsigned int batchSize = 1);
    /// Setting up the device ID.
    void setDeviceID(unsigned int devID = 0);
    /// Setting up the number of tensorRT build iterations.
    void setNbIterBuild(unsigned int nbIter = 1);

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

    /// Return number of tensorRT build iterations
    unsigned int getNbIterBuild() const
    {
        return mNbIterBuild;
    };

    void logOutput(unsigned int target);
    void getOutput(float* output, unsigned int target);
    unsigned int inputDimX();
    unsigned int inputDimY();
    unsigned int inputDimZ();

    std::vector<unsigned int> outputDimX();
    std::vector<unsigned int> outputDimY();
    std::vector<unsigned int> outputDimZ();
    std::vector<unsigned int> outputTarget();

    /// Destructor
    ~n2d2_tensorRT_inference() { /*free_memory();*/ };


#ifdef WRAPPER_PYTHON

    void executePython(np::ndarray const & input);

    void getOutputPython(np::ndarray const & output, unsigned int target);

    void estimatedPython(np::ndarray const & output,
                         unsigned int target,
                         float threshold = 0.0,
                         bool useGPU = false);
                 
    void overlayPython(np::ndarray const & overlay_data,
                         unsigned int batchSize,
                         unsigned int target,
                         float alpha);

    void envReadPython(const std::string& fileName, unsigned int size,
                                            unsigned int channelsHeight, unsigned int channelsWidth,
                                             np::ndarray const & data, bool noLabels,
                                             unsigned int outputsSize,
                                             np::ndarray const & outputTargets);
#endif


protected:
    // Batch size
    unsigned int mBatchSize;
    // Device ID
    unsigned int mDeviceID;
    // Number of tensorRT build iterations
    unsigned int mNbIterBuild;
    // Number of network targets
    unsigned int mNbTargets;
    // Profiling flag
    bool mProfiling;
    // Bit Precision
    int mBitPrecision;

};

#endif // N2D2_TENSORRT_INFERENCE_H

