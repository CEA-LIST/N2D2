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

#include "cpp_utils.hpp"
#include "../dnn/include/network.hpp"

#include "n2d2_tensorRT_inference.hpp"

n2d2_tensorRT_inference::n2d2_tensorRT_inference()
{
    //ctor
}

n2d2_tensorRT_inference::n2d2_tensorRT_inference( unsigned int batchSize,
                                                  unsigned int devID,
                                                  unsigned int nbBuildIter,
                                                  int bitPrecision,
                                                  bool profiling,
                                                  std::string inputEngineFile,
                                                  std::string outputEngineFile,
                                                  bool useINT8)
    : mBatchSize(batchSize),
      mDeviceID(devID),
      mNbIterBuild(nbBuildIter),
      mBitPrecision(bitPrecision)
{
    // ctor
    mProfiling = false;

    if (mBatchSize == 0)
        throw std::runtime_error("n2d2_tensorRT_inference() constructor: "
                                 "batchSize must be at least 1");
    if (mNbIterBuild == 0)
        throw std::runtime_error("n2d2_tensorRT_inference() constructor: "
                                 "nbBuildIter must be at least 1");

    mNbTargets = getOutputNbTargets();
    std::cout << "The network provides "
                <<  mNbTargets
                << " output targets" << std::endl;

    network_tensorRT_init(mBatchSize, mDeviceID, mNbIterBuild, mBitPrecision, inputEngineFile, outputEngineFile, useINT8);

    if(profiling)
    {
        std::cout << "A layer wise profiling is set, it can decrease real time performances" << std::endl;
        mProfiling = true;
        setProfiling();
    }

}

void n2d2_tensorRT_inference::initialize(unsigned int batchSize,
                                         unsigned int devID,
                                         unsigned int nbIter,
                                         int bitPrecision,
                                         bool profiling,
                                          std::string inputEngineFile,
                                          std::string outputEngineFile,
                                          bool useINT8)
{
    mProfiling = false;

    if (batchSize == 0)
        throw std::runtime_error("n2d2_tensorRT_inference::initialize(): "
                                 "batchSize must be at least 1");
    mBatchSize = batchSize;

    if (nbIter == 0)
        throw std::runtime_error("n2d2_tensorRT_inference::initialize(): "
                                 "nbBuildIter must be at least 1");
    mNbIterBuild = nbIter;

    mDeviceID = devID;
    mBitPrecision = bitPrecision,

    network_tensorRT_init(mBatchSize, mDeviceID, mNbIterBuild, mBitPrecision, inputEngineFile, outputEngineFile, useINT8);

    if(profiling)
    {
        std::cout << "A layer wise profiling is set, it can decrease real time performances" << std::endl;
        mProfiling = true;
        setProfiling();
    }

}

void n2d2_tensorRT_inference::getProfiling(unsigned int nbIter)
{
    if(mProfiling)
        reportProfiling(nbIter);
}


void n2d2_tensorRT_inference::setBatchSize(unsigned int batchSize)
{
    if (batchSize == 0)
        throw std::runtime_error("n2d2_tensorRT_inference::setBatchSize(): "
                                 "batchSize must be at least 1");
    if(batchSize > mBatchSize)
        initialize(batchSize, mDeviceID, mNbIterBuild, mBitPrecision, mProfiling);
    else
        mBatchSize = batchSize;

}

void n2d2_tensorRT_inference::setDeviceID(unsigned int devID)
{
    if(devID != mDeviceID)
        initialize(mBatchSize, devID, mNbIterBuild, mBitPrecision, mProfiling);
}

void n2d2_tensorRT_inference::setNbIterBuild(unsigned int nbIter)
{
    if (nbIter == 0)
        throw std::runtime_error("n2d2_tensorRT_inference::setNbIterBuild(): "
                                 "nbBuildIter must be at least 1");

    initialize(mBatchSize, mDeviceID, nbIter, mBitPrecision, mProfiling);
}
void n2d2_tensorRT_inference::execute(float* input_data)
{
    network_tensorRT_syncExe(input_data, mBatchSize);
}
void n2d2_tensorRT_inference::executeGPU(float** externalInOut)
{
    network_tensorRT_syncGPUExe(externalInOut, mBatchSize);
}

void n2d2_tensorRT_inference
    ::estimated(uint32_t* output_data,
                unsigned int target,
                float threshold,
                bool useGPU)
{
    if(target > mNbTargets)
        throw std::runtime_error("n2d2_tensorRT_inference::estimated(): "
                                 "invalid target !");
    if(!useGPU)
        network_tensorRT_output(output_data, mBatchSize, target);
    else
        network_tensorRT_fcnn_output(output_data, mBatchSize, target, threshold);

}

void n2d2_tensorRT_inference
    ::overlay(unsigned char* overlay_data,
                unsigned int batchSize,
                unsigned int target,
                float alpha)
{
    network_tensorRT_overlay_input(overlay_data, batchSize, target, alpha);

}

void* n2d2_tensorRT_inference::getDevPtr(unsigned int target)
{
    return(network_tensorRT_get_device_ptr(target));
}

unsigned int n2d2_tensorRT_inference::inputDimX()
{
    return getInputDimX();
}

unsigned int n2d2_tensorRT_inference::inputDimY()
{
    return getInputDimY();
}

unsigned int n2d2_tensorRT_inference::inputDimZ()
{
    return getInputDimZ();
}

std::vector<unsigned int> n2d2_tensorRT_inference::outputDimX()
{
    std::vector<unsigned int> dim;
    unsigned int nbTarget = getOutputNbTargets();

    for(unsigned int it = 0; it < nbTarget; ++it)
        dim.push_back(getOutputDimX(it));

    return dim;
}

std::vector<unsigned int> n2d2_tensorRT_inference::outputDimY()
{
    std::vector<unsigned int> dim;
    unsigned int nbTarget = getOutputNbTargets();

    for(unsigned int it = 0; it < nbTarget; ++it)
        dim.push_back(getOutputDimY(it));

    return dim;
}

std::vector<unsigned int> n2d2_tensorRT_inference::outputDimZ()
{
    std::vector<unsigned int> dim;
    unsigned int nbTarget = getOutputNbTargets();

    for(unsigned int it = 0; it < nbTarget; ++it)
        dim.push_back(getOutputDimZ(it));

    return dim;
}

std::vector<unsigned int> n2d2_tensorRT_inference::outputTarget()
{
    std::vector<unsigned int> dim;
    unsigned int nbTarget = getOutputNbTargets();

    for(unsigned int it = 0; it < nbTarget; ++it)
        dim.push_back(getOutputTarget(it));

    return dim;
}

void n2d2_tensorRT_inference::logOutput(unsigned int target)
{
    if(target > getOutputNbTargets())
        throw std::runtime_error("n2d2_tensorRT_inference::logOutput(unsigned int): "
                         "invalid target !");

    std::stringstream fileName;
    fileName << "target_" << target << "_tensorRT.dat";

    std::ofstream dataFile(fileName.str());

    if (!dataFile.good())
        throw std::runtime_error("Could not create data log file: "
                                    + fileName.str());


    float* out;
    out = new float[getOutputDimX(target)
                        * getOutputDimY(target)
                        * getOutputDimZ(target)
                        * mBatchSize];

    network_tensorRT_log_output(out, mBatchSize, target);
    for(unsigned int b = 0; b < mBatchSize; ++ b){
        for(unsigned int z = 0; z < getOutputDimZ(target); ++ z){
            for(unsigned int y = 0; y < getOutputDimY(target); ++ y){
                for(unsigned int x = 0; x < getOutputDimX(target); ++ x)
                {
                    unsigned int idx = x + y * getOutputDimX(target)
                                        + z * getOutputDimX(target)
                                            * getOutputDimY(target)
                                        + b * getOutputDimX(target)
                                            * getOutputDimY(target)
                                            * getOutputDimZ(target);

                    dataFile << std::setprecision(10) << out[idx] << " ";
                }
                dataFile << "\n";
            }
        }
    }
    dataFile.close();
}

void n2d2_tensorRT_inference::getOutput(float* output, unsigned int target)
{
    if(target > getOutputNbTargets())
        throw std::runtime_error("n2d2_tensorRT_inference::logOutput(unsigned int): "
                         "invalid target !");

    network_tensorRT_log_output(output, mBatchSize, target);
}




#ifdef WRAPPER_PYTHON

template<class T>
struct VecToList
{
    static PyObject* convert(const std::vector<T>& vec)
    {
        p::list* l = new p::list();
        for(size_t i = 0; i < vec.size(); i++) {
            l->append(vec[i]);
        }

        return l->ptr();
    }
};


void n2d2_tensorRT_inference::executePython(np::ndarray const & input)
{
    n2d2_tensorRT_inference::execute(reinterpret_cast<float*>(input.get_data()));
}


void n2d2_tensorRT_inference::getOutputPython(np::ndarray const & output, unsigned int target)
{
    n2d2_tensorRT_inference::getOutput(reinterpret_cast<float*>(output.get_data()), target);
}

void n2d2_tensorRT_inference::estimatedPython(np::ndarray const & estimated,
                                              unsigned int target,
                                              float threshold,
                                              bool useGPU)
{
    n2d2_tensorRT_inference::estimated(reinterpret_cast<uint32_t*>(estimated.get_data()),
                                        target,
                                        threshold,
                                        useGPU);
}

void n2d2_tensorRT_inference::overlayPython(np::ndarray const & overlay_data,
                                              unsigned int batchSize,
                                              unsigned int target,
                                              float alpha)
{
    n2d2_tensorRT_inference::overlay(reinterpret_cast<unsigned char*>(overlay_data.get_data()),
                                        batchSize,
                                        target,
                                        alpha);
}

void n2d2_tensorRT_inference::envReadPython(const std::string& fileName, unsigned int size,
                                            unsigned int channelsHeight, unsigned int channelsWidth,
                                             np::ndarray const & data, bool noLabels,
                                             unsigned int outputsSize,
                                             np::ndarray const & outputTargets)
{
    envRead(fileName, size, channelsHeight, channelsWidth, reinterpret_cast<float*>(data.get_data()), noLabels, outputsSize, reinterpret_cast<int32_t*>(outputTargets.get_data()));
}



using namespace boost::python;


BOOST_PYTHON_MODULE(n2d2_tensorRT_inference)
{
    np::initialize();

    class_<n2d2_tensorRT_inference>("n2d2_tensorRT_inference")
        .def(init<unsigned int, unsigned int, unsigned int, int, bool, std::string, std::string, bool>())
        .def("init", &n2d2_tensorRT_inference::initialize)
		.def("estimated", &n2d2_tensorRT_inference::estimatedPython)
		.def("overlay", &n2d2_tensorRT_inference::overlayPython)
		.def("getProfiling", &n2d2_tensorRT_inference::getProfiling)
		.def("setBatchSize", &n2d2_tensorRT_inference::setBatchSize)
		.def("setDeviceID", &n2d2_tensorRT_inference::setDeviceID)
		.def("setNbIterBuild", &n2d2_tensorRT_inference::setNbIterBuild)
		.def("getBatchSize", &n2d2_tensorRT_inference::getBatchSize)
		.def("getDeviceID", &n2d2_tensorRT_inference::getDeviceID)
		.def("getNbIterBuild", &n2d2_tensorRT_inference::getNbIterBuild)
		.def("logOutput", &n2d2_tensorRT_inference::logOutput)
		.def("inputDimX", &n2d2_tensorRT_inference::inputDimX)
		.def("inputDimY", &n2d2_tensorRT_inference::inputDimY)
		.def("inputDimZ", &n2d2_tensorRT_inference::inputDimZ)
		.def("outputDimX", &n2d2_tensorRT_inference::outputDimX)
		.def("outputDimY", &n2d2_tensorRT_inference::outputDimY)
		.def("outputDimZ", &n2d2_tensorRT_inference::outputDimZ)
		.def("outputTarget", &n2d2_tensorRT_inference::outputTarget)
        .def("execute", &n2d2_tensorRT_inference::executePython)
        .def("getOutput", &n2d2_tensorRT_inference::getOutputPython)
        .def("envRead", &n2d2_tensorRT_inference::envReadPython)
    ;

    p::to_python_converter<std::vector<unsigned int, std::allocator<unsigned int> >, VecToList<unsigned int> >();
}

#endif

