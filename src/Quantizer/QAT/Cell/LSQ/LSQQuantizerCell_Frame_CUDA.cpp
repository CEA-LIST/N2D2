/**
 * (C) Copyright 2020 CEA LIST. All Rights Reserved.
 *  Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
 *                  David BRIAND (david.briand@cea.fr)
 *                  Inna KUCHER (inna.kucher@cea.fr)
 *                  Olivier BICHLER (olivier.bichler@cea.fr)
 *                  Vincent TEMPLIER (vincent.templier@cea.fr)
 * 
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 * 
 */

#ifdef CUDA

#include "Quantizer/QAT/Cell/LSQ/LSQQuantizerCell_Frame_CUDA.hpp"
#include "Quantizer/QAT/Kernel/LSQQuantizer_Frame_CUDA_Kernels.hpp"
#include "Quantizer/QAT/Kernel/Quantizer_Frame_CUDA_Kernels.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"

/* Only float functions for now */

template<>
N2D2::Registrar<N2D2::LSQQuantizerCell>
N2D2::LSQQuantizerCell_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::LSQQuantizerCell_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::LSQQuantizerCell>::Type<float>());


namespace N2D2 {

template<>
LSQQuantizerCell_Frame_CUDA<float>::LSQQuantizerCell_Frame_CUDA()
    : LSQQuantizerCell(),
      QuantizerCell_Frame_CUDA()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame_CUDA<float> >();
}


template<class T>
void LSQQuantizerCell_Frame_CUDA<T>::addWeights(BaseTensor& weights, BaseTensor& diffWeights)
{
    if(mInitialized){
        //reset all refs, as they were changed by calling initialize() again
        mFullPrecisionWeights.clear();
        mDiffQuantizedWeights.clear();

        mFullPrecisionWeights.push_back(&weights);
        mDiffQuantizedWeights.push_back(&diffWeights);
        return;
    }

    mFullPrecisionWeights.push_back(&weights);
    mQuantizedWeights.push_back(new CudaTensor<T>(weights.dims()));

    mDiffQuantizedWeights.push_back(&diffWeights);
    mDiffFullPrecisionWeights.push_back(new CudaTensor<T>(diffWeights.dims()), 0);

    mDiffStepSizeInterface.push_back(new CudaTensor<T>(diffWeights.dims()), 0);
    mDiffStepSizeInterface.back().fill(0.0);

    //mDiffStepSizeTensor.resize(diffWeights.dims(), T(0.0));
}

template<class T>
void LSQQuantizerCell_Frame_CUDA<T>::addBiases(BaseTensor& biases, BaseTensor& diffBiases)
{
    if(mInitialized)
        return;

    mFullPrecisionBiases = &(dynamic_cast<CudaBaseTensor&>(biases));
    mQuantizedBiases.resize(biases.dims());

    mDiffQuantizedBiases = &(dynamic_cast<CudaBaseTensor&>(diffBiases));
    mDiffFullPrecisionBiases.resize(diffBiases.dims());
}

template<class T>
void LSQQuantizerCell_Frame_CUDA<T>::initialize()
{
    //Set weights range
    mBitRanges = std::make_pair( (int) -((mRange+1)/2), (int) ((mRange - 1)/2));

    Tensor<T> tens = cuda_tensor_cast<T>(mFullPrecisionWeights[0]);

    //init step size with fake value (needed for quant weight initialization below)
    if(mStepSize.empty()) {
        if(mSetOptInitStepSize){
            float initialValue = 10000.;
            setStepSizeValue(initialValue);
        }
        mStepSize.resize({1,1,1,1});
        mStepSize.fill(T(mStepSizeVal));
        mStepSize.synchronizeHToD();

        mDiffStepSize.resize({1,1,1,1});
        mDiffStepSize.fill(T(0.0));
        mDiffStepSize.synchronizeHToD();
    }
    mGradScaleFactor = T(0.0);

    std::cout << "      " << std::setprecision(8) <<
        "Quantizer::LSQ || " <<  
        " StepSizeVal[" << mStepSizeVal << "] || " <<
        " StepInit[" << mSetOptInitStepSize << "] || " << 
        " Range[" << mBitRanges.first << ", " << mBitRanges.second << "]" << std::endl;

    //Initialize the quantized weights
    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
        std::shared_ptr<CudaDeviceTensor<float> > fullPrecWeights
            = cuda_device_tensor_cast<float>(mFullPrecisionWeights[k]);

        LSQQuantizer_Frame_CUDA_Kernels::cudaF_quantize_propagate(fullPrecWeights->getDevicePtr(),
                                                                  mStepSize.getDevicePtr(),
                                                                  mBitRanges.first,
                                                                  mBitRanges.second,
                                                                  mQuantizedWeights[k].getDevicePtr(),
                                                                  fullPrecWeights->getDevicePtr(),
                                                                  false, /*False because we don't need to copy FP weights*/
                                                                  mFullPrecisionWeights[k].size());  
    }

    mInitialized = true;
}

template<>
void LSQQuantizerCell_Frame_CUDA<float>::propagate()
{
    //init step size using correct imported weights this time
    if(mStepSizeVal == 10000.) {
        std::cout << "Initialize the correct LSQ step size ... " << std::endl;

        Tensor<float> tens = cuda_tensor_cast<float>(mFullPrecisionWeights[0]);

        if(mSetOptInitStepSize){
            // Initialisation of the weight step size according to the LSQ paper
            // (https://arxiv.org/pdf/1902.08153.pdf)
            float initialValue = 2*(float)tens.mean(true) / sqrt((mRange-1)/2);
            setStepSizeValue(initialValue);

            // Initialisation of the weight step size according to the LSQ+ paper
            // (https://arxiv.org/pdf/2004.09576.pdf)
            //float a = (float)tens.mean(true)-3*(float)tens.std();
            //float b = (float)tens.mean(true)+3*(float)tens.std();
            //float initialValue = std::max(abs(a), abs(b))/((mRange-1)/2);
            //setStepSizeValue(initialValue);
        }
        mStepSize.resize({1,1,1,1});
        mStepSize.fill(float(mStepSizeVal));
        mStepSize.synchronizeHToD();

        std::cout << "      " << std::setprecision(8) <<
            "Quantizer::LSQ || " <<
            " StepSizeVal[" << mStepSizeVal << "] || " <<
            " StepInit[" << mSetOptInitStepSize << "] || " <<
            " Range[" << mBitRanges.first << ", " << mBitRanges.second << "]" << std::endl;
    }


    unsigned int totElementW = 0;

    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {

        std::shared_ptr<CudaDeviceTensor<float> > fullPrecWeights
            = cuda_device_tensor_cast<float>(mFullPrecisionWeights[k]);

        LSQQuantizer_Frame_CUDA_Kernels::cudaF_quantize_propagate(fullPrecWeights->getDevicePtr(),
                                                                    mStepSize.getDevicePtr(),
                                                                    mBitRanges.first,
                                                                    mBitRanges.second,
                                                                    mQuantizedWeights[k].getDevicePtr(),
                                                                    fullPrecWeights->getDevicePtr(),
                                                                    false, /*False because we don't need to copy FP weights*/
                                                                    mFullPrecisionWeights[k].size());  
        totElementW += mFullPrecisionWeights[k].size();
    }

    mGradScaleFactor = 1.0f / sqrt(totElementW * mBitRanges.second);

    if (mFullPrecisionBiases) {
        std::shared_ptr<CudaDeviceTensor<float> > fullPrecBiases
            = cuda_device_tensor_cast<float>(cuda_tensor_cast<float>(*mFullPrecisionBiases));

        Quantizer_Frame_CUDA_Kernels::cudaF_copyData(fullPrecBiases->getDevicePtr(),
                                                     mQuantizedBiases.getDevicePtr(),
                                                     mFullPrecisionBiases->size());
    }
}

template<>
void LSQQuantizerCell_Frame_CUDA<float>::back_propagate()
{
    float diffStepSize = 0.0f;
    
    if(!mSolver->isNewIteration()) {
        mDiffStepSize.synchronizeDToH();
        diffStepSize = mDiffStepSize(0,0,0,0);
        mDiffStepSize.synchronizeHToD();
    }
    for (unsigned int k = 0, size = mDiffQuantizedWeights.size(); k < size; ++k) {

        std::shared_ptr<CudaDeviceTensor<float> > diffQuantizedWeights
            = cuda_device_tensor_cast<float>(mDiffQuantizedWeights[k]);
        
        std::shared_ptr<CudaDeviceTensor<float> > fullPrecWeights
            = cuda_device_tensor_cast<float>(mFullPrecisionWeights[k]);

        LSQQuantizer_Frame_CUDA_Kernels::cudaF_quantize_back_propagate(diffQuantizedWeights->getDevicePtr(),
                                                                       fullPrecWeights->getDevicePtr(),
                                                                       mDiffFullPrecisionWeights[k].getDevicePtr(),
                                                                       mDiffStepSizeInterface[k].getDevicePtr(),
                                                                       mBitRanges.first,
                                                                       mBitRanges.second,
                                                                       mStepSize.getDevicePtr(),
                                                                       mGradScaleFactor,
                                                                       0.0f,
                                                                       mFullPrecisionWeights[k].size());

        diffStepSize += Quantizer_Frame_CUDA_Kernels::cudaF_accumulate(mDiffStepSizeInterface[k].getDevicePtr(),
                                                                       mDiffStepSizeInterface[k].size());
    }
    mDiffStepSize.synchronizeDToH();
    mDiffStepSize(0,0,0,0) = diffStepSize;
    mDiffStepSize.synchronizeHToD();

    if (mDiffQuantizedBiases) {

        std::shared_ptr<CudaDeviceTensor<float> > diffQuantBiases
            = cuda_device_tensor_cast<float>(cuda_tensor_cast<float>(*mDiffQuantizedBiases));

        Quantizer_Frame_CUDA_Kernels::cudaF_copyData(diffQuantBiases->getDevicePtr(),
                                                     mDiffFullPrecisionBiases.getDevicePtr(),
                                                     mDiffQuantizedBiases->size());
    }
}


template<>
void LSQQuantizerCell_Frame_CUDA<float>::update(unsigned int batchSize)
{
    mSolver->update(mStepSize, mDiffStepSize, batchSize);
}


template <class T>
LSQQuantizerCell_Frame_CUDA<T>::~LSQQuantizerCell_Frame_CUDA()
{
    // dtor
}


template <class T>
void LSQQuantizerCell_Frame_CUDA<T>::exportFreeParameters(const std::string& fileName) const 
{
    mStepSize.synchronizeDToH();

    const std::string dirName = Utils::dirName(fileName);
    if (!dirName.empty())
        Utils::createDirectories(dirName);

    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string alphasWFile = fileBase + "_alphaW_LSQ" + fileExt;
    std::ofstream alphasW(alphasWFile.c_str());

    if (!alphasW.good())
        throw std::runtime_error("Could not create synaptic file: "
                                 + alphasWFile);
                                 
    alphasW << mStepSize(0) << " ";
    mStepSize.synchronizeHToD();
}

template <class T>
void LSQQuantizerCell_Frame_CUDA<T>::importFreeParameters(const std::string
                                                     & fileName, bool ignoreNotExists)
{
    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string alphasWFile = fileBase + "_alphaW_LSQ" + fileExt;
    std::ifstream alphasW(alphasWFile.c_str());

    if (!alphasW.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << alphasWFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + alphasWFile);
    }

    mStepSize.synchronizeDToH();
    if(mStepSize.empty()){
        mStepSize.resize({1, 1, 1, 1});
    }

    T valueW;
    if (!(alphasW >> valueW))
        throw std::runtime_error( "Error while reading synaptic file: "
                        + alphasWFile);

    mStepSize(0) = T(valueW);
    setStepSizeValue(mStepSize(0));
    
    // Discard trailing whitespaces
    while (std::isspace(alphasW.peek()))
        alphasW.ignore();

    if (alphasW.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Synaptic file size larger than expected: "
                                 + alphasWFile);

    mStepSize.synchronizeHToD();
}


}
#endif