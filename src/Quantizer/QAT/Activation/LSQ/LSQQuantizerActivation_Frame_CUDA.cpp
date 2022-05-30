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

#include "Quantizer/QAT/Activation/LSQ/LSQQuantizerActivation_Frame_CUDA.hpp"
#include "Quantizer/QAT/Kernel/Quantizer_Frame_CUDA_Kernels.hpp"
#include "Quantizer/QAT/Kernel/LSQQuantizer_Frame_CUDA_Kernels.hpp"
#include "third_party/half.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"


template<>
N2D2::Registrar<N2D2::LSQQuantizerActivation>
N2D2::LSQQuantizerActivation_Frame_CUDA<half_float::half>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::LSQQuantizerActivation_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::LSQQuantizerActivation>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::LSQQuantizerActivation>
N2D2::LSQQuantizerActivation_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::LSQQuantizerActivation_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::LSQQuantizerActivation>::Type<float>());

template<>
N2D2::Registrar<N2D2::LSQQuantizerActivation>
N2D2::LSQQuantizerActivation_Frame_CUDA<double>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::LSQQuantizerActivation_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::LSQQuantizerActivation>::Type<double>());
    

namespace N2D2 {

template<class T>
LSQQuantizerActivation_Frame_CUDA<T>::LSQQuantizerActivation_Frame_CUDA()
    : LSQQuantizerActivation(),
      QuantizerActivation_Frame_CUDA<T>()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame_CUDA<T> >();
}

template <class T>
LSQQuantizerActivation_Frame_CUDA<T>::~LSQQuantizerActivation_Frame_CUDA()
{
}


// ----------------------------------------------------------------------------
// ---------------------------- Forward functions -----------------------------
// ----------------------------------------------------------------------------

template <>
void LSQQuantizerActivation_Frame_CUDA<half_float::half>::propagate(BaseTensor& /*baseInOut*/,
                                                                    bool /*inference*/)  
{
    // nothing here for now
}

template <>
void LSQQuantizerActivation_Frame_CUDA<float>::propagate(BaseTensor& baseInOut,
                                                         bool /*inference*/)  
{
    const CudaTensor<float>& input = dynamic_cast<const CudaTensor<float>&>(baseInOut);

    /*
    mBitRanges = std::make_pair( (int) -(std::pow(2, (int) mBitPrecision) - 1), 
                                    (int) (std::pow(2, (int) mBitPrecision - 1) - 1) );
    */
   //init the quantization range
   if(mBitRanges.first == 0 && mBitRanges.second == 0){
        mBitRanges = std::make_pair( 0, (int) mRange );
        //std::cout << " LSQ activation range :: [" << mBitRanges.first << ";" << mBitRanges.second << "]"<< std::endl;
   }

    unsigned int totElement = input.size();
    mGradScaleFactor = 1.0f / sqrt(totElement * mBitRanges.second);

    if(mStepSize.empty()) {
        input.synchronizeDToH();
        // Initialisation of the activation step size according to the LSQ paper
        // (https://arxiv.org/pdf/1902.08153.pdf)    
        if(mSetOptInitStepSize){
            float initialValue = 2 * (float)input.mean(true) / sqrt(mBitRanges.second);
            setStepSizeValue(initialValue);
        }
        mStepSize.resize({1, 1, 1, 1});
        mStepSize.fill(float(mStepSizeParameter));
        mStepSize.synchronizeHToD();
        input.synchronizeHToD();        // TODO: check if this line is really required
    }

    if(mFullPrecisionActivations.empty()) {
        mFullPrecisionActivations.resize(baseInOut.dims());
        mFullPrecisionActivations.synchronizeHToD();
    }

    LSQQuantizer_Frame_CUDA_Kernels::cudaF_quantize_propagate(input.getDevicePtr(),
                                                                mStepSize.getDevicePtr(),
                                                                mBitRanges.first,
                                                                mBitRanges.second,
                                                                input.getDevicePtr(),
                                                                mFullPrecisionActivations.getDevicePtr(),
                                                                true,
                                                                input.size());
}

template <>
void LSQQuantizerActivation_Frame_CUDA<double>::propagate(BaseTensor& /*baseInOut*/,
                                                          bool /*inference*/)  
{
    // nothing here for now
}


// ----------------------------------------------------------------------------
// --------------------------- Backward functions -----------------------------
// ----------------------------------------------------------------------------

template <>
void N2D2::LSQQuantizerActivation_Frame_CUDA<half_float::half>::back_propagate(const BaseTensor& /*baseInput*/,
                                                                               const BaseTensor& /*baseOutput*/,
                                                                               const BaseTensor& /*baseDiffInput*/,
                                                                               BaseTensor& /*baseDiffOutput*/)
{
    //nothing here for now
}

template <>
void N2D2::LSQQuantizerActivation_Frame_CUDA<float>::back_propagate(const BaseTensor& baseInput,
                                                                    const BaseTensor& /*baseOutput*/,
                                                                    const BaseTensor& baseDiffInput,
                                                                    BaseTensor& baseDiffOutput)
{
    const float beta = (mSolver->isNewIteration()) ? float(0.0) : float(1.0);

    const CudaTensor<float>& diffInput = dynamic_cast<const CudaTensor<float>&>(baseDiffInput);
    const CudaTensor<float>& input = dynamic_cast<const CudaTensor<float>&>(baseInput);
    CudaTensor<float>& diffOutput = dynamic_cast<CudaTensor<float>&>(baseDiffOutput);

    //init the quantization range, if not done before (e.g. unit test)
    if(mBitRanges.first == 0 && mBitRanges.second == 0){
        mBitRanges = std::make_pair( 0, (int) mRange );
        //std::cout << " LSQ activation range :: [" << mBitRanges.first << ";" << mBitRanges.second << "]"<< std::endl;
    }

    // Initialize mDiffStepSize at the first backpropagate
    if(mDiffStepSize.empty()) {
        mDiffStepSize.resize({1, 1, 1, 1});
        mDiffStepSize.fill(float(0));
        mDiffStepSize.synchronizeHToD();
    }
    // Initialize mDiffStepSizeTensor at the first backpropagate
    if(mDiffStepSizeTensor.empty()) {
        mDiffStepSizeTensor.resize(diffInput.dims());
        mDiffStepSizeTensor.fill(float(0));
        mDiffStepSizeTensor.synchronizeHToD();
    }
    // Initialize StepSize tensor at the first propagate
    if(mStepSize.empty()) {
        mStepSize.resize({1, 1, 1, 1});
        mStepSize.fill(float(mStepSizeParameter));
        mStepSize.synchronizeHToD();
        //this is done for unit test purpose
        unsigned int totElement = input.size();
        mGradScaleFactor = 1.0f / sqrt(totElement * mBitRanges.second);
    }

    //don't do in-place for backprop anymore
    LSQQuantizer_Frame_CUDA_Kernels::cudaF_quantize_back_propagate( diffInput.getDevicePtr(),
                                                                    input.getDevicePtr(),
                                                                    diffOutput.getDevicePtr(),
                                                                    mDiffStepSizeTensor.getDevicePtr(),
                                                                    mBitRanges.first,
                                                                    mBitRanges.second,
                                                                    mStepSize.getDevicePtr(),
                                                                    mGradScaleFactor,
                                                                    beta,
                                                                    diffInput.size());
    mDiffStepSize.synchronizeDToH();
    mDiffStepSize(0,0,0,0) 
        = Quantizer_Frame_CUDA_Kernels::cudaF_accumulate(mDiffStepSizeTensor.getDevicePtr(), 
                                                         mDiffStepSizeTensor.size())/* + beta*mDiffStepSize(0,0,0,0)*/; 
    mDiffStepSize.synchronizeHToD();
}

template <>
void N2D2::LSQQuantizerActivation_Frame_CUDA<double>::back_propagate( const BaseTensor& /*baseInput*/,
                                                            const BaseTensor& /*baseOutput*/,
                                                            const BaseTensor& /*baseDiffInput*/,
                                                            BaseTensor& /*baseDiffOutput*/)
{
    //nothing here for now
}


template<class T>
void N2D2::LSQQuantizerActivation_Frame_CUDA<T>::update(unsigned int batchSize)
{
    mSolver->update(mStepSize, mDiffStepSize, batchSize);
}


template <class T>
void LSQQuantizerActivation_Frame_CUDA<T>::exportParameters(const std::string& dirName,
                                                            const std::string& cellName) const 
{
    const std::string fileName = dirName + "/" 
                                    + cellName + "_QAct_LSQ_StepSize.syntxt";
    std::ofstream stepSize(fileName.c_str());
    if (!stepSize.good())
        throw std::runtime_error("Could not create synaptic file: "
                                 + fileName);
    if(!mStepSize.empty()) {     
        mStepSize.synchronizeDToH();                        
        stepSize << mStepSize(0);
        mStepSize.synchronizeHToD();
    }
}


template <class T>
void LSQQuantizerActivation_Frame_CUDA<T>::importParameters(const std::string& dirName,
                                                            const std::string& cellName, 
                                                            bool ignoreNotExists)
{

    const std::string stepSizeFile = dirName + "/" 
                                    + cellName + "_QAct_LSQ_StepSize.syntxt";

    std::ifstream stepSize(stepSizeFile.c_str());

    if (!stepSize.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << stepSizeFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + stepSizeFile);
    }

    if(mStepSize.empty()) {
        mStepSize.resize({1, 1, 1, 1});
    }

    T value;
    if (!(stepSize >> value))
        throw std::runtime_error( "Error while reading synaptic file: "
                        + stepSizeFile);
    mStepSize(0) = T(value);

    // Discard trailing whitespaces
    while (std::isspace(stepSize.peek()))
        stepSize.ignore();

    if (stepSize.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Synaptic file size larger than expected: "
                                 + stepSizeFile);

    mStepSize.synchronizeHToD();
}

}


namespace N2D2 {
    template class LSQQuantizerActivation_Frame_CUDA<half_float::half>;
    template class LSQQuantizerActivation_Frame_CUDA<float>;
    template class LSQQuantizerActivation_Frame_CUDA<double>;
}

#endif