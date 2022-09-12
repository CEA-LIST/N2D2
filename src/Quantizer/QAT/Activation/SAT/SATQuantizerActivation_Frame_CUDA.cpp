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

#include "Quantizer/QAT/Activation/SAT/SATQuantizerActivation_Frame_CUDA.hpp"
#include "Quantizer/QAT/Kernel/Quantizer_Frame_CUDA_Kernels.hpp"
#include "Quantizer/QAT/Kernel/SATQuantizer_Frame_CUDA_Kernels.hpp"
#include "third_party/half.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"
#include "utils/Random.hpp"

template<>
N2D2::Registrar<N2D2::SATQuantizerActivation>
N2D2::SATQuantizerActivation_Frame_CUDA<half_float::half>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::SATQuantizerActivation_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::SATQuantizerActivation>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::SATQuantizerActivation>
N2D2::SATQuantizerActivation_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::SATQuantizerActivation_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::SATQuantizerActivation>::Type<float>());

template<>
N2D2::Registrar<N2D2::SATQuantizerActivation>
N2D2::SATQuantizerActivation_Frame_CUDA<double>::mRegistrar(
    {"Frame_CUDA"},
    N2D2::SATQuantizerActivation_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::SATQuantizerActivation>::Type<double>());
    

template<class T>
N2D2::SATQuantizerActivation_Frame_CUDA<T>::SATQuantizerActivation_Frame_CUDA()
    : SATQuantizerActivation(),
      QuantizerActivation_Frame_CUDA<T>()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame_CUDA<T> >();
}

template <class T>
N2D2::SATQuantizerActivation_Frame_CUDA<T>::~SATQuantizerActivation_Frame_CUDA()
{
    // dtor
}


// ----------------------------------------------------------------------------
// ---------------------------- Forward functions -----------------------------
// ----------------------------------------------------------------------------

template <>
void N2D2::SATQuantizerActivation_Frame_CUDA<half_float::half>::propagate(BaseTensor& baseInOut,
                                                                          bool inference)  
{
    const CudaTensor<half_float::half>& input = dynamic_cast<const CudaTensor<half_float::half>&>(baseInOut);

    if(mAlphas.empty()) {
        mAlphas.resize({1, 1, 1, 1});
        mAlphas.fill(half_float::half(mAlphaParameter));
        mAlphas.synchronizeHToD();
    }

    if(!inference && mFullPrecisionActivations.empty()) {
        mFullPrecisionActivations.resize(baseInOut.dims());
        mFullPrecisionActivations.synchronizeHToD();
    }

    SATQuantizer_Frame_CUDA_Kernels::cudaH_quantize_activation_propagate(input.getDevicePtr(),
                                                                         mRange,
                                                                         mAlphas.getDevicePtr(),
                                                                         mFullPrecisionActivations.getDevicePtr(),
                                                                         input.size(),
                                                                         inference);
}

template <>
void N2D2::SATQuantizerActivation_Frame_CUDA<float>::propagate(BaseTensor& baseInOut,
                                                               bool inference)    
{
    const CudaTensor<float>& input = dynamic_cast<const CudaTensor<float>&>(baseInOut);

    if(mAlphas.empty()) {
        mAlphas.resize({1, 1, 1, 1});
        mAlphas.fill(float(mAlphaParameter));
        mAlphas.synchronizeHToD();
    }

    if(!inference && mFullPrecisionActivations.empty()) {
        mFullPrecisionActivations.resize(baseInOut.dims());
        mFullPrecisionActivations.synchronizeHToD();
    }

    SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_activation_propagate(input.getDevicePtr(),
                                                                         mRange,
                                                                         mAlphas.getDevicePtr(),
                                                                         mFullPrecisionActivations.getDevicePtr(),
                                                                         input.size(),
                                                                         inference);
}

template <>
void N2D2::SATQuantizerActivation_Frame_CUDA<double>::propagate(BaseTensor& baseInOut,
                                                                bool inference)   
{
    const CudaTensor<double>& input = dynamic_cast<const CudaTensor<double>&>(baseInOut);

    if(mAlphas.empty()) {
        mAlphas.resize({1, 1, 1, 1});
        mAlphas.fill(double(mAlphaParameter));
        mAlphas.synchronizeHToD();
    }

    if(!inference && mFullPrecisionActivations.empty()) {
        mFullPrecisionActivations.resize(baseInOut.dims());
        mFullPrecisionActivations.synchronizeHToD();
    }

    SATQuantizer_Frame_CUDA_Kernels::cudaD_quantize_activation_propagate(input.getDevicePtr(),
                                                                         mRange,
                                                                         mAlphas.getDevicePtr(),
                                                                         mFullPrecisionActivations.getDevicePtr(),
                                                                         input.size(),
                                                                         inference);
}


// ----------------------------------------------------------------------------
// --------------------------- Backward functions -----------------------------
// ----------------------------------------------------------------------------

template <>
void N2D2::SATQuantizerActivation_Frame_CUDA<half_float::half>::back_propagate(const BaseTensor& baseInput,
                                                                               const BaseTensor& /*baseOutput*/,
                                                                               const BaseTensor& baseDiffInput,
                                                                               BaseTensor& baseDiffOutput)
{
    const half_float::half beta = (mSolver->isNewIteration()) ? half_float::half(0.0) : half_float::half(1.0);

    const CudaTensor<half_float::half>& diffInput = dynamic_cast<const CudaTensor<half_float::half>&>(baseDiffInput);
    const CudaTensor<half_float::half>& input = dynamic_cast<const CudaTensor<half_float::half>&>(baseInput);
    CudaTensor<half_float::half>& diffOutput = dynamic_cast<CudaTensor<half_float::half>&>(baseDiffOutput);

    // Initialize mDiffAlphas at the first backpropagate
    if(mDiffAlphas.empty()) {
        mDiffAlphas.resize({1, 1, 1, 1});
        mDiffAlphas.fill(half_float::half(0));
        mDiffAlphas.synchronizeHToD();
    }
    // Initialize mDiffAlphasTensor at the first backpropagate
    if(mDiffAlphasTensor.empty()) {
        mDiffAlphasTensor.resize(diffInput.dims());
        mDiffAlphasTensor.fill(half_float::half(0));
        mDiffAlphasTensor.synchronizeHToD();
    }
    // Initialize Alpha tensor (if it hasn't been done before)
    if(mAlphas.empty()) {
        mAlphas.resize({1, 1, 1, 1});
        mAlphas.fill(half_float::half(mAlphaParameter));
        mAlphas.synchronizeHToD();
    }

    SATQuantizer_Frame_CUDA_Kernels::cudaH_quantize_activation_back_propagate(diffInput.getDevicePtr(),
                                                                              diffOutput.getDevicePtr(),
                                                                              mDiffAlphasTensor.getDevicePtr(),
                                                                              input.getDevicePtr(),
                                                                              mRange,
                                                                              mAlphas.getDevicePtr(),
                                                                              diffInput.size());
    if(mDeviceWorkspace.empty()) {
        mDeviceWorkspace.resize({32, 1, 1, 1});
        mDeviceWorkspace.fill(half_float::half(0.0));
        mDeviceWorkspace.synchronizeHToD();
    }

    mDiffAlphas.synchronizeDToH();
    mDiffAlphas(0,0,0,0) 
        = Quantizer_Frame_CUDA_Kernels::cudaH_accumulate(mDiffAlphasTensor.getDevicePtr(), 
                                                         mDeviceWorkspace.getDevicePtr(), 
                                                         mDiffAlphasTensor.size()) + beta * mDiffAlphas(0,0,0,0); 
    mDiffAlphas.synchronizeHToD();
}

template <>
void N2D2::SATQuantizerActivation_Frame_CUDA<float>::back_propagate(const BaseTensor& baseInput,
                                                                    const BaseTensor& /*baseOutput*/,
                                                                    const BaseTensor& baseDiffInput,
                                                                    BaseTensor& baseDiffOutput)
{
    const float beta = (mSolver->isNewIteration()) ? float(0.0) : float(1.0);

    const CudaTensor<float>& diffInput = dynamic_cast<const CudaTensor<float>&>(baseDiffInput);
    const CudaTensor<float>& input = dynamic_cast<const CudaTensor<float>&>(baseInput);
    CudaTensor<float>& diffOutput = dynamic_cast<CudaTensor<float>&>(baseDiffOutput);

    // Initialize mDiffAlphas at the first backpropagate
    if(mDiffAlphas.empty()) {
        mDiffAlphas.resize({1, 1, 1, 1});
        mDiffAlphas.fill(float(0));
        mDiffAlphas.synchronizeHToD();
    }
    // Initialize mDiffAlphasTensor at the first backpropagate
    if(mDiffAlphasTensor.empty()) {
        mDiffAlphasTensor.resize(diffInput.dims());
        mDiffAlphasTensor.fill(float(0));
        mDiffAlphasTensor.synchronizeHToD();
    }
    // Initialize Alpha tensor (if it hasn't been done before)
    if(mAlphas.empty()) {
        mAlphas.resize({1, 1, 1, 1});
        mAlphas.fill(float(mAlphaParameter));
        mAlphas.synchronizeHToD();
    }

    SATQuantizer_Frame_CUDA_Kernels::cudaF_quantize_activation_back_propagate(diffInput.getDevicePtr(),
                                                                              diffOutput.getDevicePtr(),
                                                                              mDiffAlphasTensor.getDevicePtr(),
                                                                              input.getDevicePtr(),
                                                                              mRange,
                                                                              mAlphas.getDevicePtr(),
                                                                              diffInput.size());
    mDiffAlphas.synchronizeDToH();

    mDiffAlphas(0,0,0,0) 
        = Quantizer_Frame_CUDA_Kernels::cudaF_accumulate(mDiffAlphasTensor.getDevicePtr(), mDiffAlphasTensor.size()) 
            + beta * mDiffAlphas(0,0,0,0); 

    mDiffAlphas.synchronizeHToD();
}

template <>
void N2D2::SATQuantizerActivation_Frame_CUDA<double>::back_propagate(const BaseTensor& baseInput,
                                                                     const BaseTensor& /*baseOutput*/,
                                                                     const BaseTensor& baseDiffInput,
                                                                     BaseTensor& baseDiffOutput)
{
    const double beta = (mSolver->isNewIteration()) ? double(0.0) : double(1.0);

    const CudaTensor<double>& diffInput = dynamic_cast<const CudaTensor<double>&>(baseDiffInput);
    const CudaTensor<double>& input = dynamic_cast<const CudaTensor<double>&>(baseInput);
    CudaTensor<double>& diffOutput = dynamic_cast<CudaTensor<double>&>(baseDiffOutput);

    // Initialize mDiffAlphas at the first backpropagate
    if(mDiffAlphas.empty()) {
        mDiffAlphas.resize({1, 1, 1, 1});
        mDiffAlphas.fill(double(0));
        mDiffAlphas.synchronizeHToD();
    }
    // Initialize mDiffAlphasTensor at the first backpropagate
    if(mDiffAlphasTensor.empty()) {
        mDiffAlphasTensor.resize(diffInput.dims());
        mDiffAlphasTensor.fill(double(0));
        mDiffAlphasTensor.synchronizeHToD();
    }
    // Initialize Alpha tensor (if it hasn't been done before)
    if(mAlphas.empty()) {
        mAlphas.resize({1, 1, 1, 1});
        mAlphas.fill(double(mAlphaParameter));
        mAlphas.synchronizeHToD();
    }

    SATQuantizer_Frame_CUDA_Kernels::cudaD_quantize_activation_back_propagate(diffInput.getDevicePtr(),
                                                                              diffOutput.getDevicePtr(),
                                                                              mDiffAlphasTensor.getDevicePtr(),
                                                                              input.getDevicePtr(),
                                                                              mRange,
                                                                              mAlphas.getDevicePtr(),
                                                                              diffInput.size());
    mDiffAlphas.synchronizeDToH();

    mDiffAlphas(0,0,0,0) 
        = Quantizer_Frame_CUDA_Kernels::cudaD_accumulate(mDiffAlphasTensor.getDevicePtr(), mDiffAlphasTensor.size()) 
            + beta * mDiffAlphas(0,0,0,0); 

    mDiffAlphas.synchronizeHToD();
}


template<class T>
void N2D2::SATQuantizerActivation_Frame_CUDA<T>::update(unsigned int batchSize)
{
    mSolver->update(mAlphas, mDiffAlphas, batchSize);  
}


template <class T>
void N2D2::SATQuantizerActivation_Frame_CUDA<T>::exportParameters(const std::string& dirName,
                                                                  const std::string& cellName) const 
{
    const std::string fileName = dirName + "/" 
                                    + cellName + "_QAct_SAT_Alpha.syntxt";

    std::ofstream alphas(fileName.c_str());
    if (!alphas.good())
        throw std::runtime_error("Could not create synaptic file: "
                                 + fileName);
    mAlphas.synchronizeDToH();
    if(!mAlphas.empty()){
        alphas  << std::setprecision(10) << mAlphas(0);
    }
    mAlphas.synchronizeHToD();
}

template <class T>
void N2D2::SATQuantizerActivation_Frame_CUDA<T>::importParameters(const std::string& dirName,
                                                                  const std::string& cellName, 
                                                                  bool ignoreNotExists)
{
    
    const std::string alphasFile = dirName + "/" 
                                    + cellName + "_QAct_SAT_Alpha.syntxt";

    std::ifstream alphas(alphasFile.c_str());

    if (!alphas.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << alphasFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + alphasFile);
    }

    mAlphas.synchronizeDToH();
    if(mAlphas.empty()) {
        mAlphas.resize({1, 1, 1, 1});
    }

    T value;
    if (!(alphas >> value))
        throw std::runtime_error( "Error while reading synaptic file: "
                        + alphasFile);
    mAlphas(0) = T(value);

    // Discard trailing whitespaces
    while (std::isspace(alphas.peek()))
        alphas.ignore();

    if (alphas.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Synaptic file size larger than expected: "
                                 + alphasFile);

    mAlphas.synchronizeHToD();
}


namespace N2D2 {
    template class SATQuantizerActivation_Frame_CUDA<half_float::half>;
    template class SATQuantizerActivation_Frame_CUDA<float>;
    template class SATQuantizerActivation_Frame_CUDA<double>;
}

#endif