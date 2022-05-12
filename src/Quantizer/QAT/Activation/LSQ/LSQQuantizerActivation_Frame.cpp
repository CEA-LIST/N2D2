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

#include "Quantizer/QAT/Activation/LSQ/LSQQuantizerActivation_Frame.hpp"
#include "third_party/half.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"


template<>
N2D2::Registrar<N2D2::LSQQuantizerActivation>
N2D2::LSQQuantizerActivation_Frame<half_float::half>::mRegistrar(
    {"Frame"},
    N2D2::LSQQuantizerActivation_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::LSQQuantizerActivation>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::LSQQuantizerActivation>
N2D2::LSQQuantizerActivation_Frame<float>::mRegistrar(
    {"Frame"},
    N2D2::LSQQuantizerActivation_Frame<float>::create,
    N2D2::Registrar<N2D2::LSQQuantizerActivation>::Type<float>());

template<>
N2D2::Registrar<N2D2::LSQQuantizerActivation>
N2D2::LSQQuantizerActivation_Frame<double>::mRegistrar(
    {"Frame"},
    N2D2::LSQQuantizerActivation_Frame<double>::create,
    N2D2::Registrar<N2D2::LSQQuantizerActivation>::Type<double>());
    

namespace N2D2 {

template<class T>
LSQQuantizerActivation_Frame<T>::LSQQuantizerActivation_Frame()
    : LSQQuantizerActivation(),
      QuantizerActivation_Frame<T>()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame<T> >();
}

template <class T>
LSQQuantizerActivation_Frame<T>::~LSQQuantizerActivation_Frame()
{
    // dtor
}


// ----------------------------------------------------------------------------
// ---------------------------- Forward functions -----------------------------
// ----------------------------------------------------------------------------

template <>
void LSQQuantizerActivation_Frame<half_float::half>::propagate(BaseTensor& /*baseInOut*/,
                                                               bool /*inference*/)  
{
    // nothing here for now
}

template <>
void LSQQuantizerActivation_Frame<float>::propagate(BaseTensor& /*baseInOut*/,
                                                    bool /*inference*/)  
{
    // nothing here for now
}

template <>
void LSQQuantizerActivation_Frame<double>::propagate(BaseTensor& /*baseInOut*/,
                                                     bool /*inference*/)  
{
    // nothing here for now
}


// ----------------------------------------------------------------------------
// --------------------------- Backward functions -----------------------------
// ----------------------------------------------------------------------------

template <>
void N2D2::LSQQuantizerActivation_Frame<half_float::half>::back_propagate(const BaseTensor& /*baseInput*/,
                                                                          const BaseTensor& /*baseOutput*/,
                                                                          const BaseTensor& /*baseDiffInput*/,
                                                                          BaseTensor& /*baseDiffOutput*/)
{
    // nothing here for now
}

template <>
void N2D2::LSQQuantizerActivation_Frame<float>::back_propagate(const BaseTensor& /*baseInput*/,
                                                               const BaseTensor& /*baseOutput*/,
                                                               const BaseTensor& /*baseDiffInput*/,
                                                               BaseTensor& /*baseDiffOutput*/)
{
    // nothing here for now
}

template <>
void N2D2::LSQQuantizerActivation_Frame<double>::back_propagate(const BaseTensor& /*baseInput*/,
                                                                const BaseTensor& /*baseOutput*/,
                                                                const BaseTensor& /*baseDiffInput*/,
                                                                BaseTensor& /*baseDiffOutput*/)
{
    // nothing here for now
}


template<class T>
void N2D2::LSQQuantizerActivation_Frame<T>::update(unsigned int batchSize)
{
    mSolver->update(mStepSize, mDiffStepSize, batchSize);
}


template <class T>
void LSQQuantizerActivation_Frame<T>::exportParameters(const std::string& dirName,
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
void LSQQuantizerActivation_Frame<T>::importParameters(const std::string& dirName,
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

    //Tensor<float> allAlphas(mAlphas.size());
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
    template class LSQQuantizerActivation_Frame<half_float::half>;
    template class LSQQuantizerActivation_Frame<float>;
    template class LSQQuantizerActivation_Frame<double>;
}

