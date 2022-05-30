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

#include "Quantizer/QAT/Activation/SAT/SATQuantizerActivation_Frame.hpp"
#include "Quantizer/QAT/Kernel/SATQuantizer_Frame_Kernels.hpp"
#include "third_party/half.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"
#include "containers/Tensor.hpp"


template<>
N2D2::Registrar<N2D2::SATQuantizerActivation>
N2D2::SATQuantizerActivation_Frame<half_float::half>::mRegistrar(
    {"Frame"},
    N2D2::SATQuantizerActivation_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::SATQuantizerActivation>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::SATQuantizerActivation>
N2D2::SATQuantizerActivation_Frame<float>::mRegistrar(
    {"Frame"},
    N2D2::SATQuantizerActivation_Frame<float>::create,
    N2D2::Registrar<N2D2::SATQuantizerActivation>::Type<float>());

template<>
N2D2::Registrar<N2D2::SATQuantizerActivation>
N2D2::SATQuantizerActivation_Frame<double>::mRegistrar(
    {"Frame"},
    N2D2::SATQuantizerActivation_Frame<double>::create,
    N2D2::Registrar<N2D2::SATQuantizerActivation>::Type<double>());
    

template<class T>
N2D2::SATQuantizerActivation_Frame<T>::SATQuantizerActivation_Frame()
    : SATQuantizerActivation(),
      QuantizerActivation_Frame<T>()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame<T> >();
}

template <class T>
N2D2::SATQuantizerActivation_Frame<T>::~SATQuantizerActivation_Frame()
{
    // dtor
}

//In-place method for activation quantization
template <class T>
void N2D2::SATQuantizerActivation_Frame<T>::propagate(BaseTensor& baseInOut,
                                                      bool inference)  
{
    // Initialize tensor of alphas at the first propagate
    // For now, only a single alpha is considered
    if(mAlphas.empty()) {
        mAlphas.resize({1, 1, 1, 1});
        mAlphas.fill(T(mAlphaParameter));
    }

    Tensor<T>& inOut = dynamic_cast<Tensor<T>&>(baseInOut);

    if(!inference && mFullPrecisionActivations.empty()) {
        mFullPrecisionActivations.resize(inOut.dims());
    }

    SATQuantizer_Frame_Kernels::quantize_activation_propagate(inOut,
                                                              mRange,
                                                              mAlphas,
                                                              mFullPrecisionActivations,
                                                              inference);
}

template <class T>
void N2D2::SATQuantizerActivation_Frame<T>::back_propagate(const BaseTensor& baseInput,
                                                           const BaseTensor& /*baseOutput*/,
                                                           const BaseTensor& baseDiffInput,
                                                           BaseTensor& baseDiffOutput)
{
    const T beta = (mSolver->isNewIteration()) ? T(0.0) : T(1.0);

    const Tensor<T>& input = dynamic_cast<const Tensor<T>&>(baseInput);
    //const Tensor<T>& output = dynamic_cast<const Tensor<T>&>(baseOutput);
    const Tensor<T>& diffInput = dynamic_cast<const Tensor<T>&>(baseDiffInput);
    Tensor<T>& diffOutput = dynamic_cast<Tensor<T>&>(baseDiffOutput);

    // Initialize mDiffAlphas at the first backpropagate
    if(mDiffAlphas.empty()) {
        mDiffAlphas.resize({1, 1, 1, 1});
        mDiffAlphas.fill(T(0));
    }
    // Initialize mDiffAlphasTensor at the first backpropagate
    if(mDiffAlphasTensor.empty()) {
        mDiffAlphasTensor.resize(diffInput.dims());
        mDiffAlphasTensor.fill(T(0));
    }
    // Initialize Alpha tensor at the first propagate
    if(mAlphas.empty()) {
        mAlphas.resize({1, 1, 1, 1});
        mAlphas.fill(T(mAlphaParameter));
    }

    SATQuantizer_Frame_Kernels::quantize_activation_back_propagate(diffInput,
                                                                   diffOutput,
                                                                   mDiffAlphasTensor,
                                                                   input,
                                                                   mRange,
                                                                   mAlphas);

    mDiffAlphas(0,0,0,0) = std::accumulate(mDiffAlphasTensor.begin(), 
                                           mDiffAlphasTensor.end(), T(0.0))
                            + beta * mDiffAlphas(0,0,0,0); 
}


template<class T>
void N2D2::SATQuantizerActivation_Frame<T>::update(unsigned int batchSize)
{
    mSolver->update(mAlphas, mDiffAlphas, batchSize);
}


template <class T>
void N2D2::SATQuantizerActivation_Frame<T>::exportParameters(const std::string& dirName,
                                                             const std::string& cellName) const 
{
    const std::string fileName = dirName + "/" 
                                    + cellName + "_QAct_SAT_Alpha.syntxt";
    std::ofstream alphas(fileName.c_str());
    if (!alphas.good())
        throw std::runtime_error("Could not create synaptic file: "
                                 + fileName);
    if(!mAlphas.empty())                             
        alphas << mAlphas(0);
}

template <class T>
void N2D2::SATQuantizerActivation_Frame<T>::importParameters(const std::string& dirName,
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

    //Tensor<float> allAlphas(mAlphas.size());
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
}


namespace N2D2 {
    template class SATQuantizerActivation_Frame<half_float::half>;
    template class SATQuantizerActivation_Frame<float>;
    template class SATQuantizerActivation_Frame<double>;
}
