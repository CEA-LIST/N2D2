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

/**
 * @brief Implement the forward pass of the quantization using LSQ method for layer weights.
 * 
 * @fn void LSQQuantizerActivation_Frame<T>::propagate(BaseTensor& baseInOut,
                                                    bool inference)
 * 
 * @tparam T weights type.
 * Can be: double (64 bits)
 *         float (32 bits)
 *         half_float::half (16 bits)
 * 
 * @param baseInOut Input features tensor.
 */
template <typename T>
void LSQQuantizerActivation_Frame<T>::propagate(BaseTensor& baseInOut,
                                                    bool /*inference*/)  
{
    // Conversion to Tensor to access its values
    Tensor<T>& input = dynamic_cast<Tensor<T>&>(baseInOut);
    //init the quantization range for activation
    if(mBitRanges.first == 0 && mBitRanges.second == 0) {
        mBitRanges = std::make_pair(0, (int)mRange);
        /*std::cout << " LSQ activation range: [" << mBitRanges.first << ";" << mBitRanges.second << "]" << std::endl;*/
    }

    unsigned int totElement = baseInOut.size();
    mGradScaleFactor = T(1.0) / sqrt(T(totElement * mBitRanges.second));


    if(mStepSize.empty()) {
        // Initialisation of the activation step size according to the LSQ paper
        // (https://arxiv.org/pdf/1902.08153.pdf)    
        if(mSetOptInitStepSize){
            float initialValue = 2.0f * float(input.mean(true) / sqrt(mBitRanges.second));
            setStepSizeValue(initialValue);
        }
        mStepSize.resize({1, 1, 1, 1});
        mStepSize.fill((T)(mStepSizeParameter));
    }

    if(mFullPrecisionActivations.empty()) {
        mFullPrecisionActivations.resize(baseInOut.dims());
    }

    // store activation full value
    bool saveFpData_ = true;
    if (saveFpData_) {
        mFullPrecisionActivations = input;
    }

    for( T &qData: input) {
        qData = qData / mStepSize(0);
        qData = (qData <= (T) mBitRanges.first) ? (T) mBitRanges.first :
                (qData >= (T) mBitRanges.second) ? (T) mBitRanges.second :
                qData;
        qData = rint(qData);
        qData = qData * mStepSize(0);
    }
}


// ----------------------------------------------------------------------------
// --------------------------- Backward functions -----------------------------
// ----------------------------------------------------------------------------

/**
 * @brief Implement the backward pass of the quantization step using LSQ method for layer features and step size.
 * @fn void N2D2::LSQQuantizerActivation_Frame<T>::back_propagate(const BaseTensor& baseInput,const BaseTensor& baseOutput, const BaseTensor& baseDiffInput, BaseTensor& baseDiffOutput)
 * @tparam T features and step size type.
 * Can be: double (64 bits)
 *         float (32 bits)
 *         half_float::half (16 bits)
 * @param baseInput Input features tensor from forward step.
 * @param baseOutput Output features tensor from forward step.
 * @param baseDiffInput Gradient of deeper layer to be propagated.
 * @param baseDiffOutput Gradient propagated.
 */
template <typename T>
void N2D2::LSQQuantizerActivation_Frame<T>::back_propagate(const BaseTensor& baseInput,
                                                               const BaseTensor& baseOutput,
                                                               const BaseTensor& baseDiffInput,
                                                               BaseTensor& baseDiffOutput)
{
    //init the quantization range, if not done before (e.g. unit test)
    if(mBitRanges.first == 0 && mBitRanges.second == 0){
        mBitRanges = std::make_pair( 0, (int) mRange );
        //std::cout << " LSQ activation range :: [" << mBitRanges.first << ";" << mBitRanges.second << "]"<< std::endl;
    }

    // Initialize mDiffStepSize at the first backpropagate
    if(mDiffStepSize.empty()) {
        mDiffStepSize.resize({1, 1, 1, 1});
        mDiffStepSize.fill(T(0));
    }

    // Initialize StepSize tensor at the first propagate
    if(mStepSize.empty()) {
        mStepSize.resize({1, 1, 1, 1});
        mStepSize.fill((T)mStepSizeParameter);
        //this is done for unit test purpose
        unsigned int totElement = baseInput.size();
        mGradScaleFactor = T(1.0) / sqrt((T)(totElement * mBitRanges.second));
    }

    // Initialize StepSize gradient if it is part of a running iteration
    T diffStepSize = T(0.0);
    if(!mSolver->isNewIteration()) {
        diffStepSize = mDiffStepSize(0,0,0,0);
    }

    // Converts BaseTensor to Tensor as it is not possible to access BaseTensor values
    const Tensor<T>& fullPrecInput = tensor_cast<T>(baseInput);
    const Tensor<T>& fullPrecDiffInput = tensor_cast<T>(baseDiffInput);
    Tensor<T>& fullPrecDiffOutput = dynamic_cast<Tensor<T>&>(baseDiffOutput);

    for(unsigned int i=0; i<baseInput.size(); i++) {
        const T fullPrecScale = fullPrecInput(i) / mStepSize(0);

        /*****************Step Size Gradient Computation******************/
        //1st: clip the gradient in interval [rangeMin, rangeMax] and take account of qError
        T qData = fullPrecScale;
        qData = ((qData <= (T) mBitRanges.first) ? (T) mBitRanges.first :
                (qData >= (T) mBitRanges.second) ? (T) mBitRanges.second :
                rint(qData) - qData);
        //2nd: Multiplie backward data with clipped grad
        qData = qData*fullPrecDiffInput(i);
        diffStepSize += qData;


        /*****************Features Gradient Computation********************/
        // STE method is simply applied
        fullPrecDiffOutput(i) = fullPrecDiffInput(i)*( (fullPrecScale <= (T)mBitRanges.first) ? T(0.0) :
                                                        (fullPrecScale >= (T)mBitRanges.second) ? T(0.0) :
                                                        T(1.0));
    }
    // 3rd: Multiply Step Size gradient with scale factor
    mDiffStepSize.fill(diffStepSize*mGradScaleFactor);
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

