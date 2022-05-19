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

#include "Quantizer/QAT/Cell/LSQ/LSQQuantizerCell_Frame.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"

/* Only float functions for now */

template<>
N2D2::Registrar<N2D2::LSQQuantizerCell>
N2D2::LSQQuantizerCell_Frame<half_float::half>::mRegistrar(
    {"Frame"},
    N2D2::LSQQuantizerCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::LSQQuantizerCell>::Type<half_float::half>());

template<>
N2D2::Registrar<N2D2::LSQQuantizerCell>
N2D2::LSQQuantizerCell_Frame<float>::mRegistrar(
    {"Frame"},
    N2D2::LSQQuantizerCell_Frame<float>::create,
    N2D2::Registrar<N2D2::LSQQuantizerCell>::Type<float>());

template<>
N2D2::Registrar<N2D2::LSQQuantizerCell>
N2D2::LSQQuantizerCell_Frame<double>::mRegistrar(
    {"Frame"},
    N2D2::LSQQuantizerCell_Frame<double>::create,
    N2D2::Registrar<N2D2::LSQQuantizerCell>::Type<double>());

namespace N2D2 {

template<class T>
LSQQuantizerCell_Frame<T>::LSQQuantizerCell_Frame()
    : LSQQuantizerCell(),
      QuantizerCell_Frame<T>()
{
    // ctor
    // Default Solver
    mSolver = std::make_shared<SGDSolver_Frame<T> >();
}


template<class T>
void LSQQuantizerCell_Frame<T>::addWeights(BaseTensor& weights, BaseTensor& diffWeights)
{
    if(mInitialized)
        return;

    mFullPrecisionWeights.push_back(&weights);
    mQuantizedWeights.push_back(new Tensor<T>(weights.dims()));

    mDiffQuantizedWeights.push_back(&diffWeights);
    mDiffFullPrecisionWeights.push_back(new Tensor<T>(diffWeights.dims()));

    mDiffStepSizeInterface.push_back(new Tensor<T>(diffWeights.dims()));
    mDiffStepSizeInterface.back().fill(T(0.0));

    //mDiffStepSizeTensor.resize(diffWeights.dims(), T(0.0));
}

template<class T>
void LSQQuantizerCell_Frame<T>::addBiases(BaseTensor& biases, BaseTensor& diffBiases)
{
    if(mInitialized)
        return;

    mFullPrecisionBiases = &(dynamic_cast<BaseTensor&>(biases));
    mQuantizedBiases.resize(biases.dims());

    mDiffQuantizedBiases = &(dynamic_cast<BaseTensor&>(diffBiases));
    mDiffFullPrecisionBiases.resize(diffBiases.dims());
}

/**
 * @brief Quantizes weights and biaises before the first iteration using a very high step size.
 * 
 * @tparam T weights type
 * Can be : double (64 bits)
 *          float (32 bits)
 *          half_float::half (16 bits)
 */
template<typename T>
void LSQQuantizerCell_Frame<T>::initialize()
{
    //init the quantization range for activation
    // mRange = (nb of bits)^2 - 1
    mBitRanges = std::make_pair((int) -((mRange + 1)/2), (int) (mRange - 1)/2);

    //init step size with fake value (needed for quant weight initialization below)
    if(mStepSize.empty()) {
        if(mSetOptInitStepSize){
            float initialValue = 10000.;
            setStepSizeValue(initialValue);
        }
        mStepSize.resize({1,1,1,1});
        mStepSize.fill(T(mStepSizeVal));

        mDiffStepSize.resize({1,1,1,1});
        mDiffStepSize.fill(T(0.0));
    }
    mGradScaleFactor = T(0.0);

    std::cout << "      " << std::setprecision(8) <<
        "Quantizer::LSQ || " <<  
        " StepSizeVal[" << mStepSizeVal << "] || " <<
        " StepInit[" << mSetOptInitStepSize << "] || " << 
        " Range[" << mBitRanges.first << ", " << mBitRanges.second << "]" << std::endl;

    //Initialize the quantized weights
    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
        for (size_t i = 0; i < mFullPrecisionWeights[k].size(); i++) {
            Tensor<T>& fullPrecWeights = dynamic_cast<Tensor<T>&>(mFullPrecisionWeights[k]);
            T qData = fullPrecWeights(i) / mStepSize(0);
            qData = (qData <= (T) mBitRanges.first ) ? (T) mBitRanges.first :
                    (qData >= (T) mBitRanges.second ) ? (T) mBitRanges.second :
                    qData;
            qData = rintf(qData);
            // there is no need to save the full precision weights
            mQuantizedWeights[k](i) = qData*mStepSize(0);
        }
    }
    mInitialized = true;
}


// ----------------------------------------------------------------------------
// ---------------------------- Forward functions -----------------------------
// ----------------------------------------------------------------------------

/**
 * @brief Implement the forward pass of the quantization using LSQ method for layer weights.
 * 
 * @fn void LSQQuantizerCell_Frame<T>::propagate()
 * 
 * @tparam T weights type
 * Can be : double (64 bits)
 *          float (32 bits)
 *          half_float::half (16 bits)
 */
template<typename T>
void LSQQuantizerCell_Frame<T>::propagate()
{
    //init step size using correct imported weights this time
    if(mStepSizeVal == 10000.) {
        std::cout << "Initialize the correct LSQ step size ... " << std::endl;

        Tensor<T>& fullPrecWeights = dynamic_cast<Tensor<T>&>(mFullPrecisionWeights[0]);

        if(mSetOptInitStepSize){
            // Initialisation of the weight step size according to the LSQ paper
            // (https://arxiv.org/pdf/1902.08153.pdf)
            float initialValue = 2*fullPrecWeights.mean(true) / sqrt((mRange-1)/2);
            setStepSizeValue(initialValue);

            // Initialisation of the weight step size according to the LSQ+ paper
            // (https://arxiv.org/pdf/2004.09576.pdf)
            //float a = (float)tens.mean(true)-3*(float)tens.std();
            //float b = (float)tens.mean(true)+3*(float)tens.std();
            //float initialValue = std::max(abs(a), abs(b))/((mRange-1)/2);
            //setStepSizeValue(initialValue);
        }
        mStepSize.resize({1,1,1,1});
        mStepSize.fill(T(mStepSizeVal));

        std::cout << "      " << std::setprecision(8) <<
            "Quantizer::LSQ || " <<
            " StepSizeVal[" << mStepSizeVal << "] || " <<
            " StepInit[" << mSetOptInitStepSize << "] || " <<
            " Range[" << mBitRanges.first << ", " << mBitRanges.second << "]" << std::endl;
    }

    unsigned int totElementW = 0;

    for (unsigned int k = 0, size = mFullPrecisionWeights.size(); k < size; ++k) {
        for (size_t i = 0; i < mFullPrecisionWeights[k].size(); i++) {
            Tensor<T>& fullPrecWeights = dynamic_cast<Tensor<T>&>(mFullPrecisionWeights[k]);
            T qData = fullPrecWeights(i) / mStepSize(0);
            qData = (qData <= (T) mBitRanges.first ) ? (T) mBitRanges.first :
                    (qData >= (T) mBitRanges.second ) ? (T) mBitRanges.second :
                    qData;
            qData = rint(qData);
            // there is no need to save the full precision weights
            mQuantizedWeights[k](i) = qData*mStepSize(0);
        }
        totElementW += mFullPrecisionWeights[k].size();
    }

    // compute for backward step
    mGradScaleFactor = T(1.0) / sqrt(T(totElementW * mBitRanges.second));


    if (mFullPrecisionBiases) {
        Tensor<T>& fullPrecBiases = dynamic_cast<Tensor<T>&>(*mFullPrecisionBiases);
        for (unsigned int i=0; i<mFullPrecisionBiases->size(); i++) {
            T value = fullPrecBiases(i);
            mQuantizedBiases(i) = value;
        }
    }
}


// ----------------------------------------------------------------------------
// --------------------------- Backward functions -----------------------------
// ----------------------------------------------------------------------------

/**
 * @brief Implement the backward pass of the quantization step using LSQ method for layer weights.
 * 
 * @fn void LSQQuantizerCell_Frame<T>::back_propagate()
 * 
 * @tparam T weights type
 * Can be : double (64 bits)
 *          float (32 bits)
 *          half_float::half (16 bits)
 */
template<typename T>
void LSQQuantizerCell_Frame<T>::back_propagate()
{
    // Step size gradient
    T diffStepSize = T(0.0);

    // reset the step size gradient if starting a new iteration
    if(!mSolver->isNewIteration()) {
        diffStepSize = mDiffStepSize(0,0,0,0);
    }

    for (unsigned int k = 0, size = mDiffQuantizedWeights.size(); k < size; ++k) {
        // Converts BaseTensor to Tensor as it is not possible to access BaseTensor values
        Tensor<T>& diffQuantizedWeights = dynamic_cast<Tensor<T>&>(mDiffQuantizedWeights[k]);
        Tensor<T>& fullPrecWeights = dynamic_cast<Tensor<T>&>(mFullPrecisionWeights[k]);

        // Updating gradient for each weight of the layer
        for (unsigned int i = 0; i < mFullPrecisionWeights[k].size(); i++) {
            const T fullPrecScale = fullPrecWeights(i) / mStepSize(0);
            /*****************Step Size Gradient Computation******************/
            //1st: clip the gradient in interval [rangeMin, rangeMax] and take account of qError
            T qData = fullPrecScale;
            qData = ((qData <= (T)mBitRanges.first) ? (T)mBitRanges.first :
                    (qData >= (T)mBitRanges.second) ? (T)mBitRanges.second :
                    rint(qData) - qData);
            //2nd: Multiplie backward data with clipped grad
            diffStepSize += qData*diffQuantizedWeights(i);

            /******************Weight Gradient Computation*********************/
            mDiffFullPrecisionWeights[k](i) = diffQuantizedWeights(i)* ((qData <= (T) mBitRanges.first) ? T(0.0) :
                                                                        (qData >= (T) mBitRanges.second) ? T(0.0) :
                                                                        T(1.0));
        }
    }
    mDiffStepSize.fill(diffStepSize*mGradScaleFactor);
}


template<typename T>
void LSQQuantizerCell_Frame<T>::update(unsigned int batchSize)
{
    mSolver->update(mStepSize, mDiffStepSize, batchSize);
}


template <class T>
LSQQuantizerCell_Frame<T>::~LSQQuantizerCell_Frame()
{
    // dtor
}


template <class T>
void LSQQuantizerCell_Frame<T>::exportFreeParameters(const std::string& fileName) const 
{

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
}

template <class T>
void LSQQuantizerCell_Frame<T>::importFreeParameters(const std::string& fileName, 
                                                     bool ignoreNotExists)
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


    T valueW;
    if (!(alphasW >> valueW))
        throw std::runtime_error( "Error while reading synaptic file: "
                        + alphasWFile);

    mStepSize.resize(mStepSize.dims(), T(valueW));
    
    // Discard trailing whitespaces
    while (std::isspace(alphasW.peek()))
        alphasW.ignore();

    if (alphasW.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Synaptic file size larger than expected: "
                                 + alphasWFile);

    mStepSize.synchronizeHToD();

}

}
