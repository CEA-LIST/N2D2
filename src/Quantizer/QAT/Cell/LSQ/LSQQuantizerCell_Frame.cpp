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
#pragma omp declare reduction(+ : half_float::half : omp_out = omp_out + omp_in) initializer(omp_priv=half_float::half(0.0))

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
    mBitRanges = std::make_pair((int) -((mRange + 1) >> 1), (int) ((mRange - 1) >> 1));

    //init step size with fake value (needed for quant weight initialization below)
    if(mStepSize.empty()) {
        if(mSetOptInitStepSize){
            const float initialValue = 10000.;
            setStepSizeValue(initialValue);
        }
        mStepSize.resize({1,1,1,1});
        mStepSize.fill(T(mStepSizeVal));

        mDiffStepSize.resize({1,1,1,1});
        mDiffStepSize.fill(T(0.0));
    }

    //Initialize the quantized weights
    unsigned int totElementW = 0;
    const T bitRangesLowerBound = ((T)mBitRanges.first) * mStepSize(0);
    const T bitRangesUpperBound = ((T)mBitRanges.second) * mStepSize(0);
    for (unsigned int k = 0; k < mFullPrecisionWeights.size(); ++k) {
        totElementW += mFullPrecisionWeights[k].size();
        Tensor<T>& fullPrecWeights = dynamic_cast<Tensor<T>&>(mFullPrecisionWeights[k]);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < mFullPrecisionWeights[k].size(); i++) {
            T qData = fullPrecWeights(i) / mStepSize(0);
            qData = (qData <= (T) mBitRanges.first ) ? (T) bitRangesLowerBound :
                    (qData >= (T) mBitRanges.second ) ? (T) bitRangesUpperBound :
                    round(qData)*mStepSize(0);
            // there is no need to save the full precision weights
            mQuantizedWeights[k](i) = qData;
        }
    }
    mGradScaleFactor = (T)(1.0/sqrt(totElementW * mBitRanges.second));
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

        Tensor<T> fullPrecWeights = tensor_cast<T>(mFullPrecisionWeights[0]);
        if(mSetOptInitStepSize){
            // Initialisation of the weight step size according to the LSQ paper
            // (https://arxiv.org/pdf/1902.08153.pdf)
            const float initialValue = 2.0f * float(fullPrecWeights.mean(true) / sqrt((mRange-1) >> 1));
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

    }
    const T bitRangesLowerBound = ((T)mBitRanges.first) * mStepSize(0);
    const T bitRangesUpperBound = ((T)mBitRanges.second) * mStepSize(0);
    for (unsigned int k = 0; k < mFullPrecisionWeights.size(); ++k) {
        Tensor<T>& fullPrecWeights = dynamic_cast<Tensor<T>&>(mFullPrecisionWeights[k]);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < mFullPrecisionWeights[k].size(); ++i) {
            T qData = fullPrecWeights(i) / mStepSize(0);
            qData = (qData <= (T) mBitRanges.first ) ? (T) bitRangesLowerBound :
                    (qData >= (T) mBitRanges.second ) ? (T) bitRangesUpperBound :
                    round(qData)*mStepSize(0);
            // there is no need to save the full precision weights
            mQuantizedWeights[k](i) = qData;
        }
    }

    // compute for backward step


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
//#pragma omp parallel for schedule(dynamic) reduction(+:diffStepSize)
    for (unsigned int k = 0; k < mDiffQuantizedWeights.size(); ++k) {
        // Converts BaseTensor to Tensor as it is not possible to access BaseTensor values
        Tensor<T>& diffQuantizedWeights = dynamic_cast<Tensor<T>&>(mDiffQuantizedWeights[k]);
        Tensor<T>& fullPrecWeights = dynamic_cast<Tensor<T>&>(mFullPrecisionWeights[k]);
        T diffStepSize_loc = T(0.0);
        // Updating gradient for each weight of the layer
        unsigned int weightsTensorSize = mFullPrecisionWeights[k].size();
        
        #pragma omp parallel for schedule(dynamic,256) reduction(+:diffStepSize_loc)
        for (unsigned int i = 0; i < weightsTensorSize/4; ++i) {
            const T fullPrecScale_1 = fullPrecWeights(4*i) / mStepSize(0);
            const T fullPrecScale_2 = fullPrecWeights(4*i+1) / mStepSize(0);
            const T fullPrecScale_3 = fullPrecWeights(4*i+2) / mStepSize(0);
            const T fullPrecScale_4 = fullPrecWeights(4*i+3) / mStepSize(0);

            /******************Weight Gradient Computation*********************/
            mDiffFullPrecisionWeights[k](4*i) = diffQuantizedWeights(4*i)* ((fullPrecScale_1 <= (T) mBitRanges.first) ? T(0.0) :
                                                                        (fullPrecScale_1 >= (T) mBitRanges.second) ? T(0.0) :
                                                                        T(1.0));
            mDiffFullPrecisionWeights[k](4*i+1) = diffQuantizedWeights(4*i+1)* ((fullPrecScale_2 <= (T) mBitRanges.first) ? T(0.0) :
                                                                        (fullPrecScale_2 >= (T) mBitRanges.second) ? T(0.0) :
                                                                        T(1.0));
            mDiffFullPrecisionWeights[k](4*i+2) = diffQuantizedWeights(4*i+2)* ((fullPrecScale_3 <= (T) mBitRanges.first) ? T(0.0) :
                                                                        (fullPrecScale_3 >= (T) mBitRanges.second) ? T(0.0) :
                                                                        T(1.0));
            mDiffFullPrecisionWeights[k](4*i+3) = diffQuantizedWeights(4*i+3)* ((fullPrecScale_4 <= (T) mBitRanges.first) ? T(0.0) :
                                                                        (fullPrecScale_4 >= (T) mBitRanges.second) ? T(0.0) :
                                                                        T(1.0));

            /*****************Step Size Gradient Computation******************/
            //1st: clip the gradient in interval [rangeMin, rangeMax] and take account of qError
            T qData_1 = fullPrecScale_1;
            qData_1 = ((qData_1 <= (T)mBitRanges.first) ? (T)mBitRanges.first :
                    (qData_1 >= (T)mBitRanges.second) ? (T)mBitRanges.second :
                    (round(qData_1) - qData_1));
            T qData_2 = fullPrecScale_2;
            qData_2 = ((qData_2 <= (T)mBitRanges.first) ? (T)mBitRanges.first :
                    (qData_2 >= (T)mBitRanges.second) ? (T)mBitRanges.second :
                    (round(qData_2) - qData_2));
            T qData_3 = fullPrecScale_3;
            qData_3 = ((qData_3 <= (T)mBitRanges.first) ? (T)mBitRanges.first :
                    (qData_3 >= (T)mBitRanges.second) ? (T)mBitRanges.second :
                    (round(qData_3) - qData_3));
            T qData_4 = fullPrecScale_4;
            qData_4 = ((qData_4 <= (T)mBitRanges.first) ? (T)mBitRanges.first :
                    (qData_4 >= (T)mBitRanges.second) ? (T)mBitRanges.second :
                    (round(qData_4) - qData_4));
            //2nd: Multiplie backward data with clipped grad
            diffStepSize_loc += (qData_1*diffQuantizedWeights(4*i) + qData_2*diffQuantizedWeights(4*i+1)) + (qData_3*diffQuantizedWeights(4*i+2) + qData_4*diffQuantizedWeights(4*i+3));
        }
        for (unsigned int i= weightsTensorSize-weightsTensorSize%4; i< weightsTensorSize; ++i) {
            const T fullPrecScale = fullPrecWeights(i) / mStepSize(0);
            mDiffFullPrecisionWeights[k](i) = diffQuantizedWeights(i)* ((fullPrecScale <= (T) mBitRanges.first) ? T(0.0) :
                                                                        (fullPrecScale >= (T) mBitRanges.second) ? T(0.0) :
                                                                        T(1.0));
            T qData = fullPrecScale;
            qData = ((qData <= (T)mBitRanges.first) ? (T)mBitRanges.first :
                    (qData >= (T)mBitRanges.second) ? (T)mBitRanges.second :
                    (round(qData) - qData));
            diffStepSize_loc += qData*diffQuantizedWeights(i);
        }
        diffStepSize+=diffStepSize_loc;
    }
    // reset the step size gradient if starting a new iteration
    if(!mSolver->isNewIteration()) {
        diffStepSize += mDiffStepSize(0,0,0,0);
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
}

}
