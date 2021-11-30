/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#ifndef N2D2_CONVCELL_H
#define N2D2_CONVCELL_H

#include <memory>
#include <vector>

#include "Cell.hpp"
#include "utils/Registrar.hpp"
#include "Quantizer/Cell/QuantizerCell.hpp"


namespace N2D2 {

class Activation;
class BaseInterface;
class DeepNet;
class Network;
class Filler;
class Solver;
template<typename T>
class Matrix;

class ConvCell : public virtual Cell {
public:
    typedef std::function<std::shared_ptr<ConvCell>(
        Network&,
        const DeepNet&,
        const std::string&,
        const std::vector<unsigned int>&,
        unsigned int,
        const std::vector<unsigned int>&,
        const std::vector<unsigned int>&,
        const std::vector<int>&,
        const std::vector<unsigned int>&,
        const std::shared_ptr<Activation>&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    // O = output feature map
    // C = input feature map (channel)
    // H = kernel row
    // W = kernel col
    enum WeightsExportFormat {
        // N2D2 default format
        OCHW,
        // TensorFlow format:
        // "filter: A Tensor. Must have the same type as input. A 4-D tensor of
        // shape [filter_height, filter_width, in_channels, out_channels]"
        // https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        HWCO
    };

    ConvCell(const DeepNet& deepNet, const std::string& name,
             const std::vector<unsigned int>& kernelDims,
             unsigned int nbOutputs,
             const std::vector<unsigned int>& subSampleDims
                = std::vector<unsigned int>(2, 1U),
             const std::vector<unsigned int>& strideDims
                = std::vector<unsigned int>(2, 1U),
             const std::vector<int>& paddingDims = std::vector<int>(2, 0),
             const std::vector<unsigned int>& dilationDims
                = std::vector<unsigned int>(2, 1U));
    const char* getType() const
    {
        return Type;
    };
    void setWeightsFiller(const std::shared_ptr<Filler>& filler)
    {
        mWeightsFiller = filler;
    };
    void setBiasFiller(const std::shared_ptr<Filler>& filler)
    {
        mBiasFiller = filler;
    };
    void setWeightsSolver(const std::shared_ptr<Solver>& solver)
    {
        mWeightsSolver = solver;
    };
    void setBiasSolver(const std::shared_ptr<Solver>& solver)
    {
        mBiasSolver = solver;
    };
    virtual void setQuantizer(std::shared_ptr<QuantizerCell> quant)
    {
        mQuantizer = quant;
    }
    virtual void logFreeParameters(const std::string& fileName,
                                   unsigned int output,
                                   unsigned int channel) const;
    virtual void logFreeParameters(const std::string& fileName,
                                   unsigned int output) const;
    virtual void logFreeParameters(const std::string& dirName) const;
    unsigned long long int getNbSharedSynapses(bool includeBias = true) const;
    unsigned long long int getNbVirtualSynapses(bool includeBias = true) const;
    unsigned int getKernelWidth() const
    {
        return mKernelDims[0];
    };
    unsigned int getKernelHeight() const
    {
        return mKernelDims[1];
    };
    const std::vector<unsigned int>& getKernelDims() const
    {
        return mKernelDims;
    };
    unsigned int getStrideX() const
    {
        return mStrideDims[0];
    };
    unsigned int getStrideY() const
    {
        return mStrideDims[1];
    };
    const std::vector<unsigned int>& getStrideDims() const
    {
        return mStrideDims;
    };
    unsigned int getSubSampleX() const
    {
        return mSubSampleDims[0];
    };
    unsigned int getSubSampleY() const
    {
        return mSubSampleDims[1];
    };
    const std::vector<unsigned int>& getSubSampleDims() const
    {
        return mSubSampleDims;
    };
    int getPaddingX() const
    {
        return mPaddingDims[0];
    };
    int getPaddingY() const
    {
        return mPaddingDims[1];
    };
    const std::vector<int>& getPaddingDims() const
    {
        return mPaddingDims;
    };
    const std::vector<int>& getExtendedPadding() const
    {
        return mExtPaddingDims;
    };
    virtual void setExtendedPadding(const std::vector<int>& paddingDims);
    unsigned int getDilationX() const
    {
        return mDilationDims[0];
    };
    unsigned int getDilationY() const
    {
        return mDilationDims[1];
    };
    std::shared_ptr<Filler> getWeightsFiller()
    {
        return mWeightsFiller;
    };
    std::shared_ptr<Filler> getBiasFiller()
    {
        return mBiasFiller;
    };
    const std::vector<unsigned int>& getDilationDims() const
    {
        return mDilationDims;
    };
    std::shared_ptr<Solver> getWeightsSolver()
    {
        return mWeightsSolver;
    };
    std::shared_ptr<Solver> getBiasSolver()
    {
        return mBiasSolver;
    };
    std::shared_ptr<QuantizerCell> getQuantizer()
    {
        return mQuantizer;
    };
    virtual void getWeight(unsigned int output,
                           unsigned int channel,
                           BaseTensor& value) const = 0;
    virtual void getQuantWeight(unsigned int output,
                           unsigned int channel,
                           BaseTensor& value) const = 0;
    virtual void getBias(unsigned int output, BaseTensor& value) const = 0;
    virtual void setWeight(unsigned int output,
                           unsigned int channel,
                           const BaseTensor& value) = 0;
    virtual void setBias(unsigned int output, const BaseTensor& value) = 0;
    void setKernel(unsigned int output,
                   unsigned int channel,
                   const Matrix<double>& value,
                   bool normalize);
    virtual BaseInterface* getWeights() { return NULL; };
    virtual const BaseInterface* getWeights() const { return NULL; };
    virtual void setWeights(unsigned int /*k*/,
                    BaseInterface* /*weights*/,
                    unsigned int /*offset*/) {};
    virtual const std::shared_ptr<BaseTensor> getBiases() const
    {
        return std::shared_ptr<BaseTensor>();
    };
    virtual void setBiases(const std::shared_ptr<BaseTensor>&
                           /*biases*/) {};
    virtual void exportFreeParameters(const std::string& fileName) const;
    virtual void exportQuantFreeParameters(const std::string& fileName) const;
    virtual void importFreeParameters(const std::string& fileName,
                                      bool ignoreNotExists = false);
    virtual void logFreeParametersDistrib(const std::string& fileName,
                                          FreeParametersType type = All) const;
    virtual void logQuantFreeParametersDistrib(const std::string& fileName,
                                          FreeParametersType type = All) const;
    void writeMap(const std::string& fileName) const;
    void randomizeFreeParameters(double stdDev);
    virtual std::pair<Float_T, Float_T> getFreeParametersRange(FreeParametersType type = All) const;
    virtual std::pair<Float_T, Float_T> getFreeParametersRangePerOutput(std::size_t output, 
                                                                   FreeParametersType type = All) const;
    virtual std::pair<Float_T, Float_T> getFreeParametersRangePerChannel(std::size_t channel) const;
    
    virtual void processFreeParameters(std::function<Float_T(Float_T)> func,
                                       FreeParametersType type = All);
    virtual void processFreeParametersPerOutput(std::function<Float_T(Float_T)> /*func*/,
                                                std::size_t /*output*/,
                                                FreeParametersType /*type*/ = All);
    virtual void processFreeParametersPerChannel(std::function<Float_T(Float_T)> /*func*/,
                                                std::size_t /*channel*/);
    
    void getStats(Stats& stats) const;
    std::vector<unsigned int> getReceptiveField(
                                const std::vector<unsigned int>& outputField
                                        = std::vector<unsigned int>()) const;
    std::map<unsigned int, unsigned int> outputsRemap() const;
    virtual ~ConvCell() {};

protected:
    virtual void setOutputsDims();

    /// If true, the output neurons don't have bias
    Parameter<bool> mNoBias;
    /// If true, enable backpropogation
    Parameter<bool> mBackPropagate;
    Parameter<WeightsExportFormat> mWeightsExportFormat;
    Parameter<bool> mWeightsExportFlip;
    Parameter<std::string> mOutputsRemap;

    // Kernel dims
    const std::vector<unsigned int> mKernelDims;
    // Subsampling at the output
    const std::vector<unsigned int> mSubSampleDims;
    // Stride for the convolution
    const std::vector<unsigned int> mStrideDims;
    // Padding for the convolution
    const std::vector<int> mPaddingDims;
    // Dilation for the convolution
    const std::vector<unsigned int> mDilationDims;

    std::vector<int> mExtPaddingDims;
    std::shared_ptr<Filler> mWeightsFiller;
    std::shared_ptr<Filler> mBiasFiller;
    std::shared_ptr<Solver> mWeightsSolver;
    std::shared_ptr<Solver> mBiasSolver;
    std::shared_ptr<QuantizerCell> mQuantizer;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::ConvCell::WeightsExportFormat>::data[]
    = {"OCHW", "HWCO"};
}

#endif // N2D2_CONVCELL_H
