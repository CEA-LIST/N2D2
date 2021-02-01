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

#ifndef N2D2_DECONVCELL_H
#define N2D2_DECONVCELL_H

#include <cassert>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {

class Activation;
class BaseInterface;
class DeepNet;
class Filler;
template<typename T>
class Matrix;
class Network;
class Solver;

class DeconvCell : public virtual Cell {
public:
    typedef std::function<std::shared_ptr<DeconvCell>(
        Network&,
        const DeepNet&, 
        const std::string&,
        const std::vector<unsigned int>&,
        unsigned int,
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

    DeconvCell(const DeepNet& deepNet, const std::string& name,
               const std::vector<unsigned int>& kernelDims,
               unsigned int nbOutputs,
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
    unsigned int getStrideX() const
    {
        return mStrideDims[0];
    };
    unsigned int getStrideY() const
    {
        return mStrideDims[1];
    };
    int getPaddingX() const
    {
        return mPaddingDims[0];
    };
    int getPaddingY() const
    {
        return mPaddingDims[1];
    };
    unsigned int getDilationX() const
    {
        return mDilationDims[0];
    };
    unsigned int getDilationY() const
    {
        return mDilationDims[1];
    };
    std::shared_ptr<Solver> getWeightsSolver()
    {
        return mWeightsSolver;
    };
    std::shared_ptr<Solver> getBiasSolver()
    {
        return mBiasSolver;
    };
    virtual void getWeight(unsigned int output,
                           unsigned int channel,
                           BaseTensor& value) const = 0;
    virtual void getBias(unsigned int output, BaseTensor& value) const = 0;
    void setKernel(unsigned int output,
                   unsigned int channel,
                   const Matrix<double>& value,
                   bool normalize);
    virtual BaseInterface* getWeights() { return NULL; };
    virtual void setWeights(unsigned int /*k*/,
                    BaseInterface* /*weights*/,
                    unsigned int /*offset*/) {};
    virtual std::shared_ptr<BaseTensor> getBiases()
    {
        return std::shared_ptr<BaseTensor>();
    };
    virtual void setBiases(const std::shared_ptr<BaseTensor>&
                           /*biases*/) {};
    virtual void exportFreeParameters(const std::string& fileName) const;
    virtual void importFreeParameters(const std::string& fileName,
                                      bool ignoreNotExists = false);
    virtual void logFreeParametersDistrib(const std::string& fileName,
                                          FreeParametersType type = All) const;
    void writeMap(const std::string& fileName) const;
    virtual std::pair<Float_T, Float_T> getFreeParametersRange(FreeParametersType type = All) const;
    void randomizeFreeParameters(double stdDev);
    void getStats(Stats& stats) const;
    std::vector<unsigned int> getReceptiveField(
                                const std::vector<unsigned int>& outputField
                                        = std::vector<unsigned int>()) const;
    virtual ~DeconvCell() {};

protected:
    virtual void setOutputsDims();
    virtual void setWeight(unsigned int output,
                           unsigned int channel,
                           const BaseTensor& value) = 0;
    virtual void setBias(unsigned int output, const BaseTensor& value) = 0;
    std::map<unsigned int, unsigned int> outputsRemap() const;

    /// If true, the output neurons don't have bias
    Parameter<bool> mNoBias;
    /// If true, enable backpropogation
    Parameter<bool> mBackPropagate;
    Parameter<WeightsExportFormat> mWeightsExportFormat;
    Parameter<bool> mWeightsExportFlip;
    Parameter<std::string> mOutputsRemap;

    // Kernel dims
    const std::vector<unsigned int> mKernelDims;
    // Stride for the convolution
    const std::vector<unsigned int> mStrideDims;
    // Padding for the convolution
    const std::vector<int> mPaddingDims;
    // Dilation for the convolution
    const std::vector<unsigned int> mDilationDims;

    std::shared_ptr<Filler> mWeightsFiller;
    std::shared_ptr<Filler> mBiasFiller;
    std::shared_ptr<Solver> mWeightsSolver;
    std::shared_ptr<Solver> mBiasSolver;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::DeconvCell::WeightsExportFormat>::data[]
    = {"OCHW", "HWCO"};
}

#endif // N2D2_DECONVCELL_H
