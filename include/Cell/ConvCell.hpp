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

#include <cassert>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Activation/Activation.hpp"
#include "Cell.hpp"
#include "Filler/NormalFiller.hpp"
#include "Solver/Solver.hpp"
#include "utils/Registrar.hpp"
#include "controler/Interface.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ConvCell_Frame@N2D2@@0U?$Registrar@VConvCell@N2D2@@@2@A")
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ConvCell_Frame_CUDA@N2D2@@0U?$Registrar@VConvCell@N2D2@@@2@A")
#endif
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ConvCell_Spike@N2D2@@0U?$Registrar@VConvCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ConvCell_Spike_Analog@N2D2@@0U?$Registrar@VConvCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ConvCell_Spike_PCM@N2D2@@0U?$Registrar@VConvCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ConvCell_Spike_RRAM@N2D2@@0U?$Registrar@VConvCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@?$ConvCell_Transcode@VConvCell_Frame@N2D2@@VConvCell_Spike@2@@N2D2@@0U?$Registrar@VConvCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class ConvCell : public virtual Cell {
public:
    typedef std::function<std::shared_ptr<ConvCell>(
        Network&,
        const std::string&,
        const std::vector<unsigned int>&,
        unsigned int,
        const std::vector<unsigned int>&,
        const std::vector<unsigned int>&,
        const std::vector<int>&,
        const std::shared_ptr<Activation<Float_T> >&)> RegistryCreate_T;

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

    ConvCell(const std::string& name,
             const std::vector<unsigned int>& kernelDims,
             unsigned int nbOutputs,
             const std::vector<unsigned int>& subSampleDims
                = std::vector<unsigned int>(2, 1U),
             const std::vector<unsigned int>& strideDims
                = std::vector<unsigned int>(2, 1U),
             const std::vector<int>& paddingDims = std::vector<int>(2, 0));
    const char* getType() const
    {
        return Type;
    };
    void setWeightsFiller(const std::shared_ptr<Filler<Float_T> >& filler)
    {
        mWeightsFiller = filler;
    };
    void setBiasFiller(const std::shared_ptr<Filler<Float_T> >& filler)
    {
        mBiasFiller = filler;
    };
    void setWeightsSolver(const std::shared_ptr<Solver<Float_T> >& solver)
    {
        mWeightsSolver = solver;
    };
    void setBiasSolver(const std::shared_ptr<Solver<Float_T> >& solver)
    {
        mBiasSolver = solver;
    };
    virtual void logFreeParameters(const std::string& fileName,
                                   unsigned int output,
                                   unsigned int channel) const;
    virtual void logFreeParameters(const std::string& fileName,
                                   unsigned int output) const;
    virtual void logFreeParameters(const std::string& dirName) const;
    unsigned long long int getNbSharedSynapses() const;
    unsigned long long int getNbVirtualSynapses() const;
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
    unsigned int getSubSampleX() const
    {
        return mSubSampleDims[0];
    };
    unsigned int getSubSampleY() const
    {
        return mSubSampleDims[1];
    };
    int getPaddingX() const
    {
        return mPaddingDims[0];
    };
    int getPaddingY() const
    {
        return mPaddingDims[1];
    };
    std::shared_ptr<Solver<Float_T> > getWeightsSolver()
    {
        return mWeightsSolver;
    };
    std::shared_ptr<Solver<Float_T> > getBiasSolver()
    {
        return mBiasSolver;
    };
    virtual Tensor<Float_T> getWeight(unsigned int output,
                                      unsigned int channel) const = 0;
    virtual Float_T getBias(unsigned int output) const = 0;
    void setKernel(unsigned int output,
                   unsigned int channel,
                   const Matrix<double>& value,
                   bool normalize);
    virtual Interface<Float_T>* getWeights() { return NULL; };
    virtual void setWeights(unsigned int /*k*/,
                    Interface<Float_T>* /*weights*/,
                    unsigned int /*offset*/) {};
    virtual std::shared_ptr<Tensor<Float_T> > getBiases()
    {
        return std::shared_ptr<Tensor<Float_T> >();
    };
    virtual void setBiases(const std::shared_ptr<Tensor<Float_T> >&
                           /*biases*/) {};
    virtual void exportFreeParameters(const std::string& fileName) const;
    virtual void importFreeParameters(const std::string& fileName,
                                      bool ignoreNotExists = false);
    virtual void logFreeParametersDistrib(const std::string& fileName) const;
    void writeMap(const std::string& fileName) const;
    void randomizeFreeParameters(double stdDev);
    virtual void discretizeFreeParameters(unsigned int nbLevels);
    virtual std::pair<Float_T, Float_T> getFreeParametersRange() const;
    virtual void processFreeParameters(const std::function
                               <double(const double&)>& func);
    void getStats(Stats& stats) const;
    virtual ~ConvCell() {};

protected:
    virtual void setOutputsDims();
    virtual void setWeight(unsigned int output,
                           unsigned int channel,
                           const Tensor<Float_T>& value) = 0;
    virtual void setBias(unsigned int output, Float_T value) = 0;
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
    // Subsampling at the output
    const std::vector<unsigned int> mSubSampleDims;
    // Stride for the convolution
    const std::vector<unsigned int> mStrideDims;
    // Padding for the convolution
    const std::vector<int> mPaddingDims;

    std::shared_ptr<Filler<Float_T> > mWeightsFiller;
    std::shared_ptr<Filler<Float_T> > mBiasFiller;
    std::shared_ptr<Solver<Float_T> > mWeightsSolver;
    std::shared_ptr<Solver<Float_T> > mBiasSolver;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::ConvCell::WeightsExportFormat>::data[]
    = {"OCHW", "HWCO"};
}

#endif // N2D2_CONVCELL_H
