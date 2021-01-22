/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_FCCELL_H
#define N2D2_FCCELL_H

#include <memory>
#include <map>

#include "Cell.hpp"
#include "utils/Registrar.hpp"
#include "Quantizer/Quantizer.hpp"


namespace N2D2 {

class Activation;
class DeepNet;
class Filler;
class Network;
class Solver;

class FcCell : public virtual Cell {
public:
    typedef std::function
        <std::shared_ptr<FcCell>(Network&, const DeepNet&, 
                                 const std::string&,
                                 unsigned int,
                                 const std::shared_ptr<Activation>&)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;
    // O = output feature map
    // C = input feature map (channel)
    enum WeightsExportFormat {
        // N2D2 default format
        OC,
        // TensorFlow format:
        // "filter: A Tensor. Must have the same type as input. A 4-D tensor of
        // shape [in_channels, out_channels]"
        // https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        CO
    };
    FcCell(const DeepNet& deepNet, const std::string& name, unsigned int nbOutputs);
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
    void setQuantizer(const std::shared_ptr<Quantizer>& quantizer)
    {
        mQuantizer = quantizer;
    };
    virtual void logFreeParameters(const std::string& fileName,
                                   unsigned int output) const;
    virtual void logFreeParameters(const std::string& dirName) const;
    unsigned long long int getNbSynapses(bool includeBias = true) const;
    std::shared_ptr<Solver> getWeightsSolver()
    {
        return mWeightsSolver;
    };
    std::shared_ptr<Solver> getBiasSolver()
    {
        return mBiasSolver;
    };
    std::shared_ptr<Quantizer> getQuantizer()
    {
        return mQuantizer;
    };
    virtual void getWeight(unsigned int output,
                           unsigned int channel, BaseTensor& value) const = 0;
    virtual void getBias(unsigned int output, BaseTensor& value) const = 0;
    virtual void exportFreeParameters(const std::string& fileName) const;
    virtual void importFreeParameters(const std::string& fileName,
                                      bool ignoreNotExists = false);
    virtual void logFreeParametersDistrib(const std::string& fileName,
                                          FreeParametersType type = All) const;
    void writeMap(const std::string& fileName) const;
    void randomizeFreeParameters(double stdDev);
    virtual std::pair<Float_T, Float_T> getFreeParametersRange(bool withAdditiveParameters = true) const;
    virtual std::pair<Float_T, Float_T> getFreeParametersRangePerOutput(std::size_t output, 
                                                                   bool withAdditiveParameters) const;
    virtual std::pair<Float_T, Float_T> getFreeParametersRangePerChannel(std::size_t channel) const;
    
    virtual void processFreeParameters(std::function<Float_T(Float_T)> func,
                                       FreeParametersType type = All);
    virtual void processFreeParametersPerOutput(std::function<Float_T(Float_T)> /*func*/,
                                                std::size_t /*output*/,
                                                FreeParametersType /*type*/ = All);
    virtual void processFreeParametersPerChannel(std::function<Float_T(Float_T)> /*func*/,
                                                std::size_t /*channel*/);
    
    void getStats(Stats& stats) const;
    virtual void setWeight(unsigned int output, unsigned int channel,
                           const BaseTensor& value) = 0;
    virtual void setBias(unsigned int output, const BaseTensor& value) = 0;
    virtual ~FcCell() {};

protected:
    virtual void setOutputsDims() {};
    std::map<unsigned int, unsigned int> outputsRemap() const;

    /// If true, the output neurons don't have bias
    Parameter<bool> mNoBias;
    /// If true, normalize the weights (L2 norm)
    Parameter<bool> mNormalize;
    /// If true, enable backpropogation
    Parameter<bool> mBackPropagate;
    Parameter<WeightsExportFormat> mWeightsExportFormat;
    Parameter<std::string> mOutputsRemap;

    std::shared_ptr<Filler> mWeightsFiller;
    std::shared_ptr<Filler> mBiasFiller;
    std::shared_ptr<Solver> mWeightsSolver;
    std::shared_ptr<Solver> mBiasSolver;
    std::shared_ptr<Quantizer> mQuantizer;
};
}
namespace {
    template <>
    const char* const EnumStrings<N2D2::FcCell::WeightsExportFormat>::data[]
        = {"OC", "CO"};
}

#endif // N2D2_FCCELL_H
