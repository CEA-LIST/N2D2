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
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Activation/Activation.hpp"
#include "Environment.hpp"
#include "Filler/NormalFiller.hpp"
#include "Solver/Solver.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#include "Cell.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FcCell_Frame@N2D2@@0U?$Registrar@VFcCell@N2D2@@@2@A")
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FcCell_Frame_CUDA@N2D2@@0U?$Registrar@VFcCell@N2D2@@@2@A")
#endif
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FcCell_Spike@N2D2@@0U?$Registrar@VFcCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FcCell_Spike_Analog@N2D2@@0U?$Registrar@VFcCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FcCell_Spike_PCM@N2D2@@0U?$Registrar@VFcCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FcCell_Spike_RRAM@N2D2@@0U?$Registrar@VFcCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@?$FcCell_Transcode@VFcCell_Frame@N2D2@@VFcCell_Spike@2@@N2D2@@0U?$Registrar@VFcCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class FcCell : public virtual Cell {
public:
    typedef std::function
        <std::shared_ptr<FcCell>(Network&,
                                 const std::string&,
                                 unsigned int,
                                 const std::shared_ptr<Activation<Float_T> >&)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    FcCell(const std::string& name, unsigned int nbOutputs);
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
                                   unsigned int output) const;
    virtual void logFreeParameters(const std::string& dirName) const;
    unsigned long long int getNbSynapses() const;
    std::shared_ptr<Solver<Float_T> > getWeightsSolver()
    {
        return mWeightsSolver;
    };
    std::shared_ptr<Solver<Float_T> > getBiasSolver()
    {
        return mBiasSolver;
    };
    virtual Float_T getWeight(unsigned int output,
                              unsigned int channel) const = 0;
    virtual Float_T getBias(unsigned int output) const = 0;
    virtual void exportFreeParameters(const std::string& fileName) const;
    virtual void importFreeParameters(const std::string& fileName,
                                      bool ignoreNotExists = false);
    virtual void logFreeParametersDistrib(const std::string& fileName) const;
    void writeMap(const std::string& fileName) const;
    void randomizeFreeParameters(double stdDev);
    virtual void discretizeFreeParameters(unsigned int nbLevels);
    virtual void normalizeFreeParameters(double normFactor = 1.0);
    virtual void processFreeParameters(const std::function
                               <double(const double&)>& func);
    void getStats(Stats& stats) const;
    virtual ~FcCell() {};

protected:
    virtual void setOutputsSize() {};
    virtual void
    setWeight(unsigned int output, unsigned int channel, Float_T value) = 0;
    virtual void setBias(unsigned int output, Float_T value) = 0;

    /// If true, the output neurons don't have bias
    Parameter<bool> mNoBias;
    /// If true, enable backpropogation
    Parameter<bool> mBackPropagate;

    std::shared_ptr<Filler<Float_T> > mWeightsFiller;
    std::shared_ptr<Filler<Float_T> > mBiasFiller;
    std::shared_ptr<Solver<Float_T> > mWeightsSolver;
    std::shared_ptr<Solver<Float_T> > mBiasSolver;
};
}

#endif // N2D2_FCCELL_H
