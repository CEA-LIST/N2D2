/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_BATCHNORMMAXCELL_H
#define N2D2_BATCHNORMMAXCELL_H

#include <string>
#include <vector>

#include "Activation/Activation.hpp"
#include "Cell.hpp"
#include "Solver/Solver.hpp"
#include "utils/Registrar.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@BatchNormCell_Frame@N2D2@@0U?$Registrar@VBatchNormCell@N2D2@@@2@A")
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@BatchNormCell_Frame_CUDA@N2D2@@0U?$Registrar@VBatchNormCell@N2D2@@@2@A")
#endif
#endif

namespace N2D2 {
class BatchNormCell : public virtual Cell {
public:
    typedef std::function<std::shared_ptr<BatchNormCell>(
        const std::string&,
        unsigned int,
        const std::shared_ptr<Activation<Float_T> >&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;
    BatchNormCell(const std::string& name, unsigned int nbOutputs);
    const char* getType() const
    {
        return Type;
    };
    void setScaleSolver(const std::shared_ptr<Solver<Float_T> >& solver)
    {
        mScaleSolver = solver;
    };
    void setBiasSolver(const std::shared_ptr<Solver<Float_T> >& solver)
    {
        mBiasSolver = solver;
    };
    std::shared_ptr<Solver<Float_T> > getScaleSolver()
    {
        return mScaleSolver;
    };
    std::shared_ptr<Solver<Float_T> > getBiasSolver()
    {
        return mBiasSolver;
    };
    virtual Float_T
    getScale(unsigned int channel, unsigned int sx, unsigned int sy) const = 0;
    virtual Float_T
    getBias(unsigned int channel, unsigned int sx, unsigned int sy) const = 0;
    virtual Float_T
    getMean(unsigned int channel, unsigned int sx, unsigned int sy) const = 0;
    virtual Float_T getVariance(unsigned int channel,
                                unsigned int sx,
                                unsigned int sy) const = 0;
    virtual std::shared_ptr<Tensor4d<Float_T> > getScales()
    {
        return std::shared_ptr<Tensor4d<Float_T> >();
    };
    virtual void setScales(const std::shared_ptr<Tensor4d<Float_T> >&
                           /*scales*/) {};
    virtual std::shared_ptr<Tensor4d<Float_T> > getBiases()
    {
        return std::shared_ptr<Tensor4d<Float_T> >();
    };
    virtual void setBiases(const std::shared_ptr<Tensor4d<Float_T> >&
                           /*biases*/) {};
    virtual std::shared_ptr<Tensor4d<Float_T> > getMeans()
    {
        return std::shared_ptr<Tensor4d<Float_T> >();
    };
    virtual void setMeans(const std::shared_ptr<Tensor4d<Float_T> >&
                          /*means*/) {};
    virtual std::shared_ptr<Tensor4d<Float_T> > getVariances()
    {
        return std::shared_ptr<Tensor4d<Float_T> >();
    };
    virtual void setVariances(const std::shared_ptr<Tensor4d<Float_T> >&
                              /*variances*/) {};
    virtual void exportFreeParameters(const std::string& fileName) const;
    virtual void importFreeParameters(const std::string& fileName,
                                      bool ignoreNotExists = false);
    void getStats(Stats& stats) const;
    virtual ~BatchNormCell() {};

protected:
    virtual void setOutputsSize();
    virtual void setScale(unsigned int channel,
                          unsigned int sx,
                          unsigned int sy,
                          Float_T value) = 0;
    virtual void setBias(unsigned int channel,
                         unsigned int sx,
                         unsigned int sy,
                         Float_T value) = 0;
    virtual void setMean(unsigned int channel,
                         unsigned int sx,
                         unsigned int sy,
                         Float_T value) = 0;
    virtual void setVariance(unsigned int channel,
                             unsigned int sx,
                             unsigned int sy,
                             Float_T value) = 0;

    /// Epsilon value used in the batch normalization formula
    Parameter<double> mEpsilon;

    std::shared_ptr<Solver<Float_T> > mScaleSolver;
    std::shared_ptr<Solver<Float_T> > mBiasSolver;
};
}

#endif // N2D2_BATCHNORMMAXCELL_H
