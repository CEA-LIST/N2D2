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

#include "Cell.hpp"
#include "DeepNet.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {

class Activation;
class Solver;

class BatchNormCell : public virtual Cell {
public:
    typedef std::function<std::shared_ptr<BatchNormCell>(
        const DeepNet&,
        const std::string&,
        unsigned int,
        const std::shared_ptr<Activation>&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;
    BatchNormCell(const DeepNet& deepNet, const std::string& name, unsigned int nbOutputs);
    const char* getType() const
    {
        return Type;
    };
    void setScaleSolver(const std::shared_ptr<Solver>& solver)
    {
        mScaleSolver = solver;
    };
    void setBiasSolver(const std::shared_ptr<Solver>& solver)
    {
        mBiasSolver = solver;
    };
    std::shared_ptr<Solver> getScaleSolver()
    {
        return mScaleSolver;
    };
    std::shared_ptr<Solver> getBiasSolver()
    {
        return mBiasSolver;
    };
    virtual void getScale(unsigned int index, BaseTensor& value) const = 0;
    virtual void getBias(unsigned int index, BaseTensor& value) const = 0;
    virtual void getMean(unsigned int index, BaseTensor& value) const = 0;
    virtual void getVariance(unsigned int index, BaseTensor& value) const = 0;
    virtual void setScale(unsigned int index, const BaseTensor& value) = 0;
    virtual void setBias(unsigned int index, const BaseTensor& value) = 0;
    virtual void setMean(unsigned int index, const BaseTensor& value) = 0;
    virtual void setVariance(unsigned int index, const BaseTensor& value) = 0;
    virtual std::shared_ptr<BaseTensor> getScales() const
    {
        return std::shared_ptr<BaseTensor>();
    };
    virtual void setScales(const std::shared_ptr<BaseTensor>&
                           /*scales*/) {};
    virtual std::shared_ptr<BaseTensor> getBiases() const
    {
        return std::shared_ptr<BaseTensor>();
    };
    virtual void setBiases(const std::shared_ptr<BaseTensor>&
                           /*biases*/) {};
    virtual std::shared_ptr<BaseTensor> getMeans() const
    {
        return std::shared_ptr<BaseTensor>();
    };
    virtual void setMeans(const std::shared_ptr<BaseTensor>&
                          /*means*/) {};
    virtual std::shared_ptr<BaseTensor> getVariances() const
    {
        return std::shared_ptr<BaseTensor>();
    };
    virtual void setVariances(const std::shared_ptr<BaseTensor>&
                              /*variances*/) {};
    virtual void exportFreeParameters(const std::string& fileName) const;
    virtual void importFreeParameters(const std::string& fileName,
                                      bool ignoreNotExists = false);
    void getStats(Stats& stats) const;
    virtual ~BatchNormCell() {};

protected:
    virtual void setOutputsDims();

    /// Epsilon value used in the batch normalization formula
    Parameter<double> mEpsilon;

    // Moving average rate: used for the moving average of
    // batch-wise means and standard deviations during training.
    // The closer to 1.0, the more it will depend on the last batch 
    Parameter<double> mMovingAverageMomentum;
    
    std::shared_ptr<Solver> mScaleSolver;
    std::shared_ptr<Solver> mBiasSolver;
};
}

#endif // N2D2_BATCHNORMMAXCELL_H
