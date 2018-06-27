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

#ifndef N2D2_BATCHNORMCELL_FRAME_H
#define N2D2_BATCHNORMCELL_FRAME_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "BatchNormCell.hpp"
#include "Cell_Frame.hpp"
#include "Solver/SGDSolver_Frame.hpp"

namespace N2D2 {
class BatchNormCell_Frame : public virtual BatchNormCell, public Cell_Frame {
public:
    BatchNormCell_Frame(const std::string& name,
                        unsigned int nbOutputs,
                        const std::shared_ptr<Activation<Float_T> >& activation
                        = std::make_shared<TanhActivation_Frame<Float_T> >());
    static std::shared_ptr<BatchNormCell>
    create(const std::string& name,
           unsigned int nbOutputs,
           const std::shared_ptr<Activation<Float_T> >& activation
           = std::make_shared<TanhActivation_Frame<Float_T> >())
    {
        return std::make_shared
            <BatchNormCell_Frame>(name, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline Float_T getScale(unsigned int index) const
    {
        return (*mScale)(index);
    }
    inline Float_T getBias(unsigned int index) const
    {
        return (*mBias)(index);
    }
    inline Float_T getMean(unsigned int index) const
    {
        return (*mMean)(index);
    }
    inline Float_T getVariance(unsigned int index) const
    {
        return (*mVariance)(index);
    }
    inline std::shared_ptr<Tensor<Float_T> > getScales() const
    {
        return mScale;
    };
    inline void setScales(const std::shared_ptr<Tensor<Float_T> >& scales)
    {
        mScale = scales;
    }
    inline std::shared_ptr<Tensor<Float_T> > getBiases() const
    {
        return mBias;
    };
    inline void setBiases(const std::shared_ptr<Tensor<Float_T> >& biases)
    {
        mBias = biases;
    }
    inline std::shared_ptr<Tensor<Float_T> > getMeans() const
    {
        return mMean;
    };
    inline void setMeans(const std::shared_ptr<Tensor<Float_T> >& means)
    {
        mMean = means;
    }
    inline std::shared_ptr<Tensor<Float_T> > getVariances() const
    {
        return mVariance;
    };
    inline void setVariances(const std::shared_ptr<Tensor<Float_T> >&
                             variances)
    {
        mVariance = variances;
    }
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    virtual ~BatchNormCell_Frame();

protected:
    inline void setScale(unsigned int index, Float_T value)
    {
        (*mScale)(index) = value;
    }
    inline void setBias(unsigned int index, Float_T value)
    {
        (*mBias)(index) = value;
    }
    inline void setMean(unsigned int index, Float_T value)
    {
        (*mMean)(index) = value;
    }
    inline void setVariance(unsigned int index, Float_T value)
    {
        (*mVariance)(index) = value;
    }

    unsigned int mNbPropagate;
    std::shared_ptr<Tensor<Float_T> > mScale;
    std::shared_ptr<Tensor<Float_T> > mBias;
    std::shared_ptr<Tensor<Float_T> > mMean;
    std::shared_ptr<Tensor<Float_T> > mVariance;
    Tensor<Float_T> mDiffScale;
    Tensor<Float_T> mDiffBias;
    Tensor<Float_T> mDiffSavedMean;
    Tensor<Float_T> mDiffSavedVariance;
    Tensor<Float_T> mSavedMean;
    Tensor<Float_T> mSavedVariance;

private:
    static Registrar<BatchNormCell> mRegistrar;
};
}

#endif // N2D2_BATCHNORMCELL_FRAME_H
