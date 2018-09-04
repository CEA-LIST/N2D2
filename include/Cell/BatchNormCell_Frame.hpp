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
template <class T>
class BatchNormCell_Frame : public virtual BatchNormCell, public Cell_Frame<T> {
public:
    typedef typename Utils::scaling_type<T>::type ParamT;

    using Cell_Frame<T>::mInputs;
    using Cell_Frame<T>::mOutputs;
    using Cell_Frame<T>::mDiffInputs;
    using Cell_Frame<T>::mDiffOutputs;

    BatchNormCell_Frame(const std::string& name,
                        unsigned int nbOutputs,
                        const std::shared_ptr<Activation>& activation
                        = std::make_shared<TanhActivation_Frame<T> >());
    static std::shared_ptr<BatchNormCell>
    create(const std::string& name,
           unsigned int nbOutputs,
           const std::shared_ptr<Activation>& activation
           = std::make_shared<TanhActivation_Frame<T> >())
    {
        return std::make_shared
            <BatchNormCell_Frame>(name, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline void getScale(unsigned int index, BaseTensor& value) const
    {
        // Need to specify std::initializer_list<size_t> for GCC 4.4
        value.resize(std::initializer_list<size_t>({1}));
        value = Tensor<ParamT>({1}, (*mScale)(index));
    }
    inline void getBias(unsigned int index, BaseTensor& value) const
    {
        value.resize(std::initializer_list<size_t>({1}));
        value = Tensor<ParamT>({1}, (*mBias)(index));
    }
    inline void getMean(unsigned int index, BaseTensor& value) const
    {
        value.resize(std::initializer_list<size_t>({1}));
        value = Tensor<ParamT>({1}, (*mMean)(index));
    }
    inline void getVariance(unsigned int index, BaseTensor& value) const
    {
        value.resize(std::initializer_list<size_t>({1}));
        value = Tensor<ParamT>({1}, (*mVariance)(index));
    }
    inline std::shared_ptr<BaseTensor> getScales() const
    {
        return mScale;
    };
    inline void setScales(const std::shared_ptr<BaseTensor>& scales)
    {
        mScale = std::dynamic_pointer_cast<Tensor<ParamT> >(scales);

        if (!mScale) {
            throw std::runtime_error("BatchNormCell_Frame<ParamT>::setScales():"
                                     " invalid type");
        }
    }
    inline std::shared_ptr<BaseTensor> getBiases() const
    {
        return mBias;
    };
    inline void setBiases(const std::shared_ptr<BaseTensor>& biases)
    {
        mBias = std::dynamic_pointer_cast<Tensor<ParamT> >(biases);

        if (!mBias) {
            throw std::runtime_error("BatchNormCell_Frame<ParamT>::setBiases():"
                                     " invalid type");
        }
    }
    inline std::shared_ptr<BaseTensor> getMeans() const
    {
        return mMean;
    };
    inline void setMeans(const std::shared_ptr<BaseTensor>& means)
    {
        mMean = std::dynamic_pointer_cast<Tensor<ParamT> >(means);

        if (!mMean) {
            throw std::runtime_error("BatchNormCell_Frame<ParamT>::setMeans():"
                                     " invalid type");
        }
    }
    inline std::shared_ptr<BaseTensor> getVariances() const
    {
        return mVariance;
    };
    inline void setVariances(const std::shared_ptr<BaseTensor>&
                             variances)
    {
        mVariance = std::dynamic_pointer_cast<Tensor<ParamT> >(variances);

        if (!mVariance) {
            throw std::runtime_error("BatchNormCell_Frame<ParamT>::setVariances():"
                                     " invalid type");
        }
    }
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    virtual ~BatchNormCell_Frame();

protected:
    inline void setScale(unsigned int index, const BaseTensor& value)
    {
        (*mScale)(index) = tensor_cast<ParamT>(value)(0);
    }
    inline void setBias(unsigned int index, const BaseTensor& value)
    {
        (*mBias)(index) = tensor_cast<ParamT>(value)(0);
    }
    inline void setMean(unsigned int index, const BaseTensor& value)
    {
        (*mMean)(index) = tensor_cast<ParamT>(value)(0);
    }
    inline void setVariance(unsigned int index, const BaseTensor& value)
    {
        (*mVariance)(index) = tensor_cast<ParamT>(value)(0);
    }

    unsigned int mNbPropagate;
    std::shared_ptr<Tensor<ParamT> > mScale;
    std::shared_ptr<Tensor<ParamT> > mBias;
    std::shared_ptr<Tensor<ParamT> > mMean;
    std::shared_ptr<Tensor<ParamT> > mVariance;
    Tensor<ParamT> mDiffScale;
    Tensor<ParamT> mDiffBias;
    Tensor<ParamT> mDiffSavedMean;
    Tensor<ParamT> mDiffSavedVariance;
    Tensor<ParamT> mSavedMean;
    Tensor<ParamT> mSavedVariance;

private:
    static Registrar<BatchNormCell> mRegistrar;
};
}

#endif // N2D2_BATCHNORMCELL_FRAME_H
