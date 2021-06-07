/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_DISTANCE_CELL_FRAME_CUDA_H
#define N2D2_DISTANCE_CELL_FRAME_CUDA_H

#include <memory>
#include <string>

#include "Cell_Frame_CUDA.hpp"
#include "DistanceCell.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"

namespace N2D2 {

class DeepNet;

template<class T>
class DistanceCell_Frame_CUDA : public virtual DistanceCell, public Cell_Frame_CUDA<T> {
public:
    using Cell_Frame_CUDA<T>::mInputs;
    using Cell_Frame_CUDA<T>::mOutputs;
    using Cell_Frame_CUDA<T>::mDiffInputs;
    using Cell_Frame_CUDA<T>::mDiffOutputs;
    using Cell_Frame_CUDA<T>::keepInSync;
    using Cell_Frame_CUDA<T>::mKeepInSync;
    using Cell_Frame_CUDA<T>::mDevices;

    
    DistanceCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                           unsigned int nbOutputs, double margin);
    virtual ~DistanceCell_Frame_CUDA() = default;

    static std::shared_ptr<DistanceCell> create(const DeepNet& deepNet, const std::string& name,
                                               unsigned int nbOutputs, double margin)
    {
        return std::make_shared<DistanceCell_Frame_CUDA>(deepNet, name, nbOutputs, std::move(margin));
    }

    virtual void initialize();
    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();

    void exportFreeParameters(const std::string& fileName) const;
    void importFreeParameters(const std::string& fileName,
                              bool ignoreNotExists = false);
    inline void getWeight(unsigned int output, unsigned int channel,
                          BaseTensor& value) const;
    void setWeights(const std::shared_ptr<BaseTensor>& weights);

    inline std::shared_ptr<BaseTensor> getWeights()
    {
        return mMean;
    };

    inline BaseTensor& getDist()
    {
        return mDist;
    };

    inline BaseTensor& getmDiffMean()
    {
        return mDiffMean;
    };

    inline void setMeans(const BaseTensor& value);

    void synchronizeToH(bool keepInSync_) const;
    void synchronizeToD(bool keepInSync_);

    virtual double applyLoss(double targetVal, double defaultVal);
    virtual double applyLoss();

    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);

protected:
    inline void setWeight(unsigned int output, unsigned int channel,
                          const BaseTensor& value);
    
    T mCurrentMargin;
    long long unsigned int mItCounter = 0;

    std::shared_ptr<CudaTensor<T> > mMean;
    CudaTensor<T> mDiffMean;
    CudaTensor<T> mSigma;
    CudaTensor<T> mDist;
    CudaTensor<T> mLabels;

    std::shared_ptr<Solver> mWeightsSolver;
};
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::setMeans(const BaseTensor& value)
{
    if (mMean->empty())
        mMean->resize({1, 1, mInputs[0].dimZ(), this->getNbOutputs()});

    (*mMean) = tensor_cast<T>(value);
    //mMean->synchronizeHToD();
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::setWeight(unsigned int output,
                                           unsigned int channel,
                                           const BaseTensor& value)
{
    (*mMean)(0, 0, channel, output) = tensor_cast<T>(value)(0);

    if (mKeepInSync)
        mMean->synchronizeHToD(0, 0, channel, output, 1);
}

template <class T>
void N2D2::DistanceCell_Frame_CUDA<T>::getWeight(unsigned int output,
                                           unsigned int channel,
                                           BaseTensor& value) const
{
    if (mKeepInSync)
        mMean->synchronizeDToH(0, 0, channel, output, 1);

    value.resize({1});
    value = Tensor<T>({1}, (*mMean)(0, 0, channel, output));
}

#endif // N2D2_DISTANCE_CELL_FRAME_CUDA_H
