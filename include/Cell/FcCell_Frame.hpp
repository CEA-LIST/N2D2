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

#ifndef N2D2_FCCELL_FRAME_H
#define N2D2_FCCELL_FRAME_H

#include "Activation/TanhActivation_Frame.hpp"
#include "Cell_Frame.hpp"
#include "DeepNet.hpp"
#include "FcCell.hpp"

namespace N2D2 {
template <class T>
class FcCell_Frame : public virtual FcCell, public Cell_Frame<T> {
public:
    using Cell_Frame<T>::mInputs;
    using Cell_Frame<T>::mOutputs;
    using Cell_Frame<T>::mDiffInputs;
    using Cell_Frame<T>::mDiffOutputs;

    FcCell_Frame(const DeepNet& deepNet, const std::string& name,
                 unsigned int nbOutputs,
                 const std::shared_ptr<Activation>& activation
                 = std::shared_ptr<Activation>());
    static std::shared_ptr<FcCell> create(Network& /*net*/, const DeepNet& deepNet, 
                                          const std::string& name,
                                          unsigned int nbOutputs,
                                          const std::shared_ptr
                                          <Activation>& activation
                                          = std::shared_ptr<Activation>())
    {
        return std::make_shared<FcCell_Frame>(deepNet, name, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void initializeParameters(unsigned int nbInputChannels, unsigned int nbInputs);
    virtual void initializeWeightQuantizer();
    virtual void check_input();
    virtual void initializeDataDependent();
    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();

    void resetWeights();
    void resetBias();
    void resetWeightsSolver(const std::shared_ptr<Solver>& solver)
    {
        setWeightsSolver(solver);
        for (unsigned int k = 0, size = mWeightsSolvers.size(); k < size; ++k) {
            mWeightsSolvers[k] = mWeightsSolver->clone();
        }
    };

    inline void getWeight(unsigned int output, unsigned int channel,
                          BaseTensor& value) const
    {
        // Need to specify std::initializer_list<size_t> for GCC 4.4
        value.resize(std::initializer_list<size_t>({1}));
        value = Tensor<T>({1}, mSynapses(0, 0, channel, output));
    };
    inline void getQuantWeight(unsigned int output, unsigned int channel,
                          BaseTensor& value) const
    {
        if (!mQuantizer)
            return;

        const unsigned int k = mInputs.getTensorIndex(channel);
        channel -= mInputs.getTensorDataOffset(channel);

        const Tensor<T>& synapses = tensor_cast<T>(mQuantizer->getQuantizedWeights(k));
        value.resize(std::initializer_list<size_t>({1}));
        value = Tensor<T>({1}, synapses(0, 0, channel, output));
    };
    inline void getBias(unsigned int output, BaseTensor& value) const
    {
        value.resize(std::initializer_list<size_t>({1}));
        value = Tensor<T>({1}, mBias(output));
    };
    inline BaseInterface* getWeights()
    {
        return &mSynapses;
    };
    virtual const BaseInterface* getWeights() const { return &mSynapses; };
    virtual const BaseTensor* getBiases() const { return &mBias; };
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    
    
    virtual ~FcCell_Frame();

protected:
    inline void setWeight(unsigned int output, unsigned int channel,
                          const BaseTensor& value)
    {
        mSynapses(0, 0, channel, output) = tensor_cast<T>(value)(0);
    };
    inline void setBias(unsigned int output, const BaseTensor& value)
    {
        if (!mNoBias && mBias.empty())
            mBias.resize({getNbOutputs(), 1, 1, 1});

        mBias(output) = tensor_cast<T>(value)(0);
    };

    Parameter<double> mDropConnect;

    // Internal
    std::vector<std::shared_ptr<Solver> > mWeightsSolvers;
    Interface<T> mSynapses;
    Tensor<T> mBias;
    Interface<T> mDiffSynapses;
    Tensor<T> mDiffBias;

    Interface<bool> mDropConnectMask;
    bool mLockRandom;

private:
    static Registrar<FcCell> mRegistrar;
};
}

#endif // N2D2_FCCELL_FRAME_H
