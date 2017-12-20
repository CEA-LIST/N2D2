/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Victor GACOIN

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

#ifndef N2D2_CELL_FRAME_TOP_H
#define N2D2_CELL_FRAME_TOP_H

#include <string>
#include <vector>

#include "Activation/Activation.hpp"
#include "Activation/LogisticActivation.hpp"
#include "Activation/RectifierActivation.hpp"
#include "Activation/SaturationActivation.hpp"
#include "Activation/SoftplusActivation.hpp"
#include "Activation/TanhActivation.hpp"
#include "Environment.hpp"
#include "containers/Tensor2d.hpp"
#include "containers/Tensor4d.hpp"

namespace N2D2 {
class Cell_Frame_Top {
public:
    enum Signals {
        In = 1,
        Out = 2,
        InOut = 3   // = In | Out
    };

    Cell_Frame_Top(const std::shared_ptr<Activation<Float_T> >& activation
                   = std::shared_ptr<Activation<Float_T> >())
        : mNbChannels(0),
          mActivation(activation),
          mFullMap(true),
          mFullMapInitialized(false),
          mUnitMap(true),
          mUnitMapInitialized(false)
    {
    }
    inline virtual bool isFullMap() const;
    inline virtual bool isUnitMap() const;
    virtual void addInput(Tensor4d<Float_T>& inputs,
                          Tensor4d<Float_T>& diffOutputs) = 0;
    virtual void propagate(bool inference = false) = 0;
    virtual void backPropagate() = 0;
    virtual void update() = 0;
    virtual void checkGradient(double /*epsilon*/, double /*maxError*/) = 0;
    virtual void discretizeSignals(unsigned int /*nbLevels*/,
                                   const Signals& /*signals*/ = In) = 0;
    virtual void setOutputTarget(const Tensor4d<int>& targets,
                                 double targetVal = 1.0,
                                 double defaultVal = 0.0) = 0;
    virtual void setOutputTargets(const Tensor4d<int>& targets,
                                  double targetVal = 1.0,
                                  double defaultVal = 0.0) = 0;
    virtual void setOutputTargets(const Tensor4d<Float_T>& targets) = 0;
    virtual void setOutputErrors(const Tensor4d<Float_T>& errors) = 0;
    virtual Tensor4d<Float_T>& getOutputs() = 0;
    virtual const Tensor4d<Float_T>& getOutputs() const = 0;
    virtual Tensor4d<Float_T>& getDiffInputs() = 0;
    virtual const Tensor4d<Float_T>& getDiffInputs() const = 0;
    virtual unsigned int getMaxOutput(unsigned int batchPos = 0) const = 0;
    const std::shared_ptr<Activation<Float_T> >& getActivation() const
    {
        return mActivation;
    };
    void setActivation(const std::shared_ptr<Activation<Float_T> >& activation)
    {
        mActivation = activation;
    };
    virtual ~Cell_Frame_Top() {};

protected:
    // Number of input channels
    unsigned int mNbChannels;
    // Input-output mapping
    Tensor2d<bool> mMaps;
    std::shared_ptr<Activation<Float_T> > mActivation;

private:
    mutable bool mFullMap;
    mutable bool mFullMapInitialized;
    mutable bool mUnitMap;
    mutable bool mUnitMapInitialized;
};
}

bool N2D2::Cell_Frame_Top::isFullMap() const
{
    if (!mFullMapInitialized) {
        for (unsigned int output = 0,
                          nbOutputs = mMaps.dimX(),
                          nbChannels = mMaps.dimY();
             output < nbOutputs;
             ++output) {
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                if (!mMaps(output, channel)) {
                    mFullMap = false;
                    break;
                }
            }
        }

        mFullMapInitialized = true;
    }

    return mFullMap;
}

bool N2D2::Cell_Frame_Top::isUnitMap() const
{
    if (!mUnitMapInitialized) {
        for (unsigned int output = 0,
                          nbOutputs = mMaps.dimX(),
                          nbChannels = mMaps.dimY();
             output < nbOutputs;
             ++output) {
            if (nbChannels < nbOutputs) {
                mUnitMap = false;
                break;
            }

            for (unsigned int channel = 0; channel < nbChannels; ++channel) {
                if ((channel != output && mMaps(output, channel))
                    || (channel == output && !mMaps(output, channel))) {
                    mUnitMap = false;
                    break;
                }
            }
        }

        mUnitMapInitialized = true;
    }

    return mUnitMap;
}

#endif // N2D2_CELL_FRAME_TOP_H
