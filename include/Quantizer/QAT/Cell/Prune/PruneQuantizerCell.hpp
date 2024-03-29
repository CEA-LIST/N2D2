/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#ifndef N2D2_PRUNEQUANTIZERCELL_H
#define N2D2_PRUNEQUANTIZERCELL_H

#include "Quantizer/QAT/Cell/QuantizerCell.hpp"
#include "utils/Registrar.hpp"
#include "containers/Tensor.hpp"

namespace N2D2 {

class PruneQuantizerCell : virtual public QuantizerCell {
public:

    enum PruningMode{
        Identity,
        Static,
        Gradual
    };

    enum PruningFiller{
        None,
        Random,
        IterNonStruct
    };

    typedef std::function<std::shared_ptr<PruneQuantizerCell>()> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static const char* Type;

    PruneQuantizerCell();


    // Setters

    void setPruningMode(PruningMode mode)
    {
        mPruningMode = mode;
    };
    void setPruningFiller(PruningFiller filler)
    {
        mPruningFiller = filler;
    };
    void setThreshold(float threshold)
    {
        mThreshold = threshold;
    };
    void setDelta(float delta)
    {
        mDelta = delta;
    };
    void setStartThreshold(float start)
    {
        mStartThreshold = start;
    };
    void setStepSizeThreshold(unsigned int stepsize)
    {
        mStepSizeThreshold = stepsize;
    };
    void setGammaThreshold(float gamma)
    {
        mGammaThreshold = gamma;
    }


    // Getters

    virtual BaseTensor& getMasksWeights(unsigned int k) = 0;

    virtual const char* getType() const
    {
        return Type;
    }
    PruningMode getPruningMode()
    {
        return mPruningMode;
    };
    float getThreshold()
    {
        return mThreshold;
    };  
    float getDelta()
    {
        return mDelta;
    };  
    float getCurrentThreshold()
    {
        return mCurrentThreshold;
    }

    virtual void exportFreeParameters(const std::string& /*fileName*/) const {};
    virtual void importFreeParameters(const std::string& /*fileName*/, bool /*ignoreNoExists*/) {};

    virtual ~PruneQuantizerCell() {};

protected:
    Parameter<PruningMode> mPruningMode;
    Parameter<PruningFiller> mPruningFiller;
    Parameter<float> mThreshold;

    // For IterNonStruct filler
    Parameter<float> mDelta;

    // For Gradual mode
    Parameter<float> mStartThreshold;
    Parameter<unsigned int> mStepSizeThreshold;
    Parameter<float> mGammaThreshold;

    float mCurrentThreshold;

private:
};

}

namespace {
template <>
const char* const EnumStrings<N2D2::PruneQuantizerCell::PruningMode>::data[]
    = {"Identity", "Static", "Gradual"};

template <>
const char* const EnumStrings<N2D2::PruneQuantizerCell::PruningFiller>::data[]
    = {"None", "Random", "IterNonStruct"};
}

#endif  // N2D2_PRUNEQUANTIZERCELL_H