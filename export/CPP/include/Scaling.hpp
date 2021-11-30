/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifndef N2D2_SCALING_HPP
#define N2D2_SCALING_HPP

#include "params.h"

namespace N2D2 {

int64_t toInt64(uint32_t lo, uint32_t hi) {
    return (int64_t) (((uint64_t) hi) << 32ull) | ((uint64_t) lo);
}

int64_t smlal(int32_t lhs, int32_t rhs, 
                   uint32_t accumLo, uint32_t accumHi) 
{
    return ((int64_t) lhs) * ((int64_t) rhs) + toInt64(accumLo, accumHi);
}

struct NoScaling {
    SUM_T operator()(SUM_T weightedSum, std::size_t /*output*/) const {
        return weightedSum;
    }
};


#if NB_BITS > 0
struct FloatingPointScaling {
    SUM_T operator()(SUM_T weightedSum, std::size_t /*output*/) const {
        return round(weightedSum*mScaling);
    }

    double mScaling;
};

template<std::size_t SIZE>
struct FloatingPointClippingAndScaling {
    SUM_T operator()(SUM_T weightedSum, std::size_t /*output*/) const {
        SUM_T clipValue = weightedSum;
        clipValue = (clipValue < SUM_T(0)) ?
                    SUM_T(0) : (clipValue > SUM_T(mClipping)) ?
                    SUM_T(mClipping) : clipValue;
        return round(clipValue * mScaling);
    }

    double mScaling;
    int32_t mClipping;
};

template<std::size_t SIZE>
struct FloatingPointClippingAndScalingPerChannel {
    SUM_T operator()(SUM_T weightedSum, std::size_t output) const {
        SUM_T clipValue = weightedSum;
        clipValue = (clipValue < SUM_T(0)) ? 
                    SUM_T(0) : (clipValue > SUM_T(mClipping[output])) ? 
                    SUM_T(mClipping[output]) : clipValue;
        return round(clipValue * mScaling[output]);
    }

    double mScaling[SIZE];
    int32_t mClipping[SIZE];
};

template<std::size_t SIZE>
struct FloatingPointScalingPerChannel {
    SUM_T operator()(SUM_T weightedSum, std::size_t output) const {
        return round(weightedSum * mScaling[output]);
    }

    double mScaling[SIZE];
};



template<int32_t SCALING, int64_t FRACTIONAL_BITS>
struct FixedPointScaling {
    SUM_T operator()(SUM_T weightedSum, std::size_t /*output*/) const {
        // Different rounding if weightesSum < 0
        // if(weightedSum < 0) {
        //     HALF--; 
        // }
        
        return smlal(weightedSum, SCALING, HALF_LO, HALF_HI) >> FRACTIONAL_BITS; 
    }

    static const uint32_t HALF_LO = (FRACTIONAL_BITS > 0)
        ? (1ull << (FRACTIONAL_BITS - 1)) & 0xFFFFFFFF : 0;
    static const uint32_t HALF_HI = (FRACTIONAL_BITS > 0)
        ? (1ull << (FRACTIONAL_BITS - 1)) >> 32u : 0;
};

template<std::size_t SIZE, int64_t FRACTIONAL_BITS>
struct FixedPointClippingAndScalingPerChannel {
    SUM_T operator()(SUM_T weightedSum, std::size_t output) const {
        // Different rounding if weightesSum < 0
        // if(weightedSum < 0) {
        //     HALF--; 
        // }
        SUM_T clipValue = weightedSum;
        clipValue = (clipValue < SUM_T(0)) ? 
                    SUM_T(0) : (clipValue > SUM_T(mClipping[output])) ? 
                    SUM_T(mClipping[output]) : clipValue;

        return smlal(clipValue, mScaling[output], HALF_LO, HALF_HI) >> FRACTIONAL_BITS; 
    }

    static const uint32_t HALF_LO = (1ull << (FRACTIONAL_BITS - 1)) & 0xFFFFFFFF;
    static const uint32_t HALF_HI = (1ull << (FRACTIONAL_BITS - 1)) >> 32u;

    int32_t mScaling[SIZE];
    int32_t mClipping[SIZE];
};

template<std::size_t SIZE, int64_t FRACTIONAL_BITS>
struct FixedPointScalingScalingPerChannel {
    SUM_T operator()(SUM_T weightedSum, std::size_t output) const {
        // Different rounding if weightesSum < 0
        // if(weightedSum < 0) {
        //     HALF--; 
        // }

        return smlal(weightedSum, mScaling[output], HALF_LO, HALF_HI) >> FRACTIONAL_BITS; 
    }

    static const uint32_t HALF_LO = (FRACTIONAL_BITS > 0)
        ? (1ull << (FRACTIONAL_BITS - 1)) & 0xFFFFFFFF : 0;
    static const uint32_t HALF_HI = (FRACTIONAL_BITS > 0)
        ? (1ull << (FRACTIONAL_BITS - 1)) >> 32u : 0;

    int32_t mScaling[SIZE];
};



template<SUM_T SHIFT>
struct SingleShiftScaling {
    SUM_T operator()(SUM_T weightedSum, std::size_t /*output*/) const {
        // Different rounding if weightesSum < 0
        // if(weightedSum < 0) {
        //     HALF--; 
        // }

        return weightedSum >> SHIFT;
    }
};

template<std::size_t SIZE>
struct SingleShiftScalingPerChannel {
    SUM_T operator()(SUM_T weightedSum, std::size_t output) const {
        // Different rounding if weightesSum < 0
        // if(weightedSum < 0) {
        //     HALF--; 
        // }

        return weightedSum >> mScaling[output];
    }

    unsigned char mScaling[SIZE];
};



template<SUM_T SHIFT1, SUM_T SHIFT2>
struct DoubleShiftScaling {
    SUM_T operator()(SUM_T weightedSum, std::size_t /*output*/) const {
        // Different rounding if weightesSum < 0
        // if(weightedSum < 0) {
        //     HALF--; 
        // }

        return (weightedSum + (weightedSum << SHIFT1) + HALF) >> SHIFT2;
    }

    static const SUM_T HALF = ((SUM_T) 1) << (SHIFT2 - 1);
};

template<std::size_t SIZE, bool UNSIGNED_WEIGHTED_SUM>
struct DoubleShiftScalingPerChannel {
    SUM_T operator()(SUM_T weightedSum, std::size_t output) const {
        const SUM_T SHIFT1 = mScaling[output][0];
        const SUM_T SHIFT2 = mScaling[output][1];
        const SUM_T HALF = mScaling[output][2];
        
        // Different rounding if weightesSum < 0
        // if(weightedSum < 0) {
        //     HALF--; 
        // }

        return (weightedSum + (weightedSum << SHIFT1) + HALF) >> SHIFT2;
    }

    SUM_T mScaling[SIZE][3];
};
#endif

}

#endif
