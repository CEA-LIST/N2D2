/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifndef N2D2_SCALING_HPP
#define N2D2_SCALING_HPP

#include <array>
#include "params.h"

namespace N2D2 {

std::int64_t toInt64(std::uint32_t lo, std::uint32_t hi) {
    return (std::int64_t) (((std::uint64_t) hi) << 32ull) | ((std::uint64_t) lo);
}

std::int64_t smlal(std::int32_t lhs, std::int32_t rhs, 
                   std::uint32_t accumLo, std::uint32_t accumHi) 
{
    return ((std::int64_t) lhs) * ((std::int64_t) rhs) + toInt64(accumLo, accumHi);
}

struct NoScaling {
    SUM_T operator()(SUM_T weightedSum, std::size_t /*output*/) const {
        return weightedSum;
    }
};


#if NB_BITS > 0
struct FloatingPointScaling {
    SUM_T operator()(SUM_T weightedSum, std::size_t /*output*/) const {
        return std::round(weightedSum*mScaling);
    }

    double mScaling;
};

template<std::size_t SIZE>
struct FloatingPointScalingPerChannel {
    SUM_T operator()(SUM_T weightedSum, std::size_t output) const {
        return std::round(weightedSum * mScaling[output]);
    }

    std::array<double, SIZE> mScaling;
};



template<std::int32_t SCALING, std::int64_t FRACTIONAL_BITS>
struct FixedPointScaling {
    SUM_T operator()(SUM_T weightedSum, std::size_t /*output*/) const {
        // Different rounding if weightesSum < 0
        // if(weightedSum < 0) {
        //     HALF--; 
        // }
        
        return smlal(weightedSum, SCALING, HALF_LO, HALF_HI) >> FRACTIONAL_BITS; 
    }

    static const std::uint32_t HALF_LO = (1ull << (FRACTIONAL_BITS - 1)) & 0xFFFFFFFF;
    static const std::uint32_t HALF_HI = (1ull << (FRACTIONAL_BITS - 1)) >> 32u;
};

template<std::size_t SIZE, std::int64_t FRACTIONAL_BITS>
struct FixedPointScalingScalingPerChannel {
    SUM_T operator()(SUM_T weightedSum, std::size_t output) const {
        // Different rounding if weightesSum < 0
        // if(weightedSum < 0) {
        //     HALF--; 
        // }

        return smlal(weightedSum, mScaling[output], HALF_LO, HALF_HI) >> FRACTIONAL_BITS; 
    }

    static const std::uint32_t HALF_LO = (1ull << (FRACTIONAL_BITS - 1)) & 0xFFFFFFFF;
    static const std::uint32_t HALF_HI = (1ull << (FRACTIONAL_BITS - 1)) >> 32u;

    std::array<std::int32_t, SIZE> mScaling;
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

    std::array<unsigned char, SIZE> mScaling;
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

    std::array<std::array<SUM_T, 3>, SIZE> mScaling;
};
#endif

}

#endif