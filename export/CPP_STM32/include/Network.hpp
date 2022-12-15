/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifndef N2D2_NETWORK_HPP
#define N2D2_NETWORK_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "typedefs.h" // old C header, deprecated

#ifdef N2D2_STM32_EMULATOR
#include <chrono>
#include <map>
#include <numeric>
#else
#include <cmsis_compiler.h>
#include HAL_HEADER
#endif

#ifdef N2D2_STM32_EMULATOR
#define N2D2_THROW_OR_ABORT(ex, msg) throw ex(msg)
#else
#define N2D2_THROW_OR_ABORT(ex, msg) std::abort()
#endif

#define N2D2_ALWAYS_INLINE __attribute__((always_inline))

#ifndef N2D2_SECTION_NN_MEMORY
#define N2D2_SECTION_NN_MEMORY ".nn_memory"
#endif
#ifndef N2D2_SECTION_NN_BUFFER
#define N2D2_SECTION_NN_BUFFER ".nn_buffer"
#endif
#ifndef N2D2_SECTION_NN_WEIGHTS
#define N2D2_SECTION_NN_WEIGHTS ".nn_weights"
#endif
#ifndef N2D2_SECTION_NN_BIASSES
#define N2D2_SECTION_NN_BIASSES ".nn_biasses"
#endif
#define N2D2_SECTION_ATTRIBUTE(sec) __attribute__((section(sec)))

namespace N2D2_Export {

class Network {
public:
    enum class Format {
        HWC,
        CHW
    };

    enum ElemWiseOp {
        Sum
    };

#ifdef N2D2_STM32_EMULATOR
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> Tick_T;
#else
    typedef uint32_t Tick_T;
#endif
    typedef struct {
        double mean;
        unsigned long long int count;
    } RunningMean_T;

    template<typename Input_T, typename Output_T>
    void propagate(const Input_T* inputs, Output_T* outputs) const;

    std::size_t inputHeight() const;
    std::size_t inputWidth() const;
    std::size_t inputNbChannels() const;
    std::size_t inputSize() const;

    std::size_t outputHeight(std::size_t index = 0) const;
    std::size_t outputWidth(std::size_t index = 0) const;
    std::size_t outputNbOutputs(std::size_t index = 0) const;
    std::size_t outputSize(std::size_t index = 0) const;

private:
    template<// For all inputs
            int NB_INPUTS,
            int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
            int NB_OUTPUTS,
            int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
            int OUTPUT_MEM_CONT_OFFSET,
            int OUTPUT_MEM_CONT_SIZE,
            int OUTPUT_MEM_WRAP_OFFSET,
            int OUTPUT_MEM_WRAP_SIZE,
            int OUTPUT_MEM_STRIDE,
            // For first input
            int INPUT_NB_CHANNELS,
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // For next inputs
            int... ARGS,
            typename... INPUTS,
            // Types
            typename Input_T, typename Output_T>
    N2D2_ALWAYS_INLINE void concatenatePropagate(
        Output_T* __restrict outputs,
        const Input_T* __restrict firstInputs,
        INPUTS... inputs) const;

    template<// For all inputs
            int NB_INPUTS,
            int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
            int NB_OUTPUTS,
            int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
            N2D2_Export::Network::ElemWiseOp ELEM_OP,
            ActivationFunction_T ACTIVATION,
            int OUTPUT_MEM_CONT_OFFSET,
            int OUTPUT_MEM_CONT_SIZE,
            int OUTPUT_MEM_WRAP_OFFSET,
            int OUTPUT_MEM_WRAP_SIZE,
            int OUTPUT_MEM_STRIDE,
            // For first input
            int INPUT_NB_CHANNELS,
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // For next inputs
            int... ARGS,
            typename... INPUTS,
            // Types
            typename Input_T, typename Output_T,
            typename Rescaling_T>
    N2D2_ALWAYS_INLINE void elemWisePropagate(
        Output_T* __restrict outputs,
        const Rescaling_T& __restrict rescaling,
        const Input_T* __restrict firstInputs,
        INPUTS... inputs) const;

    /**
     * inputs[CHANNELS_HEIGHT*CHANNELS_WIDTH*NB_CHANNELS]
     * outputs[OUTPUTS_HEIGHT*OUTPUTS_WIDTH*NB_OUTPUTS]
     * biasses[NB_OUTPUTS]
     * weights[NB_OUTPUTS*KERNEL_HEIGHT*KERNEL_WIDTH*NB_CHANNELS]
     */
    template<int NB_CHANNELS, 
            int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
            int NB_OUTPUTS,
            int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
            int PADDING_Y, int PADDING_X,
            int STRIDE_Y, int STRIDE_X,
            int KERNEL_HEIGHT, int KERNEL_WIDTH,
            ActivationFunction_T ACTIVATION,
            // Memory mapping: inputs
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // Memory mapping: outputs
            int OUTPUT_MEM_CONT_OFFSET,
            int OUTPUT_MEM_CONT_SIZE,
            int OUTPUT_MEM_WRAP_OFFSET,
            int OUTPUT_MEM_WRAP_SIZE,
            int OUTPUT_MEM_STRIDE,
            typename Input_T, typename Output_T,
            typename Weight_T, typename Bias_T,
            typename Rescaling_T>
    N2D2_ALWAYS_INLINE void convcellPropagate(
        const Input_T* __restrict inputs,
        Output_T* __restrict outputs,
        const Bias_T* __restrict biasses,
        const Weight_T* __restrict weights,
        const Rescaling_T& __restrict rescaling) const;

    /*
     * inputs[CHANNELS_HEIGHT*CHANNELS_WIDTH*NB_CHANNELS]
     * outputs[OUTPUTS_HEIGHT*OUTPUTS_WIDTH*NB_OUTPUTS]
     * biasses[NB_OUTPUTS]
     * weights[NB_OUTPUTS*KERNEL_HEIGHT*KERNEL_WIDTH]
     */
    template<int NB_CHANNELS, 
            int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
            int NB_OUTPUTS,
            int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
            int PADDING_Y, int PADDING_X,
            int STRIDE_Y, int STRIDE_X,
            int KERNEL_HEIGHT, int KERNEL_WIDTH,
            ActivationFunction_T ACTIVATION,
            // Memory mapping: inputs
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // Memory mapping: outputs
            int OUTPUT_MEM_CONT_OFFSET,
            int OUTPUT_MEM_CONT_SIZE,
            int OUTPUT_MEM_WRAP_OFFSET,
            int OUTPUT_MEM_WRAP_SIZE,
            int OUTPUT_MEM_STRIDE,
            typename Input_T, typename Output_T,
            typename Weight_T, typename Bias_T,
            typename Rescaling_T>
    N2D2_ALWAYS_INLINE void convcellDWPropagate(
        const Input_T* __restrict inputs,
        Output_T* __restrict outputs,
        const Bias_T* __restrict biasses,
        const Weight_T* __restrict weights,
        const Rescaling_T& __restrict rescaling) const;

    /**
     * inputs[CHANNELS_HEIGHT*CHANNELS_WIDTH*NB_CHANNELS]
     * outputs[OUTPUTS_HEIGHT*OUTPUTS_WIDTH*NB_OUTPUTS]
     */
    template<int NB_CHANNELS, 
            int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
            int NB_OUTPUTS,
            int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
            int PADDING_Y, int PADDING_X,
            int STRIDE_Y, int STRIDE_X,
            int POOL_HEIGHT, int POOL_WIDTH,
            Pooling_T POOLING_TYPE,
            ActivationFunction_T ACTIVATION,
            // Memory mapping: inputs
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // Memory mapping: outputs
            int OUTPUT_MEM_CONT_OFFSET,
            int OUTPUT_MEM_CONT_SIZE,
            int OUTPUT_MEM_WRAP_OFFSET,
            int OUTPUT_MEM_WRAP_SIZE,
            int OUTPUT_MEM_STRIDE,
            typename Input_T, typename Output_T>
    N2D2_ALWAYS_INLINE void poolcellPropagate(
        const Input_T* __restrict inputs,
        Output_T* __restrict outputs) const;

    template<int NB_CHANNELS, 
            int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
            int NB_OUTPUTS,
            int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
            ActivationFunction_T ACTIVATION,
            // Memory mapping: inputs
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // Memory mapping: outputs
            int OUTPUT_MEM_CONT_OFFSET,
            int OUTPUT_MEM_CONT_SIZE,
            int OUTPUT_MEM_WRAP_OFFSET,
            int OUTPUT_MEM_WRAP_SIZE,
            int OUTPUT_MEM_STRIDE,
            typename Input_T, typename Output_T,
            typename Weight_T, typename Bias_T,
            typename Rescaling_T>
    N2D2_ALWAYS_INLINE void fccellPropagate(
        const Input_T* __restrict inputs,
        Output_T* __restrict outputs,
        const Bias_T* __restrict biasses,
        const Weight_T* __restrict weights,
        const Rescaling_T& __restrict rescaling) const;

    template<int NB_CHANNELS, 
            int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
            int NB_OUTPUTS,
            int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
            // Memory mapping: inputs
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // Memory mapping: outputs
            int OUTPUT_MEM_CONT_OFFSET,
            int OUTPUT_MEM_CONT_SIZE,
            int OUTPUT_MEM_WRAP_OFFSET,
            int OUTPUT_MEM_WRAP_SIZE,
            int OUTPUT_MEM_STRIDE,
            typename Input_T, typename Output_T>
    N2D2_ALWAYS_INLINE void resizeNearestNeighborPropagate(
        const Input_T* __restrict inputs,
        Output_T* __restrict outputs) const;

    template<int NB_CHANNELS, 
            int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
            int NB_OUTPUTS,
            int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
            // Memory mapping: inputs
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // Memory mapping: outputs
            int OUTPUT_MEM_CONT_OFFSET,
            int OUTPUT_MEM_CONT_SIZE,
            int OUTPUT_MEM_WRAP_OFFSET,
            int OUTPUT_MEM_WRAP_SIZE,
            int OUTPUT_MEM_STRIDE,
            typename Input_T, typename Output_T,
            typename Rescaling_T>
    N2D2_ALWAYS_INLINE void scalingPropagate(
        const Input_T* __restrict inputs,
        Output_T* __restrict outputs,
        const Rescaling_T& __restrict rescaling) const;

    template<int NB_CHANNELS, 
            int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
            int NB_OUTPUTS,
            int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
            // Memory mapping: inputs
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // Memory mapping: outputs
            int OUTPUT_MEM_CONT_OFFSET,
            int OUTPUT_MEM_CONT_SIZE,
            int OUTPUT_MEM_WRAP_OFFSET,
            int OUTPUT_MEM_WRAP_SIZE,
            int OUTPUT_MEM_STRIDE,
            typename Input_T, typename Output_T>
    N2D2_ALWAYS_INLINE void transposePropagate(
        const Input_T* __restrict inputs,
        Output_T* __restrict outputs,
        const int perm[4]) const;

    template<int NB_CHANNELS,
            int INPUTS_HEIGHT, int INPUTS_WIDTH,
            // Memory mapping: outputs
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            typename Input_T>
    N2D2_ALWAYS_INLINE void maxPropagate(
        const Input_T* __restrict inputs,
        int32_t* __restrict outputs) const;

#ifdef N2D2_STM32_EMULATOR
    template<typename Output_T>
    void saveOutputs(
        int NB_OUTPUTS,
        int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
        int OUTPUT_MEM_CONT_OFFSET,
        int OUTPUT_MEM_CONT_SIZE,
        int OUTPUT_MEM_WRAP_OFFSET,
        int OUTPUT_MEM_WRAP_SIZE,
        int OUTPUT_MEM_STRIDE,
        const Output_T* __restrict outputs,
        FILE* pFile,
        Format format) const;
#endif

private:
#ifdef N2D2_STM32_EMULATOR
    mutable std::map<std::string, double> cumulativeTiming;
#endif

    template<typename Output_T>
    N2D2_ALWAYS_INLINE void concatenate(
        Output_T* __restrict /*outputs*/,
        int /*pos*/) const;

    template<// For first input
            int INPUT_NB_CHANNELS,
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // For next inputs
            int... ARGS,
            typename... INPUTS,
            // Types
            typename Input_T, typename Output_T>
    N2D2_ALWAYS_INLINE void concatenate(
        Output_T* __restrict outputs,
        int pos,
        const Input_T* __restrict firstInputs,
        INPUTS... inputs) const;

    template<ElemWiseOp ELEM_OP>
    N2D2_ALWAYS_INLINE SUM_T elemWise(
        int /*pos*/,
        int /*ch*/) const;

    template<ElemWiseOp ELEM_OP,
            // For first input
            int INPUT_NB_CHANNELS,
            int INPUT_MEM_CONT_OFFSET,
            int INPUT_MEM_CONT_SIZE,
            int INPUT_MEM_WRAP_OFFSET,
            int INPUT_MEM_WRAP_SIZE,
            int INPUT_MEM_STRIDE,
            // For next inputs
            int... ARGS,
            typename... INPUTS,
            // Types
            typename Input_T>
    N2D2_ALWAYS_INLINE SUM_T elemWise(
        int pos,
        int ch,
        const Input_T* __restrict firstInputs,
        INPUTS... inputs) const;

    template<typename T>
    N2D2_ALWAYS_INLINE static T clamp(T v, T lo, T hi) {
        if(v < lo) {
            return lo;
        }
        else if(v > hi) {
            return hi;
        }
        else {
            return v;
        }
    }
    
    template<typename T>
    N2D2_ALWAYS_INLINE static T max(T lhs, T rhs) {
        return (lhs >= rhs)?lhs:rhs;
    }

    template<typename Output_T, typename Rescaling_T>
    N2D2_ALWAYS_INLINE static Output_T sat(SUM_T weightedSum, int output, 
                                           ActivationFunction_T func, 
                                           const Rescaling_T& __restrict rescaling) 
    {
        switch(func) {
            case Linear:
            case Saturation: {
                break;
            }
            case Rectifier: {
                if(weightedSum <= 0) weightedSum = 0;
                break;
            }
            default:
                N2D2_THROW_OR_ABORT(std::runtime_error, "Unsupported activation function.");
        }

        return saturate<Output_T>(rescaling(weightedSum, output), std::numeric_limits<Output_T>::digits);
    }

#ifndef N2D2_STM32_EMULATOR
    template<typename Input_T>
    N2D2_ALWAYS_INLINE static std::uint32_t XTB16(std::uint32_t val) {
        return std::is_unsigned<Input_T>::value?__UXTB16(val):__SXTB16(val);
    }
#endif

    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T>
    N2D2_ALWAYS_INLINE static SUM_T dualMac(const Input_T* __restrict inputs, 
                                            const Weight_T* __restrict weights, 
                                            SUM_T weightedSum) 
    {
        weightedSum += inputs[0] * weights[0]
         + inputs[INPUTS_INC] * weights[WEIGHTS_INC];

        return weightedSum;
    }

    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T,
             typename std::enable_if<std::is_floating_point<Input_T>::value>::type* = nullptr>
    N2D2_ALWAYS_INLINE static SUM_T quadMac(const Input_T* __restrict inputs, 
                                            const Weight_T* __restrict weights, 
                                            SUM_T weightedSum) 
    {
        weightedSum += inputs[0*INPUTS_INC] * weights[0*WEIGHTS_INC]
         + inputs[1*INPUTS_INC] * weights[1*WEIGHTS_INC]
         + inputs[2*INPUTS_INC] * weights[2*WEIGHTS_INC]
         + inputs[3*INPUTS_INC] * weights[3*WEIGHTS_INC];

        return weightedSum;
    }
#if NB_BITS > 0
    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T, 
             typename Weight_T,
             typename std::enable_if<!std::is_floating_point<Input_T>::value>::type* = nullptr>
    N2D2_ALWAYS_INLINE static SUM_T quadMac(const Input_T* __restrict inputs, 
                                            const Weight_T* __restrict weights, 
                                            SUM_T weightedSum) 
    {
#ifdef N2D2_STM32_EMULATOR
        weightedSum += inputs[0*INPUTS_INC] * weights[0*WEIGHTS_INC]
         + inputs[1*INPUTS_INC] * weights[1*WEIGHTS_INC]
         + inputs[2*INPUTS_INC] * weights[2*WEIGHTS_INC]
         + inputs[3*INPUTS_INC] * weights[3*WEIGHTS_INC];

        return weightedSum;
#else
        if(INPUTS_INC != 1 || WEIGHTS_INC != 1) {
            weightedSum += inputs[0*INPUTS_INC] * weights[0*WEIGHTS_INC]
             + inputs[1*INPUTS_INC] * weights[1*WEIGHTS_INC]
             + inputs[2*INPUTS_INC] * weights[2*WEIGHTS_INC]
             + inputs[3*INPUTS_INC] * weights[3*WEIGHTS_INC];

            return weightedSum;
        }
/*
        static_assert(std::is_same<Input_T, data<8> >::value || 
                      std::is_same<Input_T, udata<8> >::value, "");
        static_assert((std::is_same<Weight_T, data<8> >::value));
*/

        std::uint32_t in;
        std::memcpy((void*) &in, inputs, sizeof(in));
        
        std::uint32_t in1 = XTB16<Input_T>(in);
        std::uint32_t in2 = XTB16<Input_T>(in >> 8);
        
        
        std::uint32_t wt;
        std::memcpy((void*) &wt, weights, sizeof(wt));
        
        std::uint32_t wt1 = XTB16<Weight_T>(wt);
        std::uint32_t wt2 = XTB16<Weight_T>(wt >> 8);
        
        
        static_assert((std::is_same<SUM_T, std::int32_t>::value || 
                       std::is_same<SUM_T, std::int64_t>::value), 
                      "SUM_T must be an int32_t or an int64_t.");
        
        if(std::is_same<SUM_T, std::int32_t>::value) {
            weightedSum = __SMLAD(in1, wt1, weightedSum);
            weightedSum = __SMLAD(in2, wt2, weightedSum);
        }
        else {
            weightedSum = __SMLALD(in1, wt1, weightedSum);
            weightedSum = __SMLALD(in2, wt2, weightedSum);
            
        }
        
        return weightedSum;
#endif
    }
#endif
    template<typename Output_T, typename T,  
             typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Output_T saturate(T value, std::int32_t sat) {
        return value;
    }

    template<typename Output_T, typename T,  
             typename std::enable_if<!std::is_floating_point<T>::value>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Output_T saturate(T value, std::uint32_t sat) {
#ifdef N2D2_STM32_EMULATOR
        return std::is_unsigned<Output_T>::value?clamp(value, T(0), (T(1) << sat) - 1):
                                                 clamp(value, -(T(1) << (sat - 1)), 
                                                               (T(1) << (sat - 1)) - 1);
#else
        static_assert(std::is_same<T, std::int32_t>::value || 
                      std::is_same<T, std::uint32_t>::value, "");
        
        return std::is_unsigned<Output_T>::value?__USAT(value, sat):__SSAT(value, sat);
#endif
    }

    template<typename T,  
             typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    N2D2_ALWAYS_INLINE static T threshold() {
        return 0.0;
    }

    template<typename T,  
             typename std::enable_if<!std::is_floating_point<T>::value>::type* = nullptr>
    N2D2_ALWAYS_INLINE static T threshold() {
        return (std::is_unsigned<T>::value)
            ? std::numeric_limits<T>::max() / 2 : 0;
    }

    /**
     * Multiply-accumulate the values in inputs and weights for NB_ITERATIONS.
     * Using an explicit unrolling as unrolling a for-loop with '#pragma GCC unroll NB_ITERATIONS'
     * is slower (TODO check why).
     */
    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             class Input_T, 
             class Weight_T,
             typename std::enable_if<(NB_ITERATIONS == 0)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict /*inputs*/, 
                                               const Weight_T* __restrict /*weights*/, 
                                               SUM_T& __restrict /*weightedSum*/) 
    {
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             class Input_T, 
             class Weight_T,
             typename std::enable_if<(NB_ITERATIONS == 1)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs, 
                                               const Weight_T* __restrict weights, 
                                               SUM_T& __restrict weightedSum) 
    {
        weightedSum += (*weights) * (*inputs);
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             class Input_T, 
             class Weight_T,
             typename std::enable_if<(NB_ITERATIONS >= 2 && NB_ITERATIONS < 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs, 
                                               const Weight_T* __restrict weights, 
                                               SUM_T& __restrict weightedSum) 
    {
        weightedSum = dualMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
        macsOnRange<NB_ITERATIONS - 2, INPUTS_INC, WEIGHTS_INC>(inputs + 2*INPUTS_INC, 
                                       weights + 2*WEIGHTS_INC, weightedSum);
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             class Input_T, 
             class Weight_T,
             typename std::enable_if<(NB_ITERATIONS >= 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs, 
                                               const Weight_T* __restrict weights, 
                                               SUM_T& __restrict weightedSum) 
    {
        weightedSum = quadMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
        macsOnRange<NB_ITERATIONS - 4, INPUTS_INC, WEIGHTS_INC>(inputs + 4*INPUTS_INC, 
                                       weights + 4*WEIGHTS_INC, weightedSum);
    }

    N2D2_ALWAYS_INLINE Tick_T tick() const;
    N2D2_ALWAYS_INLINE void benchmark(const char* name,
                                      const Tick_T& start,
                                      const Tick_T& end,
                                      RunningMean_T& timing) const;
};
}

template<typename Output_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::concatenate(
    Output_T* __restrict /*outputs*/,
    int /*pos*/) const {}

template<// For first input
         int INPUT_NB_CHANNELS,
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // For next inputs
         int... ARGS,
         typename... INPUTS,
         // Types
         typename Input_T, typename Output_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::concatenate(
    Output_T* __restrict outputs,
    int pos,
    const Input_T* __restrict firstInputs,
    INPUTS... inputs) const
{
    int iOffset = INPUT_MEM_STRIDE * pos;

    if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
        iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                    - INPUT_MEM_CONT_SIZE;
    }

    for (int ch = 0; ch < INPUT_NB_CHANNELS; ++ch)
        outputs[ch] = firstInputs[iOffset + ch];

    concatenate<ARGS...>(outputs + INPUT_NB_CHANNELS, pos, inputs...);
}

template<// For all inputs
         int NB_INPUTS,
         int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
         int NB_OUTPUTS,
         int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
         int OUTPUT_MEM_CONT_OFFSET,
         int OUTPUT_MEM_CONT_SIZE,
         int OUTPUT_MEM_WRAP_OFFSET,
         int OUTPUT_MEM_WRAP_SIZE,
         int OUTPUT_MEM_STRIDE,
         // For first input
         int INPUT_NB_CHANNELS,
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // For next inputs
         int... ARGS,
         typename... INPUTS,
         // Types
         typename Input_T, typename Output_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::concatenatePropagate(
    Output_T* __restrict outputs,
    const Input_T* __restrict firstInputs,
    INPUTS... inputs) const
{
    for (int oy = 0; oy < OUTPUTS_HEIGHT; oy++) {
        for (int ox = 0; ox < OUTPUTS_WIDTH; ox++) {
            const int pos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * pos;

            if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
                oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                            - OUTPUT_MEM_CONT_SIZE;
            }

            concatenate<INPUT_NB_CHANNELS,
                        INPUT_MEM_CONT_OFFSET,
                        INPUT_MEM_CONT_SIZE,
                        INPUT_MEM_WRAP_OFFSET,
                        INPUT_MEM_WRAP_SIZE,
                        INPUT_MEM_STRIDE,
                        ARGS...>(outputs + oOffset, pos, firstInputs, inputs...);
        }
    }
}

template<N2D2_Export::Network::ElemWiseOp ELEM_OP>
N2D2_ALWAYS_INLINE inline SUM_T N2D2_Export::Network::elemWise(
    int /*pos*/,
    int /*ch*/) const
{
    return 0;
}

template<N2D2_Export::Network::ElemWiseOp ELEM_OP,
         // For first input
         int INPUT_NB_CHANNELS,
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // For next inputs
         int... ARGS,
         typename... INPUTS,
         // Types
         typename Input_T>
N2D2_ALWAYS_INLINE inline SUM_T N2D2_Export::Network::elemWise(
    int pos,
    int ch,
    const Input_T* __restrict firstInputs,
    INPUTS... inputs) const
{
    int iOffset = INPUT_MEM_STRIDE * pos;

    if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
        iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                    - INPUT_MEM_CONT_SIZE;
    }

    return firstInputs[iOffset + ch]
                + elemWise<ELEM_OP, ARGS...>(pos, ch, inputs...);
}

template<// For all inputs
         int NB_INPUTS,
         int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
         int NB_OUTPUTS,
         int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
         N2D2_Export::Network::ElemWiseOp ELEM_OP,
         ActivationFunction_T ACTIVATION,
         int OUTPUT_MEM_CONT_OFFSET,
         int OUTPUT_MEM_CONT_SIZE,
         int OUTPUT_MEM_WRAP_OFFSET,
         int OUTPUT_MEM_WRAP_SIZE,
         int OUTPUT_MEM_STRIDE,
         // For first input
         int INPUT_NB_CHANNELS,
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // For next inputs
         int... ARGS,
         typename... INPUTS,
         // Types
         typename Input_T, typename Output_T,
         typename Rescaling_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::elemWisePropagate(
    Output_T* __restrict outputs,
    const Rescaling_T& __restrict rescaling,
    const Input_T* __restrict firstInputs,
    INPUTS... inputs) const
{
    static_assert(NB_INPUTS > 0, "Number of inputs must be > 0");
    static_assert(ELEM_OP == Sum, "Only Sum is supported");
    static_assert(INPUT_NB_CHANNELS == NB_OUTPUTS,
        "Number of channels and number of outputs must match");

    for (int oy = 0; oy < OUTPUTS_HEIGHT; oy++) {
        for (int ox = 0; ox < OUTPUTS_WIDTH; ox++) {
            const int pos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * pos;

            if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
                oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                            - OUTPUT_MEM_CONT_SIZE;
            }

            for (int ch = 0; ch < NB_OUTPUTS; ++ch) {
                const SUM_T val = elemWise<ELEM_OP,
                                        INPUT_NB_CHANNELS,
                                        INPUT_MEM_CONT_OFFSET,
                                        INPUT_MEM_CONT_SIZE,
                                        INPUT_MEM_WRAP_OFFSET,
                                        INPUT_MEM_WRAP_SIZE,
                                        INPUT_MEM_STRIDE,
                                        ARGS...>(pos, ch, firstInputs, inputs...);

                outputs[oOffset + ch]
                    = sat<Output_T>(val, ch, ACTIVATION, rescaling);
            }
        }
    }
}

template<int NB_CHANNELS, 
         int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
         int NB_OUTPUTS,
         int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
         int PADDING_Y, int PADDING_X,
         int STRIDE_Y, int STRIDE_X,
         int KERNEL_HEIGHT, int KERNEL_WIDTH,
         ActivationFunction_T ACTIVATION,
         // Memory mapping: inputs
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // Memory mapping: outputs
         int OUTPUT_MEM_CONT_OFFSET,
         int OUTPUT_MEM_CONT_SIZE,
         int OUTPUT_MEM_WRAP_OFFSET,
         int OUTPUT_MEM_WRAP_SIZE,
         int OUTPUT_MEM_STRIDE,
         typename Input_T, typename Output_T,
         typename Weight_T, typename Bias_T,
         typename Rescaling_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::convcellPropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs,
    const Bias_T* __restrict biasses,
    const Weight_T* __restrict weights,
    const Rescaling_T& __restrict rescaling) const
{
    constexpr int OUTPUTS_HEIGHT_NOPAD
        = (CHANNELS_HEIGHT - KERNEL_HEIGHT + STRIDE_Y) / STRIDE_Y;
    constexpr int OUTPUTS_WIDTH_NOPAD
        = (CHANNELS_WIDTH - KERNEL_WIDTH + STRIDE_X) / STRIDE_X;

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        const int syMin = (PADDING_Y == 0) ? 0
            : max(PADDING_Y - (oy * STRIDE_Y), 0);
        const int syMax = (PADDING_Y == 0
                && OUTPUTS_HEIGHT == OUTPUTS_HEIGHT_NOPAD) ? KERNEL_HEIGHT
            : clamp(CHANNELS_HEIGHT + PADDING_Y - (oy * STRIDE_Y), 
                    0, KERNEL_HEIGHT);
        const int iy = (oy * STRIDE_Y) - PADDING_Y;

        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            const int sxMin = (PADDING_X == 0) ? 0
                : max(PADDING_X - (ox * STRIDE_X), 0);
            const int sxMax = (PADDING_X == 0
                    && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD) ? KERNEL_WIDTH
                : clamp(CHANNELS_WIDTH + PADDING_X - (ox * STRIDE_X), 
                        0, KERNEL_WIDTH);
            const int ix = (ox * STRIDE_X) - PADDING_X;

            const int oPos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * oPos;

            if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
                oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                            - OUTPUT_MEM_CONT_SIZE;
            }

            for (int output = 0; output < NB_OUTPUTS; ++output) {
                SUM_T weightedSum = biasses[output];

                for (int sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                    if ((PADDING_Y != 0
                            || OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD)
                        && sy >= syMax - syMin)
                    {
                        break;
                    }

                    const int iPos = ((sxMin + ix)
                                        + CHANNELS_WIDTH * (iy + syMin + sy));
                    int iOffset = INPUT_MEM_STRIDE * iPos;

                    // Wrapping cannot occur in the middle of a line, except if
                    // there is only one line (1D)!
                    bool wrapInRange = false;

                    if (INPUT_MEM_WRAP_SIZE > 0
                        && iOffset >= INPUT_MEM_CONT_SIZE)
                    {
                        iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                                    - INPUT_MEM_CONT_SIZE;
                    }
                    else if (INPUT_MEM_WRAP_SIZE > 0 && KERNEL_WIDTH > 1
                        && CHANNELS_HEIGHT == 1 // single line (1D)!
                        && iOffset + KERNEL_WIDTH * NB_CHANNELS
                            > INPUT_MEM_CONT_SIZE)
                    {
                        wrapInRange = true;
                    }

                    const int wOffset = NB_CHANNELS * (sxMin
                        + KERNEL_WIDTH * (syMin + sy + KERNEL_HEIGHT * output));

                    if (!wrapInRange && (NB_CHANNELS == INPUT_MEM_STRIDE
                        && ((PADDING_X == 0
                            && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                                || sxMax - sxMin == KERNEL_WIDTH)))
                    {
                        macsOnRange<KERNEL_WIDTH * NB_CHANNELS>(
                            inputs + iOffset, 
                            weights + wOffset, 
                            weightedSum);
                    }
                    else {
                        for (int sx = 0; sx < KERNEL_WIDTH; ++sx) {
                            if ((PADDING_X != 0
                                    || OUTPUTS_WIDTH != OUTPUTS_WIDTH_NOPAD)
                                && sx >= sxMax - sxMin)
                            {
                                break;
                            }

                            int iOffsetInRange = iOffset
                                + sx * INPUT_MEM_STRIDE;

                            if (wrapInRange
                                && iOffsetInRange >= INPUT_MEM_CONT_SIZE)
                            {
                                iOffsetInRange += INPUT_MEM_WRAP_OFFSET
                                            - INPUT_MEM_CONT_OFFSET
                                            - INPUT_MEM_CONT_SIZE;
                            }

                            macsOnRange<NB_CHANNELS>(
                                // same input line so no wrapping can occur
                                inputs + iOffsetInRange, 
                                weights + wOffset + sx * NB_CHANNELS, 
                                weightedSum);
                        }
                    }
                }

                outputs[output + oOffset]
                    = sat<Output_T>(weightedSum, output, ACTIVATION, rescaling);
            }
        }
    }
}

template<int NB_CHANNELS, 
         int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
         int NB_OUTPUTS,
         int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
         int PADDING_Y, int PADDING_X,
         int STRIDE_Y, int STRIDE_X,
         int KERNEL_HEIGHT, int KERNEL_WIDTH,
         ActivationFunction_T ACTIVATION,
         // Memory mapping: inputs
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // Memory mapping: outputs
         int OUTPUT_MEM_CONT_OFFSET,
         int OUTPUT_MEM_CONT_SIZE,
         int OUTPUT_MEM_WRAP_OFFSET,
         int OUTPUT_MEM_WRAP_SIZE,
         int OUTPUT_MEM_STRIDE,
         typename Input_T, typename Output_T,
         typename Weight_T, typename Bias_T,
         typename Rescaling_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::convcellDWPropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs,
    const Bias_T* __restrict biasses,
    const Weight_T* __restrict weights,
    const Rescaling_T& __restrict rescaling) const
{
    static_assert(NB_OUTPUTS % NB_CHANNELS == 0,
        "NB_OUTPUTS should be a multiple of NB_CHANNELS.");

    constexpr int OUTPUTS_HEIGHT_NOPAD
        = (CHANNELS_HEIGHT - KERNEL_HEIGHT + STRIDE_Y) / STRIDE_Y;
    constexpr int OUTPUTS_WIDTH_NOPAD
        = (CHANNELS_WIDTH - KERNEL_WIDTH + STRIDE_X) / STRIDE_X;

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        const int syMin = (PADDING_Y == 0) ? 0
            : max(PADDING_Y - (oy * STRIDE_Y), 0);
        const int syMax = (PADDING_Y == 0
                && OUTPUTS_HEIGHT == OUTPUTS_HEIGHT_NOPAD) ? KERNEL_HEIGHT
            : clamp(CHANNELS_HEIGHT + PADDING_Y - (oy * STRIDE_Y), 
                    0, KERNEL_HEIGHT);
        const int iy = (oy * STRIDE_Y) - PADDING_Y;

        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            const int sxMin = (PADDING_X == 0) ? 0
                : max(PADDING_X - (ox * STRIDE_X), 0);
            const int sxMax = (PADDING_X == 0
                    && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                        ? KERNEL_WIDTH
                : clamp(CHANNELS_WIDTH + PADDING_X - (ox * STRIDE_X), 
                        0, KERNEL_WIDTH);
            const int ix = (ox * STRIDE_X) - PADDING_X;

            const int oPos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * oPos;

            if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
                oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                            - OUTPUT_MEM_CONT_SIZE;
            }

            for (int output = 0; output < NB_OUTPUTS; ++output) {
                const int channel = (output * NB_CHANNELS) / NB_OUTPUTS;

                SUM_T weightedSum = biasses[output];

                for (int sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                    if ((PADDING_Y != 0
                            || OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD)
                        && sy >= syMax - syMin)
                    {
                        break;
                    }

                    const int iPos = ((sxMin + ix)
                                        + CHANNELS_WIDTH * (iy + syMin + sy));
                    int iOffset = INPUT_MEM_STRIDE * iPos;

                    // Wrapping cannot occur in the middle of a line, except if
                    // there is only one line (1D)!
                    bool wrapInRange = false;

                    if (INPUT_MEM_WRAP_SIZE > 0
                        && iOffset >= INPUT_MEM_CONT_SIZE)
                    {
                        iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                                    - INPUT_MEM_CONT_SIZE;
                    }
                    else if (INPUT_MEM_WRAP_SIZE > 0 && KERNEL_WIDTH > 1
                        && CHANNELS_HEIGHT == 1 // single line (1D)!
                        && iOffset + KERNEL_WIDTH * INPUT_MEM_STRIDE
                            > INPUT_MEM_CONT_SIZE)
                    {
                        wrapInRange = true;
                    }

                    const int wOffset = (sxMin
                        + KERNEL_WIDTH * (syMin + sy + KERNEL_HEIGHT * output));

                    if (!wrapInRange && ((PADDING_X == 0
                            && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                        || sxMax - sxMin == KERNEL_WIDTH))
                    {
                        macsOnRange<KERNEL_WIDTH, INPUT_MEM_STRIDE>(
                            inputs + iOffset + channel, 
                            weights + wOffset, 
                            weightedSum);
                    }
                    else {
                        for (int sx = 0; sx < KERNEL_WIDTH; ++sx) {
                            if ((PADDING_X != 0
                                    || OUTPUTS_WIDTH != OUTPUTS_WIDTH_NOPAD)
                                && sx >= sxMax - sxMin)
                            {
                                break;
                            }

                            int iOffsetInRange = iOffset
                                + sx * INPUT_MEM_STRIDE;

                            if (wrapInRange &&
                                iOffsetInRange >= INPUT_MEM_CONT_SIZE)
                            {
                                iOffsetInRange += INPUT_MEM_WRAP_OFFSET
                                            - INPUT_MEM_CONT_OFFSET
                                            - INPUT_MEM_CONT_SIZE;
                            }

                            weightedSum += inputs[channel + iOffsetInRange]
                                * weights[wOffset + sx];
                        }
                    }
                }

                outputs[output + oOffset]
                    = sat<Output_T>(weightedSum, output, ACTIVATION, rescaling);
            }
        }
    }
}

template<int NB_CHANNELS, 
         int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
         int NB_OUTPUTS,
         int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
         int PADDING_Y, int PADDING_X,
         int STRIDE_Y, int STRIDE_X,
         int POOL_HEIGHT, int POOL_WIDTH,
         Pooling_T POOLING_TYPE,
         ActivationFunction_T ACTIVATION,
         // Memory mapping: inputs
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // Memory mapping: outputs
         int OUTPUT_MEM_CONT_OFFSET,
         int OUTPUT_MEM_CONT_SIZE,
         int OUTPUT_MEM_WRAP_OFFSET,
         int OUTPUT_MEM_WRAP_SIZE,
         int OUTPUT_MEM_STRIDE,
         typename Input_T, typename Output_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::poolcellPropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs) const
{
    static_assert(std::is_same<Input_T, Output_T>::value,
        "Input_T and Output_T must be the same.");
    static_assert(NB_CHANNELS == NB_OUTPUTS,
        "NB_CHANNELS should be equal to NB_OUTPUTS.");
    static_assert(POOLING_TYPE == Max || POOLING_TYPE == Average,
        "The export only supports Max and Average pooling.");
    static_assert(ACTIVATION == Linear,
        "The export only supports a Linear activation.");

    constexpr int OUTPUTS_HEIGHT_NOPAD
        = (CHANNELS_HEIGHT - POOL_HEIGHT + STRIDE_Y) / STRIDE_Y;
    constexpr int OUTPUTS_WIDTH_NOPAD
        = (CHANNELS_WIDTH - POOL_WIDTH + STRIDE_X) / STRIDE_X;

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        const int syMin = (PADDING_Y == 0) ? 0
            : max(PADDING_Y - (oy * STRIDE_Y), 0);
        const int syMax = (PADDING_Y == 0
                && OUTPUTS_HEIGHT == OUTPUTS_HEIGHT_NOPAD) ? POOL_HEIGHT
            : clamp(CHANNELS_HEIGHT + PADDING_Y - (oy * STRIDE_Y), 
                    0, POOL_HEIGHT);
        const int iy = (oy * STRIDE_Y) - PADDING_Y;

        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            const int sxMin = (PADDING_X == 0) ? 0
                : max(PADDING_X - (ox * STRIDE_X), 0);
            const int sxMax = (PADDING_X == 0
                    && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                        ? POOL_WIDTH
                : clamp(CHANNELS_WIDTH + PADDING_X - (ox * STRIDE_X), 
                        0, POOL_WIDTH);
            const int ix = (ox * STRIDE_X) - PADDING_X;

            const int oPos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * oPos;

            if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
                oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                            - OUTPUT_MEM_CONT_SIZE;
            }

            for (int output = 0; output < NB_OUTPUTS; ++output) {
                if (POOLING_TYPE == Max) {
                    Input_T maxVal = std::numeric_limits<Input_T>::lowest();

                    for (int sy = 0; sy < POOL_HEIGHT; ++sy) {
                        if ((PADDING_Y != 0
                                || OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD)
                            && sy >= syMax - syMin)
                        {
                            break;
                        }

                        const int iPos = ((sxMin + ix)
                                            + CHANNELS_WIDTH * (iy + syMin + sy));
                        int iOffset = INPUT_MEM_STRIDE * iPos;

                        // Wrapping cannot occur in the middle of a line, except if
                        // there is only one line (1D)!
                        bool wrapInRange = false;

                        if (INPUT_MEM_WRAP_SIZE > 0
                            && iOffset >= INPUT_MEM_CONT_SIZE)
                        {
                            iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                                        - INPUT_MEM_CONT_SIZE;
                        }
                        else if (INPUT_MEM_WRAP_SIZE > 0 && POOL_WIDTH > 1
                            && CHANNELS_HEIGHT == 1 // single line (1D)!
                            && iOffset + POOL_WIDTH * INPUT_MEM_STRIDE
                                > INPUT_MEM_CONT_SIZE)
                        {
                            wrapInRange = true;
                        }

                        for (int sx = 0; sx < POOL_WIDTH; ++sx) {
                            if ((PADDING_X != 0
                                    || OUTPUTS_WIDTH != OUTPUTS_WIDTH_NOPAD)
                                && sx >= sxMax - sxMin)
                            {
                                break;
                            }

                            int iOffsetInRange = iOffset + output
                                + sx * INPUT_MEM_STRIDE;

                            if (wrapInRange &&
                                iOffsetInRange >= INPUT_MEM_CONT_SIZE)
                            {
                                iOffsetInRange += INPUT_MEM_WRAP_OFFSET
                                            - INPUT_MEM_CONT_OFFSET
                                            - INPUT_MEM_CONT_SIZE;
                            }

                            if (inputs[iOffsetInRange] > maxVal)
                                maxVal = inputs[iOffsetInRange];
                        }
                    }

                    outputs[output + oOffset] = maxVal;
                }
                else if (POOLING_TYPE == Average) {
                    SUM_T sum = 0;

                    for (int sy = 0; sy < POOL_HEIGHT; ++sy) {
                        if ((PADDING_Y != 0
                                || OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD)
                            && sy >= syMax - syMin)
                        {
                            break;
                        }

                        const int iPos = ((sxMin + ix)
                                            + CHANNELS_WIDTH * (iy + syMin + sy));
                        int iOffset = INPUT_MEM_STRIDE * iPos;

                        // Wrapping cannot occur in the middle of a line, except if
                        // there is only one line (1D)!
                        bool wrapInRange = false;

                        if (INPUT_MEM_WRAP_SIZE > 0
                            && iOffset >= INPUT_MEM_CONT_SIZE)
                        {
                            iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                                        - INPUT_MEM_CONT_SIZE;
                        }
                        else if (INPUT_MEM_WRAP_SIZE > 0 && POOL_WIDTH > 1
                            && CHANNELS_HEIGHT == 1 // single line (1D)!
                            && iOffset + POOL_WIDTH * INPUT_MEM_STRIDE
                                > INPUT_MEM_CONT_SIZE)
                        {
                            wrapInRange = true;
                        }

                        for (int sx = 0; sx < POOL_WIDTH; ++sx) {
                            if ((PADDING_X != 0
                                    || OUTPUTS_WIDTH != OUTPUTS_WIDTH_NOPAD)
                                && sx >= sxMax - sxMin)
                            {
                                break;
                            }

                            int iOffsetInRange = iOffset + output
                                + sx * INPUT_MEM_STRIDE;

                            if (wrapInRange &&
                                iOffsetInRange >= INPUT_MEM_CONT_SIZE)
                            {
                                iOffsetInRange += INPUT_MEM_WRAP_OFFSET
                                            - INPUT_MEM_CONT_OFFSET
                                            - INPUT_MEM_CONT_SIZE;
                            }

                            sum += inputs[iOffsetInRange];
                        }
                    }

                    outputs[output + oOffset]
                        = (Output_T) (sum / (POOL_HEIGHT * POOL_WIDTH));
                }
                else {
                    N2D2_THROW_OR_ABORT(std::runtime_error,
                        "The export only supports Max and Average pooling.");
                }
            }
        }
    }
}

template<int NB_CHANNELS, 
         int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
         int NB_OUTPUTS,
         int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
         ActivationFunction_T ACTIVATION,
         // Memory mapping: inputs
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // Memory mapping: outputs
         int OUTPUT_MEM_CONT_OFFSET,
         int OUTPUT_MEM_CONT_SIZE,
         int OUTPUT_MEM_WRAP_OFFSET,
         int OUTPUT_MEM_WRAP_SIZE,
         int OUTPUT_MEM_STRIDE,
         typename Input_T, typename Output_T,
         typename Weight_T, typename Bias_T,
         typename Rescaling_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::fccellPropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs,
    const Bias_T* __restrict biasses,
    const Weight_T* __restrict weights,
    const Rescaling_T& __restrict rescaling) const
{
    static_assert(OUTPUTS_HEIGHT == 1, "Outputs height should be 1");
    static_assert(OUTPUTS_WIDTH == 1, "Outputs width should be 1");
    static_assert(OUTPUT_MEM_WRAP_SIZE == 0, "Output wrapping not supported");

    for (int och = 0; och < NB_OUTPUTS; och++) {
        SUM_T weightedSum = biasses[och];

        for (int iy = 0; iy < CHANNELS_HEIGHT; ++iy) {
            const int iPos = (CHANNELS_WIDTH * iy);
            int iOffset = INPUT_MEM_STRIDE * iPos;

            // Wrapping cannot occur in the middle of a line, except if
            // there is only one line (1D)!
            bool wrapInRange = false;

            if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                            - INPUT_MEM_CONT_SIZE;
            }
            else if (INPUT_MEM_WRAP_SIZE > 0 && CHANNELS_WIDTH > 1
                && CHANNELS_HEIGHT == 1 // single line (1D)!
                && iOffset + CHANNELS_WIDTH * NB_CHANNELS
                    > INPUT_MEM_CONT_SIZE)
            {
                wrapInRange = true;
            }

            const int wOffset = NB_CHANNELS * CHANNELS_WIDTH
                                    * (iy + CHANNELS_HEIGHT * och);

            if (!wrapInRange && INPUT_MEM_STRIDE == NB_CHANNELS) {
                macsOnRange<NB_CHANNELS * CHANNELS_WIDTH>(
                    inputs + iOffset, 
                    weights + wOffset, 
                    weightedSum);
            }
            else {
                for (int ix = 0; ix < CHANNELS_WIDTH; ++ix) {
                    int iOffsetInRange = iOffset + ix * INPUT_MEM_STRIDE;

                    if (wrapInRange
                        && iOffsetInRange >= INPUT_MEM_CONT_SIZE)
                    {
                        iOffsetInRange += INPUT_MEM_WRAP_OFFSET
                                    - INPUT_MEM_CONT_OFFSET
                                    - INPUT_MEM_CONT_SIZE;
                    }

                    macsOnRange<NB_CHANNELS>(
                        inputs + iOffsetInRange,
                        weights + wOffset + ix * NB_CHANNELS, 
                        weightedSum);
                }
            }
        }

        outputs[och] = sat<Output_T>(weightedSum, och, ACTIVATION, rescaling);
    }
}

#ifdef N2D2_STM32_EMULATOR
template<typename Output_T>
inline void N2D2_Export::Network::saveOutputs(
    int NB_OUTPUTS,
    int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE,
    const Output_T* __restrict outputs,
    FILE* pFile,
    Format format) const
{
    if (format == Format::HWC) {
        fprintf(pFile, "(");
        for(int oy = 0; oy < OUTPUTS_HEIGHT; oy++) {
            fprintf(pFile, "(");

            for(int ox = 0; ox < OUTPUTS_WIDTH; ox++) {
                fprintf(pFile, "(");

                const int oPos = (ox + OUTPUTS_WIDTH * oy);
                int oOffset = OUTPUT_MEM_STRIDE * oPos;

                if (OUTPUT_MEM_WRAP_SIZE > 0
                    && oOffset >= OUTPUT_MEM_CONT_SIZE)
                {
                    oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                                - OUTPUT_MEM_CONT_SIZE;
                }

                for (int output = 0; output < NB_OUTPUTS; output++) {
                    if (std::is_floating_point<Output_T>::value)
                        fprintf(pFile, "%f", outputs[oOffset + output]);
                    else
                        fprintf(pFile, "%d", outputs[oOffset + output]);

                    fprintf(pFile, ", ");
                }

                fprintf(pFile, "), \n");
            }

            fprintf(pFile, "), \n");
        }

        fprintf(pFile, ")\n");
    }
    else if (format == Format::CHW) {
        fprintf(pFile, "");
        for(int output = 0; output < NB_OUTPUTS; output++) {
            fprintf(pFile, "%d:\n", output);

            for(int oy = 0; oy < OUTPUTS_HEIGHT; oy++) {
                fprintf(pFile, "");

                for(int ox = 0; ox < OUTPUTS_WIDTH; ox++) {
                    const int oPos = (ox + OUTPUTS_WIDTH * oy);
                    int oOffset = OUTPUT_MEM_STRIDE * oPos;

                    if (OUTPUT_MEM_WRAP_SIZE > 0
                        && oOffset >= OUTPUT_MEM_CONT_SIZE)
                    {
                        oOffset += OUTPUT_MEM_WRAP_OFFSET
                            - OUTPUT_MEM_CONT_OFFSET - OUTPUT_MEM_CONT_SIZE;
                    }

                    if (std::is_floating_point<Output_T>::value)
                        fprintf(pFile, "%f", outputs[oOffset + output]);
                    else
                        fprintf(pFile, "%d", outputs[oOffset + output]);

                    fprintf(pFile, " ");
                }

                fprintf(pFile, "\n");
            }

            fprintf(pFile, "\n");
        }

        fprintf(pFile, "\n");
    }
    else {
        N2D2_THROW_OR_ABORT(std::runtime_error, "Unknown format.");
    }
}
#endif

template<int NB_CHANNELS,
         int INPUTS_HEIGHT, int INPUTS_WIDTH,
         // Memory mapping: outputs
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         typename Input_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::maxPropagate(
    const Input_T* __restrict inputs,
    int32_t* __restrict outputs) const
{
    int iMaxInput = 0;
    Input_T maxInput = std::numeric_limits<Input_T>::lowest();

    for (int iy = 0; iy < INPUTS_HEIGHT; ++iy) {
        for (int ix = 0; ix < INPUTS_WIDTH; ++ix) {
            const int oPos = (ix + INPUTS_WIDTH * iy);
            int iOffset = INPUT_MEM_STRIDE * oPos;

            if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                            - INPUT_MEM_CONT_SIZE;
            }

            if (NB_CHANNELS > 1) {
                for (int ch = 0; ch < NB_CHANNELS; ++ch) {
                    if (inputs[ch + iOffset] > maxInput) {
                        iMaxInput = ch;
                        maxInput = inputs[ch + iOffset];
                    }
                }

                outputs[oPos] = static_cast<int32_t>(iMaxInput);
            }
            else {
                outputs[oPos] = inputs[iOffset] > threshold<Input_T>();
            }
        }
    }
}

template<int NB_CHANNELS, 
         int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
         int NB_OUTPUTS,
         int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
         // Memory mapping: inputs
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // Memory mapping: outputs
         int OUTPUT_MEM_CONT_OFFSET,
         int OUTPUT_MEM_CONT_SIZE,
         int OUTPUT_MEM_WRAP_OFFSET,
         int OUTPUT_MEM_WRAP_SIZE,
         int OUTPUT_MEM_STRIDE,
         typename Input_T, typename Output_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::resizeNearestNeighborPropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs) const
{
    static_assert(NB_CHANNELS == NB_OUTPUTS,
        "NB_CHANNELS should be equal to NB_OUTPUTS.");
    static_assert(OUTPUTS_HEIGHT % CHANNELS_HEIGHT == 0,
        "Output height must be a multiple of input height.");
    static_assert(OUTPUTS_WIDTH % CHANNELS_WIDTH == 0,
        "Output width must be a multiple of input width.");

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            const int oPos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * oPos;

            if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
                oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                            - OUTPUT_MEM_CONT_SIZE;
            }

            const int ix = ox * CHANNELS_WIDTH / OUTPUTS_WIDTH;
            const int iy = oy * CHANNELS_HEIGHT / OUTPUTS_HEIGHT;

            const int iPos = (ix + CHANNELS_WIDTH * iy);
            int iOffset = INPUT_MEM_STRIDE * iPos;

            if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                            - INPUT_MEM_CONT_SIZE;
            }

            for (int output = 0; output < NB_OUTPUTS; ++output) {
                outputs[output + oOffset] = inputs[output + iOffset];
            }
        }
    }
}

template<int NB_CHANNELS, 
         int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
         int NB_OUTPUTS,
         int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
         // Memory mapping: inputs
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // Memory mapping: outputs
         int OUTPUT_MEM_CONT_OFFSET,
         int OUTPUT_MEM_CONT_SIZE,
         int OUTPUT_MEM_WRAP_OFFSET,
         int OUTPUT_MEM_WRAP_SIZE,
         int OUTPUT_MEM_STRIDE,
         typename Input_T, typename Output_T,
         typename Rescaling_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::scalingPropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs,
    const Rescaling_T& __restrict rescaling) const
{
    static_assert(NB_CHANNELS == NB_OUTPUTS,
        "NB_CHANNELS should be equal to NB_OUTPUTS.");
    static_assert(CHANNELS_HEIGHT == OUTPUTS_HEIGHT,
        "CHANNELS_HEIGHT should be equal to OUTPUTS_HEIGHT.");
    static_assert(CHANNELS_WIDTH == OUTPUTS_WIDTH,
        "CHANNELS_WIDTH should be equal to OUTPUTS_WIDTH.");

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            const int pos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * pos;

            if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
                oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
                            - OUTPUT_MEM_CONT_SIZE;
            }

            int iOffset = INPUT_MEM_STRIDE * pos;

            if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                            - INPUT_MEM_CONT_SIZE;
            }

            for (int ch = 0; ch < NB_OUTPUTS; ++ch) {
                outputs[ch + oOffset]
                    = sat<Output_T>(inputs[ch + iOffset], ch, Linear, rescaling);
            }
        }
    }
}

template<int NB_CHANNELS, 
         int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
         int NB_OUTPUTS,
         int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
         // Memory mapping: inputs
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         // Memory mapping: outputs
         int OUTPUT_MEM_CONT_OFFSET,
         int OUTPUT_MEM_CONT_SIZE,
         int OUTPUT_MEM_WRAP_OFFSET,
         int OUTPUT_MEM_WRAP_SIZE,
         int OUTPUT_MEM_STRIDE,
         typename Input_T, typename Output_T>
N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::transposePropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs,
    const int perm[4]) const
{
    constexpr std::size_t dims[3] = {CHANNELS_WIDTH, CHANNELS_HEIGHT,
                                     NB_CHANNELS};
    std::size_t coords[3];

    for (coords[2] = 0; coords[2] < dims[2]; ++coords[2]) {
        for (coords[1] = 0; coords[1] < dims[1]; ++coords[1]) {
            for (coords[0] = 0; coords[0] < dims[0]; ++coords[0]) {
                const std::size_t iOffset = coords[0]
                    + dims[0] * (coords[1] + dims[1] * (coords[2]));
                const std::size_t oOffset = coords[perm[0]]
                    + dims[perm[0]] * (coords[perm[1]]
                        + dims[perm[1]] * (coords[perm[2]]));

                outputs[oOffset] = inputs[iOffset];
            }
        }
    }
}

N2D2_ALWAYS_INLINE inline N2D2_Export::Network::Tick_T N2D2_Export::Network::tick() const {
#ifdef N2D2_STM32_EMULATOR
    return std::chrono::high_resolution_clock::now();
#else
    return HAL_GetTick();   // return ms
#endif
}

N2D2_ALWAYS_INLINE inline void N2D2_Export::Network::benchmark(const char* name,
                                                        const Tick_T& start,
                                                        const Tick_T& end,
                                                        RunningMean_T& timing) const
{
#ifdef N2D2_STM32_EMULATOR
    auto duration = std::chrono::duration_cast
                        <std::chrono::microseconds>(end - start).count();
#else
    const double duration = 1.0e3 * (double)(end - start);
#endif
    timing.mean = (timing.mean * timing.count + duration)
                    / (timing.count + 1.0);
    ++timing.count;

#ifdef N2D2_STM32_EMULATOR
    // Cumulative
    cumulativeTiming[name] = timing.mean;
    const double cumMeanTiming = std::accumulate(cumulativeTiming.begin(),
        cumulativeTiming.end(), 0, [] (double value,
                            const std::map<std::string, double>::value_type& p)
                   { return value + p.second; });

    printf("%s timing = %.02f us -- %.02f us\n", name, timing.mean, cumMeanTiming);
#else
    printf("%s timing = %.02f us\n", name, timing.mean);
#endif
}

#endif
