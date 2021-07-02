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
#include <chrono>
#include <map>
#include <numeric>

#include "typedefs.h"
#include "typedef_union.h"

#include <iostream>

#define N2D2_THROW_OR_ABORT(ex, msg) throw ex(msg)
#define N2D2_ALWAYS_INLINE __attribute__((always_inline))

#ifndef N2D2_SECTION_NN_MEMORY
#define N2D2_SECTION_NN_MEMORY ".nn_memory"
#endif
#ifndef N2D2_SECTION_NN_WEIGHTS
#define N2D2_SECTION_NN_WEIGHTS ".nn_weights"
#endif
#ifndef N2D2_SECTION_NN_BIASSES
#define N2D2_SECTION_NN_BIASSES ".nn_biasses"
#endif
#define N2D2_SECTION_ATTRIBUTE(sec) __attribute__((section(sec)))

namespace N2D2 {

class Network {
public:
    enum class Format {
        HWC,
        CHW
    };

    enum ElemWiseOp {
        Sum
    };

    typedef std::chrono::time_point<std::chrono::high_resolution_clock> Tick_T;
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

    std::size_t outputHeight() const;
    std::size_t outputWidth() const;
    std::size_t outputNbOutputs() const;
    std::size_t outputSize() const;

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
            N2D2::Network::ElemWiseOp ELEM_OP,
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
            int NB_BITS_W,
            int ACTIVATION_OUTPUT_RANGE,
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
            int NB_BITS_W,
            int ACTIVATION_OUTPUT_RANGE,
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
            int NB_BITS_W,
            int ACTIVATION_OUTPUT_RANGE,
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

private:
    mutable std::map<std::string, double> cumulativeTiming;

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
                                           const Rescaling_T& __restrict rescaling,
                                           int ACTIVATION_OUTPUT_RANGE) 
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

        return saturate<Output_T>(rescaling(weightedSum, output), ACTIVATION_OUTPUT_RANGE);
    }

    template<typename Output_T, typename T,  
             typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Output_T saturate(T value, std::int32_t sat) {
        return value;
    }

    template<typename Output_T, typename T,  
             typename std::enable_if<!std::is_floating_point<T>::value>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Output_T saturate(T value, std::uint32_t sat) {
        return std::is_unsigned<Output_T>::value?clamp(value, T(0), (T(1) << sat) - 1):
                                                 clamp(value, -(T(1) << (sat - 1)), 
                                                               (T(1) << (sat - 1)) - 1);
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

    //******************version for non-accumulated weights********************************

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs, 
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        for (int iter = 0; iter < NB_ITERATIONS; ++iter) {
            weightedSum += inputs[iter*INPUTS_INC] * weights[iter*WEIGHTS_INC];
        }
    }

    //******************recursive version for accumulated weights***************************

    /***************************************************************************************
    *****************************************4-bit******************************************
    ***************************************************************************************/

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 4 && NB_ITERATIONS == 1)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        T4_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];
        weightedSum += (*inputs)*data.sfields.op1;
    }

    template<int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(NB_BITS_W == 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T dualMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum, bool verbose)
    {
        T4_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        weightedSum += inputs[0*INPUTS_INC] * data.sfields.op1
         + inputs[1*INPUTS_INC] * data.sfields.op0;

        return weightedSum;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 4 && NB_ITERATIONS >=2)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        weightedSum = dualMac<NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum, verbose);
        macsOnRangeMixedPrecisionR<NB_ITERATIONS-2, NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs + 2*INPUTS_INC, weights + WEIGHTS_INC, weightedSum, verbose);
    }

    /***************************************************************************************
    *****************************************2-bit******************************************
    ***************************************************************************************/

    template<int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(NB_BITS_W == 2)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T dualMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum, bool verbose)
    {
        T2_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        if(verbose) std::cout << "dual" << std::flush;

        if(verbose) std::cout << "weight0 = " << +data.sfields.op3 <<  std::flush;
        if(verbose) std::cout << "weight1 = " << +data.sfields.op2 <<  std::flush;

        weightedSum += inputs[0*INPUTS_INC] * data.sfields.op3
         + inputs[1*INPUTS_INC] * data.sfields.op2;

        return weightedSum;
    }

    template<int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(NB_BITS_W == 2)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T tripleMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum,
                                            bool verbose)
    {
        T2_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        if(verbose) std::cout << "triple" << std::flush;
        if(verbose) std::cout << "weight0 = " << +data.sfields.op3 << std::flush;
        if(verbose) std::cout << "weight1 = " << +data.sfields.op2 << std::flush;
        if(verbose) std::cout << "weight2 = " << +data.sfields.op1 << std::flush;

        weightedSum += inputs[0*INPUTS_INC] * data.sfields.op3
         + inputs[1*INPUTS_INC] * data.sfields.op2
         + inputs[2*INPUTS_INC] * data.sfields.op1;

        return weightedSum;
    }

    template<int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(NB_BITS_W == 2)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T quadMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum,
                                            bool verbose)
    {
        T2_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        if(verbose) std::cout << "quad" << std::flush;
        if(verbose) std::cout << "weight0 = " << +data.sfields.op3 << std::flush;
        if(verbose) std::cout << "weight1 = " << +data.sfields.op2 << std::flush;
        if(verbose) std::cout << "weight2 = " << +data.sfields.op1 << std::flush;
        if(verbose) std::cout << "weight3 = " << +data.sfields.op0 << std::flush;

        weightedSum += inputs[0*INPUTS_INC] * data.sfields.op3
         + inputs[1*INPUTS_INC] * data.sfields.op2
         + inputs[2*INPUTS_INC] * data.sfields.op1
         + inputs[3*INPUTS_INC] * data.sfields.op0;

        return weightedSum;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 2 && NB_ITERATIONS == 1)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        T2_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];
        weightedSum += (*inputs)*data.sfields.op3;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 2 && NB_ITERATIONS == 2)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        weightedSum = dualMac<NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum, verbose);
        macsOnRangeMixedPrecisionR<NB_ITERATIONS-2, NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs + 2*INPUTS_INC, weights + WEIGHTS_INC, weightedSum, verbose);
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 2 && NB_ITERATIONS == 3)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        weightedSum = tripleMac<NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum, verbose);
        macsOnRangeMixedPrecisionR<NB_ITERATIONS-3, NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs + 3*INPUTS_INC, weights + WEIGHTS_INC, weightedSum, verbose);
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 2 && NB_ITERATIONS >=4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        weightedSum = quadMac<NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum, verbose);
        macsOnRangeMixedPrecisionR<NB_ITERATIONS-4, NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs + 4*INPUTS_INC, weights + WEIGHTS_INC, weightedSum, verbose);
    }

    /***************************************************************************************
    *****************************************1-bit******************************************
    ***************************************************************************************/


    /*
    template<int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(NB_BITS_W == 1)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T dualMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum, bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        if(verbose) std::cout << "dualMAC" << std::flush;

        int8_t weight0 = (int8_t)data.ufields.op3;
        if(weight0 == 0) weight0 = -1;

        int8_t weight1 = (int8_t)data.ufields.op2;
        if(weight1 == 0) weight1 = -1;

        if(verbose) std::cout << "weight0 = " << +weight0 <<  std::flush;
        if(verbose) std::cout << "weight1 = " << +weight1 <<  std::flush;

        weightedSum += inputs[0*INPUTS_INC] * weight0
         + inputs[1*INPUTS_INC] * weight1;

        return weightedSum;
    }
    template<int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(NB_BITS_W == 1)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T quadMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum,
                                            bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        int8_t weight0 = (int8_t)data.ufields.op7;
        if(weight0 == 0) weight0 = -1;
        int8_t weight1 = (int8_t)data.ufields.op6;
        if(weight1 == 0) weight1 = -1;
        int8_t weight2 = (int8_t)data.ufields.op5;
        if(weight2 == 0) weight2 = -1;
        int8_t weight3 = (int8_t)data.ufields.op4;
        if(weight3 == 0) weight3 = -1;

        if(verbose) std::cout << "quadMAC" << std::flush;
        if(verbose) std::cout << "weight0 = " << +weight0 << std::flush;
        if(verbose) std::cout << "weight1 = " << +weight1 << std::flush;
        if(verbose) std::cout << "weight2 = " << +weight2 << std::flush;
        if(verbose) std::cout << "weight3 = " << +weight3 << std::flush;

        weightedSum += inputs[0*INPUTS_INC] * weight0
         + inputs[1*INPUTS_INC] * weight1
         + inputs[2*INPUTS_INC] * weight2
         + inputs[3*INPUTS_INC] * weight3;

        return weightedSum;
    }
    */

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 1 && NB_ITERATIONS == 1)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        int8_t weight0 = data.ufields.op7==0 ? -1 : data.ufields.op7;

        weightedSum += (*inputs)*weight0;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 1 && NB_ITERATIONS == 2)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        int8_t weight0 = data.ufields.op7==0 ? -1 : data.ufields.op7;
        int8_t weight1 = data.ufields.op6==0 ? -1 : data.ufields.op6;

        weightedSum += inputs[0*INPUTS_INC] * weight0
         + inputs[1*INPUTS_INC] * weight1;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 1 && NB_ITERATIONS == 3)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        int8_t weight0 = data.ufields.op7==0 ? -1 : data.ufields.op7;
        int8_t weight1 = data.ufields.op6==0 ? -1 : data.ufields.op6;
        int8_t weight2 = data.ufields.op5==0 ? -1 : data.ufields.op5;

        weightedSum += inputs[0*INPUTS_INC] * weight0
         + inputs[1*INPUTS_INC] * weight1
         + inputs[2*INPUTS_INC] * weight2;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 1 && NB_ITERATIONS == 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        int8_t weight0 = data.ufields.op7==0 ? -1 : data.ufields.op7;
        int8_t weight1 = data.ufields.op6==0 ? -1 : data.ufields.op6;
        int8_t weight2 = data.ufields.op5==0 ? -1 : data.ufields.op5;
        int8_t weight3 = data.ufields.op4==0 ? -1 : data.ufields.op4;

        weightedSum += inputs[0*INPUTS_INC] * weight0
         + inputs[1*INPUTS_INC] * weight1
         + inputs[2*INPUTS_INC] * weight2
         + inputs[3*INPUTS_INC] * weight3;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 1 && NB_ITERATIONS == 5)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        int8_t weight0 = data.ufields.op7==0 ? -1 : data.ufields.op7;
        int8_t weight1 = data.ufields.op6==0 ? -1 : data.ufields.op6;
        int8_t weight2 = data.ufields.op5==0 ? -1 : data.ufields.op5;
        int8_t weight3 = data.ufields.op4==0 ? -1 : data.ufields.op4;
        int8_t weight4 = data.ufields.op3==0 ? -1 : data.ufields.op3;

        weightedSum += inputs[0*INPUTS_INC] * weight0
         + inputs[1*INPUTS_INC] * weight1
         + inputs[2*INPUTS_INC] * weight2
         + inputs[3*INPUTS_INC] * weight3
         + inputs[4*INPUTS_INC] * weight4;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 1 && NB_ITERATIONS == 6)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        int8_t weight0 = data.ufields.op7==0 ? -1 : data.ufields.op7;
        int8_t weight1 = data.ufields.op6==0 ? -1 : data.ufields.op6;
        int8_t weight2 = data.ufields.op5==0 ? -1 : data.ufields.op5;
        int8_t weight3 = data.ufields.op4==0 ? -1 : data.ufields.op4;
        int8_t weight4 = data.ufields.op3==0 ? -1 : data.ufields.op3;
        int8_t weight5 = data.ufields.op2==0 ? -1 : data.ufields.op2;

        weightedSum += inputs[0*INPUTS_INC] * weight0
         + inputs[1*INPUTS_INC] * weight1
         + inputs[2*INPUTS_INC] * weight2
         + inputs[3*INPUTS_INC] * weight3
         + inputs[4*INPUTS_INC] * weight4
         + inputs[5*INPUTS_INC] * weight5;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 1 && NB_ITERATIONS == 7)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        int8_t weight0 = data.ufields.op7==0 ? -1 : data.ufields.op7;
        int8_t weight1 = data.ufields.op6==0 ? -1 : data.ufields.op6;
        int8_t weight2 = data.ufields.op5==0 ? -1 : data.ufields.op5;
        int8_t weight3 = data.ufields.op4==0 ? -1 : data.ufields.op4;
        int8_t weight4 = data.ufields.op3==0 ? -1 : data.ufields.op3;
        int8_t weight5 = data.ufields.op2==0 ? -1 : data.ufields.op2;
        int8_t weight6 = data.ufields.op1==0 ? -1 : data.ufields.op1;

        weightedSum += inputs[0*INPUTS_INC] * weight0
         + inputs[1*INPUTS_INC] * weight1
         + inputs[2*INPUTS_INC] * weight2
         + inputs[3*INPUTS_INC] * weight3
         + inputs[4*INPUTS_INC] * weight4
         + inputs[5*INPUTS_INC] * weight5
         + inputs[6*INPUTS_INC] * weight6;
    }

    template<int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(NB_BITS_W == 1)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T octoMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum,
                                            bool verbose)
    {
        T1_8_Vector data;
        data.uvector = weights[0*WEIGHTS_INC];

        int8_t weight0 = data.ufields.op7==0 ? -1 : data.ufields.op7;
        int8_t weight1 = data.ufields.op6==0 ? -1 : data.ufields.op6;
        int8_t weight2 = data.ufields.op5==0 ? -1 : data.ufields.op5;
        int8_t weight3 = data.ufields.op4==0 ? -1 : data.ufields.op4;
        int8_t weight4 = data.ufields.op3==0 ? -1 : data.ufields.op3;
        int8_t weight5 = data.ufields.op2==0 ? -1 : data.ufields.op2;
        int8_t weight6 = data.ufields.op1==0 ? -1 : data.ufields.op1;
        int8_t weight7 = data.ufields.op0==0 ? -1 : data.ufields.op0;

        weightedSum += inputs[0*INPUTS_INC] * weight0
         + inputs[1*INPUTS_INC] * weight1
         + inputs[2*INPUTS_INC] * weight2
         + inputs[3*INPUTS_INC] * weight3
         + inputs[4*INPUTS_INC] * weight4
         + inputs[5*INPUTS_INC] * weight5
         + inputs[6*INPUTS_INC] * weight6
         + inputs[7*INPUTS_INC] * weight7;

        return weightedSum;
    }

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W == 1 && NB_ITERATIONS >=8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
        weightedSum = octoMac<NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum, verbose);
        macsOnRangeMixedPrecisionR<NB_ITERATIONS-8, NB_BITS_W, INPUTS_INC, WEIGHTS_INC>(inputs + 8*INPUTS_INC, weights + WEIGHTS_INC, weightedSum, verbose);
    }


    /***************************************************************************************
    *****************************************0-iterations***********************************
    ***************************************************************************************/

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<((NB_BITS_W == 1 || NB_BITS_W == 2 || NB_BITS_W == 4)  && NB_ITERATIONS == 0)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum,
                                               bool verbose)
    {
    }

    /***************************************************************************************
    **************************************>=8-bit******************************************
    ***************************************************************************************/

    template<int NB_ITERATIONS,
             int NB_BITS_W,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(NB_BITS_W >= 8 || NB_BITS_W <= 0)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRangeMixedPrecisionR(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum, bool verbose)
    {
        for (int iter = 0; iter < NB_ITERATIONS; ++iter) {
            weightedSum += inputs[iter*INPUTS_INC] * weights[iter*WEIGHTS_INC];
        }
    }

    N2D2_ALWAYS_INLINE Tick_T tick() const;
    N2D2_ALWAYS_INLINE void benchmark(const char* name,
                                      const Tick_T& start,
                                      const Tick_T& end,
                                      RunningMean_T& timing) const;
};
}

template<typename Output_T>
N2D2_ALWAYS_INLINE inline void N2D2::Network::concatenate(
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
N2D2_ALWAYS_INLINE inline void N2D2::Network::concatenate(
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
N2D2_ALWAYS_INLINE inline void N2D2::Network::concatenatePropagate(
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

template<N2D2::Network::ElemWiseOp ELEM_OP>
N2D2_ALWAYS_INLINE inline SUM_T N2D2::Network::elemWise(
    int /*pos*/,
    int /*ch*/) const
{
    return 0;
}

template<N2D2::Network::ElemWiseOp ELEM_OP,
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
N2D2_ALWAYS_INLINE inline SUM_T N2D2::Network::elemWise(
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
         N2D2::Network::ElemWiseOp ELEM_OP,
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
N2D2_ALWAYS_INLINE inline void N2D2::Network::elemWisePropagate(
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
                    = sat<Output_T>(val, ch, ACTIVATION, rescaling, NB_BITS);
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
         int NB_BITS_W,
         int ACTIVATION_OUTPUT_RANGE,
         typename Input_T, typename Output_T,
         typename Weight_T, typename Bias_T,
         typename Rescaling_T>
N2D2_ALWAYS_INLINE inline void N2D2::Network::convcellPropagate(
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

#pragma omp parallel for collapse(2)
        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            for (int output = 0; output < NB_OUTPUTS; ++output) {
                // moved to inner loop for collapsing -->
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
                // <--
                Bias_T weightedSum = biasses[output];

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

                    int wOffset = NB_CHANNELS * (sxMin
                        + KERNEL_WIDTH * (syMin + sy + KERNEL_HEIGHT * output));

                    bool verbose = false;

                    // NB_CHANNELS -> ((NB_CHANNELS*precision)+(NB_CHANNELS*precision)%8)/8
                    // e.g. (7*4 + (4))/8 -> 4
                    constexpr int NB_INT8 = ((NB_CHANNELS*NB_BITS_W)+(NB_CHANNELS*NB_BITS_W)%8)/8;
                    if(NB_BITS_W > 0) {
                        wOffset = NB_INT8 * (sxMin
                                + KERNEL_WIDTH * (syMin + sy + KERNEL_HEIGHT * output));
                    }

                    if (!wrapInRange && (NB_CHANNELS == INPUT_MEM_STRIDE
                        && ((PADDING_X == 0
                            && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                                || sxMax - sxMin == KERNEL_WIDTH)) && ((NB_CHANNELS*NB_BITS_W)%8 == 0))
                    {
                            macsOnRangeMixedPrecisionR<KERNEL_WIDTH * NB_CHANNELS, NB_BITS_W>(
                                inputs + iOffset,
                                weights + wOffset,
                                weightedSum,
                                verbose);
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

                            int factor = NB_CHANNELS;
                            if(NB_BITS_W > 0) factor = NB_INT8;

                            macsOnRangeMixedPrecisionR<NB_CHANNELS, NB_BITS_W>(
                                // same input line so no wrapping can occur
                                inputs + iOffsetInRange, 
                                weights + wOffset + sx * factor,
                                weightedSum,
                                verbose);
                        }
                    }
                }

                outputs[oOffset + output]
                    = sat<Output_T>(weightedSum, output, ACTIVATION, rescaling, ACTIVATION_OUTPUT_RANGE);

                //TODO: accumulate outputs here
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
         int NB_BITS_W,
         int ACTIVATION_OUTPUT_RANGE,
         typename Input_T, typename Output_T,
         typename Weight_T, typename Bias_T,
         typename Rescaling_T>
N2D2_ALWAYS_INLINE inline void N2D2::Network::convcellDWPropagate(
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

#pragma omp parallel for collapse(2)
        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            for (int output = 0; output < NB_OUTPUTS; ++output) {
                // moved to inner loop for collapsing -->
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
                // <--

                const int channel = (output * NB_CHANNELS) / NB_OUTPUTS;

                //SUM_T = Bias_T
                //SUM_T weightedSum = biasses[output];
                Bias_T weightedSum = biasses[output];

                for (int sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                    if ((PADDING_Y != 0
                            || OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD)
                        && sy >= syMax - syMin)
                    {
                        break;
                    }

                    const int iPos = (ix
                                        + CHANNELS_WIDTH * (iy + syMin + sy));

                    int iOffset = INPUT_MEM_STRIDE * iPos;

                    // Wrapping cannot occur in the middle of a line, except if
                    // there is only one line (1D)!
                    bool wrapInRange = false;

                    if (INPUT_MEM_WRAP_SIZE > 0
                        && (iOffset+INPUT_MEM_STRIDE*sxMin) >= INPUT_MEM_CONT_SIZE)
                    {
                        iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                                    - INPUT_MEM_CONT_SIZE;
                    }
                    else if (INPUT_MEM_WRAP_SIZE > 0 && KERNEL_WIDTH > 1
                        && CHANNELS_HEIGHT == 1 // single line (1D)!
                        && (iOffset+INPUT_MEM_STRIDE*sxMin) + KERNEL_WIDTH * INPUT_MEM_STRIDE
                            > INPUT_MEM_CONT_SIZE)
                    {
                        wrapInRange = true;
                    }

                    int wOffset = (sxMin
                        + KERNEL_WIDTH * (syMin + sy + KERNEL_HEIGHT * output));


                    constexpr int NB_INT8 = ((KERNEL_WIDTH*NB_BITS_W)+(KERNEL_WIDTH*NB_BITS_W)%8)/8;

                    int nbInt8 = KERNEL_WIDTH;
                    int nbSlot_per_Int8 = 1;

                    if(NB_BITS_W > 0) {
                        wOffset = (NB_INT8 * (syMin + sy + KERNEL_HEIGHT * output));
                        nbInt8 = NB_INT8;
                        nbSlot_per_Int8 = 8/(size_t)NB_BITS_W;
                    }

                    if (!wrapInRange && ((PADDING_X == 0
                            && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                        || sxMax - sxMin == KERNEL_WIDTH))
                    {
                        //not accumulated weights for DW conv!
                        /*
                        macsOnRange<KERNEL_WIDTH, INPUT_MEM_STRIDE>(
                            inputs + iOffset + channel, 
                            weights + wOffset, 
                            weightedSum);
                        */
                       //accumulated weights
                        macsOnRangeMixedPrecisionR<KERNEL_WIDTH,NB_BITS_W,INPUT_MEM_STRIDE>(
                            inputs + iOffset + channel,
                            weights + wOffset,
                            weightedSum,
                            false);
                    }
                    else {

                        int iInt8_start = sxMin/nbSlot_per_Int8;
                        int iSlot_start = (nbSlot_per_Int8 > 1)?(sxMin%nbSlot_per_Int8):0;

                        for (int iInt8 = 0; iInt8 < nbInt8; ++iInt8) {
                            for(int iSlot = 0; iSlot < nbSlot_per_Int8; ++iSlot){

                                int trueISlot = (iInt8==0)?iSlot+iSlot_start:iSlot;

                                if(trueISlot >= nbSlot_per_Int8){
                                    break;
                                }

                                int sx = (iInt8_start+iInt8)*nbSlot_per_Int8 + trueISlot;

                                if(sx >= KERNEL_WIDTH) {
                                    break;
                                }

                                if ((PADDING_X != 0
                                        || OUTPUTS_WIDTH != OUTPUTS_WIDTH_NOPAD)
                                    //&& sx >= sxMax - sxMin)
                                    && sx >= sxMax)
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

                                //not accumulated weights for DW conv!
                                /*
                                weightedSum += inputs[iOffsetInRange + channel]
                                    * weights[wOffset + sx];
                                */

                                //test for 4b only
                                if(NB_BITS_W == 4) {
                                    T4_8_Vector data;
                                    data.uvector = weights[wOffset + (iInt8+iInt8_start)];

                                    if(trueISlot == 0) {
                                        weightedSum += inputs[iOffsetInRange + channel]*data.sfields.op1;
                                    }
                                    else{
                                        weightedSum += inputs[iOffsetInRange + channel]*data.sfields.op0;
                                    }
                                }
                                else{
                                    weightedSum += inputs[iOffsetInRange + channel]
                                        * weights[wOffset + (iInt8+iInt8_start)];
                                }
                            }
                        }
                    }
                }
                outputs[oOffset + output]
                    = sat<Output_T>(weightedSum, output, ACTIVATION, rescaling, ACTIVATION_OUTPUT_RANGE);
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
N2D2_ALWAYS_INLINE inline void N2D2::Network::poolcellPropagate(
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

#pragma omp parallel for collapse(2)
        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            for (int output = 0; output < NB_OUTPUTS; ++output) {
                // moved to inner loop for collapsing -->
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
                // <--

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

                    outputs[oOffset + output] = maxVal;
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

                    outputs[oOffset + output] = (Output_T) (sum
                        / (POOL_HEIGHT * POOL_WIDTH));
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
         int NB_BITS_W,
         int ACTIVATION_OUTPUT_RANGE,
         typename Input_T, typename Output_T,
         typename Weight_T, typename Bias_T,
         typename Rescaling_T>
N2D2_ALWAYS_INLINE inline void N2D2::Network::fccellPropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs,
    const Bias_T* __restrict biasses,
    const Weight_T* __restrict weights,
    const Rescaling_T& __restrict rescaling) const
{
    static_assert(OUTPUTS_HEIGHT == 1, "Outputs height should be 1");
    static_assert(OUTPUTS_WIDTH == 1, "Outputs width should be 1");
    static_assert(OUTPUT_MEM_WRAP_SIZE == 0, "Output wrapping not supported");

#pragma omp parallel for
    for (int och = 0; och < NB_OUTPUTS; och++) {
        //SUM_T -> Bias_T
        //SUM_T weightedSum = biasses[och];
        Bias_T weightedSum = biasses[och];

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

            int wOffset = NB_CHANNELS * CHANNELS_WIDTH
                                    * (iy + CHANNELS_HEIGHT * och);

            constexpr int NB_INT8 = ((NB_CHANNELS*NB_BITS_W)+(NB_CHANNELS*NB_BITS_W)%8)/8;

            if(NB_BITS_W > 0){
                wOffset = NB_INT8 * CHANNELS_WIDTH
                                        * (iy + CHANNELS_HEIGHT * och);
            }

            bool verbose = false;

            if (!wrapInRange && INPUT_MEM_STRIDE == NB_CHANNELS &&
                ((NB_CHANNELS*NB_BITS_W)%8 == 0)) {
                macsOnRangeMixedPrecisionR<NB_CHANNELS * CHANNELS_WIDTH, NB_BITS_W>(
                                inputs + iOffset,
                                weights + wOffset,
                                weightedSum,
                                verbose);
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

                    int factor = NB_CHANNELS;
                    if(NB_BITS_W > 0) factor = NB_INT8;

                    macsOnRangeMixedPrecisionR<NB_CHANNELS, NB_BITS_W>(
                                inputs + iOffsetInRange,
                                weights + wOffset + ix * factor,
                                weightedSum,
                                verbose);
                }
            }
        }

        outputs[och] = sat<Output_T>(weightedSum, och, ACTIVATION, rescaling, ACTIVATION_OUTPUT_RANGE);
    }
}

template<typename Output_T>
inline void N2D2::Network::saveOutputs(
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
            fprintf(pFile, "");

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

template<int NB_CHANNELS,
         int INPUTS_HEIGHT, int INPUTS_WIDTH,
         // Memory mapping: outputs
         int INPUT_MEM_CONT_OFFSET,
         int INPUT_MEM_CONT_SIZE,
         int INPUT_MEM_WRAP_OFFSET,
         int INPUT_MEM_WRAP_SIZE,
         int INPUT_MEM_STRIDE,
         typename Input_T>
N2D2_ALWAYS_INLINE inline void N2D2::Network::maxPropagate(
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
                    if (inputs[iOffset + ch] > maxInput) {
                        iMaxInput = ch;
                        maxInput = inputs[iOffset + ch];
                    }
                }

                outputs[oPos] = static_cast<int32_t>(iMaxInput);
            }
            else {
                outputs[oPos] = (inputs[iOffset] > threshold<Input_T>());
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
N2D2_ALWAYS_INLINE inline void N2D2::Network::resizeNearestNeighborPropagate(
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
                outputs[oOffset + output] = inputs[iOffset + output];
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
N2D2_ALWAYS_INLINE inline void N2D2::Network::scalingPropagate(
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
                outputs[oOffset + ch]
                    = sat<Output_T>(inputs[iOffset + ch], ch, Linear, rescaling, NB_BITS);
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
N2D2_ALWAYS_INLINE inline void N2D2::Network::transposePropagate(
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

N2D2_ALWAYS_INLINE inline N2D2::Network::Tick_T N2D2::Network::tick() const {
    return std::chrono::high_resolution_clock::now();
}

N2D2_ALWAYS_INLINE inline void N2D2::Network::benchmark(const char* name,
                                                        const Tick_T& start,
                                                        const Tick_T& end,
                                                        RunningMean_T& timing) const
{
    auto duration = std::chrono::duration_cast
                        <std::chrono::microseconds>(end - start).count();
    timing.mean = (timing.mean * timing.count + duration)
                    / (timing.count + 1.0);
    ++timing.count;

    // Cumulative
    cumulativeTiming[name] = timing.mean;
    const double cumMeanTiming = std::accumulate(cumulativeTiming.begin(),
        cumulativeTiming.end(), 0, [] (double value,
                            const std::map<std::string, double>::value_type& p)
                   { return value + p.second; });

    printf("%s timing = %.02f us -- %.02f us\n", name, timing.mean, cumMeanTiming);
}

#endif
