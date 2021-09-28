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

#include "typedefs.h" // old C header, deprecated
#include "typedefs.hpp"

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

    //******************recursive version for accumulated weights***************************

    /***************************************************************************************
    *****************************************4-bit******************************************
    ***************************************************************************************/

    //******************accumulated activations are in 8 bits***************************

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 4
                && NB_ITERATIONS == 1 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += (*inputs) * weights[0*WEIGHTS_INC].fields.op1;
    }

    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 4
                && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T dualMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum)
    {
        weightedSum += inputs[0*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op1
            + inputs[1*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op0;

        return weightedSum;
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 4
                && NB_ITERATIONS >= 2 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum = dualMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
        macsOnRange<NB_ITERATIONS-2, INPUTS_INC, WEIGHTS_INC>(
            inputs + 2*INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
    }

    //******************accumulated activations are in 4 bits***************************
    //******************accumulated weights are in 4 bits***************************

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 4
                && NB_ITERATIONS == 1 && std::numeric_limits<Input_T>::digits == 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += inputs[0*INPUTS_INC].fields.op1
                        * weights[0*WEIGHTS_INC].fields.op1;
    }

    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 4
                && std::numeric_limits<Input_T>::digits == 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T dualMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum)
    {
        weightedSum
            += inputs[0*INPUTS_INC].fields.op1
                * weights[0*WEIGHTS_INC].fields.op1
            + inputs[0*INPUTS_INC].fields.op0
                * weights[0*WEIGHTS_INC].fields.op0;

        return weightedSum;
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 4
                && NB_ITERATIONS >=2 && std::numeric_limits<Input_T>::digits == 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum = dualMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
        macsOnRange<NB_ITERATIONS-2, INPUTS_INC, WEIGHTS_INC>(
            inputs + INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
    }

    //for the last FC case
    //******************accumulated activations are in 4 bits***************************
    //******************weights are in 8 bits***************************

       template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 8
                && NB_ITERATIONS == 1 && std::numeric_limits<Input_T>::digits == 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += inputs[0*INPUTS_INC].fields.op1 * weights[0*WEIGHTS_INC];
    }

    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 8
                && std::numeric_limits<Input_T>::digits == 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T dualMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum)
    {
        weightedSum += inputs[0*INPUTS_INC].fields.op1 * weights[0*WEIGHTS_INC]
            + inputs[0*INPUTS_INC].fields.op0 * weights[1*WEIGHTS_INC];

        return weightedSum;
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 8
                && NB_ITERATIONS >= 2 && std::numeric_limits<Input_T>::digits == 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum = dualMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
        macsOnRange<NB_ITERATIONS-2, INPUTS_INC, WEIGHTS_INC>(
            inputs + INPUTS_INC, weights + 2*WEIGHTS_INC, weightedSum);
    }

    /***************************************************************************************
    *****************************************2-bit******************************************
    ***************************************************************************************/

    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 2
                && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T dualMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum)
    {
        weightedSum += inputs[0*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op3
            + inputs[1*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op2;

        return weightedSum;
    }

    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 2
                && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T tripleMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum)
    {
        weightedSum += inputs[0*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op3
            + inputs[1*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op2
            + inputs[2*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op1;

        return weightedSum;
    }

    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 2
                && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T quadMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum)
    {
        weightedSum += inputs[0*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op3
            + inputs[1*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op2
            + inputs[2*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op1
            + inputs[3*INPUTS_INC] * weights[0*WEIGHTS_INC].fields.op0;

        return weightedSum;
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 2
                && NB_ITERATIONS == 1 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += (*inputs) * weights[0*WEIGHTS_INC].fields.op3;
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 2
                && NB_ITERATIONS == 2 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum = dualMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
        macsOnRange<NB_ITERATIONS-2, INPUTS_INC, WEIGHTS_INC>(
            inputs + 2*INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 2
                && NB_ITERATIONS == 3 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum = tripleMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
        macsOnRange<NB_ITERATIONS-3, INPUTS_INC, WEIGHTS_INC>(
            inputs + 3*INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 2
                && NB_ITERATIONS >= 4)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum = quadMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
        macsOnRange<NB_ITERATIONS-4, INPUTS_INC, WEIGHTS_INC>(
            inputs + 4*INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
    }

    /***************************************************************************************
    *****************************************1-bit******************************************
    ***************************************************************************************/

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 1
                && NB_ITERATIONS == 1 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += ((weights[0*WEIGHTS_INC].fields.op7 == 0)
                ? (Bias_T)(-(*inputs)) : (Bias_T)(*inputs));
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 1
                && NB_ITERATIONS == 2 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += ((weights[0*WEIGHTS_INC].fields.op7 == 0)
                ? (Bias_T)(-inputs[0*INPUTS_INC]) : (Bias_T)inputs[0*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op6 == 0)
                ? (Bias_T)(-inputs[1*INPUTS_INC]) : (Bias_T)inputs[1*INPUTS_INC]);
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 1
                && NB_ITERATIONS == 3 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += ((weights[0*WEIGHTS_INC].fields.op7 == 0)
                ? (Bias_T)(-inputs[0*INPUTS_INC]) : (Bias_T)inputs[0*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op6 == 0)
                ? (Bias_T)(-inputs[1*INPUTS_INC]) : (Bias_T)inputs[1*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op5 == 0)
                ? (Bias_T)(-inputs[2*INPUTS_INC]) : (Bias_T)inputs[2*INPUTS_INC]);
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 1
                && NB_ITERATIONS == 4 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += ((weights[0*WEIGHTS_INC].fields.op7 == 0)
                ? (Bias_T)(-inputs[0*INPUTS_INC]) : (Bias_T)inputs[0*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op6 == 0)
                ? (Bias_T)(-inputs[1*INPUTS_INC]) : (Bias_T)inputs[1*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op5 == 0)
                ? (Bias_T)(-inputs[2*INPUTS_INC]) : (Bias_T)inputs[2*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op4 == 0)
                ? (Bias_T)(-inputs[3*INPUTS_INC]) : (Bias_T)inputs[3*INPUTS_INC]);
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 1
                && NB_ITERATIONS == 5 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += ((weights[0*WEIGHTS_INC].fields.op7 == 0)
                ? (Bias_T)(-inputs[0*INPUTS_INC]) : (Bias_T)inputs[0*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op6 == 0)
                ? (Bias_T)(-inputs[1*INPUTS_INC]) : (Bias_T)inputs[1*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op5 == 0)
                ? (Bias_T)(-inputs[2*INPUTS_INC]) : (Bias_T)inputs[2*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op4 == 0)
                ? (Bias_T)(-inputs[3*INPUTS_INC]) : (Bias_T)inputs[3*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op3 == 0)
                ? (Bias_T)(-inputs[4*INPUTS_INC]) : (Bias_T)inputs[4*INPUTS_INC]);
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 1
                && NB_ITERATIONS == 6 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += ((weights[0*WEIGHTS_INC].fields.op7 == 0)
                ? (Bias_T)(-inputs[0*INPUTS_INC]) : (Bias_T)inputs[0*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op6 == 0)
                ? (Bias_T)(-inputs[1*INPUTS_INC]) : (Bias_T)inputs[1*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op5 == 0)
                ? (Bias_T)(-inputs[2*INPUTS_INC]) : (Bias_T)inputs[2*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op4 == 0)
                ? (Bias_T)(-inputs[3*INPUTS_INC]) : (Bias_T)inputs[3*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op3 == 0)
                ? (Bias_T)(-inputs[4*INPUTS_INC]) : (Bias_T)inputs[4*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op2 == 0)
                ? (Bias_T)(-inputs[5*INPUTS_INC]) : (Bias_T)inputs[5*INPUTS_INC]);
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 1
                && NB_ITERATIONS == 7 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum += ((weights[0*WEIGHTS_INC].fields.op7 == 0)
                ? (Bias_T)(-inputs[0*INPUTS_INC]) : (Bias_T)inputs[0*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op6 == 0)
                ? (Bias_T)(-inputs[1*INPUTS_INC]) : (Bias_T)inputs[1*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op5 == 0)
                ? (Bias_T)(-inputs[2*INPUTS_INC]) : (Bias_T)inputs[2*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op4 == 0)
                ? (Bias_T)(-inputs[3*INPUTS_INC]) : (Bias_T)inputs[3*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op3 == 0)
                ? (Bias_T)(-inputs[4*INPUTS_INC]) : (Bias_T)inputs[4*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op2 == 0)
                ? (Bias_T)(-inputs[5*INPUTS_INC]) : (Bias_T)inputs[5*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op1 == 0)
                ? (Bias_T)(-inputs[6*INPUTS_INC]) : (Bias_T)inputs[6*INPUTS_INC]);
    }

    template<int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Input_T,
             typename Weight_T, typename Bias_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 1
                && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static Bias_T octoMac(const Input_T* __restrict inputs,
                                            const Weight_T* __restrict weights,
                                            Bias_T weightedSum)
    {
        weightedSum += ((weights[0*WEIGHTS_INC].fields.op7 == 0)
                ? (Bias_T)(-inputs[0*INPUTS_INC]) : (Bias_T)inputs[0*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op6 == 0)
                ? (Bias_T)(-inputs[1*INPUTS_INC]) : (Bias_T)inputs[1*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op5 == 0)
                ? (Bias_T)(-inputs[2*INPUTS_INC]) : (Bias_T)inputs[2*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op4 == 0)
                ? (Bias_T)(-inputs[3*INPUTS_INC]) : (Bias_T)inputs[3*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op3 == 0)
                ? (Bias_T)(-inputs[4*INPUTS_INC]) : (Bias_T)inputs[4*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op2 == 0)
                ? (Bias_T)(-inputs[5*INPUTS_INC]) : (Bias_T)inputs[5*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op1 == 0)
                ? (Bias_T)(-inputs[6*INPUTS_INC]) : (Bias_T)inputs[6*INPUTS_INC])
            + ((weights[0*WEIGHTS_INC].fields.op0 == 0)
                ? (Bias_T)(-inputs[7*INPUTS_INC]) : (Bias_T)inputs[7*INPUTS_INC]);

        return weightedSum;
    }

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits == 1
                && NB_ITERATIONS >= 8 && std::numeric_limits<Input_T>::digits == 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        weightedSum = octoMac<INPUTS_INC, WEIGHTS_INC>(inputs, weights, weightedSum);
        macsOnRange<NB_ITERATIONS-8, INPUTS_INC, WEIGHTS_INC>(
            inputs + 8*INPUTS_INC, weights + WEIGHTS_INC, weightedSum);
    }


    /***************************************************************************************
    *****************************************0-iterations***********************************
    ***************************************************************************************/

    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<NB_ITERATIONS == 0>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
    }

    /***************************************************************************************
    **************************************>=8-bit******************************************
    ***************************************************************************************/
    template<int NB_ITERATIONS,
             int INPUTS_INC = 1,
             int WEIGHTS_INC = 1,
             typename Weight_T, typename Bias_T,
             class Input_T,
             typename std::enable_if<(std::numeric_limits<Weight_T>::digits >= 8
                && std::numeric_limits<Input_T>::digits >= 8)>::type* = nullptr>
    N2D2_ALWAYS_INLINE static void macsOnRange(const Input_T* __restrict inputs,
                                               const Weight_T* __restrict weights,
                                               Bias_T& __restrict weightedSum)
    {
        for (int iter = 0; iter < NB_ITERATIONS; ++iter) {
            weightedSum += inputs[iter*INPUTS_INC] * weights[iter*WEIGHTS_INC];
        }
    }

    /***************************************************************************************
    **************************************PACK_ACTIVATIONS**********************************
    ***************************************************************************************/
    template<typename Output_T>
    N2D2_ALWAYS_INLINE static void compact_data_during_loop (const uint8_t value,
                               Output_T* __restrict outputs,
                               int* outputOffset,
                               PackSupport* infoPack)
    {
        constexpr unsigned int mask = (1L << std::numeric_limits<Output_T>::digits) - 1;
        constexpr unsigned int nbSlot = ceil((double)8/std::numeric_limits<Output_T>::digits);

        infoPack->accumulator |= value & mask;
        infoPack->cptAccumulator += 1;

        if (infoPack->cptAccumulator == nbSlot) {
            outputs[*(outputOffset)] = infoPack->accumulator;
            infoPack->cptAccumulator = 0;
            infoPack->accumulator = 0;
            //infoPack->indexToWrite = 0;
            //*(outputOffset) += 1;
        }
        else {
            infoPack->accumulator <<= std::numeric_limits<Output_T>::digits;
        }
    }

    template<typename Output_T>
    N2D2_ALWAYS_INLINE static void compact_data_end_loop (Output_T* __restrict outputs,
                                int* outputOffset,
                                PackSupport* infoPack)
    {
        // if data still accumulated but not stored
        if (infoPack->cptAccumulator != 0) {
            constexpr unsigned int nbSlot = ceil((double)8/std::numeric_limits<Output_T>::digits);

            // Add extra zero to shift data to the left
            infoPack->cptAccumulator += 1;
            while (infoPack->cptAccumulator < nbSlot) {
                infoPack->accumulator <<= std::numeric_limits<Output_T>::digits;
                infoPack->cptAccumulator += 1;
            }
            outputs[*(outputOffset)] = infoPack->accumulator;
            infoPack->cptAccumulator = 0;
            infoPack->accumulator = 0;
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
N2D2_ALWAYS_INLINE inline void N2D2::Network::convcellPropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs,
    const Bias_T* __restrict biasses,
    const Weight_T* __restrict weights,
    const Rescaling_T& __restrict rescaling) const
{
    constexpr int NB_INPUT_COMPACT
        = ((NB_CHANNELS * std::numeric_limits<Input_T>::digits)
            + (NB_CHANNELS * std::numeric_limits<Input_T>::digits) % 8) / 8;

    PackSupport infoPack = { 0, 0 };

    int outputOffset = 0;

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

//#pragma omp parallel for collapse(2)
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
                    //int iOffset = NB_INPUT_COMPACT * iPos;
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

                    constexpr int NB_CHANNELS_BYTES
                        = ((NB_CHANNELS * std::numeric_limits<Weight_T>::digits)
                            + (NB_CHANNELS * std::numeric_limits<Weight_T>::digits)
                                % 8) / 8;
                    constexpr int W_BYTES = ((std::numeric_limits<Weight_T>::digits < 8)
                            ? NB_CHANNELS_BYTES : NB_CHANNELS);
                    const int wOffset = W_BYTES * (sxMin
                        + KERNEL_WIDTH * (syMin + sy + KERNEL_HEIGHT * output));

                    //if (!wrapInRange && (NB_CHANNELS == INPUT_MEM_STRIDE
                    if (!wrapInRange && (NB_CHANNELS == NB_INPUT_COMPACT
                        && ((PADDING_X == 0
                            && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                                || sxMax - sxMin == KERNEL_WIDTH)) && ((NB_CHANNELS*std::numeric_limits<Weight_T>::digits)%8 == 0)
                                && ((NB_CHANNELS*std::numeric_limits<Input_T>::digits)%8 == 0))
                    {
                            macsOnRange<KERNEL_WIDTH * NB_CHANNELS>(
                                (Input_T*)((uint8_t*)inputs + iOffset),
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

                            /*
                            if(std::numeric_limits<Input_T>::digits < 8){
                                //if(output == 0 && std::numeric_limits<Input_T>::digits == 4) std::cout << "std::numeric_limits<Input_T>::digits>0" << std::flush;
                                iOffsetInRange = iOffset
                                + sx * NB_INPUT_COMPACT;
                            }
                            */

                            if (wrapInRange
                                && iOffsetInRange >= INPUT_MEM_CONT_SIZE)
                            {
                                iOffsetInRange += INPUT_MEM_WRAP_OFFSET
                                            - INPUT_MEM_CONT_OFFSET
                                            - INPUT_MEM_CONT_SIZE;
                            }

                            macsOnRange<NB_CHANNELS>(
                                // same input line so no wrapping can occur
                                (Input_T*)((uint8_t*)inputs + iOffsetInRange), 
                                weights + wOffset + sx * W_BYTES,
                                weightedSum);
                        }
                    }
                }
               if(std::numeric_limits<Output_T>::digits < 8) {
                    int32_t output_val
                        = sat<Output_T>(weightedSum, output, ACTIVATION, rescaling);

                    unsigned int nbSlot = ceil((double)8/std::numeric_limits<Output_T>::digits);
                    outputOffset = oOffset + std::floor(output/nbSlot);
                    compact_data_during_loop(output_val, outputs, &outputOffset, &infoPack);
               }
               else{
                ((Output_T*)((uint8_t*)outputs + oOffset))[output]
                    = sat<Output_T>(weightedSum, output, ACTIVATION, rescaling);
               }
            }

            if(std::numeric_limits<Output_T>::digits < 8)
                compact_data_end_loop(outputs, &outputOffset, &infoPack);
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
N2D2_ALWAYS_INLINE inline void N2D2::Network::convcellDWPropagate(
    const Input_T* __restrict inputs,
    Output_T* __restrict outputs,
    const Bias_T* __restrict biasses,
    const Weight_T* __restrict weights,
    const Rescaling_T& __restrict rescaling) const
{
    PackSupport infoPack = { 0, 0 };

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

//#pragma omp parallel for collapse(2)
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


                    constexpr int NB_INT8 = ((KERNEL_WIDTH*std::numeric_limits<Weight_T>::digits)+(KERNEL_WIDTH*std::numeric_limits<Weight_T>::digits)%8)/8;

                    int nbInt8 = KERNEL_WIDTH;
                    int nbSlot_per_Int8 = 1;

                    if(std::numeric_limits<Weight_T>::digits < 8) {
                        wOffset = (NB_INT8 * (syMin + sy + KERNEL_HEIGHT * output));
                        nbInt8 = NB_INT8;
                        nbSlot_per_Int8 = 8/(size_t)std::numeric_limits<Weight_T>::digits;
                    }

                    if (!wrapInRange && ((PADDING_X == 0
                            && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                        || sxMax - sxMin == KERNEL_WIDTH))
                    {
                        macsOnRange<KERNEL_WIDTH, INPUT_MEM_STRIDE / sizeof(Input_T)>(
                            (Input_T*)((uint8_t*)inputs + iOffset) + channel, 
                            weights + wOffset, 
                            weightedSum);
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
                                weightedSum += ((Input_T*)((uint8_t*)inputs + iOffsetInRange))[channel]
                                    * weights[wOffset + sx];
                                */

                                //test for 4b only
                                if constexpr(std::numeric_limits<Weight_T>::digits == 4) {
                                    const Weight_T w = weights[wOffset + (iInt8+iInt8_start)];

                                    if(trueISlot == 0) {
                                        weightedSum += ((Input_T*)((uint8_t*)inputs + iOffsetInRange))[channel]
                                            *w.fields.op1;
                                    }
                                    else{
                                        weightedSum += ((Input_T*)((uint8_t*)inputs + iOffsetInRange))[channel]
                                            *w.fields.op0;
                                    }
                                }
                                else{
                                    weightedSum += ((Input_T*)((uint8_t*)inputs + iOffsetInRange))[channel]
                                        * weights[wOffset + (iInt8+iInt8_start)];
                                }
                            }
                        }
                    }
                }
                ((Output_T*)((uint8_t*)outputs + oOffset))[output]
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

    const int NB_COMPACT   = ( (NB_OUTPUTS * std::numeric_limits<Input_T>::digits)
                                      + (NB_OUTPUTS * std::numeric_limits<Input_T>::digits) % 8) / 8;

    //real nb slots for pooling accumulation in QAT case
    const unsigned int NB_SLOT_REAL = ceil((double)8/std::numeric_limits<Input_T>::digits);
    //max slot possible, needed for not QAT > 8 bits case! TODO optimize this!
    const unsigned int NB_SLOT = 8;

    PackSupport infoPack = { 0, 0 };

    int outputOffset = 0;

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

//#pragma omp parallel for collapse(2)
        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            for (int output = 0; output < NB_OUTPUTS; ++output) {
            //TODO: change when no extra inputs 0 are present in memory
            //for (int output = 0; output < NB_COMPACT; ++output) {
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
                    
                    if constexpr(std::numeric_limits<Input_T>::digits == 4) {
                        maxVal.fields.op0 = std::numeric_limits<Input_T>::lowest();
                        maxVal.fields.op1 = std::numeric_limits<Input_T>::lowest();
                    }

                    //std::array<Input_T,NB_SLOT> maxVal;
                    //maxVal.fill(std::numeric_limits<Input_T>::lowest());

                    for (int sy = 0; sy < POOL_HEIGHT; ++sy) {
                        if ((PADDING_Y != 0
                                || OUTPUTS_HEIGHT != OUTPUTS_HEIGHT_NOPAD)
                            && sy >= syMax - syMin)
                        {
                            break;
                        }

                        const int iPos = ((sxMin + ix)
                                            + CHANNELS_WIDTH * (iy + syMin + sy));
                        //int iOffset = NB_COMPACT * iPos;
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

                            //int iOffsetInRange = iOffset + output
                            //    + sx * NB_COMPACT;
                            int iOffsetInRange = iOffset + output * sizeof(Input_T)
                                + sx * INPUT_MEM_STRIDE;

                            if (wrapInRange &&
                                iOffsetInRange >= INPUT_MEM_CONT_SIZE)
                            {
                                iOffsetInRange += INPUT_MEM_WRAP_OFFSET
                                            - INPUT_MEM_CONT_OFFSET
                                            - INPUT_MEM_CONT_SIZE;
                            }

                            //4 bits example for now : unpack and find max for each input
                            //TODO :: write separate methods for each input range
                            if constexpr(std::numeric_limits<Input_T>::digits == 4){
                                const Input_T& in = inputs[iOffsetInRange];

                                if(in.fields.op1 > maxVal.fields.op1){
                                    maxVal.fields.op1 = in.fields.op1;
                                }
                                if(in.fields.op0 > maxVal.fields.op0){
                                    maxVal.fields.op0 = in.fields.op0;
                                }
                            }
                            else{
                                if(*((Input_T*)((uint8_t*)inputs + iOffsetInRange)) > maxVal) {
                                    maxVal = *((Input_T*)((uint8_t*)inputs + iOffsetInRange));
                                }
                            }
                        }
                    }

                    outputOffset = oOffset + output;
                    int32_t output_max[NB_SLOT];

                    if constexpr(std::numeric_limits<Input_T>::digits < 8){
                        compact_data_during_loop(maxVal.fields.op1, outputs, &outputOffset, &infoPack);
                        compact_data_during_loop(maxVal.fields.op0, outputs, &outputOffset, &infoPack);

                        //for(int iSlot = 0; iSlot < NB_SLOT_REAL; iSlot++){
                        //    output_max[iSlot] = maxVal[iSlot];
                        //    compact_data_during_loop(output_max[iSlot], outputs, &outputOffset, &infoPack);
                        //}
                    }
                    else{
                        ((Output_T*)((uint8_t*)outputs + oOffset))[output] = maxVal;
                    }
                }
                //TODO :: adapt and test average pooling with packed inputs
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

                            int iOffsetInRange = iOffset + output * sizeof(Input_T)
                                + sx * INPUT_MEM_STRIDE;

                            if (wrapInRange &&
                                iOffsetInRange >= INPUT_MEM_CONT_SIZE)
                            {
                                iOffsetInRange += INPUT_MEM_WRAP_OFFSET
                                            - INPUT_MEM_CONT_OFFSET
                                            - INPUT_MEM_CONT_SIZE;
                            }

                            sum += *((Input_T*)((uint8_t*)inputs + iOffsetInRange));
                        }
                    }

                    ((Output_T*)((uint8_t*)outputs + oOffset))[output]
                        = (Output_T) (sum / (POOL_HEIGHT * POOL_WIDTH));
                }
                else {
                    N2D2_THROW_OR_ABORT(std::runtime_error,
                        "The export only supports Max and Average pooling.");
                }
            }
            if(std::numeric_limits<Input_T>::digits < 8)
                compact_data_end_loop(outputs, &outputOffset, &infoPack);
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

    PackSupport infoPack = { 0, 0 };

    int outputOffset = 0;

//#pragma omp parallel for
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

            constexpr int NB_CHANNELS_BYTES
                = ((NB_CHANNELS * std::numeric_limits<Weight_T>::digits)
                    + (NB_CHANNELS * std::numeric_limits<Weight_T>::digits)
                        % 8) / 8;
            constexpr int W_BYTES = ((std::numeric_limits<Weight_T>::digits < 8)
                    ? NB_CHANNELS_BYTES : NB_CHANNELS);
            const int wOffset = W_BYTES * CHANNELS_WIDTH
                                    * (iy + CHANNELS_HEIGHT * och);

            if (!wrapInRange && INPUT_MEM_STRIDE == NB_CHANNELS &&
                ((NB_CHANNELS*std::numeric_limits<Weight_T>::digits)%8 == 0)) {
                macsOnRange<NB_CHANNELS * CHANNELS_WIDTH>(
                                (Input_T*)((uint8_t*)inputs + iOffset),
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
                                (Input_T*)((uint8_t*)inputs + iOffsetInRange),
                                weights + wOffset + ix * W_BYTES,
                                weightedSum);
                }
            }
        }
        if (std::numeric_limits<Output_T>::digits < 8) {
            int8_t output_val = sat<Output_T>(weightedSum, och, ACTIVATION, rescaling);
            unsigned int nbSlot = ceil((double)8/std::numeric_limits<Output_T>::digits);
            outputOffset = std::floor(och/nbSlot);
            compact_data_during_loop(output_val, outputs, &outputOffset, &infoPack);
        }
        //do not accumulate for the last fc which is int32
        else{
            outputs[och] = sat<Output_T>(weightedSum, och, ACTIVATION, rescaling);
        }
    }

    if (std::numeric_limits<Output_T>::digits < 8)
        compact_data_end_loop(outputs, &outputOffset, &infoPack);
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
    constexpr unsigned int NB_SLOT = std::ceil((double)8
        / std::numeric_limits<Output_T>::digits);
    unsigned int NB_OUTPUTS_COMPACT = NB_OUTPUTS / NB_SLOT;

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

                // if no "+" it is printed always as unsigned!
                for (int output = 0; output < NB_OUTPUTS_COMPACT; output++) {
                    if (std::is_floating_point<Output_T>::value)
                        fprintf(pFile, "%f", +(float)((Output_T*)((uint8_t*)outputs + oOffset))[output]);
                    else
                        fprintf(pFile, "%d", +((Output_T*)((uint8_t*)outputs + oOffset))[output]);

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
        for(int output = 0; output < NB_OUTPUTS_COMPACT; output++) {
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

                    // if no "+" it is printed always as unsigned!
                    if (std::is_floating_point<Output_T>::value)
                        fprintf(pFile, "%f", +(float)((Output_T*)((uint8_t*)outputs + oOffset))[output]);
                    else
                        fprintf(pFile, "%d", ((Output_T*)((uint8_t*)outputs + oOffset))[output]);
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
                    if (((Input_T*)((uint8_t*)inputs + iOffset))[ch] > maxInput) {
                        iMaxInput = ch;
                        maxInput = ((Input_T*)((uint8_t*)inputs + iOffset))[ch];
                    }
                }

                outputs[oPos] = static_cast<int32_t>(iMaxInput);
            }
            else {
                outputs[oPos] = (*((Input_T*)((uint8_t*)inputs + iOffset)) > threshold<Input_T>());
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
                    = sat<Output_T>(inputs[iOffset + ch], ch, Linear, rescaling);
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
