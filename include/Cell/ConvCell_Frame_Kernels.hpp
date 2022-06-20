/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_CONVCELL_FRAME_KERNELS_H
#define N2D2_CONVCELL_FRAME_KERNELS_H

#include <vector>
#include "containers/Tensor.hpp"
#include "Cell/Cell.hpp"

namespace N2D2 {

namespace ConvCell_Frame_Kernels {
    /**
     * @brief Convolution layer features description
     */
    struct Descriptor {
        const std::vector<unsigned int> subSample; // sub-sampling sizes to apply after the convolution ({subSampleX, subSampleY})
        const std::vector<unsigned int> stride; // stride value ({strideX, strideY})
        std::vector<int> padding; // One-value vector or four-value (left, top, right and bottom) if 2D features
        const std::vector<unsigned int> dilation; // dilatation sizes ({dilatationX, dilatationY})

        Descriptor(const std::vector<unsigned int>& subSample_,
                   const std::vector<unsigned int>& stride_,
                   const std::vector<int>& padding_,
                   const std::vector<unsigned int>& dilation_)
            : subSample(subSample_),
              stride(stride_),
              padding(padding_),
              dilation(dilation_)
        {
            if (padding.size() == stride.size()) {
                // Duplicate left, top padding for right, bottom padding
                for (std::size_t i = 0; i < stride.size(); ++i)
                    padding.push_back(padding[i]);
            }
        }
    };

    /*===================
            Forward
    =====================*/

    /**
     * @brief Convolutional layer forward function.
     * 
     * @tparam T Features type.
     * @param alpha Propagation coefficient applied to the weighted sum.
     * @param inputs One of the input tensors.
     * @param sharedSynapses Weight tensor for the expected input.
     * @param desc Convolutional layer feature description.
     * @param beta Accumulation coefficient to sum the output from several input tensors.
     * @param outputs Output tensor.
     * @param maps Connections from input channels to output channels.
     */
    template <class T>
    void forward(const T* alpha,
                 const Tensor<T>& inputs,
                 const Tensor<T>& sharedSynapses,
                 const Descriptor& desc,
                 const T* beta,
                 Tensor<T>& outputs,
                 const Tensor<bool>& maps = Tensor<bool>());

    /**
     * @brief Convolutional layer forward function.
     * 
     * @tparam T Biais type.
     * @param alpha Propagation coefficient applied to the biais before summing it with the weighted sum.
     * @param bias Layer's biais tensor.
     * @param beta Propagation coefficient applied to the weighted sum before summing with the biais.
     * @param outputs Output tensor.
     */
    template <class T>
    void forwardBias(const T* alpha,
                     const Tensor<T>& bias,
                     const T* beta,
                     Tensor<T>& outputs);

    /*===================
            Backward
    =====================*/
    /**
     * @brief Implements gradient backpropagation for features.
     * 
     * @tparam T Data type
     * @param alpha Propagation coefficient applied to the computed gradient before being summed and passed on.
     * @param sharedSynapses Weight tensor for the expected input.
     * @param diffInputs Input gradient tensor.
     * @param desc Convolutional layer feature description.
     * @param beta Accumulation coefficient to sum the output gradient from several input gradient tensors.
     * @param diffOutputs Output gradient tensor.
     * @param maps Connections from input channels to output channels.
     */
    template <class T>
    void backwardData(const T* alpha,
                      const Tensor<T>& sharedSynapses,
                      const Tensor<T>& diffInputs,
                      const Descriptor& desc,
                      const T* beta,
                      Tensor<T>& diffOutputs,
                      const Tensor<bool>& maps = Tensor<bool>());

    /**
     * @brief Implements gradient backpropagation for weights.
     * 
     * @tparam T Data type
     * @param alpha Coefficient applied to the computed gradient before being summed.
     * @param inputs Input tensor computed during the forward step for the current layer.
     * @param diffInputs Input gradient tensor.
     * @param desc Convolutional layer feature description.
     * @param beta Accumulation coefficient to sum the computed weight gradient from several input gradient tensors.
     * @param diffSharedSynapses Weights gradient tensor for this layer for a single input features tensor.
     * @param maps Connections from input channels to output channels.
     */
    template <class T>
    void backwardFilter(const T* alpha,
                        const Tensor<T>& inputs,
                        const Tensor<T>& diffInputs,
                        const Descriptor& desc,
                        const T* beta,
                        Tensor<T>& diffSharedSynapses,
                        const Tensor<bool>& maps = Tensor<bool>());
    
    /**
     * @brief Implements gradient backpropagation for features.
     * 
     * @tparam T Data type
     * @param alpha Coefficient applied to the computed gradient before being summed.
     * @param diffInputs Input gradient tensor.
     * @param beta Accumulation coefficient to sum the bias gradient from several input gradient tensors.
     * @param diffBias Output bias gradient tensor.
     */
    template <class T>
    void backwardBias(const T* alpha,
                      const Tensor<T>& diffInputs,
                      const T* beta,
                      Tensor<T>& diffBias);
}
}

#endif // N2D2_CONVCELL_FRAME_KERNELS_H
