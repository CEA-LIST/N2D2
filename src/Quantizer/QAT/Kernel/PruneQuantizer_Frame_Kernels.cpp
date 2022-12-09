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

#include "Quantizer/QAT/Kernel/PruneQuantizer_Frame_Kernels.hpp"
#include "third_party/half.hpp"
#include "utils/Utils.hpp"
#include "utils/Random.hpp"

namespace N2D2 {
namespace PruneQuantizer_Frame_Kernels {

template <class T>
void apply_pruning_with_masks(Tensor<T>& data,
                              Tensor<T>& dataPruned,
                              Tensor<unsigned int>& masks)
{
#pragma omp parallel for if (dataPruned.size() > 1024)
    for (unsigned int i = 0; i < dataPruned.size(); ++i) {
        dataPruned(i) = data(i) * (T)masks(i);
    }
}

void update_masks_random(Tensor<unsigned int>& masks,
                         float threshold)
{
    unsigned int nbzero = (masks.eq(0)).sum();

    // Possible not to add 0 if threshold already satisfied
    unsigned int nbzero_to_add = (unsigned int)std::max((int)0, (int)std::ceil(threshold * masks.size()) - (int)nbzero);

    for (unsigned int i = 0; i < nbzero_to_add; ++i) {
        bool isZeroAdded = false;

        while (!isZeroAdded) {

            // Choice of the index
            unsigned int ind = Random::randUniform(0, masks.size());

            while (ind != masks.size() && masks(ind) == 0U) {
                ++ind;
            }

            if (ind != masks.size() && masks(ind) == 1U) {
                masks(ind) = 0U;
                isZeroAdded = true;
            }

        }
    }
}

template <class T>
void update_masks_iter_nonstruct(Tensor<T>& data,
                                 Tensor<unsigned int>& masks,
                                 float threshold,
                                 float delta)
{
    unsigned int zero_counter = 0;
    unsigned int iter_delta = 0;

    // Find the correct iter delta
    while ((float)zero_counter / data.size() < threshold) {
        ++iter_delta;
        zero_counter = 0;
        for (unsigned int i = 0; i < data.size(); ++i) {
            if (abs(data(i)) <= delta * iter_delta) {
                ++zero_counter;
            }
        }
    }

    // Fill the mask tensor
    for (unsigned int i = 0; i < data.size(); ++i) {
        masks(i) = (abs(data(i)) <= delta * iter_delta) ? 0U : 1U;
    }
}

}

// ----------------------------------------------------------------------------
// ----------------------------- SPECIALIZATIONS ------------------------------
// ----------------------------------------------------------------------------


template void PruneQuantizer_Frame_Kernels::apply_pruning_with_masks<half_float::half>(Tensor<half_float::half>& data,
                                                                                       Tensor<half_float::half>& dataPruned,
                                                                                       Tensor<unsigned int>& masks);
template void PruneQuantizer_Frame_Kernels::apply_pruning_with_masks<float>(Tensor<float>& data,
                                                                            Tensor<float>& dataPruned,
                                                                            Tensor<unsigned int>& masks);
template void PruneQuantizer_Frame_Kernels::apply_pruning_with_masks<double>(Tensor<double>& data,
                                                                             Tensor<double>& dataPruned,
                                                                             Tensor<unsigned int>& masks);


template void PruneQuantizer_Frame_Kernels::update_masks_iter_nonstruct<half_float::half>(Tensor<half_float::half>& data,
                                                                                          Tensor<unsigned int>& masks,
                                                                                          float threshold,
                                                                                          float delta);
template void PruneQuantizer_Frame_Kernels::update_masks_iter_nonstruct<float>(Tensor<float>& data,
                                                                               Tensor<unsigned int>& masks,
                                                                               float threshold,
                                                                               float delta);
template void PruneQuantizer_Frame_Kernels::update_masks_iter_nonstruct<double>(Tensor<double>& data,
                                                                                Tensor<unsigned int>& masks,
                                                                                float threshold,
                                                                                float delta);


}