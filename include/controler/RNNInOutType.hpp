/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): Thibault ALLENET (thibault.allenet@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
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
#ifndef N2D2_RNNInOutType_H
#define N2D2_RNNInOutType_H

#include <vector>
#include "CudaContext.hpp"


namespace N2D2 {
/**
 * @class   RNNInOutType
 * @brief   Gives a vector of tensor descriptors.
*/
    template <typename T>
    class RNNInOutType {
        public:

        RNNInOutType(size_t n,
                     const std::vector<int>& dim,
                     const std::vector<int>& stride)
        {
            descs.resize(n);
            for (size_t i = 0; i < n; ++i) {
                cudnnCreateTensorDescriptor(&descs[i]);
                cudnnSetTensorNdDescriptor(descs[i],
                                           CudaContext::data_type<T>::value,
                                           dim.size(),
                                           dim.data(),
                                           stride.data());
            }
        };

        ~RNNInOutType() {
            for (auto desc : descs) {
                cudnnDestroyTensorDescriptor(desc);
            }
        };

        const cudnnTensorDescriptor_t* getdescs() const {
            return descs.data();
        }

        private:
        std::vector<cudnnTensorDescriptor_t> descs;
    };

}

#endif // N2D2_RNNInOutType_H
