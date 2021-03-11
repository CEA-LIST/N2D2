/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifdef CUDA

#include "N2D2.hpp"

#include "Cell/ReshapeCell_Frame_CUDA.hpp"
#include "containers/Tensor.hpp"
#include "DeepNet.hpp"
#include "Xnet/Network.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Random.hpp"

#include <limits>
#include <string>
#include <tuple>
#include <vector>

using namespace N2D2;

template <class T>
class ReshapeCell_Frame_CUDA_Test: public ReshapeCell_Frame_CUDA<T> {
public:
    ReshapeCell_Frame_CUDA_Test(const DeepNet& deepNet,
                          const std::string& name,
                          unsigned int nbOutputs,
                          const std::vector<int>& dims):
        Cell(deepNet, name, nbOutputs),
        ReshapeCell(deepNet, name, nbOutputs, dims),
        ReshapeCell_Frame_CUDA<T>(deepNet, name, nbOutputs, dims) 
    {                                
    }

    friend class UnitTest_ReshapeCell_Frame_CUDA_float_propagate;
};

TEST(ReshapeCell_Frame_CUDA, propagate)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    // NHWC -> NCHW
    const unsigned int nbOutputs = 10;
    const std::vector<int> shape = {2, 1, nbOutputs, 1};

    Network net;
    DeepNet dn(net);

    Random::mtSeed(0);

    ReshapeCell_Frame_CUDA_Test<Float_T> reshape(dn, "reshape",
        nbOutputs,
        shape);

    ASSERT_EQUALS(reshape.getName(), "reshape");
    ASSERT_EQUALS(reshape.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputs({nbOutputs, 2, 1, 1});
    Tensor<Float_T> diffOutputs({nbOutputs, 2, 1, 1});

    for (unsigned int index = 0; index < inputs.size(); ++index)
        inputs(index) = Random::randUniform(-1.0, 1.0);


    reshape.addInput(inputs, diffOutputs);
    reshape.initialize();

    reshape.propagate();
    reshape.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(reshape.getOutputs());
    //ASSERT_NOTHROW_ANY(reshape.checkGradient(1.0e-3, 1.0e-3));

    ASSERT_EQUALS(outputs.dimX(), shape[0]);
    ASSERT_EQUALS(outputs.dimY(), shape[1]);
    ASSERT_EQUALS(outputs.dimZ(), shape[2]);
    ASSERT_EQUALS(outputs.dimB(), shape[3]);

    for (unsigned int index = 0; index < outputs.size(); ++index) {
        ASSERT_EQUALS_DELTA(outputs(index), inputs(index), 1.0e-12);
    }
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
