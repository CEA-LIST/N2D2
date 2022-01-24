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

#include "Cell/TransposeCell_Frame_CUDA.hpp"
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
class TransposeCell_Frame_CUDA_Test: public TransposeCell_Frame_CUDA<T> {
public:
    TransposeCell_Frame_CUDA_Test(const DeepNet& deepNet,
                          const std::string& name,
                          unsigned int nbOutputs,
                          const std::vector<int>& perm):
        Cell(deepNet, name, nbOutputs),
        TransposeCell(deepNet, name, nbOutputs, perm),
        TransposeCell_Frame_CUDA<T>(deepNet, name, nbOutputs, perm) 
    {                                
    }

    friend class UnitTest_TransposeCell_Frame_CUDA_float_propagate;
};

TEST(TransposeCell_Frame_CUDA, propagate)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    // NHWC -> NCHW
    //const std::vector<int> permutation = {1, 2, 0, 3};
    const std::vector<int> permutation = {2, 0, 1, 3};

    Network net(0U,false);
    DeepNet dn(net);

    Random::mtSeed(0);

    const unsigned int nbOutputs = 10;
    const unsigned int batchSize = 8;
    const unsigned int dimX = 3;
    const unsigned int dimY = 3;
    TransposeCell_Frame_CUDA_Test<Float_T> transpose(dn, "transpose",
        nbOutputs,
        permutation);

    const std::vector<int> perm = transpose.getPermutation();
    ASSERT_EQUALS(perm[0], permutation[0]);
    ASSERT_EQUALS(perm[1], permutation[1]);
    ASSERT_EQUALS(perm[2], permutation[2]);
    ASSERT_EQUALS(perm[3], permutation[3]);

    const std::vector<int> invPerm = transpose.getInversePermutation();
    ASSERT_EQUALS(invPerm[0], 1);
    ASSERT_EQUALS(invPerm[1], 2);
    ASSERT_EQUALS(invPerm[2], 0);
    ASSERT_EQUALS(invPerm[3], 3);

    ASSERT_EQUALS(transpose.getName(), "transpose");
    ASSERT_EQUALS(transpose.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputs({dimX, dimY, nbOutputs, batchSize});
    Tensor<Float_T> diffOutputs({dimX, dimY, nbOutputs, batchSize});

    for (unsigned int index = 0; index < inputs.size(); ++index)
        inputs(index) = Random::randUniform(-1.0, 1.0);


    transpose.addInput(inputs, diffOutputs);
    transpose.initialize();

    transpose.propagate();
    transpose.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(transpose.getOutputs());
    //ASSERT_NOTHROW_ANY(transpose.checkGradient(1.0e-3, 1.0e-3));

    ASSERT_EQUALS(outputs.dimX(), nbOutputs);
    ASSERT_EQUALS(outputs.dimY(), dimX);
    ASSERT_EQUALS(outputs.dimZ(), dimY);
    ASSERT_EQUALS(outputs.dimB(), batchSize);

    ASSERT_EQUALS_DELTA(std::accumulate(outputs.begin(), outputs.end(), 0.0),
    std::accumulate(inputs.begin(), inputs.end(), 0.0), 1.0e-12);

    std::size_t coords[4];
    for (coords[3] = 0; coords[3] < inputs.dims()[3]; ++coords[3]) {
        for (coords[2] = 0; coords[2] < inputs.dims()[2]; ++coords[2]) {
            for (coords[1] = 0; coords[1] < inputs.dims()[1]; ++coords[1]) {
                for (coords[0] = 0; coords[0] < inputs.dims()[0]; ++coords[0])
                {
                    ASSERT_EQUALS_DELTA(
                        outputs(coords[perm[0]], coords[perm[1]],
                                coords[perm[2]], coords[perm[3]]),
                        inputs(coords[0], coords[1],
                               coords[2], coords[3]),
                        1.0e-12);
                }
            }
        }
    }


    transpose.mDiffInputs = transpose.getOutputs();
    transpose.mDiffInputs.setValid();
    transpose.mDiffInputs.synchronizeHToD();

    transpose.backPropagate();
    transpose.mDiffOutputs[0].synchronizeDToH();

    ASSERT_EQUALS_DELTA(std::accumulate(diffOutputs.begin(), diffOutputs.end(), 0.0),
    std::accumulate(inputs.begin(), inputs.end(), 0.0), 1.0e-12);

    for (coords[3] = 0; coords[3] < inputs.dims()[3]; ++coords[3]) {
        for (coords[2] = 0; coords[2] < inputs.dims()[2]; ++coords[2]) {
            for (coords[1] = 0; coords[1] < inputs.dims()[1]; ++coords[1]) {
                for (coords[0] = 0; coords[0] < inputs.dims()[0]; ++coords[0])
                {
                    ASSERT_EQUALS_DELTA(
                        diffOutputs(coords[0], coords[1],
                                coords[2], coords[3]),
                        inputs(coords[0], coords[1],
                               coords[2], coords[3]),
                        1.0e-12);
                }
            }
        }
    }
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
