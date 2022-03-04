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

#include "Cell/NormalizeCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "Xnet/Network.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class NormalizeCell_Frame_Test_CUDA : public NormalizeCell_Frame_CUDA<Float_T> {
public:
    NormalizeCell_Frame_Test_CUDA(const DeepNet& deepNet, 
                            const std::string& name,
                            unsigned int nbOutputs,
                            Norm norm)
        : Cell(deepNet, name, nbOutputs),
          NormalizeCell(deepNet, name, nbOutputs, norm),
          NormalizeCell_Frame_CUDA(deepNet, name, nbOutputs, norm)
    {}
};

TEST(NormalizeCell_Frame_CUDA,
     propagate)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    NormalizeCell_Frame_Test_CUDA normalize(dn, "normalize",
                                     nbOutputs,
                                     NormalizeCell::L2);

    ASSERT_EQUALS(normalize.getName(), "normalize");
    ASSERT_EQUALS(normalize.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputs({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputs({8, 8, nbOutputs, 2});

    std::vector<Float_T> norms;

    for (unsigned int channel = 0; channel < inputs[0][0].size(); ++channel) {
        float norm = 0.0f;

        for (unsigned int output = 0; output < nbOutputs; ++output) {
            const float w = output + channel + 1;

            inputs[0](channel, output) = w;
            norm += w * w;
        }

        norms.push_back(std::sqrt(norm + 1.0e-6));
    }

    inputs.synchronizeHToD();

    normalize.addInput(inputs, diffOutputs);
    normalize.initialize();

    normalize.propagate();
    normalize.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(normalize.getOutputs());

    ASSERT_EQUALS(outputs.dimX(), inputs.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputs.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputs.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputs.dimB());

    for (unsigned int channel = 0; channel < inputs[0][0].size(); ++channel) {
        float sumSq = 0.0f;

        for (unsigned int output = 0; output < nbOutputs; ++output) {
            sumSq += outputs[0](channel, output) * outputs[0](channel, output);

            ASSERT_EQUALS_DELTA(outputs[0](channel, output),
                (output + channel + 1) / norms[channel], 1e-5);
        }

        ASSERT_EQUALS_DELTA(sumSq, 1.0f, 1e-5);
    }

    ASSERT_NOTHROW_ANY(normalize.checkGradient(1.0e-3, 1.0e-2));
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
