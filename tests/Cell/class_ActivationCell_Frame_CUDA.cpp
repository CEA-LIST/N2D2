/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#include "Cell/ActivationCell_Frame_CUDA.hpp"
#include "Activation/LinearActivation_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "Xnet/Network.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;
template <class T>
class ActivationCell_Frame_CUDA_Test : public ActivationCell_Frame_CUDA<T> {
public:
    ActivationCell_Frame_CUDA_Test(const DeepNet& deepNet, 
                                 const std::string& name,
                                 unsigned int nbOutputs,
                   const std::shared_ptr<Activation>& activation
                   = std::shared_ptr<Activation>())
        : Cell(deepNet, name, nbOutputs),
          ActivationCell(deepNet, name, nbOutputs),
          ActivationCell_Frame_CUDA<T>(deepNet, name, nbOutputs, activation) {}
    friend class UnitTest_ActivationCell_Frame_CUDA_float_propagate;
};

TEST(ActivationCell_Frame_CUDA_float, propagate)
{
    REQUIRED(UnitTest::CudaDeviceExists(0));

    Network net;
    DeepNet dn(net);

    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;
    const std::shared_ptr<Activation>& actOperator 
            = std::make_shared<RectifierActivation_Frame_CUDA<float> >();
    const auto actOperator_ptr
        = std::dynamic_pointer_cast<RectifierActivation>(actOperator);
    actOperator_ptr->setParameter("Clipping", 6.0);
    ActivationCell_Frame_CUDA_Test<float> activation(  dn, 
                                                "activation",
                                                nbOutputs,
                                                actOperator);

    ASSERT_EQUALS(activation.getName(), "activation");
    ASSERT_EQUALS(activation.getNbOutputs(), nbOutputs);

    Tensor<float> inputs({8, 8, nbOutputs, 2});
    Tensor<float> diffOutputs({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputs.size(); ++index) {
        inputs(index) = Random::randUniform(-1.0, 1.0);
    }

    inputs.synchronizeHToD();

    activation.addInput(inputs, diffOutputs);
    activation.initialize();

    activation.propagate();
    activation.getOutputs().synchronizeDToH();
    const Tensor<float>& outputs = tensor_cast<float>(activation.getOutputs());

    ASSERT_EQUALS(outputs.dimX(), inputs.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputs.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputs.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputs.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        if(inputs(o) > 0) {
            ASSERT_EQUALS_DELTA(outputs(o), inputs(o), 1.0e-9);
        }
        else {
            ASSERT_EQUALS_DELTA(outputs(o), 0.0f, 1.0e-9);
        }
    }

    //ASSERT_NOTHROW_ANY(activation.checkGradient(1.0e-3, 1.0e-2));
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
