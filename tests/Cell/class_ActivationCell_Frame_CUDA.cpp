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

TEST_DATASET(ActivationCell_Frame_CUDA_float, propagate,
    (double clipping, unsigned int nbOutputs),
    std::make_tuple(1.0, 4U),
    std::make_tuple(10.0, 10U),
    std::make_tuple(30.0, 128U))
{
    REQUIRED(UnitTest::CudaDeviceExists(0));

    Network net;
    DeepNet dn(net);

    Random::mtSeed(0);

    const std::shared_ptr<Activation>& actOperator 
            = std::make_shared<RectifierActivation_Frame_CUDA<float> >();
    const auto actOperator_ptr
        = std::dynamic_pointer_cast<RectifierActivation>(actOperator);
    actOperator_ptr->setParameter("Clipping", clipping);
    ActivationCell_Frame_CUDA_Test<float> activation(  dn, 
                                                "activation",
                                                nbOutputs,
                                                actOperator);

    ASSERT_EQUALS(activation.getName(), "activation");
    ASSERT_EQUALS(activation.getNbOutputs(), nbOutputs);

    Tensor<float> inputs({8, 8, nbOutputs, 2});
    Tensor<float> diffOutputs({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputs.size(); ++index) {
        inputs(index) = (float) Random::randUniform(-3.0*clipping, clipping*2.0);
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
        ASSERT_EQUALS_DELTA(std::min(std::max(0.0f, inputs(o)), (float) clipping), outputs(o), 1.0e-9);
    }

    //ASSERT_NOTHROW_ANY(activation.checkGradient(1.0e-3, 1.0e-2));
}

TEST_DATASET(ActivationCell_Frame_CUDA_float, propagate_2_inputs,
    (double clipping, unsigned int nbInputs_1, unsigned int nbInputs_2, unsigned int batchSize),
    std::make_tuple(0.0, 4U, 4U, 1U)
    ,std::make_tuple(0.0, 10U, 3U, 3U)
    ,std::make_tuple(7034.0, 53U, 128U, 12U))
{
    REQUIRED(UnitTest::CudaDeviceExists(0));

    Network net;
    DeepNet dn(net);

    Random::mtSeed(0);

    const std::shared_ptr<Activation>& actOperator 
            = std::make_shared<RectifierActivation_Frame_CUDA<float> >();
    const auto actOperator_ptr
        = std::dynamic_pointer_cast<RectifierActivation>(actOperator);
    actOperator_ptr->setParameter("Clipping", clipping);
    const unsigned int nbOutputs = nbInputs_1 + nbInputs_2;
    ActivationCell_Frame_CUDA_Test<float> activation(  dn, 
                                                "activation",
                                                nbOutputs,
                                                actOperator);

    ASSERT_EQUALS(activation.getName(), "activation");
    ASSERT_EQUALS(activation.getNbOutputs(), nbOutputs);

    Tensor<float> inputs_1({8, 8, nbInputs_1, batchSize});
    Tensor<float> diffOutputs_1({8, 8, nbInputs_1, batchSize});

    Tensor<float> inputs_2({8, 8, nbInputs_2, batchSize});
    Tensor<float> diffOutputs_2({8, 8, nbInputs_2, batchSize});
    const size_t input_1_length = inputs_1.dimX() * inputs_1.dimY() * inputs_1.dimZ();
    const size_t input_2_length = inputs_2.dimX() * inputs_2.dimY() * inputs_2.dimZ();

    for (unsigned int b = 0; b < batchSize; ++b) {
        for(unsigned int z = 0; z < inputs_1.dimZ(); ++z) {
            for(unsigned int y = 0; y < inputs_1.dimY(); ++y) {
                for(unsigned int x = 0; x < inputs_1.dimX(); ++x) {
                    const int idx = b*(input_2_length + input_1_length)
                                + z * inputs_1.dimX() * inputs_1.dimY()
                                + y * inputs_1.dimX()
                                + x;
                    inputs_1(x,y,z,b) = idx;
                }
            }
        }
    }

    for (unsigned int b = 0; b < batchSize; ++b) {
        for(unsigned int z = 0; z < inputs_2.dimZ(); ++z) {
            for(unsigned int y = 0; y < inputs_2.dimY(); ++y) {
                for(unsigned int x = 0; x < inputs_2.dimX(); ++x) {
                    const int idx = input_1_length
                                + b*(input_2_length + input_1_length)
                                + z * inputs_2.dimX() * inputs_2.dimY()
                                + y * inputs_2.dimX()
                                + x;

                    inputs_2(x,y,z,b) = idx;
                }
            }
        }
    }
        

    inputs_1.synchronizeHToD();
    inputs_2.synchronizeHToD();

    activation.addInput(inputs_1, diffOutputs_1);
    activation.addInput(inputs_2, diffOutputs_2);

    activation.initialize();

    activation.propagate();
    activation.getOutputs().synchronizeDToH();
    const Tensor<float>& outputs = tensor_cast<float>(activation.getOutputs());
    ASSERT_EQUALS(outputs.dimX(), inputs_1.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputs_1.dimY());

    //ASSERT_EQUALS(outputs.dimX(), inputs_2.dimX());
    //ASSERT_EQUALS(outputs.dimY(), inputs_2.dimY());
    //ASSERT_EQUALS(outputs.dimZ(), inputs_1.dimZ() + inputs_2.dimZ());
    ASSERT_EQUALS(outputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(outputs.dimB(), inputs_1.dimB());
    //ASSERT_EQUALS(outputs.dimB(), inputs_2.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        if(clipping == 0.0) {
            ASSERT_EQUALS_DELTA(o, outputs(o), 1.0e-9);
        } else {
            ASSERT_EQUALS_DELTA(std::min((double) o, clipping), outputs(o), 1.0e-9);
        }
    }

    //ASSERT_NOTHROW_ANY(activation.checkGradient(1.0e-3, 1.0e-2));
}


TEST_DATASET(ActivationCell_Frame_CUDA_float, propagate_3_inputs,
    (double clipping, unsigned int nbInputs_1, unsigned int nbInputs_2, unsigned int nbInputs_3,  unsigned int batchSize),
    std::make_tuple(0.0, 4U, 4U, 1U, 1U)
    ,std::make_tuple(0.0, 10U, 3U, 22U, 3U)
    ,std::make_tuple(14023.0, 53U, 128U, 12U, 12U)
    ,std::make_tuple(32.0, 29U, 56U, 512U, 3U)
    ,std::make_tuple(0.0, 156U, 32U, 63U, 12U))
{
    REQUIRED(UnitTest::CudaDeviceExists(0));

    Network net;
    DeepNet dn(net);

    Random::mtSeed(0);

    const std::shared_ptr<Activation>& actOperator 
            = std::make_shared<RectifierActivation_Frame_CUDA<float> >();
    const auto actOperator_ptr
        = std::dynamic_pointer_cast<RectifierActivation>(actOperator);
    actOperator_ptr->setParameter("Clipping", clipping);
    const unsigned int nbOutputs = nbInputs_1 + nbInputs_2 + nbInputs_3;
    ActivationCell_Frame_CUDA_Test<float> activation(  dn, 
                                                "activation",
                                                nbOutputs,
                                                actOperator);

    ASSERT_EQUALS(activation.getName(), "activation");
    ASSERT_EQUALS(activation.getNbOutputs(), nbOutputs);

    Tensor<float> inputs_1({8, 8, nbInputs_1, batchSize});
    Tensor<float> diffOutputs_1({8, 8, nbInputs_1, batchSize});

    Tensor<float> inputs_2({8, 8, nbInputs_2, batchSize});
    Tensor<float> diffOutputs_2({8, 8, nbInputs_2, batchSize});

    Tensor<float> inputs_3({8, 8, nbInputs_3, batchSize});
    Tensor<float> diffOutputs_3({8, 8, nbInputs_3, batchSize});

    const size_t input_1_length = inputs_1.dimX() * inputs_1.dimY() * inputs_1.dimZ();
    const size_t input_2_length = inputs_2.dimX() * inputs_2.dimY() * inputs_2.dimZ();
    const size_t input_3_length = inputs_3.dimX() * inputs_3.dimY() * inputs_3.dimZ();

    for (unsigned int b = 0; b < batchSize; ++b) {
        for(unsigned int z = 0; z < inputs_1.dimZ(); ++z) {
            for(unsigned int y = 0; y < inputs_1.dimY(); ++y) {
                for(unsigned int x = 0; x < inputs_1.dimX(); ++x) {
                    const int idx = b*(input_2_length + input_1_length + input_3_length)
                                + z * inputs_1.dimX() * inputs_1.dimY()
                                + y * inputs_1.dimX()
                                + x;
                    inputs_1(x,y,z,b) = idx;
                }
            }
        }
    }

    for (unsigned int b = 0; b < batchSize; ++b) {
        for(unsigned int z = 0; z < inputs_2.dimZ(); ++z) {
            for(unsigned int y = 0; y < inputs_2.dimY(); ++y) {
                for(unsigned int x = 0; x < inputs_2.dimX(); ++x) {
                    const int idx = input_1_length
                                + b*(input_2_length + input_1_length + input_3_length)
                                + z * inputs_2.dimX() * inputs_2.dimY()
                                + y * inputs_2.dimX()
                                + x;

                    inputs_2(x,y,z,b) = idx;
                }
            }
        }
    }
        
    for (unsigned int b = 0; b < batchSize; ++b) {
        for(unsigned int z = 0; z < inputs_3.dimZ(); ++z) {
            for(unsigned int y = 0; y < inputs_3.dimY(); ++y) {
                for(unsigned int x = 0; x < inputs_3.dimX(); ++x) {
                    const int idx = input_1_length + input_2_length
                                + b*(input_2_length + input_1_length + input_3_length)
                                + z * inputs_3.dimX() * inputs_3.dimY()
                                + y * inputs_3.dimX()
                                + x;

                    inputs_3(x,y,z,b) = idx;
                }
            }
        }
    }
        
    inputs_1.synchronizeHToD();
    inputs_2.synchronizeHToD();
    inputs_3.synchronizeHToD();

    activation.addInput(inputs_1, diffOutputs_1);
    activation.addInput(inputs_2, diffOutputs_2);
    activation.addInput(inputs_3, diffOutputs_3);

    activation.initialize();

    activation.propagate();
    activation.getOutputs().synchronizeDToH();
    const Tensor<float>& outputs = tensor_cast<float>(activation.getOutputs());
    ASSERT_EQUALS(outputs.dimX(), inputs_1.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputs_1.dimY());

    ASSERT_EQUALS(outputs.dimX(), inputs_2.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputs_2.dimY());
    ASSERT_EQUALS(outputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(outputs.dimB(), inputs_1.dimB());
    ASSERT_EQUALS(outputs.dimB(), inputs_2.dimB());
    ASSERT_EQUALS(outputs.dimB(), inputs_3.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        if(clipping == 0.0) {
            ASSERT_EQUALS_DELTA(o, outputs(o), 1.0e-9);
        } else {
            ASSERT_EQUALS_DELTA(std::min((double) o, clipping), outputs(o), 1.0e-9);
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
