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

#include "N2D2.hpp"

#include "Activation/RectifierActivation_Frame.hpp"
#include "Cell/ActivationCell_Frame.hpp"
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
class ActivationCell_Frame_Test: public ActivationCell_Frame<T> {
public:
    ActivationCell_Frame_Test(const DeepNet& deepNet,
                          const std::string& name,
                          unsigned int nbOutputs,
                          const std::shared_ptr<Activation>& activation):
        Cell(deepNet, name, nbOutputs),
        ActivationCell(deepNet, name, nbOutputs),
        ActivationCell_Frame<T>(deepNet, name, nbOutputs, activation) 
    {                                
    }

    friend class UnitTest_ActivationCell_Frame_float_propagate;
};

TEST(ActivationCell_Frame, propagate)
{
    Network net;
    DeepNet dn(net);

    Random::mtSeed(0);

    const unsigned int nbOutputs = 10;
    ActivationCell_Frame_Test<Float_T> act(dn, "act",
        nbOutputs,
        std::make_shared<RectifierActivation_Frame<Float_T> >());

    ASSERT_EQUALS(act.getName(), "act");
    ASSERT_EQUALS(act.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputs({1, 1, nbOutputs, 1});
    Tensor<Float_T> diffOutputs({1, 1, nbOutputs, 1});

    for (unsigned int index = 0; index < inputs.size(); ++index)
        inputs(index) = Random::randUniform(-1.0, 1.0);


    act.addInput(inputs, diffOutputs);
    act.initialize();

    act.propagate();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(act.getOutputs());
    //ASSERT_NOTHROW_ANY(act.checkGradient(1.0e-3, 1.0e-3));

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            std::max(0.0f, inputs(o)),
                            1.0e-6);
    }

    ASSERT_EQUALS(outputs.dimB(), 1);
    ASSERT_EQUALS(outputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(outputs.dimX(), 1);
    ASSERT_EQUALS(outputs.dimY(), 1);

}

RUN_TESTS()
