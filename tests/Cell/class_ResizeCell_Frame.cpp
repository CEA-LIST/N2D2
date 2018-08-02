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

#include "Cell/ResizeCell_Frame.hpp"
#include "Network.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Random.hpp"

using namespace N2D2;


class ResizeCell_Frame_Test : public ResizeCell_Frame {
public:
    ResizeCell_Frame_Test(const std::string& name,
                            unsigned int outputWidth,
                            unsigned int outputHeight,
                            unsigned int nbOutputs,
                            ResizeMode resizeMode)
        : Cell(name, nbOutputs),
          ResizeCell(name,
                     outputWidth,
                     outputHeight,
                     nbOutputs,
                     resizeMode),

          ResizeCell_Frame(name,
                            outputWidth,
                            outputHeight,
                            nbOutputs,
                            resizeMode) {};

    friend class UnitTest_ResizeCell_Frame_propagate_bilinearTF_aligned_checkGradient;

};

TEST_DATASET(ResizeCell_Frame,
     propagate_bilinearTF_aligned_checkGradient,
     (unsigned int outputWidth,
     unsigned int outputHeight,
     unsigned int inputWidth,
     unsigned int inputHeight,
     unsigned int batchSize),
     std::make_tuple(24, 24, 12, 12, 1),
     std::make_tuple(92, 92, 53, 53, 1),
     std::make_tuple(32, 32, 96, 96, 1),
     std::make_tuple(32, 32, 96, 96, 4))
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 3;
    ResizeCell::ResizeMode mode = ResizeCell::ResizeMode::BilinearTF;
    ResizeCell_Frame_Test resize("resize",
                                  outputWidth,
                                  outputHeight,
                                  nbOutputs,
                                  mode);

    ASSERT_EQUALS(resize.getName(), "resize");
    ASSERT_EQUALS(resize.getNbOutputs(), nbOutputs);
    Tensor<Float_T> inputs({inputWidth, inputHeight, nbOutputs, batchSize});
    Tensor<Float_T> diffOutputs({inputWidth, inputHeight, nbOutputs, batchSize});

    for (unsigned int index = 0; index < inputs.size(); ++index)
        inputs(index) = Random::randUniform(-1.0, 1.0);


    resize.addInput(inputs, diffOutputs);
    resize.setOutputsDims();
    resize.initialize();

    resize.propagate();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(resize.getOutputs());
    //ASSERT_NOTHROW_ANY(resize.checkGradient(1.0e-3, 1.0e-3));

    ASSERT_EQUALS(outputs.dimB(), batchSize);
    ASSERT_EQUALS(outputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(outputs.dimX(), outputWidth);
    ASSERT_EQUALS(outputs.dimY(), outputHeight);

}

RUN_TESTS()
