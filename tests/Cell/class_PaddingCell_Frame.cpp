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

#include "Cell/PaddingCell_Frame.hpp"
#include "Network.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Random.hpp"

using namespace N2D2;

class PaddingCell_Frame_Test : public PaddingCell_Frame {
public:
    PaddingCell_Frame_Test( const std::string& name,
                            unsigned int nbOutputs,
                            int topPad,
                            int botPad,
                            int leftPad,
                            int rightPad)

        : Cell(name, nbOutputs),
          PaddingCell(name, 
                      nbOutputs,
                      topPad,
                      botPad,
                      leftPad,
                      rightPad),
          PaddingCell_Frame(name, 
                            nbOutputs,
                            topPad,
                            botPad,
                            leftPad,
                            rightPad) {};

    friend class UnitTest_PaddingCell_Frame_addInput_1;
    friend class UnitTest_PaddingCell_Frame_addInput_2;
    friend class UnitTest_PaddingCell_Frame_checkGradient_1;
    friend class UnitTest_PaddingCell_Frame_checkGradient_2;
};

TEST_DATASET(PaddingCell_Frame,
             addInput_1,
             (int topPad,
              int botPad,
              int leftPad,
              int rightPad,
              unsigned int inputWidth,
              unsigned int inputHeight,
              unsigned int batchSize),
             std::make_tuple(-3, -2, 4, 1, 48, 48, 1),
             std::make_tuple(3, 2, -4, -1, 48, 48, 1),
             std::make_tuple(0, 0, 0, 0, 48, 48, 1),
             std::make_tuple(-1, 2, 4, 1, 48, 48, 1),
             std::make_tuple(0, 0, 1, -4, 48, 48, 2),
             std::make_tuple(0, -8, 0, -8, 48, 48, 2))
{
    const unsigned int nbOutputs = 10;
    Tensor4d<Float_T> inputs(inputWidth, inputHeight, nbOutputs, batchSize);
    Tensor4d<Float_T> diffOutputs(inputWidth, inputHeight, nbOutputs, batchSize);
    for(unsigned int i = 0; i < inputs.size(); ++i)
        inputs(i) = Random::randNormal();

    PaddingCell_Frame padding1("padding1", 
                                nbOutputs,
                                topPad,
                                botPad,
                                leftPad,
                                rightPad);
    ASSERT_EQUALS(padding1.getName(), "padding1");
    ASSERT_EQUALS(padding1.getNbOutputs(), nbOutputs);
    padding1.addInput(inputs, diffOutputs);
    padding1.initialize();

    ASSERT_EQUALS(padding1.getOutputsWidth(), 
                  leftPad + rightPad + inputWidth);

    ASSERT_EQUALS(padding1.getOutputsHeight(), 
                  topPad + botPad + inputHeight);

    padding1.propagate();


    const Tensor4d<Float_T>& outputs1 = padding1.getOutputs();

    for(unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        for (unsigned int output = 0; output < inputs.dimZ(); ++output) {
            for (unsigned int oy = 0; oy < outputs1.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < outputs1.dimX(); ++ox) {
                    
                    int ix = (int) ox - leftPad;
                    int iy = (int) oy - topPad;

                    if( ix >= 0  && ix < (int) inputs.dimX()
                        && iy >= 0  && iy < (int) inputs.dimY())
                    {
                        ASSERT_EQUALS_DELTA(outputs1(ox, oy, output, batchPos), inputs(ix, iy , output, batchPos), 1.0e-9);
                    }
                    else
                    {
                        ASSERT_EQUALS_DELTA(outputs1(ox, oy, output, batchPos), 0.0, 1.0e-9);
                    }
                }
            }
        }
    }
}


TEST_DATASET(PaddingCell_Frame,
             checkGradient_1,
             (int topPad,
              int botPad,
              int leftPad,
              int rightPad,
              unsigned int inputWidth,
              unsigned int inputHeight,
              unsigned int batchSize),
             std::make_tuple(-3, -2, 4, 1, 12, 12, 1),
             std::make_tuple(3, 2, -4, -1, 12, 12, 1),
             std::make_tuple(0, 0, 1, -4, 12, 12, 2),
             std::make_tuple(0, -8, 0, -8, 12, 12, 2))
{
    const unsigned int nbOutputs = 10;
    Tensor4d<Float_T> inputs(inputWidth, inputHeight, nbOutputs, batchSize);
    Tensor4d<Float_T> diffOutputs(inputWidth, inputHeight, nbOutputs, batchSize);
    for(unsigned int i = 0; i < inputs.size(); ++i)
        inputs(i) = Random::randNormal();

    PaddingCell_Frame padding1("padding1", 
                                nbOutputs,
                                topPad,
                                botPad,
                                leftPad,
                                rightPad);
    ASSERT_EQUALS(padding1.getName(), "padding1");
    ASSERT_EQUALS(padding1.getNbOutputs(), nbOutputs);
    padding1.addInput(inputs, diffOutputs);
    padding1.initialize();

    ASSERT_EQUALS(padding1.getOutputsWidth(), 
                  leftPad + rightPad + inputWidth);

    ASSERT_EQUALS(padding1.getOutputsHeight(), 
                  topPad + botPad + inputHeight);

    padding1.propagate();

    ASSERT_NOTHROW_ANY(padding1.checkGradient(1.0e-3, 1.0e-2));
}

TEST_DATASET(PaddingCell_Frame,
             addInput_2,
             (int topPad,
              int botPad,
              int leftPad,
              int rightPad,
              unsigned int nbInputA,
              unsigned int nbInputB,
              unsigned int inputWidth,
              unsigned int inputHeight,
              unsigned int batchSize),
             std::make_tuple(-3, -2, 4, 1, 2, 2, 48, 48, 1),
             std::make_tuple(3, 2, -4, -1, 2, 1, 48, 48, 1),
             std::make_tuple(0, 0, 0, 0, 1, 2, 48, 48, 1),
             std::make_tuple(-1, 2, 4, 1, 4, 8, 48, 48, 1),
             std::make_tuple(0, 0, 1, -4, 5, 3, 48, 48, 2),
             std::make_tuple(0, -8, 0, -8, 1, 1, 48, 48, 2))
{
    const unsigned int nbOutputs = nbInputA + nbInputB;
    Tensor4d<Float_T> inputsA(inputWidth, inputHeight, nbInputA, batchSize);
    Tensor4d<Float_T> inputsB(inputWidth, inputHeight, nbInputB, batchSize);

    Tensor4d<Float_T> diffOutputsA(inputWidth, inputHeight, nbInputA, batchSize);
    Tensor4d<Float_T> diffOutputsB(inputWidth, inputHeight, nbInputB, batchSize);

    for(unsigned int i = 0; i < inputsA.size(); ++i)
        inputsA(i) = Random::randNormal();

    for(unsigned int i = 0; i < inputsB.size(); ++i)
        inputsB(i) = Random::randNormal();

    PaddingCell_Frame padding1("padding1", 
                                nbOutputs,
                                topPad,
                                botPad,
                                leftPad,
                                rightPad);
    ASSERT_EQUALS(padding1.getName(), "padding1");
    ASSERT_EQUALS(padding1.getNbOutputs(), nbOutputs);
    padding1.addInput(inputsA, diffOutputsA);
    padding1.addInput(inputsB, diffOutputsB);

    padding1.initialize();

    ASSERT_EQUALS(padding1.getOutputsWidth(), 
                  leftPad + rightPad + inputWidth);

    ASSERT_EQUALS(padding1.getOutputsHeight(), 
                  topPad + botPad + inputHeight);

    padding1.propagate();


    const Tensor4d<Float_T>& outputs1 = padding1.getOutputs();

    for(unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        for (unsigned int output = 0; output < inputsA.dimZ(); ++output) {
            for (unsigned int oy = 0; oy < outputs1.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < outputs1.dimX(); ++ox) {
                    
                    int ix = (int) ox - leftPad;
                    int iy = (int) oy - topPad;

                    if( ix >= 0  && ix < (int) inputsA.dimX()
                        && iy >= 0  && iy < (int) inputsA.dimY())
                    {
                        ASSERT_EQUALS_DELTA(outputs1(ox, oy, output, batchPos), 
                                            inputsA(ix, iy , output, batchPos), 
                                            1.0e-9);
                    }
                    else
                    {
                        ASSERT_EQUALS_DELTA(outputs1(ox, oy, output, batchPos), 0.0, 1.0e-9);
                    }
                }
            }
        }
    }

    for(unsigned int batchPos = 0; batchPos < batchSize; ++batchPos) {
        for (unsigned int output = 0; output < inputsB.dimZ(); ++output) {
            for (unsigned int oy = 0; oy < outputs1.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < outputs1.dimX(); ++ox) {
                    
                    int ix = (int) ox - leftPad;
                    int iy = (int) oy - topPad;

                    if( ix >= 0  && ix < (int) inputsB.dimX()
                        && iy >= 0  && iy < (int) inputsB.dimY())
                    {
                        ASSERT_EQUALS_DELTA(outputs1(ox, oy, output + nbInputA, batchPos), 
                                            inputsB(ix, iy , output, batchPos), 
                                            1.0e-9);
                    }
                    else
                    {
                        ASSERT_EQUALS_DELTA(outputs1(ox, oy, output + nbInputA, batchPos), 0.0, 1.0e-9);
                    }
                }
            }
        }
    }
}

TEST_DATASET(PaddingCell_Frame,
             checkGradient_2,
             (int topPad,
              int botPad,
              int leftPad,
              int rightPad,
              unsigned int nbInputA,
              unsigned int nbInputB,
              unsigned int inputWidth,
              unsigned int inputHeight,
              unsigned int batchSize),
             std::make_tuple(-3, -2, 4, 1, 2, 2, 12, 12, 1),
             std::make_tuple(3, 2, -4, -1, 2, 1, 12, 12, 1),
             std::make_tuple(0, 0, 1, -4, 5, 3, 12, 12, 2),
             std::make_tuple(0, -8, 0, -8, 1, 1, 12, 12, 2))
{
    const unsigned int nbOutputs = nbInputA + nbInputB;
    Tensor4d<Float_T> inputsA(inputWidth, inputHeight, nbInputA, batchSize);
    Tensor4d<Float_T> inputsB(inputWidth, inputHeight, nbInputB, batchSize);

    Tensor4d<Float_T> diffOutputsA(inputWidth, inputHeight, nbInputA, batchSize);
    Tensor4d<Float_T> diffOutputsB(inputWidth, inputHeight, nbInputB, batchSize);

    for(unsigned int i = 0; i < inputsA.size(); ++i)
        inputsA(i) = Random::randNormal();

    for(unsigned int i = 0; i < inputsB.size(); ++i)
        inputsB(i) = Random::randNormal();

    PaddingCell_Frame padding1("padding1", 
                                nbOutputs,
                                topPad,
                                botPad,
                                leftPad,
                                rightPad);
    ASSERT_EQUALS(padding1.getName(), "padding1");
    ASSERT_EQUALS(padding1.getNbOutputs(), nbOutputs);
    padding1.addInput(inputsA, diffOutputsA);
    padding1.addInput(inputsB, diffOutputsB);

    padding1.initialize();

    ASSERT_EQUALS(padding1.getOutputsWidth(), 
                  leftPad + rightPad + inputWidth);

    ASSERT_EQUALS(padding1.getOutputsHeight(), 
                  topPad + botPad + inputHeight);

    padding1.propagate();

    ASSERT_NOTHROW_ANY(padding1.checkGradient(1.0e-3, 1.0e-2));
}

RUN_TESTS()
