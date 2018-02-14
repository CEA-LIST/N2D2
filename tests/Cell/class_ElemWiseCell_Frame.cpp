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

#include "Cell/ElemWiseCell_Frame.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class ElemWiseCell_Frame_Test : public ElemWiseCell_Frame {
public:
    ElemWiseCell_Frame_Test(const std::string& name,
                            unsigned int nbOutputs,
                            Operation operation,
                   const std::vector<Float_T>& weights = std::vector<Float_T>(),
                   const std::shared_ptr<Activation<Float_T> >& activation
                   = std::shared_ptr<Activation<Float_T> >())
        : Cell(name, nbOutputs),
          ElemWiseCell(name, nbOutputs, operation, weights),
          ElemWiseCell_Frame(name, nbOutputs, operation, weights, activation)
    {}
};

TEST(ElemWiseCell_Frame,
     propagate_sum2)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Sum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o), inputsA(o) + inputsB(o), 1.0e-9);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST(ElemWiseCell_Frame,
     propagate_sum3)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Sum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsC(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsC(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o), inputsA(o) + inputsB(o) + inputsC(o),
                            1.0e-9);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST_DATASET(ElemWiseCell_Frame,
             propagate_sum3_w,
             (double wA, double wB, double wC),
             std::make_tuple(1.0, 1.0, 1.0),
             std::make_tuple(0.33, 0.66, 0.99),
             std::make_tuple(0.0, 2.0, -1.0))
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;
    std::vector<Float_T> weights;
    weights.push_back(wA);
    weights.push_back(wB);
    weights.push_back(wC);

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Sum,
                                     weights);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsC(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsC(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            wA * inputsA(o) + wB * inputsB(o) + wC * inputsC(o),
                            1.0e-6);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST(ElemWiseCell_Frame,
     propagate_abs_sum2)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::AbsSum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            std::abs(inputsA(o)) + std::abs(inputsB(o)),
                            1.0e-9);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST(ElemWiseCell_Frame,
     propagate_abs_sum3)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::AbsSum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsC(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsC(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            std::abs(inputsA(o))
                                + std::abs(inputsB(o))
                                + std::abs(inputsC(o)),
                            1.0e-9);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-5, 1.0e-2));
}

TEST_DATASET(ElemWiseCell_Frame,
             propagate_abs_sum3_w,
             (double wA, double wB, double wC),
             std::make_tuple(1.0, 1.0, 1.0),
             std::make_tuple(0.33, 0.66, 0.99),
             std::make_tuple(0.0, 2.0, -1.0))
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;
    std::vector<Float_T> weights;
    weights.push_back(wA);
    weights.push_back(wB);
    weights.push_back(wC);

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::AbsSum,
                                     weights);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsC(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsC(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            wA * std::abs(inputsA(o))
                                + wB * std::abs(inputsB(o))
                                + wC * std::abs(inputsC(o)),
                            1.0e-6);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-5, 1.0e-1));
}

TEST(ElemWiseCell_Frame,
     propagate_euclidean_sum2)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::EuclideanSum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            std::sqrt(inputsA(o) * inputsA(o)
                                      + inputsB(o) * inputsB(o)),
                            1.0e-6);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST(ElemWiseCell_Frame,
     propagate_euclidean_sum3)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::EuclideanSum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsC(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsC(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            std::sqrt(inputsA(o) * inputsA(o)
                                + inputsB(o) * inputsB(o)
                                + inputsC(o) * inputsC(o)),
                            1.0e-6);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST_DATASET(ElemWiseCell_Frame,
             propagate_euclidean_sum3_w,
             (double wA, double wB, double wC),
             std::make_tuple(1.0, 1.0, 1.0),
             std::make_tuple(0.33, 0.66, 0.99),
             std::make_tuple(0.0, 2.0, -1.0))
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;
    std::vector<Float_T> weights;
    weights.push_back(wA);
    weights.push_back(wB);
    weights.push_back(wC);

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::EuclideanSum,
                                     weights);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsC(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsC(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            std::sqrt(wA * wA * inputsA(o) * inputsA(o)
                                + wB * wB * inputsB(o) * inputsB(o)
                                + wC * wC * inputsC(o) * inputsC(o)),
                            1.0e-6);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST(ElemWiseCell_Frame,
     propagate_prod2)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Prod);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o)
        ASSERT_EQUALS_DELTA(outputs(o), inputsA(o) * inputsB(o), 1.0e-9)

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-3));
}

TEST(ElemWiseCell_Frame,
     propagate_prod3)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Prod);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> inputsC(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(8, 8, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsC(8, 8, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o)
        ASSERT_EQUALS_DELTA(outputs(o), inputsA(o) * inputsB(o) * inputsC(o),
                            1.0e-9)

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-3));
}

TEST(ElemWiseCell_Frame,
     propagate_max2)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 2;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Max);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(4, 4, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(4, 4, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(4, 4, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(4, 4, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o), std::max(inputsA(o), inputsB(o)),
                            1.0e-9);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-4, 1.0e-3));
}

TEST(ElemWiseCell_Frame,
     propagate_max3)
{
    Random::mtSeed(0);

    const unsigned int nbOutputs = 2;

    ElemWiseCell_Frame_Test elemWise("elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Max);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor4d<Float_T> inputsA(4, 4, nbOutputs, 2);
    Tensor4d<Float_T> inputsB(4, 4, nbOutputs, 2);
    Tensor4d<Float_T> inputsC(4, 4, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsA(4, 4, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsB(4, 4, nbOutputs, 2);
    Tensor4d<Float_T> diffOutputsC(4, 4, nbOutputs, 2);

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    const Tensor4d<Float_T>& outputs = elemWise.getOutputs();

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            std::max(std::max(inputsA(o), inputsB(o)),
                                     inputsC(o)),
                            1.0e-15);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-4, 1.0e-3));
}

RUN_TESTS()
