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

#include "Cell/ElemWiseCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "Xnet/Network.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class ElemWiseCell_Frame_CUDA_Test : public ElemWiseCell_Frame_CUDA {
public:
    ElemWiseCell_Frame_CUDA_Test(const DeepNet& deepNet, 
                                 const std::string& name,
                                 unsigned int nbOutputs,
                                 Operation operation,
                                 CoeffMode mode = ElemWiseCell::PerLayer,
                   const std::vector<Float_T>& weights = std::vector<Float_T>(),
                   const std::vector<Float_T>& shifts = std::vector<Float_T>(),
                   const std::shared_ptr<Activation>& activation
                   = std::shared_ptr<Activation>())
        : Cell(deepNet, name, nbOutputs),
          ElemWiseCell(deepNet, name, nbOutputs, operation, mode, weights, shifts),
          ElemWiseCell_Frame_CUDA(deepNet, name, nbOutputs, operation, mode, weights, shifts, activation)
    {}
};

TEST(ElemWiseCell_Frame_CUDA,
     propagate_sum2)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);

    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Sum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o), inputsA(o) + inputsB(o), 1.0e-9);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST(ElemWiseCell_Frame_CUDA,
     propagate_sum3)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Sum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

TEST_DATASET(ElemWiseCell_Frame_CUDA,
             propagate_sum3_w,
             (double wA, double wB, double wC),
             std::make_tuple(1.0, 1.0, 1.0),
             std::make_tuple(0.33, 0.66, 0.99),
             std::make_tuple(0.0, 2.0, -1.0))
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;
    std::vector<Float_T> weights;
    weights.push_back(wA);
    weights.push_back(wB);
    weights.push_back(wC);

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Sum,
                                     ElemWiseCell::PerInput,
                                     weights);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

TEST_DATASET(ElemWiseCell_Frame_CUDA,
             propagate_sum3_w_s,
             (double wA, double wB, double wC,
              double sA, double sB, double sC),
             std::make_tuple(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
             std::make_tuple(0.33, 0.66, 0.99, 0.4, 0.67, 1.256),
             std::make_tuple(0.0, 2.0, -1.0, -0.58, 0.39, 4.2))
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;
    std::vector<Float_T> weights;
    weights.push_back(wA);
    weights.push_back(wB);
    weights.push_back(wC);
    std::vector<Float_T> shifts;
    shifts.push_back(sA);
    shifts.push_back(sB);
    shifts.push_back(sC);

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Sum,
                                     ElemWiseCell::PerInput,
                                     weights,
                                     shifts);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            wA * inputsA(o) + sA
                            + wB * inputsB(o) + sB
                            + wC * inputsC(o) + sC,
                            1.0e-6);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST(ElemWiseCell_Frame_CUDA,
     propagate_abs_sum2)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                          nbOutputs,
                                          ElemWiseCell::AbsSum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

TEST(ElemWiseCell_Frame_CUDA,
     propagate_abs_sum3)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                          nbOutputs,
                                          ElemWiseCell::AbsSum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

TEST_DATASET(ElemWiseCell_Frame_CUDA,
             propagate_abs_sum3_w,
             (double wA, double wB, double wC),
             std::make_tuple(1.0, 1.0, 1.0),
             std::make_tuple(0.33, 0.66, 0.99),
             std::make_tuple(0.0, 2.0, -1.0))
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(2);

    const unsigned int nbOutputs = 4;
    std::vector<Float_T> weights;
    weights.push_back(wA);
    weights.push_back(wB);
    weights.push_back(wC);

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                          nbOutputs,
                                          ElemWiseCell::AbsSum,
                                          ElemWiseCell::PerInput,
                                          weights);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-5, 1.0e-2));
}

TEST(ElemWiseCell_Frame_CUDA,
     propagate_euclidean_sum2)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                          nbOutputs,
                                          ElemWiseCell::EuclideanSum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

TEST(ElemWiseCell_Frame_CUDA,
     propagate_euclidean_sum3)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                          nbOutputs,
                                          ElemWiseCell::EuclideanSum);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

TEST_DATASET(ElemWiseCell_Frame_CUDA,
             propagate_euclidean_sum3_w,
             (double wA, double wB, double wC),
             std::make_tuple(1.0, 1.0, 1.0),
             std::make_tuple(0.33, 0.66, 0.99),
             std::make_tuple(0.0, 2.0, -1.0))
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;
    std::vector<Float_T> weights;
    weights.push_back(wA);
    weights.push_back(wB);
    weights.push_back(wC);

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                          nbOutputs,
                                          ElemWiseCell::EuclideanSum,
                                          ElemWiseCell::PerInput,
                                          weights);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

TEST_DATASET(ElemWiseCell_Frame_CUDA,
             propagate_euclidean_sum3_w_s,
             (double wA, double wB, double wC,
              double sA, double sB, double sC),
             std::make_tuple(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
             std::make_tuple(0.33, 0.66, 0.99, 0.4, 0.67, 1.256),
             std::make_tuple(0.0, 2.0, -1.0, -0.58, 0.39, 4.2))
{
    REQUIRED(UnitTest::CudaDeviceExists(0));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;
    std::vector<Float_T> weights;
    weights.push_back(wA);
    weights.push_back(wB);
    weights.push_back(wC);
    std::vector<Float_T> shifts;
    shifts.push_back(sA);
    shifts.push_back(sB);
    shifts.push_back(sC);

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                          nbOutputs,
                                          ElemWiseCell::EuclideanSum,
                                          ElemWiseCell::PerInput,
                                          weights,
                                          shifts);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o) {
        ASSERT_EQUALS_DELTA(outputs(o),
                            std::sqrt(wA * wA * inputsA(o) * inputsA(o) + sA*sA
                                + wB * wB * inputsB(o) * inputsB(o) + sB*sB
                                + wC * wC * inputsC(o) * inputsC(o) + sC*sC),
                            1.0e-6);
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST_DATASET(ElemWiseCell_Frame_CUDA,
             propagate_sum3_per_channel_w,
             (double wA, double wB, double wC),
             std::make_tuple(1.0, 1.0, 1.0),
             std::make_tuple(0.33, 0.66, 0.99),
             std::make_tuple(0.0, 2.0, -1.0))
{
    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;
    std::vector<Float_T> weights;
    weights.push_back(wA);
    weights.push_back(wB);
    weights.push_back(wC);
    weights.push_back(wA + wB + wC);

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Sum,
                                     ElemWiseCell::PerChannel,
                                     weights);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }
    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();

    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int b = 0; b < inputsA.dimB(); ++b) {
        for (unsigned int o = 0; o < nbOutputs; ++o) {
            for(unsigned int y = 0; y < inputsA.dimY(); ++y) {
                for(unsigned int x = 0; x < inputsA.dimX(); ++x) {
                    double factor = o == 0 ? wA 
                                    : o == 1 ? wB 
                                    : o == 2 ? wC 
                                    : o == 3 ? wA + wB + wC 
                                    : 0.0;

                    ASSERT_EQUALS_DELTA(outputs(x,y,o,b),
                                        factor * inputsA(x,y,o,b) + factor * inputsB(x,y,o,b) + factor * inputsC(x,y,o,b),
                                        1.0e-6);

                }
            }
        }
    }

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-2));
}

TEST(ElemWiseCell_Frame_CUDA,
     propagate_prod2)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Prod);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o)
        ASSERT_EQUALS_DELTA(outputs(o), inputsA(o) * inputsB(o), 1.0e-9)

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-3));
}

TEST(ElemWiseCell_Frame_CUDA,
     propagate_prod3)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 4;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Prod);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> inputsC({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({8, 8, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({8, 8, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

    ASSERT_EQUALS(outputs.dimX(), inputsA.dimX());
    ASSERT_EQUALS(outputs.dimY(), inputsA.dimY());
    ASSERT_EQUALS(outputs.dimZ(), inputsA.dimZ());
    ASSERT_EQUALS(outputs.dimB(), inputsA.dimB());

    for (unsigned int o = 0; o < outputs.size(); ++o)
        ASSERT_EQUALS_DELTA(outputs(o), inputsA(o) * inputsB(o) * inputsC(o),
                            1.0e-9)

    ASSERT_NOTHROW_ANY(elemWise.checkGradient(1.0e-3, 1.0e-3));
}

TEST(ElemWiseCell_Frame_CUDA,
     propagate_max2)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 2;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Max);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({4, 4, nbOutputs, 2});
    Tensor<Float_T> inputsB({4, 4, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({4, 4, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({4, 4, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

TEST(ElemWiseCell_Frame_CUDA,
     propagate_max3)
{
    REQUIRED(UnitTest::CudaDeviceExists(3));

    Network net(0U,false);
    DeepNet dn(net);
    
    Random::mtSeed(0);

    const unsigned int nbOutputs = 2;

    ElemWiseCell_Frame_CUDA_Test elemWise(dn, "elemwise",
                                     nbOutputs,
                                     ElemWiseCell::Max);

    ASSERT_EQUALS(elemWise.getName(), "elemwise");
    ASSERT_EQUALS(elemWise.getNbOutputs(), nbOutputs);

    Tensor<Float_T> inputsA({4, 4, nbOutputs, 2});
    Tensor<Float_T> inputsB({4, 4, nbOutputs, 2});
    Tensor<Float_T> inputsC({4, 4, nbOutputs, 2});
    Tensor<Float_T> diffOutputsA({4, 4, nbOutputs, 2});
    Tensor<Float_T> diffOutputsB({4, 4, nbOutputs, 2});
    Tensor<Float_T> diffOutputsC({4, 4, nbOutputs, 2});

    for (unsigned int index = 0; index < inputsA.size(); ++index) {
        inputsA(index) = Random::randUniform(-1.0, 1.0);
        inputsB(index) = Random::randUniform(-1.0, 1.0);
        inputsC(index) = Random::randUniform(-1.0, 1.0);
    }

    inputsA.synchronizeHToD();
    inputsB.synchronizeHToD();
    inputsC.synchronizeHToD();

    elemWise.addInput(inputsA, diffOutputsA);
    elemWise.addInput(inputsB, diffOutputsB);
    elemWise.addInput(inputsC, diffOutputsC);
    elemWise.initialize();

    elemWise.propagate();
    elemWise.getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(elemWise.getOutputs());

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

#else

int main()
{
    return 0;
}

#endif
