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

#include "DeepNet.hpp"
#include "Environment.hpp"
#include "Network.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "Cell/FcCell_Frame.hpp"
#include "third_party/half.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

template <class T>
class FcCell_Frame_Test : public FcCell_Frame<T> {
public:
    FcCell_Frame_Test(const DeepNet& deepNet, 
                      const std::string& name,
                      unsigned int nbOutputs,
                      const std::shared_ptr<Activation>& activation)
        : Cell(deepNet, name, nbOutputs),
          FcCell(deepNet, name, nbOutputs),
          FcCell_Frame<T>(deepNet, name, nbOutputs, activation) {};

    friend class UnitTest_FcCell_Frame_float_addInput__env;
    friend class UnitTest_FcCell_Frame_float_addInput;
    friend class UnitTest_FcCell_Frame_float_addInput_multi_outputs;
    friend class UnitTest_FcCell_Frame_float_addInput_multi_inputs;
    friend class UnitTest_FcCell_Frame_float_setWeight;
    friend class UnitTest_FcCell_Frame_float_propagate_input_check;
    friend class UnitTest_FcCell_Frame_float_propagate_normalize_check;
    friend class UnitTest_FcCell_Frame_float_propagate_2_input_check;
    friend class UnitTest_FcCell_Frame_float_propagate_weight_check;
    friend class UnitTest_FcCell_Frame_double_addInput__env;
    friend class UnitTest_FcCell_Frame_double_addInput;
    friend class UnitTest_FcCell_Frame_double_addInput_multi_outputs;
    friend class UnitTest_FcCell_Frame_double_addInput_multi_inputs;
    friend class UnitTest_FcCell_Frame_double_setWeight;
    friend class UnitTest_FcCell_Frame_double_propagate_input_check;
    friend class UnitTest_FcCell_Frame_double_propagate_normalize_check;
    friend class UnitTest_FcCell_Frame_double_propagate_2_input_check;
    friend class UnitTest_FcCell_Frame_double_propagate_weight_check;
    friend class UnitTest_FcCell_Frame_half_addInput__env;
    friend class UnitTest_FcCell_Frame_half_addInput;
    friend class UnitTest_FcCell_Frame_half_addInput_multi_outputs;
    friend class UnitTest_FcCell_Frame_half_addInput_multi_inputs;
    friend class UnitTest_FcCell_Frame_half_setWeight;
    friend class UnitTest_FcCell_Frame_half_propagate_input_check;
    friend class UnitTest_FcCell_Frame_half_propagate_normalize_check;
    friend class UnitTest_FcCell_Frame_half_propagate_2_input_check;
    friend class UnitTest_FcCell_Frame_half_propagate_weight_check;
};

static MNIST_IDX_Database& getDatabase() {
    static MNIST_IDX_Database database(N2D2_DATA("mnist"));
    return database;
}

////////////////////////////////////////////////////////////////////////////////
// float
////////////////////////////////////////////////////////////////////////////////
TEST_DATASET(FcCell_Frame_float,
             FcCell_Frame,
             (unsigned int nbOutputs),
             std::make_tuple(0U),
             std::make_tuple(1U),
             std::make_tuple(3U),
             std::make_tuple(10U),
             std::make_tuple(253U))
{
    Network net;
    DeepNet dn(net);
    FcCell_Frame<float> fc1(
        dn, "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<float> >());

    ASSERT_EQUALS(fc1.getName(), "fc1");
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
}

TEST_DATASET(FcCell_Frame_float,
             addInput__env,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 24U, 24U),
             std::make_tuple(1U, 24U, 32U),
             std::make_tuple(1U, 32U, 24U),
             std::make_tuple(3U, 24U, 24U),
             std::make_tuple(3U, 24U, 32U),
             std::make_tuple(3U, 32U, 24U),
             std::make_tuple(10U, 24U, 24U),
             std::make_tuple(10U, 24U, 32U),
             std::make_tuple(10U, 32U, 24U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<float> fc1(
        dn, "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<float> >());
    fc1.setParameter("NoBias", true);
    fc1.addInput(env);
    fc1.initialize();

    ASSERT_EQUALS(fc1.getNbChannels(), 1U);
    ASSERT_EQUALS(fc1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc1.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc1.getOutputsHeight(), 1U);
    // ASSERT_NOTHROW_ANY(fc1.checkGradient(1.0e-3, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(fc1.mInputs.dataSize(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(fc1.mOutputs.size(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffInputs.size(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffOutputs.dataSize(), 0U);
}

TEST_DATASET(FcCell_Frame_float,
             addInput,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 24U, 24U),
             std::make_tuple(1U, 24U, 32U),
             std::make_tuple(1U, 32U, 24U),
             std::make_tuple(3U, 24U, 24U),
             std::make_tuple(3U, 24U, 32U),
             std::make_tuple(3U, 32U, 24U),
             std::make_tuple(10U, 24U, 24U),
             std::make_tuple(10U, 24U, 32U),
             std::make_tuple(10U, 32U, 24U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<float> fc1(
        dn, "fc1", 16, std::make_shared<TanhActivation_Frame<float> >());
    FcCell_Frame_Test<float> fc2(
        dn, "fc2", nbOutputs, std::make_shared<TanhActivation_Frame<float> >());

    fc1.addInput(env);
    fc2.addInput(&fc1);
    fc1.initialize();
    fc2.initialize();

    ASSERT_EQUALS(fc2.getNbSynapses(), (16U + 1U) * nbOutputs);
    ASSERT_EQUALS(fc2.getNbChannels(), 16U);
    ASSERT_EQUALS(fc2.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc2.getChannelsHeight(), 1U);
    ASSERT_EQUALS(fc2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc2.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc2.getOutputsHeight(), 1U);
    // ASSERT_NOTHROW_ANY(fc2.checkGradient(1.0e-3, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(fc2.mInputs.dataSize(), 16U);
    ASSERT_EQUALS(fc2.mOutputs.size(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffInputs.size(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffOutputs.dataSize(), 16U);
}

TEST_DATASET(FcCell_Frame_float,
             addInput_multi_outputs,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<float> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<float> fc2(
        dn, "fc2", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<float> fc3(
        dn, "fc3", nbOutputs, std::shared_ptr<Activation>());

    fc1.addInput(env);
    fc2.addInput(&fc1);
    fc3.addInput(&fc1);

    fc1.initialize();
    fc2.initialize();
    fc3.initialize();

    // fc1 external
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc1.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc1.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc1.getNbChannels(), 1U);
    ASSERT_EQUALS(fc1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc1.getChannelsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(fc1.checkGradient(1.0e-3, 1.0e-3));

    // fc2 external
    ASSERT_EQUALS(fc2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc2.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc2.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc2.getNbChannels(), nbOutputs);
    ASSERT_EQUALS(fc2.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc2.getChannelsHeight(), 1U);

    // fc3 external
    ASSERT_EQUALS(fc3.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc3.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc3.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc3.getNbChannels(), nbOutputs);
    ASSERT_EQUALS(fc3.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc3.getChannelsHeight(), 1U);

    // fc1 internal
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), channelsWidth);
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), channelsHeight);
    ASSERT_EQUALS(fc1.mInputs.dimZ(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffInputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimZ(), nbOutputs);

    // fc2 internal
    ASSERT_EQUALS(fc2.mInputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc2.mInputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc2.mInputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc2.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffOutputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc2.mDiffOutputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc2.mDiffOutputs.dimZ(), nbOutputs);

    // fc3 internal
    ASSERT_EQUALS(fc3.mInputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mInputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mInputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc3.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs.dimZ(), nbOutputs);
}

TEST_DATASET(FcCell_Frame_float,
             addInput_multi_inputs,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<float> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<float> fc2(
        dn, "fc2", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<float> fc3(
        dn, "fc3", nbOutputs, std::shared_ptr<Activation>());

    fc1.addInput(env);
    fc2.addInput(env);
    fc3.addInput(&fc1);
    fc3.addInput(&fc2);

    fc1.initialize();
    fc2.initialize();
    fc3.initialize();

    // fc1 external
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc1.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc1.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc1.getNbChannels(), 1U);
    ASSERT_EQUALS(fc1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc1.getChannelsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(fc1.checkGradient(1.0e-3, 1.0e-3));

    // fc2 external
    ASSERT_EQUALS(fc2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc2.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc2.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc2.getNbChannels(), 1U);
    ASSERT_EQUALS(fc2.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc2.getChannelsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(fc2.checkGradient(1.0e-3, 1.0e-3));

    // fc3 external
    ASSERT_EQUALS(fc3.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc3.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc3.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc3.getNbChannels(), nbOutputs + nbOutputs);
    ASSERT_EQUALS(fc3.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc3.getChannelsHeight(), 1U);

    // fc1 internal
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), channelsWidth);
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), channelsHeight);
    ASSERT_EQUALS(fc1.mInputs.dimZ(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffInputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimZ(), nbOutputs);

    // fc2 internal
    ASSERT_EQUALS(fc2.mInputs[0].dimX(), channelsWidth);
    ASSERT_EQUALS(fc2.mInputs[0].dimY(), channelsHeight);
    ASSERT_EQUALS(fc2.mInputs.dimZ(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffInputs.dimX(), 1U);
    ASSERT_EQUALS(fc2.mDiffInputs.dimY(), 1U);
    ASSERT_EQUALS(fc2.mDiffInputs.dimZ(), nbOutputs);

    // fc3 internal
    ASSERT_EQUALS(fc3.mInputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mInputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mInputs.dimZ(), nbOutputs + nbOutputs);
    ASSERT_EQUALS(fc3.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs.dimZ(), nbOutputs + nbOutputs);
}

TEST_DATASET(FcCell_Frame_float,
             setWeight,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<float> fc1(
        dn, "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<float> >());

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<float> weight({1}, (float)output + channel);
            fc1.setWeight(output, channel, weight);
        }
    }

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<float> weight;
            fc1.getWeight(output, channel, weight);

            ASSERT_EQUALS(weight(0), (float)output + channel);
        }
    }
}

TEST_DATASET(FcCell_Frame_float,
             propagate_input_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;
    DeepNet dn(net);
    FcCell_Frame_Test<float> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);

    Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<Float_T>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    ASSERT_EQUALS(inputSize, channelsWidth * channelsHeight);
    ASSERT_EQUALS(outputSize, nbOutputs);

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<float> weight({1}, 1.0);
            fc1.setWeight(output, channel, weight);
        }
    }

    fc1.propagate();

    const Tensor<float>& out = tensor_cast<float>(fc1.getOutputs());

    ASSERT_EQUALS(out.dimZ(), nbOutputs);
    ASSERT_EQUALS(out.dimX(), 1U);
    ASSERT_EQUALS(out.dimY(), 1U);

    const float sum
        = std::accumulate(in.begin(),
                          in.begin() + inputSize,
                          0.0f); // Warning: 0.0 leads to wrong results!

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        ASSERT_EQUALS_DELTA(out(output, 0), sum, 1e-5);
    }
}

TEST_DATASET(FcCell_Frame_float,
             propagate_normalize_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;
    DeepNet dn(net);
    FcCell_Frame_Test<float> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);
    fc1.setParameter("Normalize", true);

    Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<Float_T>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    ASSERT_EQUALS(inputSize, channelsWidth * channelsHeight);
    ASSERT_EQUALS(outputSize, nbOutputs);

    std::vector<float> norms;

    for (unsigned int output = 0; output < outputSize; ++output) {
        float norm = 0.0f;

        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            const float w = output + channel + 1;

            Tensor<float> weight({1}, w);
            fc1.setWeight(output, channel, weight);

            norm += w * w;
        }

        norms.push_back(std::sqrt(norm + 1.0e-6));
    }

    fc1.propagate();

    for (unsigned int output = 0; output < outputSize; ++output) {
        float sumSq = 0.0f;

        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<float> weight;
            fc1.getWeight(output, channel, weight);

            sumSq += weight(0) * weight(0);

            ASSERT_EQUALS_DELTA(weight(0),
                (output + channel + 1) / norms[output], 1e-5);
        }

        ASSERT_EQUALS_DELTA(sumSq, 1.0f, 1e-5);
    }
}

TEST_DATASET(FcCell_Frame_float,
             propagate_2_input_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;
    DeepNet dn(net);
    FcCell_Frame_Test<float> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);

    Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    fc1.addInput(env);
    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<float> weight({1}, 1.0);
            fc1.setWeight(output, channel, weight);
        }
    }

    fc1.propagate();

    ASSERT_EQUALS(fc1.mInputs.dimZ(), fc1.getNbChannels());
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), fc1.getChannelsWidth());
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), fc1.getChannelsHeight());

    float sum = 0.0;

    for (unsigned int k = 0; k < fc1.mInputs.size(); ++k) {
        const Tensor<float>& input = tensor_cast<float>(fc1.mInputs[k]);

        for (unsigned int channel = 0; channel < input.dimZ(); ++channel) {
            for (unsigned int y = 0; y < fc1.getChannelsHeight(); ++y) {
                for (unsigned int x = 0; x < fc1.getChannelsWidth(); ++x)
                    sum += input(x, y, channel, 0);
            }
        }
    }

    const Tensor<float>& out = tensor_cast<float>(fc1.getOutputs());

    ASSERT_EQUALS(out.dimZ(), fc1.getNbOutputs());
    ASSERT_EQUALS(out.dimX(), fc1.getOutputsWidth());
    ASSERT_EQUALS(out.dimY(), fc1.getOutputsHeight());

    for (unsigned int ox = 0; ox < fc1.getOutputsWidth(); ++ox) {
        for (unsigned int oy = 0; oy < fc1.getOutputsHeight(); ++oy) {
            for (unsigned int output = 0; output < fc1.getNbOutputs(); ++output)
                ASSERT_EQUALS_DELTA(out(ox, oy, output, 0), sum, 1e-3);
        }
    }
}

TEST_DATASET(FcCell_Frame_float,
             propagate_weight_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(
        net, EmptyDatabase, {channelsWidth, channelsHeight, 1}, 2, false);

    FcCell_Frame_Test<float> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);

    const cv::Mat img0(
        channelsHeight, channelsWidth, CV_32FC1, cv::Scalar(1.0));
    const cv::Mat img1(
        channelsHeight, channelsWidth, CV_32FC1, cv::Scalar(0.5));

    env.streamStimulus(img0, Database::Learn, 0);
    env.streamStimulus(img1, Database::Learn, 1);

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();

    fc1.propagate();

    const Tensor<float>& out = tensor_cast<float>(fc1.getOutputs());

    ASSERT_EQUALS(out.dimZ(), nbOutputs);
    ASSERT_EQUALS(out.dimX(), 1U);
    ASSERT_EQUALS(out.dimY(), 1U);

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        float sum = 0.0;

        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<float> weight;
            fc1.getWeight(output, channel, weight);

            sum += weight(0);
        }

        ASSERT_EQUALS_DELTA(out(output, 0), sum, 1e-5);
    }
}

////////////////////////////////////////////////////////////////////////////////
// double
////////////////////////////////////////////////////////////////////////////////
TEST_DATASET(FcCell_Frame_double,
             FcCell_Frame,
             (unsigned int nbOutputs),
             std::make_tuple(0U),
             std::make_tuple(1U),
             std::make_tuple(3U),
             std::make_tuple(10U),
             std::make_tuple(253U))
{
    Network net;
    DeepNet dn(net);
    FcCell_Frame<double> fc1(
        dn, "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<double> >());

    ASSERT_EQUALS(fc1.getName(), "fc1");
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
}

TEST_DATASET(FcCell_Frame_double,
             addInput__env,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 24U, 24U),
             std::make_tuple(1U, 24U, 32U),
             std::make_tuple(1U, 32U, 24U),
             std::make_tuple(3U, 24U, 24U),
             std::make_tuple(3U, 24U, 32U),
             std::make_tuple(3U, 32U, 24U),
             std::make_tuple(10U, 24U, 24U),
             std::make_tuple(10U, 24U, 32U),
             std::make_tuple(10U, 32U, 24U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<double> fc1(
        dn, "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<double> >());
    fc1.setParameter("NoBias", true);
    fc1.addInput(env);
    fc1.initialize();

    ASSERT_EQUALS(fc1.getNbChannels(), 1U);
    ASSERT_EQUALS(fc1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc1.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc1.getOutputsHeight(), 1U);
    // ASSERT_NOTHROW_ANY(fc1.checkGradient(1.0e-3, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(fc1.mInputs.dataSize(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(fc1.mOutputs.size(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffInputs.size(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffOutputs.dataSize(), 0U);
}

TEST_DATASET(FcCell_Frame_double,
             addInput,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 24U, 24U),
             std::make_tuple(1U, 24U, 32U),
             std::make_tuple(1U, 32U, 24U),
             std::make_tuple(3U, 24U, 24U),
             std::make_tuple(3U, 24U, 32U),
             std::make_tuple(3U, 32U, 24U),
             std::make_tuple(10U, 24U, 24U),
             std::make_tuple(10U, 24U, 32U),
             std::make_tuple(10U, 32U, 24U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<double> fc1(
        dn, "fc1", 16, std::make_shared<TanhActivation_Frame<double> >());
    FcCell_Frame_Test<double> fc2(
        dn, "fc2", nbOutputs, std::make_shared<TanhActivation_Frame<double> >());

    fc1.addInput(env);
    fc2.addInput(&fc1);
    fc1.initialize();
    fc2.initialize();

    ASSERT_EQUALS(fc2.getNbSynapses(), (16U + 1U) * nbOutputs);
    ASSERT_EQUALS(fc2.getNbChannels(), 16U);
    ASSERT_EQUALS(fc2.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc2.getChannelsHeight(), 1U);
    ASSERT_EQUALS(fc2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc2.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc2.getOutputsHeight(), 1U);
    // ASSERT_NOTHROW_ANY(fc2.checkGradient(1.0e-3, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(fc2.mInputs.dataSize(), 16U);
    ASSERT_EQUALS(fc2.mOutputs.size(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffInputs.size(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffOutputs.dataSize(), 16U);
}

TEST_DATASET(FcCell_Frame_double,
             addInput_multi_outputs,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<double> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<double> fc2(
        dn, "fc2", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<double> fc3(
        dn, "fc3", nbOutputs, std::shared_ptr<Activation>());

    fc1.addInput(env);
    fc2.addInput(&fc1);
    fc3.addInput(&fc1);

    fc1.initialize();
    fc2.initialize();
    fc3.initialize();

    // fc1 external
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc1.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc1.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc1.getNbChannels(), 1U);
    ASSERT_EQUALS(fc1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc1.getChannelsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(fc1.checkGradient(1.0e-3, 1.0e-3));

    // fc2 external
    ASSERT_EQUALS(fc2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc2.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc2.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc2.getNbChannels(), nbOutputs);
    ASSERT_EQUALS(fc2.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc2.getChannelsHeight(), 1U);

    // fc3 external
    ASSERT_EQUALS(fc3.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc3.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc3.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc3.getNbChannels(), nbOutputs);
    ASSERT_EQUALS(fc3.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc3.getChannelsHeight(), 1U);

    // fc1 internal
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), channelsWidth);
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), channelsHeight);
    ASSERT_EQUALS(fc1.mInputs.dimZ(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffInputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimZ(), nbOutputs);

    // fc2 internal
    ASSERT_EQUALS(fc2.mInputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc2.mInputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc2.mInputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc2.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffOutputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc2.mDiffOutputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc2.mDiffOutputs.dimZ(), nbOutputs);

    // fc3 internal
    ASSERT_EQUALS(fc3.mInputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mInputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mInputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc3.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs.dimZ(), nbOutputs);
}

TEST_DATASET(FcCell_Frame_double,
             addInput_multi_inputs,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<double> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<double> fc2(
        dn, "fc2", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<double> fc3(
        dn, "fc3", nbOutputs, std::shared_ptr<Activation>());

    fc1.addInput(env);
    fc2.addInput(env);
    fc3.addInput(&fc1);
    fc3.addInput(&fc2);

    fc1.initialize();
    fc2.initialize();
    fc3.initialize();

    // fc1 external
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc1.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc1.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc1.getNbChannels(), 1U);
    ASSERT_EQUALS(fc1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc1.getChannelsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(fc1.checkGradient(1.0e-3, 1.0e-3));

    // fc2 external
    ASSERT_EQUALS(fc2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc2.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc2.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc2.getNbChannels(), 1U);
    ASSERT_EQUALS(fc2.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc2.getChannelsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(fc2.checkGradient(1.0e-3, 1.0e-3));

    // fc3 external
    ASSERT_EQUALS(fc3.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc3.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc3.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc3.getNbChannels(), nbOutputs + nbOutputs);
    ASSERT_EQUALS(fc3.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc3.getChannelsHeight(), 1U);

    // fc1 internal
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), channelsWidth);
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), channelsHeight);
    ASSERT_EQUALS(fc1.mInputs.dimZ(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffInputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimZ(), nbOutputs);

    // fc2 internal
    ASSERT_EQUALS(fc2.mInputs[0].dimX(), channelsWidth);
    ASSERT_EQUALS(fc2.mInputs[0].dimY(), channelsHeight);
    ASSERT_EQUALS(fc2.mInputs.dimZ(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffInputs.dimX(), 1U);
    ASSERT_EQUALS(fc2.mDiffInputs.dimY(), 1U);
    ASSERT_EQUALS(fc2.mDiffInputs.dimZ(), nbOutputs);

    // fc3 internal
    ASSERT_EQUALS(fc3.mInputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mInputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mInputs.dimZ(), nbOutputs + nbOutputs);
    ASSERT_EQUALS(fc3.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs.dimZ(), nbOutputs + nbOutputs);
}

TEST_DATASET(FcCell_Frame_double,
             setWeight,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<double> fc1(
        dn, "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<double> >());

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<double> weight({1}, (double)output + channel);
            fc1.setWeight(output, channel, weight);
        }
    }

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<double> weight;
            fc1.getWeight(output, channel, weight);

            ASSERT_EQUALS(weight(0), (double)output + channel);
        }
    }
}

TEST_DATASET(FcCell_Frame_double,
             propagate_input_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;
    DeepNet dn(net);
    FcCell_Frame_Test<double> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);

    Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<Float_T>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    ASSERT_EQUALS(inputSize, channelsWidth * channelsHeight);
    ASSERT_EQUALS(outputSize, nbOutputs);

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<double> weight({1}, 1.0);
            fc1.setWeight(output, channel, weight);
        }
    }

    fc1.propagate();

    const Tensor<double>& out = tensor_cast<double>(fc1.getOutputs());

    ASSERT_EQUALS(out.dimZ(), nbOutputs);
    ASSERT_EQUALS(out.dimX(), 1U);
    ASSERT_EQUALS(out.dimY(), 1U);

    const double sum
        = std::accumulate(in.begin(),
                          in.begin() + inputSize,
                          0.0);

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        ASSERT_EQUALS_DELTA(out(output, 0), sum, 1e-5);
    }
}

TEST_DATASET(FcCell_Frame_double,
             propagate_normalize_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;
    DeepNet dn(net);
    FcCell_Frame_Test<double> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);
    fc1.setParameter("Normalize", true);

    Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<Float_T>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    ASSERT_EQUALS(inputSize, channelsWidth * channelsHeight);
    ASSERT_EQUALS(outputSize, nbOutputs);

    std::vector<double> norms;

    for (unsigned int output = 0; output < outputSize; ++output) {
        double norm = 0.0;

        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            const double w = output + channel + 1;

            Tensor<double> weight({1}, w);
            fc1.setWeight(output, channel, weight);

            norm += w * w;
        }

        norms.push_back(std::sqrt(norm + 1.0e-6));
    }

    fc1.propagate();

    for (unsigned int output = 0; output < outputSize; ++output) {
        double sumSq = 0.0;

        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<double> weight;
            fc1.getWeight(output, channel, weight);

            sumSq += weight(0) * weight(0);

            ASSERT_EQUALS_DELTA(weight(0),
                (output + channel + 1) / norms[output], 1e-5);
        }

        ASSERT_EQUALS_DELTA(sumSq, 1.0, 1e-5);
    }
}

TEST_DATASET(FcCell_Frame_double,
             propagate_2_input_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;
    DeepNet dn(net);
    FcCell_Frame_Test<double> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);

    Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    fc1.addInput(env);
    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<double> weight({1}, 1.0);
            fc1.setWeight(output, channel, weight);
        }
    }

    fc1.propagate();

    ASSERT_EQUALS(fc1.mInputs.dimZ(), fc1.getNbChannels());
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), fc1.getChannelsWidth());
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), fc1.getChannelsHeight());

    double sum = 0.0;

    for (unsigned int k = 0; k < fc1.mInputs.size(); ++k) {
        const Tensor<double>& input = tensor_cast<double>(fc1.mInputs[k]);

        for (unsigned int channel = 0; channel < input.dimZ(); ++channel) {
            for (unsigned int y = 0; y < fc1.getChannelsHeight(); ++y) {
                for (unsigned int x = 0; x < fc1.getChannelsWidth(); ++x)
                    sum += input(x, y, channel, 0);
            }
        }
    }

    const Tensor<double>& out = tensor_cast<double>(fc1.getOutputs());

    ASSERT_EQUALS(out.dimZ(), fc1.getNbOutputs());
    ASSERT_EQUALS(out.dimX(), fc1.getOutputsWidth());
    ASSERT_EQUALS(out.dimY(), fc1.getOutputsHeight());

    for (unsigned int ox = 0; ox < fc1.getOutputsWidth(); ++ox) {
        for (unsigned int oy = 0; oy < fc1.getOutputsHeight(); ++oy) {
            for (unsigned int output = 0; output < fc1.getNbOutputs(); ++output)
                ASSERT_EQUALS_DELTA(out(ox, oy, output, 0), sum, 1e-3);
        }
    }
}

TEST_DATASET(FcCell_Frame_double,
             propagate_weight_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(
        net, EmptyDatabase, {channelsWidth, channelsHeight, 1}, 2, false);

    FcCell_Frame_Test<double> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);

    const cv::Mat img0(
        channelsHeight, channelsWidth, CV_32FC1, cv::Scalar(1.0));
    const cv::Mat img1(
        channelsHeight, channelsWidth, CV_32FC1, cv::Scalar(0.5));

    env.streamStimulus(img0, Database::Learn, 0);
    env.streamStimulus(img1, Database::Learn, 1);

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();

    fc1.propagate();

    const Tensor<double>& out = tensor_cast<double>(fc1.getOutputs());

    ASSERT_EQUALS(out.dimZ(), nbOutputs);
    ASSERT_EQUALS(out.dimX(), 1U);
    ASSERT_EQUALS(out.dimY(), 1U);

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        double sum = 0.0;

        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<double> weight;
            fc1.getWeight(output, channel, weight);

            sum += weight(0);
        }

        ASSERT_EQUALS_DELTA(out(output, 0), sum, 1e-5);
    }
}

////////////////////////////////////////////////////////////////////////////////
// half
////////////////////////////////////////////////////////////////////////////////
TEST_DATASET(FcCell_Frame_half,
             FcCell_Frame,
             (unsigned int nbOutputs),
             std::make_tuple(0U),
             std::make_tuple(1U),
             std::make_tuple(3U),
             std::make_tuple(10U),
             std::make_tuple(253U))
{
    Network net;
    DeepNet dn(net);
    FcCell_Frame<half_float::half> fc1(
        dn, "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<half_float::half> >());

    ASSERT_EQUALS(fc1.getName(), "fc1");
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
}

TEST_DATASET(FcCell_Frame_half,
             addInput__env,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 24U, 24U),
             std::make_tuple(1U, 24U, 32U),
             std::make_tuple(1U, 32U, 24U),
             std::make_tuple(3U, 24U, 24U),
             std::make_tuple(3U, 24U, 32U),
             std::make_tuple(3U, 32U, 24U),
             std::make_tuple(10U, 24U, 24U),
             std::make_tuple(10U, 24U, 32U),
             std::make_tuple(10U, 32U, 24U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<half_float::half> fc1(
        dn, "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<half_float::half> >());
    fc1.setParameter("NoBias", true);
    fc1.addInput(env);
    fc1.initialize();

    ASSERT_EQUALS(fc1.getNbChannels(), 1U);
    ASSERT_EQUALS(fc1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc1.getChannelsHeight(), channelsHeight);
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc1.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc1.getOutputsHeight(), 1U);
    // ASSERT_NOTHROW_ANY(fc1.checkGradient(1.0e-3, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(fc1.mInputs.dataSize(), channelsWidth * channelsHeight);
    ASSERT_EQUALS(fc1.mOutputs.size(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffInputs.size(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffOutputs.dataSize(), 0U);
}

TEST_DATASET(FcCell_Frame_half,
             addInput,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 24U, 24U),
             std::make_tuple(1U, 24U, 32U),
             std::make_tuple(1U, 32U, 24U),
             std::make_tuple(3U, 24U, 24U),
             std::make_tuple(3U, 24U, 32U),
             std::make_tuple(3U, 32U, 24U),
             std::make_tuple(10U, 24U, 24U),
             std::make_tuple(10U, 24U, 32U),
             std::make_tuple(10U, 32U, 24U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<half_float::half> fc1(
        dn, "fc1", 16, std::make_shared<TanhActivation_Frame<half_float::half> >());
    FcCell_Frame_Test<half_float::half> fc2(
        dn, "fc2", nbOutputs, std::make_shared<TanhActivation_Frame<half_float::half> >());

    fc1.addInput(env);
    fc2.addInput(&fc1);
    fc1.initialize();
    fc2.initialize();

    ASSERT_EQUALS(fc2.getNbSynapses(), (16U + 1U) * nbOutputs);
    ASSERT_EQUALS(fc2.getNbChannels(), 16U);
    ASSERT_EQUALS(fc2.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc2.getChannelsHeight(), 1U);
    ASSERT_EQUALS(fc2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc2.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc2.getOutputsHeight(), 1U);
    // ASSERT_NOTHROW_ANY(fc2.checkGradient(1.0e-3, 1.0e-3));

    // Internal state testing
    ASSERT_EQUALS(fc2.mInputs.dataSize(), 16U);
    ASSERT_EQUALS(fc2.mOutputs.size(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffInputs.size(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffOutputs.dataSize(), 16U);
}

TEST_DATASET(FcCell_Frame_half,
             addInput_multi_outputs,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<half_float::half> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<half_float::half> fc2(
        dn, "fc2", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<half_float::half> fc3(
        dn, "fc3", nbOutputs, std::shared_ptr<Activation>());

    fc1.addInput(env);
    fc2.addInput(&fc1);
    fc3.addInput(&fc1);

    fc1.initialize();
    fc2.initialize();
    fc3.initialize();

    // fc1 external
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc1.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc1.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc1.getNbChannels(), 1U);
    ASSERT_EQUALS(fc1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc1.getChannelsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(fc1.checkGradient(1.0e-3, 1.0e-3));

    // fc2 external
    ASSERT_EQUALS(fc2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc2.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc2.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc2.getNbChannels(), nbOutputs);
    ASSERT_EQUALS(fc2.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc2.getChannelsHeight(), 1U);

    // fc3 external
    ASSERT_EQUALS(fc3.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc3.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc3.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc3.getNbChannels(), nbOutputs);
    ASSERT_EQUALS(fc3.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc3.getChannelsHeight(), 1U);

    // fc1 internal
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), channelsWidth);
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), channelsHeight);
    ASSERT_EQUALS(fc1.mInputs.dimZ(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffInputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimZ(), nbOutputs);

    // fc2 internal
    ASSERT_EQUALS(fc2.mInputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc2.mInputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc2.mInputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc2.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffOutputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc2.mDiffOutputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc2.mDiffOutputs.dimZ(), nbOutputs);

    // fc3 internal
    ASSERT_EQUALS(fc3.mInputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mInputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mInputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc3.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs.dimZ(), nbOutputs);
}

TEST_DATASET(FcCell_Frame_half,
             addInput_multi_inputs,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<half_float::half> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<half_float::half> fc2(
        dn, "fc2", nbOutputs, std::shared_ptr<Activation>());
    FcCell_Frame_Test<half_float::half> fc3(
        dn, "fc3", nbOutputs, std::shared_ptr<Activation>());

    fc1.addInput(env);
    fc2.addInput(env);
    fc3.addInput(&fc1);
    fc3.addInput(&fc2);

    fc1.initialize();
    fc2.initialize();
    fc3.initialize();

    // fc1 external
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc1.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc1.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc1.getNbChannels(), 1U);
    ASSERT_EQUALS(fc1.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc1.getChannelsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(fc1.checkGradient(1.0e-3, 1.0e-3));

    // fc2 external
    ASSERT_EQUALS(fc2.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc2.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc2.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc2.getNbChannels(), 1U);
    ASSERT_EQUALS(fc2.getChannelsWidth(), channelsWidth);
    ASSERT_EQUALS(fc2.getChannelsHeight(), channelsHeight);
    // ASSERT_NOTHROW_ANY(fc2.checkGradient(1.0e-3, 1.0e-3));

    // fc3 external
    ASSERT_EQUALS(fc3.getNbOutputs(), nbOutputs);
    ASSERT_EQUALS(fc3.getOutputsWidth(), 1U);
    ASSERT_EQUALS(fc3.getOutputsHeight(), 1U);
    ASSERT_EQUALS(fc3.getNbChannels(), nbOutputs + nbOutputs);
    ASSERT_EQUALS(fc3.getChannelsWidth(), 1U);
    ASSERT_EQUALS(fc3.getChannelsHeight(), 1U);

    // fc1 internal
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), channelsWidth);
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), channelsHeight);
    ASSERT_EQUALS(fc1.mInputs.dimZ(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc1.mDiffInputs.dimX(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimY(), 1U);
    ASSERT_EQUALS(fc1.mDiffInputs.dimZ(), nbOutputs);

    // fc2 internal
    ASSERT_EQUALS(fc2.mInputs[0].dimX(), channelsWidth);
    ASSERT_EQUALS(fc2.mInputs[0].dimY(), channelsHeight);
    ASSERT_EQUALS(fc2.mInputs.dimZ(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc2.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc2.mDiffInputs.dimX(), 1U);
    ASSERT_EQUALS(fc2.mDiffInputs.dimY(), 1U);
    ASSERT_EQUALS(fc2.mDiffInputs.dimZ(), nbOutputs);

    // fc3 internal
    ASSERT_EQUALS(fc3.mInputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mInputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mInputs.dimZ(), nbOutputs + nbOutputs);
    ASSERT_EQUALS(fc3.mOutputs.dimX(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimY(), 1U);
    ASSERT_EQUALS(fc3.mOutputs.dimZ(), nbOutputs);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimX(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs[0].dimY(), 1U);
    ASSERT_EQUALS(fc3.mDiffOutputs.dimZ(), nbOutputs + nbOutputs);
}

TEST_DATASET(FcCell_Frame_half,
             setWeight,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {channelsWidth, channelsHeight, 1});

    FcCell_Frame_Test<half_float::half> fc1(
        dn, "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<half_float::half> >());

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<half_float::half> weight({1},
                                    half_float::half((float)output + channel));
            fc1.setWeight(output, channel, weight);
        }
    }

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<half_float::half> weight;
            fc1.getWeight(output, channel, weight);

            ASSERT_EQUALS(weight(0), (float)output + channel);
        }
    }
}

TEST_DATASET(FcCell_Frame_half,
             propagate_input_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;
    DeepNet dn(net);
    FcCell_Frame_Test<half_float::half> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);

    Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    const Tensor<half_float::half>& in
        = tensor_cast<half_float::half>(env.getData());

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    ASSERT_EQUALS(inputSize, channelsWidth * channelsHeight);
    ASSERT_EQUALS(outputSize, nbOutputs);

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<half_float::half> weight({1}, half_float::half(1.0f));
            fc1.setWeight(output, channel, weight);
        }
    }

    fc1.propagate();

    const Tensor<half_float::half>& out = tensor_cast<half_float::half>(fc1.getOutputs());

    ASSERT_EQUALS(out.dimZ(), nbOutputs);
    ASSERT_EQUALS(out.dimX(), 1U);
    ASSERT_EQUALS(out.dimY(), 1U);

    const half_float::half sum
        = std::accumulate(in.begin(),
                          in.begin() + inputSize,
                          half_float::half(0.0f));

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        ASSERT_EQUALS_DELTA(out(output, 0), sum, 1e-4);
    }
}

TEST_DATASET(FcCell_Frame_half,
             propagate_normalize_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;
    DeepNet dn(net);
    FcCell_Frame_Test<half_float::half> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);
    fc1.setParameter("Normalize", true);

    Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor<Float_T>& in = env.getData();

    ASSERT_EQUALS(in.dimZ(), 1U);
    ASSERT_EQUALS(in.dimX(), channelsWidth);
    ASSERT_EQUALS(in.dimY(), channelsHeight);

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    ASSERT_EQUALS(inputSize, channelsWidth * channelsHeight);
    ASSERT_EQUALS(outputSize, nbOutputs);

    std::vector<half_float::half> norms;

    for (unsigned int output = 0; output < outputSize; ++output) {
        half_float::half norm(0.0f);

        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            const half_float::half w((output + channel + 1) / 100.0);

            Tensor<half_float::half> weight({1}, w);
            fc1.setWeight(output, channel, weight);

            norm += w * w;
        }

        norms.push_back((half_float::half)std::sqrt(norm + 1.0e-6));
    }

    fc1.propagate();

    for (unsigned int output = 0; output < outputSize; ++output) {
        half_float::half sumSq(0.0f);

        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<half_float::half> weight;
            fc1.getWeight(output, channel, weight);

            sumSq += weight(0) * weight(0);

            ASSERT_EQUALS_DELTA(weight(0),
                ((output + channel + 1) / 100.0) / norms[output], 1e-2);
        }

        ASSERT_EQUALS_DELTA(sumSq, 1.0f, 1e-1);
    }
}

TEST_DATASET(FcCell_Frame_half,
             propagate_2_input_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;
    DeepNet dn(net);
    FcCell_Frame_Test<half_float::half> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);

    Environment env(net, getDatabase(), {channelsWidth, channelsHeight, 1}, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    fc1.addInput(env);
    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<half_float::half> weight({1}, half_float::half(1.0f));
            fc1.setWeight(output, channel, weight);
        }
    }

    fc1.propagate();

    ASSERT_EQUALS(fc1.mInputs.dimZ(), fc1.getNbChannels());
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), fc1.getChannelsWidth());
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), fc1.getChannelsHeight());

    half_float::half sum(0.0f);

    for (unsigned int k = 0; k < fc1.mInputs.size(); ++k) {
        const Tensor<half_float::half>& input
            = tensor_cast<half_float::half>(fc1.mInputs[k]);

        for (unsigned int channel = 0; channel < input.dimZ(); ++channel) {
            for (unsigned int y = 0; y < fc1.getChannelsHeight(); ++y) {
                for (unsigned int x = 0; x < fc1.getChannelsWidth(); ++x)
                    sum += input(x, y, channel, 0);
            }
        }
    }

    const Tensor<half_float::half>& out
        = tensor_cast<half_float::half>(fc1.getOutputs());

    ASSERT_EQUALS(out.dimZ(), fc1.getNbOutputs());
    ASSERT_EQUALS(out.dimX(), fc1.getOutputsWidth());
    ASSERT_EQUALS(out.dimY(), fc1.getOutputsHeight());

    for (unsigned int ox = 0; ox < fc1.getOutputsWidth(); ++ox) {
        for (unsigned int oy = 0; oy < fc1.getOutputsHeight(); ++oy) {
            for (unsigned int output = 0; output < fc1.getNbOutputs(); ++output)
                ASSERT_EQUALS_DELTA(out(ox, oy, output, 0), sum, 1e2);
        }
    }
}

TEST_DATASET(FcCell_Frame_half,
             propagate_weight_check,
             (unsigned int nbOutputs,
              unsigned int channelsWidth,
              unsigned int channelsHeight),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(2U, 2U, 1U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(1U, 10U, 10U),
             std::make_tuple(2U, 25U, 25U),
             std::make_tuple(1U, 25U, 30U),
             std::make_tuple(1U, 30U, 25U),
             std::make_tuple(1U, 30U, 30U))
{
    Network net;
    DeepNet dn(net);
    Environment env(
        net, EmptyDatabase, {channelsWidth, channelsHeight, 1}, 2, false);

    FcCell_Frame_Test<half_float::half> fc1(
        dn, "fc1", nbOutputs, std::shared_ptr<Activation>());
    fc1.setParameter("NoBias", true);

    const cv::Mat img0(
        channelsHeight, channelsWidth, CV_32FC1, cv::Scalar(1.0));
    const cv::Mat img1(
        channelsHeight, channelsWidth, CV_32FC1, cv::Scalar(0.5));

    env.streamStimulus(img0, Database::Learn, 0);
    env.streamStimulus(img1, Database::Learn, 1);

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();

    fc1.propagate();

    const Tensor<half_float::half>& out = tensor_cast<half_float::half>(fc1.getOutputs());

    ASSERT_EQUALS(out.dimZ(), nbOutputs);
    ASSERT_EQUALS(out.dimX(), 1U);
    ASSERT_EQUALS(out.dimY(), 1U);

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        half_float::half sum(0.0f);

        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            Tensor<half_float::half> weight;
            fc1.getWeight(output, channel, weight);

            sum += weight(0);
        }

        ASSERT_EQUALS_DELTA(out(output, 0), sum, 1e-5);
    }
}

RUN_TESTS()
