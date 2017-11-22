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

#include "Database/MNIST_IDX_Database.hpp"
#include "Cell/FcCell_Frame.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class FcCell_Frame_Test : public FcCell_Frame {
public:
    FcCell_Frame_Test(const std::string& name,
                      unsigned int nbOutputs,
                      const std::shared_ptr<Activation<Float_T> >& activation)
        : Cell(name, nbOutputs),
          FcCell(name, nbOutputs),
          FcCell_Frame(name, nbOutputs, activation) {};

    friend class UnitTest_FcCell_Frame_addInput__env;
    friend class UnitTest_FcCell_Frame_addInput;
    friend class UnitTest_FcCell_Frame_addInput_multi_outputs;
    friend class UnitTest_FcCell_Frame_addInput_multi_inputs;
    friend class UnitTest_FcCell_Frame_setWeight;
    friend class UnitTest_FcCell_Frame_propagate_input_check;
    friend class UnitTest_FcCell_Frame_propagate_2_input_check;
    friend class UnitTest_FcCell_Frame_propagate_weight_check;
};

TEST_DATASET(FcCell_Frame,
             FcCell_Frame,
             (unsigned int nbOutputs),
             std::make_tuple(0U),
             std::make_tuple(1U),
             std::make_tuple(3U),
             std::make_tuple(10U),
             std::make_tuple(253U))
{
    FcCell_Frame fc1(
        "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<Float_T> >());

    ASSERT_EQUALS(fc1.getName(), "fc1");
    ASSERT_EQUALS(fc1.getNbOutputs(), nbOutputs);
}

TEST_DATASET(FcCell_Frame,
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
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    FcCell_Frame_Test fc1(
        "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<Float_T> >());
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

TEST_DATASET(FcCell_Frame,
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
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    FcCell_Frame_Test fc1(
        "fc1", 16, std::make_shared<TanhActivation_Frame<Float_T> >());
    FcCell_Frame_Test fc2(
        "fc2", nbOutputs, std::make_shared<TanhActivation_Frame<Float_T> >());

    fc1.addInput(env);
    fc2.addInput(&fc1);
    fc1.initialize();
    fc2.initialize();

    ASSERT_EQUALS(fc2.getNbSynapses(), 16U * nbOutputs);
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

TEST_DATASET(FcCell_Frame,
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
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    FcCell_Frame_Test fc1(
        "fc1", nbOutputs, std::shared_ptr<Activation<Float_T> >());
    FcCell_Frame_Test fc2(
        "fc2", nbOutputs, std::shared_ptr<Activation<Float_T> >());
    FcCell_Frame_Test fc3(
        "fc3", nbOutputs, std::shared_ptr<Activation<Float_T> >());

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

TEST_DATASET(FcCell_Frame,
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
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    FcCell_Frame_Test fc1(
        "fc1", nbOutputs, std::shared_ptr<Activation<Float_T> >());
    FcCell_Frame_Test fc2(
        "fc2", nbOutputs, std::shared_ptr<Activation<Float_T> >());
    FcCell_Frame_Test fc3(
        "fc3", nbOutputs, std::shared_ptr<Activation<Float_T> >());

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

TEST_DATASET(FcCell_Frame,
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
    Environment env(net, EmptyDatabase, channelsWidth, channelsHeight);

    FcCell_Frame_Test fc1(
        "fc1", nbOutputs, std::make_shared<TanhActivation_Frame<Float_T> >());

    fc1.addInput(env);
    fc1.initialize();

    const unsigned int inputSize = fc1.getNbChannels() * fc1.getChannelsWidth()
                                   * fc1.getChannelsHeight();
    const unsigned int outputSize = fc1.getNbOutputs() * fc1.getOutputsWidth()
                                    * fc1.getOutputsHeight();

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel)
            fc1.setWeight(output, channel, (float)output + channel);
    }

    for (unsigned int output = 0; output < outputSize; ++output) {
        for (unsigned int channel = 0; channel < inputSize; ++channel) {
            ASSERT_EQUALS(fc1.getWeight(output, channel),
                          (float)output + channel);
        }
    }
}

TEST_DATASET(FcCell_Frame,
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

    FcCell_Frame_Test fc1(
        "fc1", nbOutputs, std::shared_ptr<Activation<Float_T> >());
    fc1.setParameter("NoBias", true);

    MNIST_IDX_Database database;
    database.load(N2D2_DATA("mnist"));

    Environment env(net, database, channelsWidth, channelsHeight, 1, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);

    Tensor4d<Float_T>& in = env.getData();

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
        for (unsigned int channel = 0; channel < inputSize; ++channel)
            fc1.setWeight(output, channel, 1.0);
    }

    fc1.propagate();

    const Tensor4d<Float_T>& out = fc1.getOutputs();

    ASSERT_EQUALS(out.dimZ(), nbOutputs);
    ASSERT_EQUALS(out.dimX(), 1U);
    ASSERT_EQUALS(out.dimY(), 1U);

    const Float_T sum
        = std::accumulate(in.begin(),
                          in.begin() + inputSize,
                          0.0f); // Warning: 0.0 leads to wrong results!

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        ASSERT_EQUALS_DELTA(out(output, 0), sum, 1e-5);
    }
}

TEST_DATASET(FcCell_Frame,
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

    FcCell_Frame_Test fc1(
        "fc1", nbOutputs, std::shared_ptr<Activation<Float_T> >());
    fc1.setParameter("NoBias", true);

    MNIST_IDX_Database database;
    database.load(N2D2_DATA("mnist"));

    Environment env(net, database, channelsWidth, channelsHeight, 1, 2, false);
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
        for (unsigned int channel = 0; channel < inputSize; ++channel)
            fc1.setWeight(output, channel, 1.0);
    }

    fc1.propagate();

    ASSERT_EQUALS(fc1.mInputs.dimZ(), fc1.getNbChannels());
    ASSERT_EQUALS(fc1.mInputs[0].dimX(), fc1.getChannelsWidth());
    ASSERT_EQUALS(fc1.mInputs[0].dimY(), fc1.getChannelsHeight());

    Float_T sum = 0.0;

    for (unsigned int channel = 0; channel < fc1.getNbChannels(); ++channel) {
        for (unsigned int y = 0; y < fc1.getChannelsHeight(); ++y) {
            for (unsigned int x = 0; x < fc1.getChannelsWidth(); ++x)
                sum += fc1.mInputs(x, y, channel, 0);
        }
    }

    const Tensor4d<Float_T>& out = fc1.getOutputs();

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

TEST_DATASET(FcCell_Frame,
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
    Environment env(
        net, EmptyDatabase, channelsWidth, channelsHeight, 1, 2, false);

    FcCell_Frame_Test fc1(
        "fc1", nbOutputs, std::shared_ptr<Activation<Float_T> >());
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

    const Tensor4d<Float_T>& out = fc1.getOutputs();

    ASSERT_EQUALS(out.dimZ(), nbOutputs);
    ASSERT_EQUALS(out.dimX(), 1U);
    ASSERT_EQUALS(out.dimY(), 1U);

    for (unsigned int output = 0; output < out.dimZ(); ++output) {
        Float_T sum = 0.0;

        for (unsigned int channel = 0; channel < inputSize; ++channel)
            sum += fc1.getWeight(output, channel);

        ASSERT_EQUALS_DELTA(out(output, 0), sum, 1e-5);
    }
}

RUN_TESTS()
