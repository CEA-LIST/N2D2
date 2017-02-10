/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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

/** @file
 * This program simulate the learning of actual AER retina data with the
 * NodeNeuron_Behavioral model.
*/

#include <list>

#include "N2D2.hpp"

using namespace N2D2;

int main(int argc, char* argv[])
{
    // Program command line options
    ProgramOptions opts(argc, argv);
    const unsigned int parNbPass
        = opts.parse("-p", 10U, "number of learning pass");
    const unsigned int parNbNeurons
        = opts.parse("-n", 20U, "number of neurons per cell");
    const bool genConfig
        = opts.parse("-cfg", "save base configuration and exit");
    const bool noStepLog
        = opts.parse("-no-step-log", "don't log intermediate steps");
    const bool noise = opts.parse("-noise", "add noise and jitter");
    const std::string aerFile
        = opts.grab<std::string>("dvs/events20051221T014416 freeway.mat.dat",
                                 "<aer file>",
                                 "learning AER data file (in N2D2_DATA path)");
    opts.done();

    Network net;
    std::shared_ptr
        <Environment> env(new Environment(net, EmptyDatabase, 128, 128));
    env->addChannelTransformation(FilterTransformationAerPositive);
    env->addChannelTransformation(FilterTransformationAerNegative);

    Aer aer(env);

    if (noise) {
        aer.setParameter("AerUniformNoise", 0.5);
        aer.setParameter("AerJitter", 5 * TimeMs);
    }

    // Network topology construction
    Xcell l1(net);
    l1.populate<NodeNeuron_Behavioral>(parNbNeurons);
    l1.setNeuronsParameter<Weight_T>("WeightsMax", 1000, 200.0);
    l1.setNeuronsParameter
        <Weight_T>("WeightsInit",
                   500,
                   200.0); // ~ 650 events/trajectory = 650 events/100 ms
    l1.setNeuronsParameter
        <Weight_T>("WeightIncrement", 100, 5.0); // /!\ Attention
    l1.setNeuronsParameter<Weight_T>("WeightDecrement", 50, 5.0);
    l1.setNeuronsParameter("WeightIncrementDamping", 0.0);
    l1.setNeuronsParameter("WeightDecrementDamping", 0.0);

    l1.setNeuronsParameter("Threshold",
                           1000000.0); // @ W=50/event = ~ 200 events
    l1.setNeuronsParameter("StdpLtp",
                           12 * TimeMs); // @ 50 events/ms = ~ 5 events
    // l1.setNeuronsParameter("StdpLtd", 500*TimeMs);       // @ 50 events/ms =
    // ~ 10 events
    l1.setNeuronsParameter("Refractory", 300 * TimeMs);
    l1.setNeuronsParameter("InhibitRefractory", 50 * TimeMs);
    l1.setNeuronsParameter("Leak", 450 * TimeMs);
    // l1.setNeuronsParameter("InhibitIntegration",
    // std::numeric_limits<double>::max());  // > threshold => reset integration
    // to 0

    if (genConfig) {
        env->saveParameters("config_env.cfg");
        l1.saveParameters("config_xcell.cfg");
        l1.saveNeuronsParameters("config_neurons.cfg");
        std::exit(0);
    } else {
        env->loadParameters("config_env.cfg", true);
        l1.loadParameters("config_xcell.cfg", true);
        l1.loadNeuronsParameters("config_neurons.cfg", true);
    }

    l1.addInput(*env, 0, 0, 128, 128);

    const std::pair<Time_T, Time_T> aerTime = aer.getTimes(N2D2_DATA(aerFile));

    Monitor monitorL1(net);
    monitorL1.add(l1);
    std::list<cv::Mat> frames;

    // Simulation
    for (unsigned int n = 0; n < parNbPass; ++n) {
        const bool log = (!noStepLog || n == parNbPass - 1);

        if (log)
            monitorL1.clearAll();

        for (unsigned int i = aerTime.first / (double)TimeS;
             i <= aerTime.second / (double)TimeS;
             ++i) {
            std::cout << "Learning from " << i << " s to " << (i + 1)
                      << " s (pass " << (n + 1) << ")";
            aer.read(N2D2_DATA(aerFile),
                     AerEvent::Dvs128,
                     false,
                     0,
                     i * TimeS,
                     (i + 1) * TimeS);
            net.run((i + 1) * TimeS);

            monitorL1.update(log);
            std::cout << "   [" << monitorL1.getTotalActivity() << "]"
                      << std::endl;

            // frames.push_back(l1.reconstructPattern(0, true));
        }

        net.reset();

        if (log) {
            monitorL1.logActivity("activity.dat", true);

            // Save final weights representation
            std::cout << "Saving weights reconstruction... " << std::endl;
            l1.reconstructPatterns("res");
        }
    }

    // Create a nice animation (but takes time)
    if (frames.size() > 0) {
        cv::VideoWriter video("aer_cars.weights.avi",
                              CV_FOURCC('H', 'F', 'Y', 'U'),
                              10.0,
                              cv::Size(512, 512));

        if (!video.isOpened())
            throw std::runtime_error(
                "Unable to write video file: aer_cars.weights.avi");

        for (std::list<cv::Mat>::iterator it = frames.begin(),
                                          itEnd = frames.end();
             it != itEnd;
             ++it) {
            cv::Mat resized;
            cv::resize((*it),
                       resized,
                       cv::Size(512, 512),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
            video << resized;
        }
    }

    return 0;
}
