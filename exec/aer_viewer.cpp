/*
    (C) Copyright 2012 CEA LIST. All Rights Reserved.
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

using namespace N2D2;

int main(int argc, char* argv[])
{
    // Program command line options
    ProgramOptions opts(argc, argv);
    const unsigned int sizeX
        = opts.parse("-sx", 128U, "environment input width");
    const unsigned int sizeY
        = opts.parse("-sy", 128U, "environment input height");
    const bool dvs128
        = opts.parse("-dvs128", true, "AER file is in DVS128 format");
    const std::string outputVideo = opts.parse<std::string>(
        "-o", "", "save AER sequence to output video");
    const bool accDiff
        = opts.parse("-diff", true, "conversion difference accumulation");
    const double threshold = opts.parse("-thres", 0.01, "conversion threshold");
    const unsigned int fps
        = opts.parse("-fps", 25, "conversion frame per second (fps)");
    const std::string videoFile = opts.grab<std::string>(
        "<video file>", "AER file to read or video file to convert to AER");
    opts.done();

    Network net;
    std::shared_ptr
        <Environment> env(new Environment(net, EmptyDatabase, sizeX, sizeY));

    if (accDiff) {
        env->addChannelTransformation(FilterTransformationAerPositive);
        env->addChannelTransformation(FilterTransformationAerNegative);
    } else {
        env->addChannelTransformation(FilterTransformationLaplacian);
        env->addChannelTransformation(-FilterTransformationLaplacian);
    }

    Aer aer(env);

    if (Utils::fileExtension(videoFile) != "dat") {
        aer.loadVideo(videoFile,
                      fps,
                      threshold,
                      (accDiff) ? Aer::AccumulateDiff : Aer::Accumulate);

        const std::string aerFile = Utils::fileBaseName(videoFile) + ".dat";
        aer.viewer(aerFile, AerEvent::N2D2Env, "", 0, outputVideo);
    } else
        aer.viewer(videoFile,
                   (dvs128) ? AerEvent::Dvs128 : AerEvent::N2D2Env,
                   "",
                   0,
                   outputVideo);

    return 0;
}
