/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_AER_H
#define N2D2_AER_H

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifndef CV_WINDOW_NORMAL
#define CV_WINDOW_NORMAL 0
#endif

#include "AerEvent.hpp"
#include "HeteroEnvironment.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Random.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class Aer : public Parameterizable {
public:
    enum AerCodingMode {
        Accumulate,
        AccumulateDiff
    };
    typedef std::vector<std::pair<Time_T, unsigned int> > AerData_T;

    Aer(const std::shared_ptr<HeteroEnvironment>& environment);
    Aer(const std::shared_ptr<Environment>& environment);
    std::pair<Time_T, Time_T> getTimes(const std::string& fileName) const;

    /**
     * Read an AER file.
     *
     * @param fileName      AER file name
     * @param format        AER format used for the spike addresses (see
     *AerEvent::AerFormat)
     * @param ret           If true, return the list of events as a vector
     *(AerData_T), else add the AER events to the event
     *                      queue of the simulator in preparation of
     *Network::run() (the returned value is then an empty vector)
     * @param offset        Offset to add to the timestamp of the events
     * @param start         Read events starting from this time
     * @param end           Stop reading events after this time (only if > 0)
     * @return If @p ret is true, the list of events, else empty vector
     *
     * @exception std::runtime_error Unable to read the AER file
     * @exception std::runtime_error The input AER data is non-monotonic
    */
    AerData_T read(const std::string& fileName,
                   AerEvent::AerFormat format = AerEvent::N2D2Env,
                   bool ret = false,
                   Time_T offset = 0,
                   Time_T start = 0,
                   Time_T end = 0);

    void merge(const std::string& source1,
               const std::string& source2,
               AerEvent::AerFormat format,
               const std::string& destination);

    /**
     * Save an AER sequence (of type AerData_T) to a file.
     *
     * @param fileName      AER file name
     * @param data          AER data
     * @param append        If true, appends data to an existing file instead of
     *creating a new one
     * @param version       AER file version (version 3.0 is specific to N2D2
     *and uses 64 bits integers to store the timestamps)
    */
    static void save(const std::string& fileName,
                     const AerData_T& data,
                     bool append = false,
                     double version = 3.0);

    /**
     * Convert a video to an AER sequence emulating the output of a spiking
     *retina.
     * Preprocessing: convert the video in grayscale and resize it to the
     *correct size for each map of the Environment.
     * Environment filters are applied on each frame as for static images.
     * The resulting AER data is stored in a new file with the same name but
     *with the ".dat" extension.
     *
     * @param fileName      Video file name
     * @param fps           Frame rate of the input video, used as a scaling
     *factor for the timing of the generated events
     * @param threshold     Minimum accumulated pixel's value to trigger an
     *event
     * @param mode          AER coding mode: Accumulate, or AccumulateDiff,
     *which accumulates the difference between consecutive
     *                      frames (for a DVS128-like coding)
     * @return The number of events generated
     *
     * @exception std::runtime_error Unable to read the input video
     * @exception std::domain_error The @p threshold value must be positive
    */
    unsigned int loadVideo(const std::string& fileName,
                           unsigned int fps,
                           double threshold = 0.1,
                           AerCodingMode mode = AccumulateDiff);

    /**
     * Read an AER data file and display it on screen.
     * This function can also display labeling information on top of AER
     *sequences. Labeling data must be stored in a plain
     * text file containing a list of labels in the following format, with one
     *label per line:
     * @verbatim <timestamp in fs> @endverbatim
     * or
     * @verbatim <timestamp in fs> <x> <y> <width> <height> @endverbatim
     *
     * @param fileName      AER file name
     * @param format        AER format used for the spike addresses (see
     *AerEvent::AerFormat)
     * @param labelFile     Labeling data file name
     * @param labelTime     Stop time for displaying a label (in ms). If 0, wait
     *indefinitely until a key is pressed
     * @param videoName     If specified, in addition to displaying the AER
     *sequences on screen, save it into a file
    */
    void viewer(const std::string& fileName,
                AerEvent::AerFormat format = AerEvent::N2D2Env,
                const std::string& labelFile = "",
                unsigned int labelTime = 500,
                const std::string& videoName = "");
    virtual ~Aer() {};

private:
    double readVersion(std::ifstream& data) const;

    const std::shared_ptr<HeteroEnvironment> mEnvironment;

    // Parameters
    /// Additional standard deviation on spike timing (jitter) when reading an
    /// AER sequence
    Parameter<Time_T> mAerJitter;
    /// Additional uniform spiking noise density when reading an AER sequence
    Parameter<double> mAerUniformNoise;
};
}

#endif // N2D2_AER_H
