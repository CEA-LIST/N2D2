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

#ifndef N2D2_COCHLEA_H
#define N2D2_COCHLEA_H

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Aer.hpp"
#include "AerEvent.hpp"
#include "Sound.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Random.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class Cochlea : public Parameterizable {
public:
    enum FilterSpace {
        LinearSpace,
        ErbSpace
    };

    Cochlea(unsigned int nbChannels);

    /**
     * Convert an audio WAV file to an AER sequence emulating the output of a
     *cochlea.
     * The resulting AER data is stored in a new file with the same name but
     *with the ".dat" extension.
     *
     * @param fileName      Audio file name (must be in the WAV format)
     * @param order         Filter order used for each filter in the filterbank.
     *If > 0, uses order 1 & 2 Butterworth filters,
     *                      else uses the Gammatone filters defined by Patterson
     *and Holdworth for simulating the cochlea.
     * @param lowFreq       Lowest center frequency of the filterbank
     * @param upFreq        Upper center frequency of the filterbank
     * @param threshold     Input LIF neurons threshold (WARNING: is made
     *dependent on the relative bandwidth of the filter)
     * @param leak          Input LIF neurons leak time constant
     * @param refractory    Input LIF neurons refractory time
     * @param filterSpace   Filter space: LinearSpace or ErbSpace
     * @param earQ          Quality factor of the filters (default is 9.26449,
     *value from Glasberg and Moore)
     * @param minBw         Minimum bandwidth of the filters (default is 24.7,
     *value from Glasberg and Moore)
     * @param start         Process the audio file starting from this time
     * @param end           Stop processing the audio file after this time (only
     *if > 0)
     *
     * @return The total number of events generated.
    */
    unsigned int load(const std::string& fileName,
                      unsigned int order,
                      double lowFreq,
                      double upFreq,
                      double threshold,
                      Time_T leak = 1 * TimeMs,
                      Time_T refractory = 5 * TimeMs,
                      FilterSpace filterSpace = ErbSpace,
                      double earQ = 9.26449,
                      double minBw = 24.7,
                      double start = 0.0,
                      double end = 0.0);
    virtual ~Cochlea() {};

private:
    unsigned int mNbChannels;

    // Parameters
    Parameter<bool> mNormalize;
};
}

#endif // N2D2_COCHLEA_H
