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

#ifndef N2D2_RANDOM_H
#define N2D2_RANDOM_H

#include <cmath>
#include <cstdlib>
#include <stdexcept>

#define MT_RAND_MAX 0xFFFFFFFF

namespace N2D2 {
namespace Random {
    extern unsigned int _mt[624];
    extern unsigned int _mt_index;
    extern unsigned int _mt_init;

    enum Endpoints {
        ClosedInterval,
        LeftHalfOpenInterval,
        RightHalfOpenInterval,
        OpenInterval
    };

    /**
     * Initialize the internal Mersenne Twister MT19937 pseudorandom number
     *generator from a seed.
     * Drop-in, improved and platform independent replacement of std::srand().
     *
     * @param seed          Seed value
    */
    void mtSeed(unsigned int seed = 1);

    /**
     * Generates uniformly distributed 32-bit integers in the range [0,
     *(2^32)-1] with the internal Mersenne Twister MT19937
     * algorithm.
     * Drop-in, improved and platform independent replacement of std::rand().
     *
     * @return Random number in the closed interval [0, MT_RAND_MAX] = [0,
     *(2^32)-1]
    */
    unsigned int mtRand();
    double randUniform(double vmin = 0.0,
                       double vmax = 1.0,
                       Endpoints endpoints = ClosedInterval);

    /**
     * Generates uniformly distributed integers in the range [min, max]
     *
     * @param vmin           Minimum interval value
     * @param vmax           Maximum interval value
     * @return Random number in the closed interval [min, max]
     *
     * @exception std::domain_error vmax must be > vmin
    */
    int randUniform(int vmin, int vmax);

    /**
     * Generates uniformly distributed integers in the range [0, max-1].
     * This function can be used with std::random_shuffle().
     *
     * @param value         Maximum value (excluded from the range)
     * @return Random number in the closed interval [0, max-1]
    */
    int randShuffle(int value);
    double randNormal(double mean = 0.0, double stdDev = 1.0);
    double randNormal(double mean, double stdDev, double vmin);
    double randNormal(double mean, double stdDev, double vmin, double vmax);
    double randLogNormal(double mean, double stdDev);

    /**
     * Generates exponentially distributed numbers, with rate = 1/mean.
     * This distribution describes the time between events in a Poisson process,
     *i.e. a process in which events occur
     * continuously and independently at a constant average rate.
     *
     * @param mean            Mean time between events (= 1/rate)
     * @return Random number
    */
    double randExponential(double mean);

    /**
     * Generates Bernoulli random number.
     *
     * @return 1 with probability p and 0 with probability 1-p
    */
    bool randBernoulli(double p = 0.5);
}
}

#endif // N2D2_RANDOM_H
