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

#ifndef RANDOM_H
#define RANDOM_H

#include <cmath>
#include <cstdlib>
#include <stdexcept>

#define MT_RAND_MAX 0xFFFFFFFF

namespace Random {
extern unsigned int _mt[624];
extern unsigned int _mt_index;

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
 * Generates uniformly distributed 32-bit integers in the range [0, (2^32)-1]
 *with the internal Mersenne Twister MT19937
 * algorithm.
 * Drop-in, improved and platform independent replacement of std::rand().
 *
 * @return Random number in the closed interval [0, MT_RAND_MAX] = [0, (2^32)-1]
*/
unsigned int mtRand();
inline double randUniform(double vmin = 0.0,
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
inline int randUniform(int vmin, int vmax);

/**
 * Generates exponentially distributed numbers, with rate = 1/mean.
 * This distribution describes the time between events in a Poisson process,
 *i.e. a process in which events occur
 * continuously and independently at a constant average rate.
 *
 * @param mean            Mean time between events (= 1/rate)
 * @return Random number
*/
inline double randExponential(double mean);

/**
 * Generates Bernoulli random number.
 *
 * @return 1 with probability p and 0 with probability 1-p
*/
inline bool randBernoulli(double p = 0.5);
}

double Random::randUniform(double vmin, double vmax, Endpoints endpoints)
{
    if (vmax < vmin)
        throw std::domain_error("Random::randUniform(): vmax must be >= vmin.");

    if (endpoints == ClosedInterval) // [vmin,vmax]
        return vmin + (double)Random::mtRand() / MT_RAND_MAX * (vmax - vmin);
    else if (endpoints == LeftHalfOpenInterval) // ]vmin,vmax] = (vmin,vmax]
        return vmin + ((double)Random::mtRand() + 1.0) / (MT_RAND_MAX + 1.0)
                      * (vmax - vmin);
    else if (endpoints == RightHalfOpenInterval) // [vmin,vmax[ = [vmin,vmax)
        return vmin + (double)Random::mtRand() / (MT_RAND_MAX + 1.0)
                      * (vmax - vmin);
    else // ]vmin,vmax[ = (vmin,vmax)
        return vmin + ((double)Random::mtRand() + 0.5) / (MT_RAND_MAX + 1.0)
                      * (vmax - vmin);
}

int Random::randUniform(int vmin, int vmax)
{
    if (vmax < vmin)
        throw std::domain_error("Random::randUniform(): vmax must be >= vmin.");

    return vmin + (int)((double)Random::mtRand() / (MT_RAND_MAX + 1.0)
                        * (vmax - vmin + 1.0));
}

double Random::randExponential(double mean)
{
    return (-mean * std::log(Random::randUniform(
                        0.0, 1.0, Random::LeftHalfOpenInterval)));
}

bool Random::randBernoulli(double p)
{
    // uniform random number x is in [0,1[
    // return 1 if x is in [0,p[ (p = 0 => return always 0)
    // return 0 if x is in [p,1[ (p = 1 => return always 1)
    return (Random::randUniform(0.0, 1.0, Random::RightHalfOpenInterval) < p);
}

#endif // N2D2_RANDOM_H
