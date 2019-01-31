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

#include "utils/Random.hpp"
#include "utils/Utils.hpp"

// Create a length 624 array to store the state of the generator
unsigned int N2D2::Random::_mt[624];
unsigned int N2D2::Random::_mt_index = 0;

// Initialize the generator from a seed
void N2D2::Random::mtSeed(unsigned int seed)
{
    _mt[0] = seed;
    _mt_index = 0; // Reset also the index to always start at the same point
    // when we re-initialize the generator

    for (unsigned int i = 1; i < 624; ++i)
        _mt[i] = (0x6C078965 * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + i)
                 & 0xFFFFFFFF;
}

// Extract a tempered pseudorandom number based on the index-th value,
unsigned int N2D2::Random::mtRand()
{
    unsigned int y;

#pragma omp critical(Random__mtRand)
{
    if (_mt_index == 0) {
        // Generate an array of 624 untempered numbers
        for (unsigned int i = 0; i < 624; ++i) {
            // bit 31 (32nd bit) of MT[i] + bits 0-30 (first 31 bits) of MT[...]
            const unsigned int y = (_mt[i] & 0x80000000)
                                   + (_mt[(i + 1) % 624] & 0x7FFFFFFF);
            _mt[i] = _mt[(i + 397) % 624] ^ (y >> 1);

            if ((y % 2) != 0)
                _mt[i] ^= 0x9908B0DF;
        }
    }

    y = _mt[_mt_index];
    _mt_index = (_mt_index + 1) % 624;
}

    y ^= y >> 11;
    y ^= (y << 7) & 0x9D2C5680;
    y ^= (y << 15) & 0xEFC60000;
    y ^= y >> 18;
    return y;
}

double N2D2::Random::randNormal(double mean, double stdDev)
{
    static bool availableDeviate = false;
    static double storedDeviate;

    if (stdDev < 0.0)
        throw std::domain_error(
            "Random::randNormal(): standard deviation must be >= 0.");

    if (stdDev == 0.0)
        return mean;

    if (availableDeviate) {
        availableDeviate = false;
        return (mean + stdDev * storedDeviate);
    } else {
        const double u1
            = randUniform(0.0, 1.0, LeftHalfOpenInterval); // u1 range is (0,1]
        const double u2
            = randUniform(0.0, 1.0, LeftHalfOpenInterval); // u2 range is (0,1]

        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * M_PI * u2;

        storedDeviate = r * std::sin(theta);
        availableDeviate = true;

        return (mean + stdDev * (r * std::cos(theta)));
    }
}

double N2D2::Random::randUniform(double vmin, double vmax, Endpoints endpoints)
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

int N2D2::Random::randUniform(int vmin, int vmax)
{
    if (vmax < vmin)
        throw std::domain_error("Random::randUniform(): vmax must be >= vmin.");

    return vmin + (int)((double)Random::mtRand() / (MT_RAND_MAX + 1.0)
                        * (vmax - vmin + 1.0));
}

double N2D2::Random::randNormal(double mean, double stdDev, double vmin)
{
    return std::max(vmin, Random::randNormal(mean, stdDev));
}

double
N2D2::Random::randNormal(double mean, double stdDev, double vmin, double vmax)
{
    if (vmax < vmin)
        throw std::domain_error("Random::randNormal(): vmax must be >= vmin.");

    return Utils::clamp(Random::randNormal(mean, stdDev), vmin, vmax);
}

double N2D2::Random::randLogNormal(double mean, double stdDev)
{
    return std::exp(randNormal(mean, stdDev));
}

double N2D2::Random::randExponential(double mean)
{
    return (-mean * std::log(Random::randUniform(
                        0.0, 1.0, Random::LeftHalfOpenInterval)));
}

bool N2D2::Random::randBernoulli(double p)
{
    // uniform random number x is in [0,1[
    // return 1 if x is in [0,p[ (p = 0 => return always 0)
    // return 0 if x is in [p,1[ (p = 1 => return always 1)
    return (Random::randUniform(0.0, 1.0, Random::RightHalfOpenInterval) < p);
}
