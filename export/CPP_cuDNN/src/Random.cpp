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

#include "Random.hpp"

// Create a length 624 array to store the state of the generator
unsigned int Random::_mt[624];
unsigned int Random::_mt_index = 0;

// Initialize the generator from a seed
void Random::mtSeed(unsigned int seed)
{
    _mt[0] = seed;
    _mt_index = 0; // Reset also the index to always start at the same point
    // when we re-initialize the generator

    for (unsigned int i = 1; i < 624; ++i)
        _mt[i] = (0x6C078965 * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + i)
                 & 0xFFFFFFFF;
}

// Extract a tempered pseudorandom number based on the index-th value,
unsigned int Random::mtRand()
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

    unsigned int y = _mt[_mt_index];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9D2C5680;
    y ^= (y << 15) & 0xEFC60000;
    y ^= y >> 18;

    _mt_index = (_mt_index + 1) % 624;
    return y;
}
