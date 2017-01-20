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

#include "AerEvent.hpp"

N2D2::AerEvent::AerEvent(double version)
    : time(0),
      addr(0),
      map(0),
      channel(0),
      node(0),
      mVersion(version),
      mRawTimeNeg(false),
      mRawTimeOffset(0)
{
    // ctor
}

std::ifstream& N2D2::AerEvent::read(std::ifstream& data)
{
    return ((int)mVersion == 2)
               ? read<unsigned int, int>(data)
               : ((int)mVersion == 3) ? read
                     <unsigned int, unsigned long long int>(data)
                                      : read<unsigned short, int>(data);
}

std::ofstream& N2D2::AerEvent::write(std::ofstream& data) const
{
    return ((int)mVersion == 2)
               ? write<unsigned int, int>(data)
               : ((int)mVersion == 3) ? write
                     <unsigned int, unsigned long long int>(data)
                                      : write<unsigned short, int>(data);
}

int N2D2::AerEvent::size() const
{
    return ((int)mVersion == 2)
               ? (sizeof(unsigned int) + sizeof(int))
               : ((int)mVersion == 3)
                     ? (sizeof(unsigned int) + sizeof(unsigned long long int))
                     : (sizeof(unsigned short) + sizeof(int));
}

void N2D2::AerEvent::maps(AerFormat format)
{
    if (format == N2D2Env) {
        map = addr >> 28;
        channel = (addr >> 24) & 0xF;
        node = addr & 0xFFFFFF;
    } else if (format == Dvs128) {
        map = 0;
        channel = addr & 1;
        node = 128 * 128 - (addr >> 1) - 1;
    } else
        throw std::runtime_error("Unknown AER format");
}

void N2D2::AerEvent::unmaps(AerFormat format)
{
    if (format == N2D2Env)
        addr = unmaps(map, channel, node);
    else if (format == Dvs128) {
        if (channel > 1)
            throw std::domain_error("AerEvent::unmaps(): out of range");

        addr = ((128 * 128 - node - 1) << 1) | channel;
    } else
        throw std::runtime_error("Unknown AER format");
}

unsigned int N2D2::AerEvent::unmaps(unsigned int map,
                                    unsigned int channel,
                                    unsigned int node)
{
    if (map > 0xF || channel > 0xF || node > 0xFFFFFF)
        throw std::domain_error("AerEvent::unmaps(): out of range");

    return ((map << 28) | (channel << 24) | node);
}
