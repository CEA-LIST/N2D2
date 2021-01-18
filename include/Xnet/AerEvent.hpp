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

#ifndef N2D2_AEREVENT_H
#define N2D2_AEREVENT_H

#include <fstream>
#include <stdexcept>

#include "Network.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
/**
 * This class is a wrapper for reading and writing AER files, used internally by
 * Environment::readAer(), Environment::saveAer()
 * and Environment::aerViewer().
*/
struct AerEvent {
    enum AerFormat {
        N2D2Env,
        Dvs128
    };

    AerEvent(double version = 3.0);
    std::ifstream& read(std::ifstream& data);
    std::ofstream& write(std::ofstream& data) const;
    int size() const;
    void maps(AerFormat format = N2D2Env);
    void unmaps(AerFormat format = N2D2Env);
    static unsigned int
    unmaps(unsigned int map, unsigned int channel, unsigned int node);

    Time_T time;
    unsigned int addr;

    unsigned int map;
    unsigned int channel;
    unsigned int node;

private:
    template <class T1, class T2>
    typename std::enable_if<std::is_unsigned<T2>::value, std::ifstream&>::type
    read(std::ifstream& data);
    template <class T1, class T2>
    typename std::enable_if<!std::is_unsigned<T2>::value, std::ifstream&>::type
    read(std::ifstream& data);
    template <class T1, class T2>
    std::ofstream& write(std::ofstream& data) const;

    const double mVersion;
    bool mRawTimeNeg;
    unsigned long long int mRawTimeOffset;
};
}

template <class T1, class T2>
typename std::enable_if<std::is_unsigned<T2>::value, std::ifstream&>::type
N2D2::AerEvent::read(std::ifstream& data)
{
    T1 rawAddr;
    T2 rawTime;

    data.read(reinterpret_cast<char*>(&rawAddr), sizeof(rawAddr));
    data.read(reinterpret_cast<char*>(&rawTime), sizeof(rawTime));

    if (!Utils::isBigEndian()) {
        Utils::swapEndian(rawAddr);
        Utils::swapEndian(rawTime);
    }

    addr = static_cast<unsigned int>(rawAddr);
    // AER version = 3
    time = rawTime;

    return data;
}

template <class T1, class T2>
typename std::enable_if<!std::is_unsigned<T2>::value, std::ifstream&>::type
N2D2::AerEvent::read(std::ifstream& data)
{
    T1 rawAddr;
    T2 rawTime;

    data.read(reinterpret_cast<char*>(&rawAddr), sizeof(rawAddr));
    data.read(reinterpret_cast<char*>(&rawTime), sizeof(rawTime));

    if (!Utils::isBigEndian()) {
        Utils::swapEndian(rawAddr);
        Utils::swapEndian(rawTime);
    }

    addr = static_cast<unsigned int>(rawAddr);

    // Check & correct for overflow
    // (required for "Tmpdiff128-2007-02-28T15-08-15-0800-0 3 flies 2m 1f.dat"
    // for example)
    if (rawTime < 0 && !mRawTimeNeg) {
        // std::cout << "AER time overflow detected! (time is " << rawTime << "
        // us, offset is "
        //          << mRawTimeOffset << " us)" << std::endl;
        mRawTimeOffset += (1ULL << 8 * sizeof(rawTime));
        mRawTimeNeg = true;
    } else if (rawTime >= 0 && mRawTimeNeg)
        mRawTimeNeg = false;

    time = (mRawTimeOffset + rawTime) * TimeUs;

    return data;
}

template <class T1, class T2>
std::ofstream& N2D2::AerEvent::write(std::ofstream& data) const
{
    T1 rawAddr = static_cast<T1>(addr);
    T2 rawTime = ((int)mVersion == 3) ? time : time / TimeUs;

    if (!Utils::isBigEndian()) {
        Utils::swapEndian(rawAddr);
        Utils::swapEndian(rawTime);
    }

    data.write(reinterpret_cast<char*>(&rawAddr), sizeof(rawAddr));
    data.write(reinterpret_cast<char*>(&rawTime), sizeof(rawTime));

    if (!data.good())
        throw std::runtime_error("AerEvent::write(): error writing data");

    return data;
}

#endif // N2D2_AEREVENT_H
