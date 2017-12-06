/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_FMPCELL_H
#define N2D2_FMPCELL_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Activation/Activation.hpp"
#include "Environment.hpp"
#include "utils/Registrar.hpp"

#include "Cell.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@FMPCell_Frame@N2D2@@0U?$Registrar@VFMPCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class FMPCell : public virtual Cell {
public:
    typedef std::function
        <std::shared_ptr<FMPCell>(Network&,
                                  const std::string&,
                                  double,
                                  unsigned int,
                                  const std::shared_ptr<Activation<Float_T> >&)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    FMPCell(const std::string& name,
            double scalingRatio,
            unsigned int nbOutputs);
    const char* getType() const
    {
        return Type;
    };
    virtual void initialize();
    unsigned long long int getNbConnections() const;
    void writeMap(const std::string& fileName) const;
    void getStats(Stats& stats) const;
    virtual ~FMPCell() {};

protected:
    virtual void setOutputsSize();

    /// If true, use overlapping regions, else use disjoint regions
    Parameter<bool> mOverlapping;
    /// If true, use pseudorandom sequences, else use random sequences
    Parameter<bool> mPseudoRandom;

    // Scaling ratio
    const double mScalingRatio;
    // mPoolNbChannels[output channel] -> number of input channels connected to
    // this output channel
    std::vector<unsigned int> mPoolNbChannels;
};
}

#endif // N2D2_FMPCELL_H
