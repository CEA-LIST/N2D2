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

#ifndef N2D2_POOLCELL_H
#define N2D2_POOLCELL_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Activation/Activation.hpp"
#include "Environment.hpp"
#include "utils/Registrar.hpp"

#include "Cell.hpp"
#include "Cell/PoolCell_Frame_Kernels.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@PoolCell_Frame@N2D2@@0U?$Registrar@VPoolCell@N2D2@@@2@A")
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@PoolCell_Frame_CUDA@N2D2@@0U?$Registrar@VPoolCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@PoolCell_Frame_EXT_CUDA@N2D2@@0U?$Registrar@VPoolCell@N2D2@@@2@A")
#endif
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@PoolCell_Spike@N2D2@@0U?$Registrar@VPoolCell@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@?$PoolCell_Transcode@VPoolCell_Frame@N2D2@@VPoolCell_Spike@2@@N2D2@@0U?$Registrar@VPoolCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class PoolCell : public virtual Cell {
public:
    enum Pooling {
        Max,
        Average
    };

    typedef std::function
        <std::shared_ptr<PoolCell>(Network&,
                                   const std::string&,
                                   unsigned int,
                                   unsigned int,
                                   unsigned int,
                                   unsigned int,
                                   unsigned int,
                                   unsigned int,
                                   unsigned int,
                                   Pooling,
                                   const std::shared_ptr<Activation<Float_T> >&
                                       activation)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    PoolCell(const std::string& name,
             unsigned int poolWidth,
             unsigned int poolHeight,
             unsigned int nbOutputs,
             unsigned int strideX = 1,
             unsigned int strideY = 1,
             unsigned int paddingX = 0,
             unsigned int paddingY = 0,
             Pooling pooling = Max);
    const char* getType() const
    {
        return Type;
    };
    unsigned long long int getNbConnections() const;
    unsigned int getPoolWidth() const
    {
        return mPoolWidth;
    };
    unsigned int getPoolHeight() const
    {
        return mPoolHeight;
    };
    unsigned int getStrideX() const
    {
        return mStrideX;
    };
    unsigned int getStrideY() const
    {
        return mStrideY;
    };
    unsigned int getPaddingX() const
    {
        return mPaddingX;
    };
    unsigned int getPaddingY() const
    {
        return mPaddingY;
    };
    Pooling getPooling() const
    {
        return mPooling;
    };
    void writeMap(const std::string& fileName) const;
    void getStats(Stats& stats) const;
    virtual Interface<PoolCell_Frame_Kernels::ArgMax>* getArgMax()
    {
        return NULL;
    };
    virtual ~PoolCell() {};

protected:
    virtual void setOutputsSize();

    // Pool width
    const unsigned int mPoolWidth;
    // Pool height
    const unsigned int mPoolHeight;
    // Horizontal stride for the pooling
    const unsigned int mStrideX;
    // Vertical stride for the pooling
    const unsigned int mStrideY;
    // Horizontal padding for the pooling
    const unsigned int mPaddingX;
    // Vertical padding for the pooling
    const unsigned int mPaddingY;
    // Pooling type
    const Pooling mPooling;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::PoolCell::Pooling>::data[]
    = {"Max", "Average"};
}

#endif // N2D2_POOLCELL_H
