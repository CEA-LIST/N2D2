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
#include <vector>

#include "utils/Registrar.hpp"

#include "Cell.hpp"
#include "Cell/PoolCell_Frame_Kernels.hpp"
#include "controler/Interface.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {

class Activation;
class DeepNet;
class Network;

class PoolCell : public virtual Cell {
public:
    enum Pooling {
        Max,
        Average
    };

    typedef std::function
        <std::shared_ptr<PoolCell>(Network&, const DeepNet&, 
                                   const std::string&,
                                   const std::vector<unsigned int>&,
                                   unsigned int,
                                   const std::vector<unsigned int>&,
                                   const std::vector<unsigned int>&,
                                   Pooling,
                                   const std::shared_ptr<Activation>&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    PoolCell(const DeepNet& deepNet, const std::string& name,
             const std::vector<unsigned int>& poolDims,
             unsigned int nbOutputs,
             const std::vector<unsigned int>& strideDims
                = std::vector<unsigned int>(2, 1U),
             const std::vector<unsigned int>& paddingDims
                = std::vector<unsigned int>(2, 0),
             Pooling pooling = Max);
    const char* getType() const
    {
        return Type;
    };
    unsigned long long int getNbConnections() const;
    unsigned int getPoolWidth() const
    {
        return mPoolDims[0];
    };
    unsigned int getPoolHeight() const
    {
        return mPoolDims[1];
    };
    unsigned int getStrideX() const
    {
        return mStrideDims[0];
    };
    unsigned int getStrideY() const
    {
        return mStrideDims[1];
    };
    int getPaddingX() const
    {
        return mPaddingDims[0];
    };
    int getPaddingY() const
    {
        return mPaddingDims[1];
    };
    Pooling getPooling() const
    {
        return mPooling;
    };
    void writeMap(const std::string& fileName) const;
    void getStats(Stats& stats) const;
    std::vector<unsigned int> getReceptiveField(
                                const std::vector<unsigned int>& outputField
                                        = std::vector<unsigned int>()) const;
    virtual Interface<PoolCell_Frame_Kernels::ArgMax>* getArgMax()
    {
        return NULL;
    };
    virtual ~PoolCell() {};

protected:
    virtual void setOutputsDims();
    std::pair<double, double> getOutputsRange() const;

protected:
    // Pool dims
    const std::vector<unsigned int> mPoolDims;
    // Stride for the pooling
    const std::vector<unsigned int> mStrideDims;
    // Padding for the pooling
    const std::vector<unsigned int> mPaddingDims;
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
