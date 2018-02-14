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

#ifndef N2D2_ELEMWISECELL_H
#define N2D2_ELEMWISECELL_H

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
    "/include:?mRegistrar@ElemWiseCell_Frame@N2D2@@0U?$Registrar@VElemWiseCell@N2D2@@@2@A")
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ElemWiseCell_Frame_CUDA@N2D2@@0U?$Registrar@VElemWiseCell@N2D2@@@2@A")

#endif
#endif

namespace N2D2 {
class ElemWiseCell : public virtual Cell {
public:
    enum Operation {
        Sum,
        AbsSum,
        EuclideanSum,
        Prod,
        Max
    };

    typedef std::function
        <std::shared_ptr<ElemWiseCell>(Network&,
                                   const std::string&,
                                   unsigned int,
                                   Operation,
                                   const std::vector<Float_T>&,
                                   const std::shared_ptr<Activation<Float_T> >&
                                       activation)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    ElemWiseCell(const std::string& name,
             unsigned int nbOutputs,
             Operation operation = Sum,
             const std::vector<Float_T>& weights = std::vector<Float_T>());
    const char* getType() const
    {
        return Type;
    };
    Operation getOperation() const
    {
        return mOperation;
    };
    void getStats(Stats& stats) const;
    virtual ~ElemWiseCell() {};

protected:
    virtual void setOutputsSize();

    // Operation type
    const Operation mOperation;
    // Block coefficients
    std::vector<Float_T> mWeights;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::ElemWiseCell::Operation>::data[]
    = {"Sum", "AbsSum", "EuclideanSum", "Prod", "Max"};
}

#endif // N2D2_ELEMWISECELL_H
