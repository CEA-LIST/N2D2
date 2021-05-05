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

#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

#include "Cell.hpp"

namespace N2D2 {

class Activation;
class DeepNet;
class Network;

class ElemWiseCell : public virtual Cell {
public:
    enum Operation {
        Sum,
        AbsSum,
        EuclideanSum,
        Prod,
        Max
    };
    enum CoeffMode {
        PerLayer,
        PerInput,
        PerChannel
    };

    typedef std::function
        <std::shared_ptr<ElemWiseCell>(Network&, const DeepNet&, 
                                   const std::string&,
                                   unsigned int,
                                   Operation,
                                   CoeffMode,
                                   const std::vector<Float_T>&,
                                   const std::vector<Float_T>&,
                                   const std::shared_ptr<Activation>&
                                       activation)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    ElemWiseCell(const DeepNet& deepNet, const std::string& name,
             unsigned int nbOutputs,
             Operation operation = Sum,
             CoeffMode mode = PerLayer,
             const std::vector<Float_T>& weights = std::vector<Float_T>(),
             const std::vector<Float_T>& shifts = std::vector<Float_T>());
    const char* getType() const
    {
        return Type;
    };
    Operation getOperation() const
    {
        return mOperation;
    };
    CoeffMode getCoeffMode() const 
    {
        return mCoeffMode;
    };
    std::vector<Float_T> getWeights() const
    {
        return mWeights;
    };
    std::vector<Float_T> getShifts() const
    {
        return mShifts;
    };

    void getStats(Stats& stats) const;
    std::vector<unsigned int> getReceptiveField(
                                const std::vector<unsigned int>& outputField
                                        = std::vector<unsigned int>()) const;
    virtual ~ElemWiseCell() {};

protected:
    virtual void setOutputsDims();
    std::pair<double, double> getOutputsRange() const;

protected:
    // Operation type
    const Operation mOperation;
    // Coeff Mode type
    const CoeffMode mCoeffMode;
    // Block coefficients
    std::vector<Float_T> mWeights;
    // Block shifts
    std::vector<Float_T> mShifts;

};
}

namespace {
template <>
const char* const EnumStrings<N2D2::ElemWiseCell::Operation>::data[]
    = {"Sum", "AbsSum", "EuclideanSum", "Prod", "Max"};
}
namespace {
template <>
const char* const EnumStrings<N2D2::ElemWiseCell::CoeffMode>::data[]
    = {"PerLayer", "PerInput", "PerChannel"};
}

#endif // N2D2_ELEMWISECELL_H
