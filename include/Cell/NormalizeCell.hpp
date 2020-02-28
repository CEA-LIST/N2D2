/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_NORMALIZE_CELL_H
#define N2D2_NORMALIZE_CELL_H

#include <functional>
#include <memory>
#include <string>

#include "Cell.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {

class DeepNet;
class Stats;

class NormalizeCell: public virtual Cell {
public:
    enum Norm {
        L1,
        L2
    };

    using RegistryCreate_T =
        std::function<std::shared_ptr<NormalizeCell>(const DeepNet& deepNet, 
                                                   const std::string& name,
                                                   unsigned int nbOutputs,
                                                   Norm norm)>;

    static RegistryMap_T& registry();
    static const char* Type;

    NormalizeCell(const DeepNet& deepNet, const std::string& name,
                unsigned int nbOutputs, Norm norm);
    virtual ~NormalizeCell() = default;

    const char* getType() const;
    void getStats(Stats& stats) const;

protected:
    virtual void setOutputsDims();

protected:
    Norm mNorm;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::NormalizeCell::Norm>::data[]
    = {"L1", "L2"};
}

#endif // N2D2_NORMALIZE_CELL_H
