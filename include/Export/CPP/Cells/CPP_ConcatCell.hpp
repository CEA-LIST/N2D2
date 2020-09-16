/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#ifndef N2D2_CPP_CONCAT_CELL_H
#define N2D2_CPP_CONCAT_CELL_H

#include <string>
#include <vector>

#include "Cell/Cell.hpp"

namespace N2D2 {

class DeepNet;
class Stats;

/**
 * Artificial cell used only in the CPP export.
 */
class CPP_ConcatCell: public virtual Cell {
public:
    using RegistryCreate_T =
        std::function<std::shared_ptr<CPP_ConcatCell>(const DeepNet& deepNet, 
                                                              const std::string& name,
                                                              unsigned int nbOutputs)>;

    static RegistryMap_T& registry();
    static const char* Type;
    
    CPP_ConcatCell(const DeepNet& deepNet, const std::string& name, 
                           unsigned int nbOutputs);
    virtual ~CPP_ConcatCell() = default;

    const char* getType() const;
    void getStats(Stats& /*stats*/) const;

protected:
    virtual void setOutputsDims();
    std::pair<double, double> getOutputsRange() const;
};

}

#endif
