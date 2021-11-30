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

#ifndef N2D2_DISTANCE_CELL_H
#define N2D2_DISTANCE_CELL_H

#include <functional>
#include <memory>
#include <string>

#include "Cell.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {

class DeepNet;
class Stats;
class Filler;
class Solver;

class DistanceCell: public virtual Cell {
public:
    using RegistryCreate_T =
        std::function<std::shared_ptr<DistanceCell>(const DeepNet& deepNet, 
                                                   const std::string& name,
                                                   unsigned int nbOutputs,
                                                   double margin,
                                                   double centercoef)>;

    static RegistryMap_T& registry();
    static const char* Type;

    DistanceCell(const DeepNet& deepNet, const std::string& name,
                unsigned int nbOutputs, double margin, double centercoef);
    virtual ~DistanceCell() = default;

    const char* getType() const;

    virtual void getWeight(unsigned int output,
                           unsigned int channel, BaseTensor& value) const = 0;
    virtual void setWeight(unsigned int output, unsigned int channel,
                           const BaseTensor& value) = 0;
    virtual void exportFreeParameters(const std::string& fileName) const;
    virtual void importFreeParameters(const std::string& fileName,
                                      bool ignoreNotExists = false);

    void setWeightsFiller(const std::shared_ptr<Filler>& filler)
    {
        mWeightsFiller = filler;
    };

    void setWeightsSolver(const std::shared_ptr<Solver>& solver)
    {
        mWeightsSolver = solver;
    };
    std::shared_ptr<Solver> getWeightsSolver()
    {
        return mWeightsSolver;
    };

    void getStats(Stats& stats) const;

protected:
    virtual void setOutputsDims();

    double mMargin;
    double mCenterCoef;
    Parameter<std::size_t> mEndIT;
    
    std::shared_ptr<Filler> mWeightsFiller;
    std::shared_ptr<Solver> mWeightsSolver;
    

};
}

#endif // N2D2_DISTANCE_CELL_H
