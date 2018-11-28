/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_SOLVER_H
#define N2D2_SOLVER_H

#include "Environment.hpp"
#include "containers/Tensor.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class Solver : public Parameterizable {
public:
    virtual const char* getType() const = 0;
    virtual void update(BaseTensor& data,
                        BaseTensor& diffData,
                        unsigned int batchSize) = 0;
    std::shared_ptr<Solver> clone() const
    {
        return std::shared_ptr<Solver>(doClone());
    }
    virtual bool isNewIteration() const = 0;
    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);
    virtual std::pair<double, double> getRange() const = 0;
    virtual std::pair<double, double> getQuantizedRange() const = 0;
    virtual ~Solver() {};

protected:
    virtual void saveInternal(std::ostream& /*state*/,
                              std::ostream& /*log*/) const {};
    virtual void loadInternal(std::istream& /*state*/) {};

private:
    virtual Solver* doClone() const = 0;
};
}

#endif // N2D2_SOLVER_H
