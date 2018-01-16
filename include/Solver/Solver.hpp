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
#include "containers/Tensor4d.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
template <class T> class Solver : public Parameterizable {
public:
    virtual void update(Tensor4d<T>* data,
                        Tensor4d<T>* diffData,
                        unsigned int batchSize) = 0;
    virtual void exportFreeParameters(const std::string& fileName) const = 0;
    std::shared_ptr<Solver<T> > clone() const
    {
        return std::shared_ptr<Solver<T> >(doClone());
    }
    virtual bool isNewIteration() const = 0;
    virtual ~Solver() {};

private:
    virtual Solver<T>* doClone() const = 0;
};
}

#endif // N2D2_SOLVER_H
