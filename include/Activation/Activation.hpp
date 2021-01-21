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

#ifndef N2D2_ACTIVATION_H
#define N2D2_ACTIVATION_H

#include <iosfwd>
#include <vector>

#include "Scaling.hpp"
#include "utils/Parameterizable.hpp"

namespace N2D2 {

class BaseTensor;
class Cell;

class Activation : public Parameterizable {
public:
    Activation();
    virtual ~Activation() {};

    virtual const char* getType() const = 0;

    virtual void propagate(const Cell& cell, BaseTensor& data, bool inference = false) = 0;
    virtual void backPropagate(const Cell& cell, BaseTensor& data, BaseTensor& diffData) = 0;

    /**
     * Return the possible range of the activation's output as a pair of min-max. 
     */
    virtual std::pair<double, double> getOutputRange() const = 0;

    virtual void save(const std::string& dirName) const;
    virtual void load(const std::string& dirName);

    const Scaling& getActivationScaling() const;
    void setActivationScaling(Scaling scaling);

protected:
    virtual void saveInternal(std::ostream& /*state*/,
                              std::ostream& /*log*/) const {};
    virtual void loadInternal(std::istream& /*state*/) {};

    Scaling mScaling;
};
}

#endif // N2D2_ACTIVATION_H
