/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_ACTIVATIONCELL_H
#define N2D2_ACTIVATIONCELL_H

#include <memory>
#include <vector>

#include "Cell.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {

class Activation;
class BaseInterface;
class DeepNet;

class ActivationCell : public virtual Cell {
public:
    typedef std::function<std::shared_ptr<ActivationCell>(
        const DeepNet&,
        const std::string&,
        unsigned int,
        const std::shared_ptr<Activation>&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    ActivationCell(const DeepNet& deepNet, const std::string& name,
             unsigned int nbOutputs);
    const char* getType() const
    {
        return Type;
    };

    void getStats(Stats& stats) const;
    std::vector<unsigned int> getReceptiveField(
                                const std::vector<unsigned int>& outputField
                                        = std::vector<unsigned int>()) const;
    virtual ~ActivationCell() {};

protected:
    virtual void setOutputsDims();
};
}

#endif // N2D2_ACTIVATIONCELL_H
