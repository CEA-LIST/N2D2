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

#ifndef N2D2_DROPOUTCELL_H
#define N2D2_DROPOUTCELL_H

#include <string>
#include <vector>

#include "Cell.hpp"
#include "utils/Registrar.hpp"

#ifdef WIN32
// For static library
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@DropoutCell_Frame_CUDA@N2D2@@0U?$Registrar@VDropoutCell@N2D2@@@2@A")
#endif
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@DropoutCell_Frame@N2D2@@0U?$Registrar@VDropoutCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class DropoutCell : public virtual Cell {
public:
    typedef std::function
        <std::shared_ptr<DropoutCell>(const std::string&, unsigned int)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    DropoutCell(const std::string& name, unsigned int nbOutputs);
    const char* getType() const
    {
        return Type;
    };
    void getStats(Stats& stats) const;
    virtual ~DropoutCell() {};

protected:
    virtual void setOutputsSize();

    /// The probability with which the value from input would be propagated
    /// through the dropout layer
    Parameter<double> mDropout;
};
}

#endif // N2D2_DROPOUTCELL_H
