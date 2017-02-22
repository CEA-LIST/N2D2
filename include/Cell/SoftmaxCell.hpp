/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Victor GACOIN

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

#ifndef N2D2_SOFTMAXCELL_H
#define N2D2_SOFTMAXCELL_H

#include <string>
#include <vector>

#include "Cell.hpp"
#include "utils/Registrar.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@SoftmaxCell_Frame@N2D2@@0U?$Registrar@VSoftmaxCell@N2D2@@@2@A")
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@SoftmaxCell_Frame_CUDA@N2D2@@0U?$Registrar@VSoftmaxCell@N2D2@@@2@A")
#endif
#endif

namespace N2D2 {
class SoftmaxCell : public virtual Cell {
public:
    typedef std::function
        <std::shared_ptr<SoftmaxCell>(const std::string&, unsigned int, bool)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    SoftmaxCell(const std::string& name,
                unsigned int nbOutputs,
                bool withLoss = false);
    const char* getType() const
    {
        return Type;
    };
    void getStats(Stats& stats) const;
    virtual ~SoftmaxCell() {};

protected:
    const bool mWithLoss;

    virtual void setOutputsSize();
};
}

#endif // N2D2_SOFTMAXCELL_H
