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

#ifndef N2D2_STIMULIPROVIDEREXPORT_H
#define N2D2_STIMULIPROVIDEREXPORT_H

#include "CellExport.hpp"
#include "DeepNet.hpp"
#include "DeepNetExport.hpp"
#include "StimuliProvider.hpp"

namespace N2D2 {
class StimuliProviderExport {
public:
    typedef std::function
        <void(StimuliProvider& sp,
              const std::string& dirName,
              Database::StimuliSet set,
              double stimuliRange,
              DeepNet* deepNet)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static void generate(StimuliProvider& sp,
                         const std::string& dirName,
                         const std::string& type,
                         Database::StimuliSet set,
                         double stimuliRange = 1.0,
                         DeepNet* deepNet = NULL);
    static void generate(StimuliProvider& sp,
                         const std::string& dirName,
                         Database::StimuliSet set,
                         double stimuliRange = 1.0,
                         DeepNet* deepNet = NULL);
    static double getStimuliRange(StimuliProvider& sp,
                                  const std::string& dirName,
                                  Database::StimuliSet set);
};
}

#endif // N2D2_STIMULIPROVIDEREXPORT_H
