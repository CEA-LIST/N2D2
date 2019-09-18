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

#include "StimuliData.hpp"
#include "Export/CellExport.hpp"
#include "utils/Registrar.hpp"

#include <iosfwd>

namespace N2D2 {

class DeepNet;
class StimuliProvider;

class StimuliProviderExport {
public:
    typedef std::function
        <void(StimuliProvider& sp,
              const std::string& dirName,
              Database::StimuliSet set,
              int nbStimuliMax,
              bool normalize,
              DeepNet* deepNet)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    enum ExportFormat {
        HWC,
        CHW
    };

    static void generate(StimuliProvider& sp,
                         const std::string& dirName,
                         const std::string& type,
                         Database::StimuliSet set,
                         int nbStimuliMax = -1,
                         bool normalize = false,
                         DeepNet* deepNet = NULL,
                         ExportFormat exportFormat = CHW);

    static void generate(StimuliProvider& sp,
                         const std::string& dirName,
                         Database::StimuliSet set,
                         int nbStimuliMax = -1,
                         bool normalize = false,
                         DeepNet* deepNet = NULL,
                         ExportFormat exportFormat = CHW);

    static void generate(StimuliProvider& sp,
                         const std::string& dirName,
                         Database::StimuliSet set,
                         double scaling,
                         bool unsignedData,
                         int nbStimuliMax = -1,
                         DeepNet* deepNet = NULL,
                         ExportFormat exportFormat = CHW);

    static double getStimuliRange(StimuliProvider& sp,
                                  const std::string& dirName,
                                  Database::StimuliSet set);

    /**
     * Return a pair of scaling and a bool indicating if the stimuli are unsigned.
     */
    static std::pair<double, bool> getScaling(StimuliProvider& sp,
                                              const std::string& dirName,
                                              Database::StimuliSet set,
                                              bool normalize = false);
protected:
    static void writeStimulusValue(Float_T value, bool unsignedData, double scaling, 
                                   CellExport::IntApprox approxMethod, 
                                   std::ofstream& envStimuli,
                                   bool asBinary = true);

    template<typename T>
    static void writeToStream(const T& value, std::ofstream& envStimuli, bool asBinary);
    
    template<typename T>
    static void generateStimulus(StimuliProvider& sp, 
                                 Database::StimuliSet set, 
                                 std::size_t iStimulus,
                                 double scaling,
                                 std::ostream& ostream);
    
    template<typename T>
    static T getScaledData(Float_T data, double scaling, 
                           CellExport::IntApprox approxMethod);

    static StimuliData getStimuliData(StimuliProvider& sp,
                                      const std::string& dirName,
                                      Database::StimuliSet set);
};

template<typename T>
inline void StimuliProviderExport::writeToStream(const T& value, std::ofstream& envStimuli, 
                                                 bool asBinary) 
{
    if(asBinary) {
        envStimuli.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    else {
        envStimuli.operator<<(value);
        envStimuli << ", ";
    }
}

}

#endif // N2D2_STIMULIPROVIDEREXPORT_H
