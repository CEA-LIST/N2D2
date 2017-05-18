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

#ifndef N2D2_TARGET_H
#define N2D2_TARGET_H

#include <map>
#include <string>
#include <vector>

#include "Cell/Cell.hpp"
#include "Cell/Cell_CSpike_Top.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Database/Database.hpp"
#include "N2D2.hpp"
#include "StimuliProvider.hpp"
#include "utils/Parameterizable.hpp"

#ifdef WIN32
// For static library
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@Target@N2D2@@0U?$Registrar@VTarget@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@TargetROIs@N2D2@@0U?$Registrar@VTarget@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@TargetRP@N2D2@@0U?$Registrar@VTarget@N2D2@@@2@A")
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@TargetScore@N2D2@@0U?$Registrar@VTarget@N2D2@@@2@A")
#endif

namespace N2D2 {
class Target : public Parameterizable {
public:
    typedef std::function
        <std::shared_ptr<Target>(const std::string&,
                                 const std::shared_ptr<Cell>&,
                                 const std::shared_ptr<StimuliProvider>&,
                                 double,
                                 double,
                                 unsigned int,
                                 const std::string&)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static std::shared_ptr<Target> create(const std::string& name,
                                          const std::shared_ptr<Cell>& cell,
                                          const std::shared_ptr
                                          <StimuliProvider>& sp,
                                          double targetValue = 1.0,
                                          double defaultValue = 0.0,
                                          unsigned int targetTopN = 1,
                                          const std::string& labelsMapping = "")
    {
        return std::make_shared<Target>(name,
                                        cell,
                                        sp,
                                        targetValue,
                                        defaultValue,
                                        targetTopN,
                                        labelsMapping);
    }
    static const char* Type;

    Target(const std::string& name,
           const std::shared_ptr<Cell>& cell,
           const std::shared_ptr<StimuliProvider>& sp,
           double targetValue = 1.0,
           double defaultValue = 0.0,
           unsigned int targetTopN = 1,
           const std::string& labelsMapping = "");
    /// Returns cell name
    const std::string& getName() const
    {
        return mName;
    };
    virtual const char* getType() const
    {
        return Type;
    };
    std::shared_ptr<Cell> getCell() const
    {
        return mCell;
    }
    std::shared_ptr<StimuliProvider> getStimuliProvider() const
    {
        return mStimuliProvider;
    }
    virtual unsigned int getNbTargets() const;
    unsigned int getTargetTopN() const
    {
        return mTargetTopN;
    }
    double getTargetValue() const
    {
        return mTargetValue;
    }
    double getDefaultValue() const
    {
        return mDefaultValue;
    }
    void setMaskLabelTarget(const std::shared_ptr<Target>& target)
    {
        mMaskLabelTarget = target;
    };
    void labelsMapping(const std::string& fileName);
    void setLabelTarget(int label, int output);
    void setDefaultTarget(int output);
    int getLabelTarget(int label) const;
    int getDefaultTarget() const;
    std::vector<int> getTargetLabels(int output) const;
    std::vector<std::string> getTargetLabelsName() const;
    void logLabelsMapping(const std::string& fileName) const;
    virtual void process(Database::StimuliSet set);
    virtual void logEstimatedLabels(const std::string& dirName) const;
    virtual void logLabelsLegend(const std::string& fileName) const;
    const Tensor4d<int>& getEstimatedLabels() const
    {
        return mEstimatedLabels;
    };
    const Tensor4d<Float_T>& getEstimatedLabelsValue() const
    {
        return mEstimatedLabelsValue;
    };
    std::pair<int, Float_T> getEstimatedLabel(const std::shared_ptr<ROI>& roi,
                                              unsigned int batchPos = 0) const;
    virtual void log(const std::string& /*fileName*/,
                     Database::StimuliSet /*set*/) {};
    virtual void clear(Database::StimuliSet /*set*/) {};
    virtual ~Target() {};

protected:
    Parameter<bool> mDataAsTarget;
    Parameter<int> mNoDisplayLabel;
    Parameter<int> mLabelsHueOffset;
    Parameter<int> mMaskedLabel;

    const std::string mName;
    const std::shared_ptr<Cell> mCell;
    const std::shared_ptr<StimuliProvider> mStimuliProvider;
    const double mTargetValue;
    const double mDefaultValue;
    const unsigned int mTargetTopN;

    std::map<int, int> mLabelsMapping;
    int mDefaultTarget;
    Tensor4d<int> mTargets;
    Tensor4d<int> mEstimatedLabels;
    Tensor4d<Float_T> mEstimatedLabelsValue;
    Tensor4d<Float_T> mEstimatedValues;
    std::shared_ptr<Target> mMaskLabelTarget;
    bool mPopulateTargets;

private:
    static Registrar<Target> mRegistrar;
};
}

#endif // N2D2_TARGET_H
