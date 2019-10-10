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
#include <numeric>

#include "containers/Tensor.hpp"
#include "Database/Database.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Registrar.hpp"
#include "FloatT.hpp"

#ifdef CUDA
#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"
#endif

namespace N2D2 {

class Cell;
class StimuliProvider;

class Target : public Parameterizable, public std::enable_shared_from_this<Target> {
public:
    typedef std::function
        <std::shared_ptr<Target>(const std::string&,
                                 const std::shared_ptr<Cell>&,
                                 const std::shared_ptr<StimuliProvider>&,
                                 double,
                                 double,
                                 unsigned int,
                                 const std::string&,
                                 bool)> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

#ifdef CUDA
    typedef CudaTensor<int> TensorLabels_T;
    typedef CudaTensor<Float_T> TensorLabelsValue_T;
#else
    typedef Tensor<int> TensorLabels_T;
    typedef Tensor<Float_T> TensorLabelsValue_T;
#endif

    static std::shared_ptr<Target> create(const std::string& name,
                                          const std::shared_ptr<Cell>& cell,
                                          const std::shared_ptr
                                          <StimuliProvider>& sp,
                                          double targetValue = 1.0,
                                          double defaultValue = 0.0,
                                          unsigned int targetTopN = 1,
                                          const std::string& labelsMapping = "",
                                          bool createMissingLabels = false)
    {
        return std::make_shared<Target>(name,
                                        cell,
                                        sp,
                                        targetValue,
                                        defaultValue,
                                        targetTopN,
                                        labelsMapping,
                                        createMissingLabels);
    }
    static const char* Type;

    Target(const std::string& name,
           const std::shared_ptr<Cell>& cell,
           const std::shared_ptr<StimuliProvider>& sp,
           double targetValue = 1.0,
           double defaultValue = 0.0,
           unsigned int targetTopN = 1,
           const std::string& labelsMapping = "",
           bool createMissingLabels = false);
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
    void labelsMapping(const std::string& fileName,
                       bool createMissingLabels = false);
    void setLabelTarget(int label, int output);
    void setDefaultTarget(int output);
    int getLabelTarget(int label) const;
    int getDefaultTarget() const;
    std::vector<int> getTargetLabels(int output) const;
    const std::vector<std::string>& getTargetLabelsName() const;
    void logLabelsMapping(const std::string& fileName) const;
    void targetLabelProvider(Database::StimuliSet set);
    virtual void process(Database::StimuliSet set);
    virtual void logEstimatedLabels(const std::string& dirName) const;
    virtual void logEstimatedLabelsJSON(const std::string& dirName,
                                        const std::string& fileName = "",
                                        unsigned int xOffset = 0,
                                        unsigned int yOffset = 0,
                                        bool append = false) const;
    virtual void logLabelsLegend(const std::string& fileName) const;
    const TensorLabels_T& getEstimatedLabels() const
    {
        return mEstimatedLabels;
    };
    const TensorLabelsValue_T& getEstimatedLabelsValue() const
    {
        return mEstimatedLabelsValue;
    };
    TensorLabelsValue_T getEstimatedLabels(const std::shared_ptr<ROI>& roi,
                                            unsigned int batchPos = 0) const;
    std::pair<int, Float_T> getEstimatedLabel(const std::shared_ptr<ROI>& roi,
                                              unsigned int batchPos = 0) const;
    const std::vector<Float_T>& getLoss() const
    {
        return mLoss;
    }
    virtual void log(const std::string& /*fileName*/,
                     Database::StimuliSet /*set*/) {};
    virtual void clear(Database::StimuliSet set);
    virtual ~Target() {};

protected:
    void process_Frame(BaseTensor& values,
                       const int batchSize);
#ifdef CUDA
    void process_Frame_CUDA(Float_T* valuesDevPtr,
                            const int batchSize);
#endif

    Parameter<bool> mDataAsTarget;
    Parameter<int> mNoDisplayLabel;
    Parameter<int> mLabelsHueOffset;
    /// If true, the value in the HSV colorspace is equal to the estimated
    /// value. Otherwise, displayed value is 255 regardless of the confidence.
    Parameter<bool> mEstimatedLabelsValueDisplay;
    Parameter<int> mMaskedLabel;
    Parameter<bool> mMaskedLabelValue;
    /// Threshold for single output (binary classification). Default is 0.5.
    Parameter<double> mBinaryThreshold;
    /// If left empty, use the database image origin format
    Parameter<std::string> mImageLogFormat;
    /// When attributing a target to an output macropixel, any target other than
    /// mWeakTarget in the macropixel takes precedence over mWeakTarget, 
    /// regardless of their respective occurrence.
    /// Value can be -1 (meaning any target other than "ignore" takes 
    /// precedence).
    /// Default value is -2 (meaning that there is no weak target, as a target
    /// is >= -1).
    Parameter<int> mWeakTarget;

    const std::string mName;
    const std::shared_ptr<Cell> mCell;
    const std::shared_ptr<StimuliProvider> mStimuliProvider;
    const double mTargetValue;
    const double mDefaultValue;
    const unsigned int mTargetTopN;

    std::map<int, int> mLabelsMapping;
    int mDefaultTarget;
    Tensor<int> mTargets;
    TensorLabels_T mEstimatedLabels;
    TensorLabelsValue_T mEstimatedLabelsValue;
    std::shared_ptr<Target> mMaskLabelTarget;
    bool mPopulateTargets;
    std::vector<Float_T> mLoss;

    mutable std::vector<std::string> mLabelsName;

private:
    static Registrar<Target> mRegistrar;
};
}

#endif // N2D2_TARGET_H
