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

#include "Target/TargetMatching.hpp"
#include "StimuliProvider.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/Cell_CSpike_Top.hpp"

N2D2::Registrar<N2D2::Target>
N2D2::TargetMatching::mRegistrar("TargetMatching", N2D2::TargetMatching::create);

const char* N2D2::TargetMatching::Type = "TargetMatching";

N2D2::TargetMatching::TargetMatching(const std::string& name,
                               const std::shared_ptr<Cell>& cell,
                               const std::shared_ptr<StimuliProvider>& sp,
                               double targetValue,
                               double defaultValue,
                               unsigned int targetTopN,
                               const std::string& labelsMapping,
                               bool createMissingLabels)
    : Target(
          name, cell, sp, targetValue, defaultValue, targetTopN, labelsMapping, createMissingLabels),
      mEER_NbSteps(this, "EER_NbSteps", 5U),
      mEER_Precision(this, "EER_Precision", 0.01),
      mMinValidationEER(1.0)
{
    // ctor
}

bool N2D2::TargetMatching::newValidationEER(double validationEER)
{
    mValidationEER.push_back(validationEER);

    if (validationEER < mMinValidationEER) {
        mMinValidationEER = validationEER;
        return true;
    } else
        return false;
}

double N2D2::TargetMatching::computeDistance(
    const Tensor<Float_T>& a,
    const Tensor<Float_T>& b) const
{
    assert(a.size() == b.size());

    double distance = 0.0;

    for (unsigned int i = 0, size = a.size(); i < size; ++i) {
        const double diff = a(i) - b(i);
        distance += diff * diff;
    }

    distance = std::sqrt(distance);
    return distance;
}

void N2D2::TargetMatching::computeDistances()
{
    mDistanceMin = std::numeric_limits<Float_T>::max();
    mDistanceMax = std::numeric_limits<Float_T>::lowest();
    mDistances.resize({mSignatures.dimB(), mSignatures.dimB()});

    for (unsigned int i = 0; i < mSignatures.dimB(); ++i) {
        for (unsigned int j = 0; j < i; ++j) {
            const Float_T dist = computeDistance(mSignatures[i],
                                                 mSignatures[j]);

            if (dist < mDistanceMin)
                mDistanceMin = dist;

            if (dist > mDistanceMax)
                mDistanceMax = dist;

            mDistances(i, j) = dist;
        }
    }
}

std::pair<double, double> N2D2::TargetMatching::computeFAR_FRR(double threshold)
    const
{
    unsigned long long int FA = 0;
    unsigned long long int FR = 0;
    unsigned long long int countGenuine = 0;
    unsigned long long int countImposter = 0;

    for (unsigned int i = 0; i < mSignatures.dimB(); ++i) {
        for (unsigned int j = 0; j < i; ++j) {
            if (mIDs(j) == mIDs(i)) {
                if (mDistances(i, j) > threshold)
                    ++FR;

                ++countGenuine;
            }
            else {
                if (mDistances(i, j) <= threshold)
                    ++FA;

                ++countImposter;
            }
        }
    }

    const double FAR = FA / (double)countImposter;
    const double FRR = FR / (double)countGenuine;
    return std::pair<double, double>(FAR, FRR);
}

double N2D2::TargetMatching::getEER() {
    assert(mEER_NbSteps > 1);
    assert(mEER_Precision > 0.0);

    computeDistances();

    mROC.clear();
    Float_T start = mDistanceMin;
    Float_T end = mDistanceMax;

    while (true) {
        std::vector<double> diff;
        std::vector<double> thresholds;

        for (unsigned int k = 0; k < mEER_NbSteps; ++k) {
            const double threshold = start
                + (end - start) * k / (double)(mEER_NbSteps - 1);

            const std::map<double, std::pair<double, double> >::const_iterator
                itROC = mROC.find(threshold);
            std::pair<double, double> FAR_FRR;

            if (itROC != mROC.end())
                FAR_FRR = (*itROC).second;
            else {
                FAR_FRR = computeFAR_FRR(threshold);
                mROC[threshold] = FAR_FRR;
            }

            double FAR, FRR;
            std::tie(FAR, FRR) = FAR_FRR;

            if (std::abs(FAR - FRR) < mEER_Precision)
                return (FAR + FRR) / 2.0;

            diff.push_back(FAR - FRR);
            thresholds.push_back(threshold);
        }

        for (unsigned int pivot = 0; pivot < mEER_NbSteps - 1; ++pivot) {
            if (std::copysign(1.0, diff[pivot])
                != std::copysign(1.0, diff[pivot + 1]))
            {
                start = thresholds[pivot];
                end = thresholds[pivot + 1];
                break;
            }
        }
    }
}

// TODO: using a gross approximation for the FRR right now...
double N2D2::TargetMatching::getFRR(double targetFAR) {
    // Reverse iterator to start with a FAR = 1.0
    for (auto it = mROC.rbegin(); it != mROC.rend(); ++it) {
        double FAR, FRR;
        std::tie(FAR, FRR) = (*it).second;

        if (FAR < targetFAR) {
            return FRR;
        }
    }

    return 1.0;
}

void N2D2::TargetMatching::process(Database::StimuliSet /*set*/)
{
    std::shared_ptr<Cell_Frame_Top> targetCell 
        = std::dynamic_pointer_cast<Cell_Frame_Top>(mCell);
    std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
        = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);

    if (targetCell)
        targetCell->getOutputs().synchronizeDToH();
    else
        targetCellCSpike->getOutputsActivity().synchronizeDToH();

    const Tensor<int>& labels = mStimuliProvider->getLabelsData();
    const Tensor<Float_T>& values
        = (targetCell) ? tensor_cast<Float_T>(targetCell->getOutputs())
                        : tensor_cast<Float_T>
                            (targetCellCSpike->getOutputsActivity());

    for (int batchPos = 0; batchPos < (int)values.dimB(); ++batchPos) {
        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        Tensor<int> label = labels[batchPos];
        Tensor<Float_T> value = values[batchPos];
        value.reshape({value.size()});

        mIDs.push_back(label(0));
        mSignatures.push_back(value);
    }
}

void N2D2::TargetMatching::log(const std::string& fileName,
                               Database::StimuliSet /*set*/)
{
    if (mROC.empty())
        return;

    const std::string name = mName + "/ROC_" + fileName + ".dat";
    std::ofstream data(name.c_str());

    if (!data.good())
        throw std::runtime_error("Could not log ROC file.");

    for (auto it = mROC.begin(); it != mROC.end(); ++it) {
        double FAR, FRR;
        std::tie(FAR, FRR) = (*it).second;
        const double threshold = (*it).first;

        data << FAR << " " << FRR << " " << threshold << "\n";
    }

    data.close();

    Gnuplot gnuplot;
    gnuplot.set("grid").set("key off");
    gnuplot.setYlabel("1 - False Rejection Rate (FRR)");
    gnuplot.setXlabel("False Acceptance Rate (FAR)");

    gnuplot.saveToFile(name);
    gnuplot.plot(name, "using 1:(1.0-$2) with linespoints,"
        " '' using 1:(1.0-$2):3 with labels offset char 5,-0.5");
}

void N2D2::TargetMatching::clear(Database::StimuliSet /*set*/)
{
    mIDs.clear();
    mSignatures.clear();
}


#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_TargetMatching(py::module &m) {
    py::class_<TargetMatching, std::shared_ptr<TargetMatching>, Target>(m, "TargetMatching", py::multiple_inheritance());
}
}
#endif
