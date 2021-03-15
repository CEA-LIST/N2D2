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
      mDistance(this, "Distance", L2),
      mROC_NbSteps(this, "ROC_NbSteps", 5U),
      mROC_Precision(this, "ROC_Precision", 1.0e-3),
      mTargetFAR(this, "TargetFAR", 1.0e-3),
      nbBatches(0U),
      mMinValidationEER(1.0),
      mMinValidationFRR(1.0)
{
    // ctor
}

bool N2D2::TargetMatching::newValidationEER(double validationEER)
{
    mValidationEER.push_back(std::make_pair(nbBatches, validationEER));

    if (validationEER < mMinValidationEER) {
        mMinValidationEER = validationEER;
        return true;
    } else
        return false;
}

bool N2D2::TargetMatching::newValidationEER(double validationEER,
                                            double validationFRR)
{
    mValidationEER.push_back(std::make_pair(nbBatches, validationEER));
    mValidationFRR.push_back(std::make_pair(nbBatches, validationFRR));

    if (validationEER < mMinValidationEER) {
        mMinValidationEER = validationEER;
        mMinValidationFRR = validationFRR;
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

    if (mDistance == L1) {
        for (std::size_t i = 0, size = a.size(); i < size; ++i)
            distance += std::abs(a(i) - b(i));
    }
    else if (mDistance == L2) {
        for (std::size_t i = 0, size = a.size(); i < size; ++i) {
            const double diff = a(i) - b(i);
            distance += diff * diff;
        }

        distance = std::sqrt(distance);
    }
    else if (mDistance == Linf) {
        for (std::size_t i = 0, size = a.size(); i < size; ++i)
            distance = std::max(distance, std::abs((double)a(i) - b(i)));
    }

    return distance;
}

void N2D2::TargetMatching::computeDistances()
{
    mDistanceMin = std::numeric_limits<Float_T>::max();
    mDistanceMax = std::numeric_limits<Float_T>::lowest();
    mDistances.resize({(mSignatures.dimB() * (mSignatures.dimB() - 1)) / 2});

    std::size_t k = 0;

    for (std::size_t i = 1; i < mSignatures.dimB(); ++i) {
        const Tensor<Float_T>& signature = mSignatures[i];

        for (std::size_t j = 0; j < i; ++j) {
            const Float_T dist = computeDistance(signature,
                                                 mSignatures[j]);

            if (dist < mDistanceMin)
                mDistanceMin = dist;

            if (dist > mDistanceMax)
                mDistanceMax = dist;

            mDistances(k) = dist;
            ++k;
        }
    }

    assert(k == mDistances.size());
}

std::pair<double, double> N2D2::TargetMatching::computeFAR_FRR(double threshold)
    const
{
    assert(mDistances.size()
        == (mSignatures.dimB() * (mSignatures.dimB() - 1)) / 2);

    unsigned long long int falseAcceptance = 0;
    unsigned long long int falseRejection = 0;
    unsigned long long int countGenuine = 0;
    unsigned long long int countImposter = 0;

    std::size_t k = 0;

    for (std::size_t i = 1; i < mSignatures.dimB(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (mIDs(j) == mIDs(i)) {
                if (mDistances(k) > threshold)
                    ++falseRejection;

                ++countGenuine;
            }
            else {
                if (mDistances(k) <= threshold)
                    ++falseAcceptance;

                ++countImposter;
            }

            ++k;
        }
    }

    assert(k == mDistances.size());

    const double FA_Rate = falseAcceptance / (double)countImposter;
    const double FR_Rate = falseRejection / (double)countGenuine;
    return std::pair<double, double>(FA_Rate, FR_Rate);
}

/**
 * Fast method to compute EER.
 * Ref: Romain Giot, Mohamad El-Abed, Christophe Rosenberger. 
 * Fast computation of the performanceevaluation of biometric systems: 
 * application to multibiometric. Future Generation Computer Systems, 
 * Elsevier, 2013, pp.10.1016/j.future.2012.02.003. 
*/
double N2D2::TargetMatching::getEER() {
    if (mROC.empty())
        computeDistances();

    double FA_Rate, FR_Rate;
    std::tie(FA_Rate, FR_Rate) = computeROC(1.0, -1.0, 0.0);

    return (FA_Rate + FR_Rate) / 2.0;
}

double N2D2::TargetMatching::getFRR() {
    return getFRR(mTargetFAR);
}

double N2D2::TargetMatching::getFRR(double targetFAR) {
    if (mROC.empty())
        computeDistances();

    double FA_Rate, FR_Rate;
    std::tie(FA_Rate, FR_Rate) = computeROC(1.0, 0.0, -targetFAR);

    return FR_Rate;
}

std::pair<double, double> N2D2::TargetMatching::computeROC(
    double a,
    double b,
    double c)
{
    assert(mROC_NbSteps > 1);
    assert(mROC_Precision > 0.0);

    Float_T start = mDistanceMin;
    Float_T end = mDistanceMax;

    while (true) {
        std::vector<double> diff;
        std::vector<double> thresholds;

        for (unsigned int k = 0; k < mROC_NbSteps; ++k) {
            const double threshold = start
                + (end - start) * k / (double)(mROC_NbSteps - 1);

            const std::map<double, std::pair<double, double> >::const_iterator
                itROC = mROC.find(threshold);
            std::pair<double, double> FAR_FRR;

            if (itROC != mROC.end())
                FAR_FRR = (*itROC).second;
            else {
                FAR_FRR = computeFAR_FRR(threshold);
                mROC[threshold] = FAR_FRR;
            }

            double FA_Rate, FR_Rate;
            std::tie(FA_Rate, FR_Rate) = FAR_FRR;

            if (std::abs(a * FA_Rate + b * FR_Rate + c) < mROC_Precision)
                return FAR_FRR;

            diff.push_back(a * FA_Rate + b * FR_Rate + c);
            thresholds.push_back(threshold);
        }

        for (unsigned int pivot = 0; pivot < mROC_NbSteps - 1; ++pivot) {
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

void N2D2::TargetMatching::process(Database::StimuliSet set)
{
    if (set == Database::Learn) {
        ++nbBatches;
        return;
    }

    std::shared_ptr<Cell_Frame_Top> targetCell 
        = std::dynamic_pointer_cast<Cell_Frame_Top>(mCell);
    std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
        = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);

    const Tensor<int>& labels = mStimuliProvider->getLabelsData();
    BaseTensor& valuesBaseTensor = (targetCell)
        ? targetCell->getOutputs() : targetCellCSpike->getOutputsActivity();
    Tensor<Float_T> values;
    valuesBaseTensor.synchronizeToH(values);

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
                               Database::StimuliSet set)
{
    if (set == Database::Learn)
        return;

    if (!mROC.empty() && (mValidationEER.empty()
        // Ensures that the ROC is saved only if this is the best result
        || mValidationEER.back().second == mMinValidationEER))
    {
        const std::string dataFileName = mName + "/ROC_" + fileName + ".dat";

        std::ofstream data(dataFileName.c_str());

        if (!data.good()) {
            throw std::runtime_error("Could not create ROC log file: "
                                     + dataFileName);
        }

        for (auto it = mROC.begin(); it != mROC.end(); ++it) {
            double FA_Rate, FR_Rate;
            std::tie(FA_Rate, FR_Rate) = (*it).second;
            const double threshold = (*it).first;

            data << FA_Rate << " " << FR_Rate << " " << threshold << "\n";
        }

        data.close();

        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.set("logscale x");
        gnuplot.setYlabel("1 - False Rejection Rate (FRR)");
        gnuplot.setXlabel("False Acceptance Rate (FAR)");

        gnuplot.saveToFile(dataFileName);
        gnuplot.plot(dataFileName, "using 1:(1.0-$2) with linespoints,"
            " '' using 1:(1.0-$2):3 with labels offset char 5,-0.5");
    }

    if (!mValidationEER.empty()) {
        const std::string dataFileName = mName + "/EER_" + fileName + ".dat";

        std::ofstream dataFile(dataFileName);

        if (!dataFile.good()) {
            throw std::runtime_error("Could not create EER log file: "
                                     + dataFileName);
        }

        for (std::vector<std::pair<unsigned int, double> >::const_iterator it
             = mValidationEER.begin(), itEnd = mValidationEER.end();
             it != itEnd; ++it)
        {
            dataFile << (*it).first << " " << (*it).second << "\n";
        }

        dataFile.close();

        // Plot validation
        std::ostringstream label;
        label << "\"Best validation EER: " << 100.0 * mMinValidationEER
            << "%";

        if (!mValidationFRR.empty()) {
            label << "\\nCorresponding FRR @ FAR=" << mTargetFAR << ": "
                << 100.0 * mMinValidationFRR << "%";
        }

        label << "\" at graph 0.5, graph 0.15 front";

        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setYlabel("Equal Error Rate (EER)");
        gnuplot.setXlabel("# steps");

        gnuplot.saveToFile(dataFileName);
        gnuplot.set("label", label.str());
        gnuplot
            << ("plot \"" + dataFileName
                + "\" using 1:2 with linespoints lt 7");
    }

    if (!mValidationFRR.empty()) {
        const std::string dataFileName = mName + "/FRR_" + fileName + ".dat";

        std::ofstream dataFile(dataFileName);

        if (!dataFile.good()) {
            throw std::runtime_error("Could not create FRR log file: "
                                     + dataFileName);
        }

        for (std::vector<std::pair<unsigned int, double> >::const_iterator it
             = mValidationFRR.begin(), itEnd = mValidationFRR.end();
             it != itEnd; ++it)
        {
            dataFile << (*it).first << " " << (*it).second << "\n";
        }

        dataFile.close();

        // Plot validation
        std::ostringstream label;
        label << "\"Best validation EER: " << 100.0 * mMinValidationEER
            << "%";
        label << "\\nCorresponding FRR @ FAR=" << mTargetFAR << ": "
            << 100.0 * mMinValidationFRR << "%";
        label << "\" at graph 0.5, graph 0.15 front";

        std::ostringstream yLabel;
        yLabel << "False Rejection Rate (FRR) @ FAR=" << mTargetFAR;

        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setYlabel(yLabel.str());
        gnuplot.setXlabel("# steps");

        gnuplot.saveToFile(dataFileName);
        gnuplot.set("label", label.str());
        gnuplot
            << ("plot \"" + dataFileName
                + "\" using 1:2 with linespoints lt 7");
    }
}

void N2D2::TargetMatching::clear(Database::StimuliSet /*set*/)
{
    mIDs.clear();
    mSignatures.clear();
    mROC.clear();
}
