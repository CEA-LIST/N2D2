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

#include "Target/TargetScore.hpp"
#include "StimuliProvider.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/Cell_CSpike_Top.hpp"

N2D2::Registrar<N2D2::Target>
N2D2::TargetScore::mRegistrar("TargetScore", N2D2::TargetScore::create);

const char* N2D2::TargetScore::Type = "TargetScore";

N2D2::TargetScore::TargetScore(const std::string& name,
                               const std::shared_ptr<Cell>& cell,
                               const std::shared_ptr<StimuliProvider>& sp,
                               double targetValue,
                               double defaultValue,
                               unsigned int targetTopN,
                               const std::string& labelsMapping,
                               bool createMissingLabels)
    : Target(
          name, cell, sp, targetValue, defaultValue, targetTopN, labelsMapping, createMissingLabels),
      mConfusionRangeMin(this, "ConfusionRangeMin", 0.0),
      mConfusionRangeMax(this, "ConfusionRangeMax", 1.0),
      mConfusionQuantSteps(this, "ConfusionQuantSteps", 10U),
      mMaxValidationScore(0.0),
      mMaxValidationTopNScore(0.0)
{
    // ctor
}

bool N2D2::TargetScore::newValidationScore(double validationScore)
{
    const std::deque<double>& success = getScore(Database::Learn).success;

    mValidationScore.push_back(std::make_pair(success.size(), validationScore));

    if (validationScore > mMaxValidationScore) {
        mMaxValidationScore = validationScore;
        return true;
    } else
        return false;
}

bool N2D2::TargetScore::newValidationTopNScore(double validationTopNScore)
{
    if (!mDataAsTarget && mTargetTopN > 1) {
        const std::deque<double>& success
            = getTopNScore(Database::Learn).success;
        mValidationTopNScore.push_back(
            std::make_pair(success.size(), validationTopNScore));

        if (validationTopNScore > mMaxValidationTopNScore) {
            mMaxValidationTopNScore = validationTopNScore;
            return true;
        } else
            return false;
    } else
        return false;
}

double N2D2::TargetScore::getBatchAverageSuccess() const
{
    return (mBatchSuccess.size() > 0)
               ? std::accumulate(mBatchSuccess.begin(),
                                 mBatchSuccess.end(),
                                 0.0) / (double)mBatchSuccess.size()
               : 1.0;
}

double N2D2::TargetScore::getBatchAverageTopNSuccess() const
{
    if (!mDataAsTarget && mTargetTopN > 1)
        return (mBatchTopNSuccess.size() > 0)
                   ? std::accumulate(mBatchTopNSuccess.begin(),
                                     mBatchTopNSuccess.end(),
                                     0.0) / (double)mBatchTopNSuccess.size()
                   : 1.0;
    else
        return 0.0;
}

double N2D2::TargetScore::getAverageSuccess(Database::StimuliSet set,
                                            unsigned int avgWindow) const
{
    assert(mScoreSet.find(set) != mScoreSet.end());

    const std::deque<double>& success = (*mScoreSet.find(set)).second.success;

    if (success.empty())
        return 0.0;

    return (avgWindow > 0 && success.size() > avgWindow)
               ? std::accumulate(success.end() - avgWindow, success.end(), 0.0)
                 / (double)avgWindow
               : std::accumulate(success.begin(), success.end(), 0.0)
                 / (double)success.size();
}

double N2D2::TargetScore::getAverageTopNSuccess(Database::StimuliSet set,
                                                unsigned int avgWindow) const
{
    if (mDataAsTarget || !(mTargetTopN > 1))
        return 0.0;

    assert(mScoreTopNSet.find(set) != mScoreTopNSet.end());

    const std::deque<double>& success
        = (*mScoreTopNSet.find(set)).second.success;

    if (success.empty())
        return 0.0;

    return (avgWindow > 0 && success.size() > avgWindow)
               ? std::accumulate(success.end() - avgWindow, success.end(), 0.0)
                 / (double)avgWindow
               : std::accumulate(success.begin(), success.end(), 0.0)
                 / (double)success.size();
}

double N2D2::TargetScore::getAverageScore(Database::StimuliSet set,
                                          ConfusionTableMetric metric) const
{
    assert(mScoreSet.find(set) != mScoreSet.end());

    const ConfusionMatrix<unsigned long long int>& confusionMatrix
        = (*mScoreSet.find(set)).second.confusionMatrix;

    if (confusionMatrix.empty())
        return 0.0;

    const std::vector<ConfusionTable<unsigned long long int> > conf
        = confusionMatrix.getConfusionTables();

    double avgScore = 0.0;

    for (unsigned int target = 0; target < conf.size(); ++target)
        avgScore += conf[target].getMetric(metric);

    if (conf.size() > 0)
        avgScore /= conf.size();

    return avgScore;
}

double N2D2::TargetScore::getAverageTopNScore(Database::StimuliSet set,
    ConfusionTableMetric metric) const
{
    if (mDataAsTarget || !(mTargetTopN > 1))
        return 0.0;

    assert(mScoreTopNSet.find(set) != mScoreTopNSet.end());

    const ConfusionMatrix<unsigned long long int>& confusionMatrix
        = (*mScoreTopNSet.find(set)).second.confusionMatrix;

    if (confusionMatrix.empty())
        return 0.0;

    const std::vector<ConfusionTable<unsigned long long int> > conf
        = confusionMatrix.getConfusionTables();

    double avgScore = 0.0;

    for (unsigned int target = 0; target < conf.size(); ++target)
        avgScore += conf[target].getMetric(metric);

    if (conf.size() > 0)
        avgScore /= conf.size();

    return avgScore;
}

void N2D2::TargetScore::logSuccess(const std::string& fileName,
                                   Database::StimuliSet set,
                                   unsigned int avgWindow) const
{
    assert(mScoreSet.find(set) != mScoreSet.end());

    const std::string dataFileName = mName + "/Success_" + fileName + ".dat";

    if (set == Database::Validation) {
        assert(mScoreSet.find(Database::Learn) != mScoreSet.end());

        // Save validation scores file
        std::ofstream dataFile(dataFileName);

        if (!dataFile.good())
            throw std::runtime_error("Could not create data rate log file: "
                                     + dataFileName);

        for (std::vector<std::pair<unsigned int, double> >::const_iterator it
             = mValidationScore.begin(),
             itEnd = mValidationScore.end();
             it != itEnd;
             ++it) {
            dataFile << (*it).first << " " << (*it).second << "\n";
        }

        dataFile.close();

        // Plot validation
        const double lastValidation = (!mValidationScore.empty())
            ? mValidationScore.back().second : 0.0;
        const double lastLearn = (!getScore(Database::Learn).success.empty())
            ? getScore(Database::Learn).success.back() : 0.0;

        const double minFinalRate = std::min(lastValidation, lastLearn);
        const double maxFinalRate = std::max(lastValidation, lastLearn);

        std::ostringstream label;
        label << "\"Best validation: " << 100.0 * mMaxValidationScore
              << "%\" at graph 0.5, graph 0.15 front";

        Gnuplot multiplot;
        multiplot.saveToFile(dataFileName);
        multiplot.setMultiplot();

        Monitor::logDataRate((*mScoreSet.find(Database::Learn)).second.success,
                             mName + "/" + fileName + "_LearningSuccess.dat",
                             avgWindow,
                             true);

        multiplot << "clear";
        multiplot.setYrange(
            Utils::clamp(minFinalRate - (1.0 - minFinalRate), 0.0, 0.99),
            Utils::clamp(2.0 * maxFinalRate, 0.01, 1.0));
        multiplot.set("label", label.str());
        multiplot
            << ("replot \"" + dataFileName
                + "\" using 1:2 with linespoints lt 7 title \"Validation\"");
    }
    else {
        Monitor::logDataRate((*mScoreSet.find(set)).second.success,
                             dataFileName,
                             avgWindow,
                             true);
    }
}
// Top-n log success
void N2D2::TargetScore::logTopNSuccess(const std::string& fileName,
                                       Database::StimuliSet set,
                                       unsigned int avgWindow) const
{
    if (!mDataAsTarget && mTargetTopN > 1) {
        assert(mScoreTopNSet.find(set) != mScoreTopNSet.end());

        const std::string dataFileName = mName + "/SuccessTopN_" + fileName
                                         + ".dat";

        if (set == Database::Validation) {
            assert(mScoreTopNSet.find(Database::Learn) != mScoreTopNSet.end());

            // Save validation scores file
            std::ofstream dataFile(dataFileName);

            if (!dataFile.good())
                throw std::runtime_error("Could not create data rate log file: "
                                         + dataFileName);

            for (std::vector<std::pair<unsigned int, double> >::const_iterator
                     it = mValidationTopNScore.begin(),
                     itEnd = mValidationTopNScore.end();
                 it != itEnd;
                 ++it) {
                dataFile << (*it).first << " " << (*it).second << "\n";
            }

            dataFile.close();

            // Plot validation
            const double lastValidation = (!mValidationTopNScore.empty())
                ? mValidationTopNScore.back().second : 0.0;
            const double lastLearn
                = (!getTopNScore(Database::Learn).success.empty())
                    ? getTopNScore(Database::Learn).success.back() : 0.0;

            const double minFinalRate = std::min(lastValidation, lastLearn);
            const double maxFinalRate = std::max(lastValidation, lastLearn);

            std::ostringstream label;
            label << "\"Best top-" << mTargetTopN
                  << " accuracy validation: " << 100.0 * mMaxValidationTopNScore
                  << "%\" at graph 0.5, graph 0.15 front";

            Gnuplot multiplot;
            multiplot.saveToFile(dataFileName);
            multiplot.setMultiplot();

            Monitor::logDataRate(
                (*mScoreTopNSet.find(Database::Learn)).second.success,
                mName + "/" + fileName + "_LearningTopNSuccess.dat",
                avgWindow,
                true);

            multiplot << "clear";
            multiplot.setYrange(
                Utils::clamp(minFinalRate - (1.0 - minFinalRate), 0.0, 0.99),
                Utils::clamp(2.0 * maxFinalRate, 0.01, 1.0));
            multiplot.set("label", label.str());
            multiplot << ("replot \"" + dataFileName + "\" using 1:2 with "
                                                       "linespoints lt 7 title "
                                                       "\"Validation\"");
        } else
            Monitor::logDataRate((*mScoreTopNSet.find(set)).second.success,
                                 dataFileName,
                                 avgWindow,
                                 true);
    }
}

void N2D2::TargetScore::logConfusionMatrix(const std::string& fileName,
                                           Database::StimuliSet set) const
{
    assert(mScoreSet.find(set) != mScoreSet.end());

    const std::vector<std::string>& labelsName = (!mDataAsTarget)
        ? getTargetLabelsName()
        : std::vector<std::string>();

    (*mScoreSet.find(set)).second.confusionMatrix.log(
        mName + "/ConfusionMatrix_" + fileName + ".dat", labelsName);
}

void N2D2::TargetScore::logMisclassified(const std::string& fileName,
                                         Database::StimuliSet set) const
{
    assert(mScoreSet.find(set) != mScoreSet.end());

    if (mDataAsTarget)
        return;

    const unsigned int nbTargets = getNbTargets();
    const unsigned int nbTargetsConfusion
        = (mMaskLabelTarget && mMaskedLabel >= 0) ? (nbTargets + 1) : nbTargets;

    std::ofstream data((mName + "/Misclassified_" + fileName + ".dat").c_str());

    if (!data.good())
        throw std::runtime_error("Could not log misclassified stimuli file.");

    //data << "# name target target-labels estimated estimated-labels count\n";
    data << "# name target estimated count\n";

    const std::map<unsigned int,
                 std::map<unsigned int,
                          std::vector<unsigned int> > >& misclassified
        = (*mScoreSet.find(set)).second.misclassified;

    for (std::map<unsigned int, std::map<unsigned int,
        std::vector<unsigned int> > >::const_iterator
        it = misclassified.begin(), itEnd = misclassified.end(); it != itEnd;
        ++it)
    {
        // For each stimulus
        bool firstOccurrence = true;
        const std::string name
            = mStimuliProvider->getDatabase().getStimulusName((*it).first);
        //const int label
        //    = mStimuliProvider->getDatabase().getStimulusLabel((*it).first);

        for (std::map<unsigned int, std::vector<unsigned int> >::const_iterator
            itMisclass = (*it).second.begin(),
            itMisclassEnd = (*it).second.end(); itMisclass != itMisclassEnd;
            ++itMisclass)
        {
            // For each stimulus' target
            const unsigned int target = (*itMisclass).first;
/*
            const std::vector<int> targetLabels = getTargetLabels(target);

            std::ostringstream targetLabelsName;

            if (targetLabels[0]
                < (int)mStimuliProvider->getDatabase().getNbLabels())
            {
                targetLabelsName << mStimuliProvider->getDatabase()
                    .getLabelName(targetLabels[0]);
            }
            else
                targetLabelsName << "";

            if (targetLabels.size() > 1)
                targetLabelsName << "...";
*/
            for (unsigned int estimated = 0; estimated < nbTargetsConfusion;
                ++estimated)
            {
                if ((*itMisclass).second[estimated] > 0) {
/*
                    const std::vector<int> estLabels
                        = getTargetLabels(estimated);

                    std::ostringstream estLabelsName;

                    if (estLabels[0]
                        < (int)mStimuliProvider->getDatabase().getNbLabels())
                    {
                        estLabelsName << mStimuliProvider->getDatabase()
                            .getLabelName(estLabels[0]);
                    }
                    else
                        estLabelsName << estLabels[0];

                    if (estLabels.size() > 1)
                        estLabelsName << "...";
*/
                    data << (*it).first;

                    if (firstOccurrence)
                        data << " " << Utils::quoted(name);
                    else
                        data << " ";

                    data << " " << target
                        //<< " " << targetLabelsName.str()
                        << " " << estimated
                        //<< " " << estLabelsName.str()
                        << " " << (*itMisclass).second[estimated]
                        << "\n";

                    firstOccurrence = false;
                }
            }
        }
    }
}

void N2D2::TargetScore::clearSuccess(Database::StimuliSet set)
{
    mScoreSet[set].success.clear();
    mScoreTopNSet[set].success.clear();
}

void N2D2::TargetScore::clearConfusionMatrix(Database::StimuliSet set)
{
    mScoreSet[set].confusionMatrix.clear();
}

void N2D2::TargetScore::clearMisclassified(Database::StimuliSet set)
{
    mScoreSet[set].misclassified.clear();
}

void N2D2::TargetScore::clearScore(Database::StimuliSet set)
{
    clearSuccess(set);
    clearConfusionMatrix(set);
    clearMisclassified(set);
}
void N2D2::TargetScore::computeScore(Database::StimuliSet set)
{
    if (mDataAsTarget) {
        ConfusionMatrix<unsigned long long int>& confusionMatrix
            = mScoreSet[set].confusionMatrix;

        if (confusionMatrix.empty()) {
            confusionMatrix.resize(mConfusionQuantSteps,
                                   mConfusionQuantSteps, 0);
        }

        std::shared_ptr<Cell_Frame_Top> targetCell 
            = std::dynamic_pointer_cast<Cell_Frame_Top>(mCell);
        std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
            = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);

        BaseTensor& valuesBaseTensor = (targetCell)
            ? targetCell->getOutputs() : targetCellCSpike->getOutputsActivity();
        Tensor<Float_T> values;
        valuesBaseTensor.synchronizeToH(values);

        mBatchSuccess.assign(values.dimB(), -1.0);

#pragma omp parallel for if (values.dimB() > 4 && values[0].size() > 1)
        for (int batchPos = 0; batchPos < (int)values.dimB(); ++batchPos) {
            const int id = mStimuliProvider->getBatch()[batchPos];

            if (id < 0) {
                // Invalid stimulus in batch (can occur for the last batch of the
                // set)
                continue;
            }

            const Tensor<Float_T> target = mStimuliProvider->getTargetData()[batchPos];
            const Tensor<Float_T> estimated = values[batchPos];

            ConfusionMatrix<unsigned long long int> confusion(
                mConfusionQuantSteps, mConfusionQuantSteps, 0);

            double mse = 0.0;

            for (size_t index = 0; index < target.size(); ++index) {
                const double err = target(index) - estimated(index);
                mse += err * err;

                const unsigned int t = Utils::clamp<unsigned int>(
                    Utils::round((mConfusionQuantSteps - 1)
                        * (target(index) - mConfusionRangeMin)
                            / (double)(mConfusionRangeMax - mConfusionRangeMin)),
                    0U, mConfusionQuantSteps - 1);
                const unsigned int e = Utils::clamp<unsigned int>(
                    Utils::round((mConfusionQuantSteps - 1)
                        * (estimated(index) - mConfusionRangeMin)
                            / (double)(mConfusionRangeMax - mConfusionRangeMin)),
                    0U, mConfusionQuantSteps - 1);

                confusion(t, e) += 1ULL;
            }

            if (target.size() > 0)
                mse /= target.size();

            mBatchSuccess[batchPos] = mse;

#pragma omp critical(TargetScore__process)
            std::transform(confusionMatrix.begin(), confusionMatrix.end(),
                        confusion.begin(),
                        confusionMatrix.begin(),
                        std::plus<unsigned long long int>());
        }
    }
    else {
        const unsigned int nbTargets = getNbTargets();
        const unsigned int nbTargetsConfusion
            = (mMaskLabelTarget && mMaskedLabel >= 0)
                ? (nbTargets + 1) : nbTargets;

        ConfusionMatrix<unsigned long long int>& confusionMatrix
            = mScoreSet[set].confusionMatrix;

        if (confusionMatrix.empty())
            confusionMatrix.resize(nbTargetsConfusion, nbTargetsConfusion, 0);

        std::map<unsigned int,
                 std::map<unsigned int, std::vector<unsigned int> > >&
            misclassified = mScoreSet[set].misclassified;

        int dev = 0;
#ifdef CUDA
        CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

        const Tensor<int>& targets = mTargetData[dev].targets;
        const TensorLabels_T& estimatedLabels = mTargetData[dev].estimatedLabels;

        estimatedLabels.synchronizeDBasedToH();

        mBatchSuccess.assign(targets.dimB(), -1.0);

        if (mTargetTopN > 1)
            mBatchTopNSuccess.assign(targets.dimB(), -1.0);

#pragma omp parallel for if (targets.dimB() > 4 && targets[0].size() > 1)
        for (int batchPos = 0; batchPos < (int)targets.dimB(); ++batchPos) {
            const int id = mStimuliProvider->getBatch()[batchPos];

            if (id < 0) {
                // Invalid stimulus in batch (can occur for the last batch of the
                // set)
                continue;
            }

            const Tensor<int> target = targets[batchPos][0];
            const Tensor<int> estLabels = estimatedLabels[batchPos];
            const TensorLabels_T mask = (mMaskLabelTarget && mMaskedLabel >= 0)
                ? mMaskLabelTarget->getEstimatedLabels()[batchPos][0]
                : TensorLabels_T();

            if (!mask.empty() && mask.dims() != target.dims()) {
                std::ostringstream errorStr;
                errorStr << "Mask dims (" << mask.dims() << ") from MaskLabelTarget"
                    " does not match target dims (" << target.dims() << ") for"
                    " target \"" << mName << "\"";

#pragma omp critical(TargetScore__process)
                throw std::runtime_error(errorStr.str());
            }

            std::map<unsigned int, std::vector<unsigned int> > misclass;

            if (target.size() == 1) {
                if (target(0) >= 0) {
#pragma omp atomic
                    confusionMatrix(target(0), estLabels(0)) += 1ULL;

                    mBatchSuccess[batchPos] = (estLabels(0) == target(0));

                    if (!mBatchSuccess[batchPos]) {
                        // Misclassified
                        std::map<unsigned int, std::vector<unsigned int> >
                            ::iterator itMisclass;
                        std::tie(itMisclass, std::ignore)
                            = misclass.insert(std::make_pair(target(0),
                                std::vector<unsigned int>(nbTargets, 0U)));
                        (*itMisclass).second[estLabels(0)] = 1U;
                    }

                    // Top-N case :
                    if (mTargetTopN > 1) {
                        unsigned int topNscore = 0;

                        for (unsigned int n = 0; n < mTargetTopN; ++n) {
                            if (estLabels(n) == target(0))
                                ++topNscore;
                        }

                        mBatchTopNSuccess[batchPos] = (topNscore > 0);
                    }
                }
            } else {
                ConfusionMatrix<unsigned long long int> confusion(nbTargetsConfusion,
                                                                nbTargetsConfusion,
                                                                0);

                std::vector<unsigned int> nbHits(nbTargets, 0);
                std::vector<unsigned int> nbHitsTopN(nbTargets, 0);
                std::vector<unsigned int> nbLabels(nbTargets, 0);

                for (unsigned int oy = 0; oy < targets.dimY(); ++oy) {
                    for (unsigned int ox = 0; ox < targets.dimX(); ++ox) {
                        if (target(ox, oy) >= 0) {
                            ++nbLabels[target(ox, oy)];

                            if (mask.empty() || mask(ox, oy) == mMaskedLabel) {
                                confusion(target(ox, oy),
                                          estLabels(ox, oy, 0)) += 1;

                                if (target(ox, oy)
                                    == (int)estLabels(ox, oy, 0))
                                    ++nbHits[target(ox, oy)];

                                // Top-N case :
                                if (mTargetTopN > 1) {
                                    unsigned int topNscore = 0;

                                    for (unsigned int n = 0; n < mTargetTopN; ++n) {
                                        if (estLabels(ox, oy, n)
                                            == target(ox, oy))
                                            ++topNscore;
                                    }

                                    if (topNscore > 0)
                                        ++nbHitsTopN[target(ox, oy)];
                                }
                            }
                            else {
                                // Masked target = masked false negative
                                // Should affect the recall
                                confusion(target(ox, oy), nbTargets) += 1;
                            }
                        }
                        else if (!mask.empty() && mask(ox, oy) == mMaskedLabel)
                        {
                            // Masked no target = masked false positive
                            // Should affect the precision
                            confusion(nbTargets, estLabels(ox, oy, 0)) += 1;
                        }
                    }
                }

                double success = 0.0;
                double successTopN = 0.0;
                unsigned int nbValidTargets = 0;

                for (unsigned int t = 0; t < nbTargets; ++t) {
                    if (nbLabels[t] > 0) {
                        success += nbHits[t] / (double)nbLabels[t];
                        successTopN += nbHitsTopN[t] / (double)nbLabels[t];
                        ++nbValidTargets;

                        // Misclassified
                        std::map<unsigned int, std::vector<unsigned int> >
                            ::iterator itMisclass;
                        std::tie(itMisclass, std::ignore)
                            = misclass.insert(std::make_pair(t,
                                std::vector<unsigned int>(nbTargetsConfusion, 0U)));

                        for (unsigned int e = 0; e < nbTargetsConfusion; ++e)
                            (*itMisclass).second[e] = confusion(t, e);
                    }
                }

                if (nbTargetsConfusion > nbTargets) {
                    std::map<unsigned int, std::vector<unsigned int> >
                        ::iterator itMisclass;
                    std::tie(itMisclass, std::ignore)
                        = misclass.insert(std::make_pair(nbTargets,
                            std::vector<unsigned int>(nbTargetsConfusion, 0U)));

                    for (unsigned int e = 0; e < nbTargetsConfusion; ++e)
                        (*itMisclass).second[e] = confusion(nbTargets, e);
                }

                mBatchSuccess[batchPos] = (nbValidTargets > 0) ?
                    (success / nbValidTargets) : 1.0;

                if (mTargetTopN > 1) {
                    mBatchTopNSuccess[batchPos] = (nbValidTargets > 0) ?
                        (successTopN / nbValidTargets) : 1.0;
                }

#pragma omp critical(TargetScore__process)
                std::transform(confusionMatrix.begin(), confusionMatrix.end(),
                            confusion.begin(),
                            confusionMatrix.begin(),
                            std::plus<unsigned long long int>());
            }

#pragma omp critical(TargetScore__process)
            misclassified[id].swap(misclass);
        }
    }

    // Remove invalid/ignored batch positions before computing the score
    correctLastBatch(mBatchSuccess, mScoreSet[set].success);
    mScoreSet[set].success.push_back(getBatchAverageSuccess());

    if (!mDataAsTarget && mTargetTopN > 1) {
        correctLastBatch(mBatchTopNSuccess, mScoreTopNSet[set].success);
        mScoreTopNSet[set].success.push_back(getBatchAverageTopNSuccess());
    }
}

void N2D2::TargetScore::correctLastBatch(
    std::vector<double>& batchSuccess,
    const std::deque<double>& success)
{
    const size_t batchSize = batchSuccess.size();
    batchSuccess.erase(std::remove(batchSuccess.begin(),
                                    batchSuccess.end(), -1.0),
                        batchSuccess.end());

    if (!success.empty() && !batchSuccess.empty()
        && batchSuccess.size() < batchSize)
    {
        // Compute the correct cumulative score sum
        const double scoreSum = batchSize
                * std::accumulate(success.begin(), success.end(), 0.0)
            + std::accumulate(batchSuccess.begin(), batchSuccess.end(), 0.0);

        // Compute the true average
        const double x = scoreSum
            / (success.size() * batchSize + batchSuccess.size());

        // Fill batchSuccess with true average
        batchSuccess.resize(batchSize, x);
    }
}

void N2D2::TargetScore::process(Database::StimuliSet set)
{
    Target::process(set);
    computeScore(set);
}

void N2D2::TargetScore::log(const std::string& fileName,
                            Database::StimuliSet set)
{
    logConfusionMatrix(fileName, set);
    logMisclassified(fileName, set);
}

void N2D2::TargetScore::clear(Database::StimuliSet set)
{
    Target::clear(set);
    clearConfusionMatrix(set);
    clearMisclassified(set);
}


#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace N2D2 {
void init_TargetScore(py::module &m) {
    py::class_<TargetScore, std::shared_ptr<TargetScore>, Target>(m, "TargetScore", py::multiple_inheritance());
}
}
#endif
