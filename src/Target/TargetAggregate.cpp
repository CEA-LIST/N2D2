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

#include "Xnet/Network.hpp"
#include "N2D2.hpp"
#include "StimuliProvider.hpp"
#include "Cell/Cell.hpp"
#include "ROI/RectangularROI.hpp"
#include "Target/TargetAggregate.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

N2D2::Registrar<N2D2::Target>
N2D2::TargetAggregate::mRegistrar("TargetAggregate", N2D2::TargetAggregate::create);

const char* N2D2::TargetAggregate::Type = "TargetAggregate";

N2D2::TargetAggregate::TargetAggregate(const std::string& name,
                             const std::shared_ptr<Cell>& cell,
                             const std::shared_ptr<StimuliProvider>& sp,
                             double targetValue,
                             double defaultValue,
                             unsigned int targetTopN,
                             const std::string& labelsMapping,
                             bool createMissingLabels)
    : Target(
          name, cell, sp, targetValue, defaultValue, targetTopN, labelsMapping, createMissingLabels),
      mScoreTopN(this, "ScoreTopN", 1U)
{
    // ctor
}

unsigned int N2D2::TargetAggregate::getNbTargets() const
{
    return (mROIsLabelTarget) ? mROIsLabelTarget->getNbTargets()
                              : Target::getNbTargets();
}

void N2D2::TargetAggregate::setROIsLabelTarget(const std::shared_ptr<Target>& target)
{
    mROIsLabelTarget = target;

    // Create target_rois_label.dat for target_rois_viewer.py
    const std::string fileName = mName + "/target_rois_label.dat";
    std::ofstream roisLabelData(fileName);

    if (!roisLabelData.good())
        throw std::runtime_error("Could not save data file: " + fileName);

    roisLabelData << mROIsLabelTarget->getName();
    roisLabelData.close();
}

void N2D2::TargetAggregate::logConfusionMatrix(const std::string& fileName,
                                          Database::StimuliSet set) const
{
    (*mScoreSet.find(set)).second.confusionMatrix.log(
        mName + "/ConfusionMatrix_" + fileName + ".dat", getTargetLabelsName());
}

void N2D2::TargetAggregate::logMisclassified(const std::string& fileName,
                                         Database::StimuliSet set) const
{
    assert(mScoreSet.find(set) != mScoreSet.end());

    if (mDataAsTarget)
        return;

    const unsigned int nbTargets = getNbTargets();

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

        for (std::map<unsigned int, std::vector<unsigned int> >::const_iterator
            itMisclass = (*it).second.begin(),
            itMisclassEnd = (*it).second.end(); itMisclass != itMisclassEnd;
            ++itMisclass)
        {
            // For each stimulus' target
            const unsigned int target = (*itMisclass).first;

            for (unsigned int estimated = 0; estimated < nbTargets;
                ++estimated)
            {
                if ((*itMisclass).second[estimated] > 0) {
                    data << (*it).first;

                    if (firstOccurrence)
                        data << " " << Utils::quoted(name);
                    else
                        data << " ";

                    data << " " << target
                        << " " << estimated
                        << " " << (*itMisclass).second[estimated]
                        << "\n";

                    firstOccurrence = false;
                }
            }
        }
    }
}

void N2D2::TargetAggregate::clearConfusionMatrix(Database::StimuliSet set)
{
    mScoreSet[set].confusionMatrix.clear();
}

void N2D2::TargetAggregate::clearMisclassified(Database::StimuliSet set)
{
    mScoreSet[set].misclassified.clear();
}

void N2D2::TargetAggregate::processEstimatedLabels(Database::StimuliSet set, Float_T* values)
{
    const bool validDatabase
        = (mStimuliProvider->getDatabase().getNbStimuli() > 0);

    const unsigned int nbTargets = getNbTargets();
    ConfusionMatrix<unsigned long long int>& confusionMatrix
        = mScoreSet[set].confusionMatrix;

    if (confusionMatrix.empty())
        confusionMatrix.resize(nbTargets, nbTargets, 0);

    std::map<unsigned int,
                std::map<unsigned int, std::vector<unsigned int> > >&
        misclassified = mScoreSet[set].misclassified;

    mEstimatedLabels.synchronizeDBasedToH();

#pragma omp parallel for if (mTargets.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mTargets.dimB(); ++batchPos) {
        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

#ifdef CUDA
        CudaContext::setDevice();
#endif

        std::map<unsigned int, std::vector<unsigned int> > misclass;

        const std::shared_ptr<ROI> globalBB
            = std::make_shared<RectangularROI<int> >(
                            0,
                            // RectangularROI<>() bottom right is exclusive
                            cv::Point(0, 0),
                            cv::Point(mEstimatedLabels.dimX(),
                                      mEstimatedLabels.dimY()));

        Float_T score = 0.0;
        std::vector<std::pair<int, double> > scoreTopN;

        if (mScoreTopN > 1) {
            TensorLabelsValue_T bbLabels = (mROIsLabelTarget)
                ? mROIsLabelTarget->getEstimatedLabels(globalBB, batchPos, values)
                : getEstimatedLabels(globalBB, batchPos, values);

            std::vector<int> labelIdx(bbLabels.size());
            std::iota(labelIdx.begin(), labelIdx.end(), 0);

            // sort indexes based on comparing values in
            // bbLabels. Sort in descending order.
            std::partial_sort(labelIdx.begin(),
                labelIdx.begin() + mScoreTopN,
                labelIdx.end(),
                [&bbLabels](int i1, int i2)
                    {return bbLabels(i1) > bbLabels(i2);});

            globalBB->setLabel(labelIdx[0]);
            score = bbLabels(labelIdx[0]);

            for (unsigned int i = 1; i < mScoreTopN; ++i) {
                scoreTopN.push_back(
                    std::make_pair(labelIdx[i], bbLabels(labelIdx[i])));
            }
        }
        else {
            int label;

            std::tie(label, score) = (mROIsLabelTarget)
                ? mROIsLabelTarget->getEstimatedLabel(globalBB, batchPos, values)
                : getEstimatedLabel(globalBB, batchPos, values);

            globalBB->setLabel(label);
        }

        if (validDatabase) {
            const Tensor<int> target = mTargets[batchPos];
            const int bbLabel = globalBB->getLabel();

            // Extract ground true ROIs
            std::vector<std::shared_ptr<ROI> > labelROIs
                = mStimuliProvider->getLabelsROIs(batchPos);

            if (labelROIs.empty() && target.size() == 1) {
                // The whole image has a single label
                if (target(0) >= 0) {
                    // Confusion computation
#pragma omp atomic
                    confusionMatrix(target(0), bbLabel) += 1ULL;

                    std::map<unsigned int, std::vector<unsigned int> >
                        ::iterator itMisclass;
                    std::tie(itMisclass, std::ignore)
                        = misclass.insert(std::make_pair(target(0),
                            std::vector<unsigned int>(nbTargets, 0U)));
                    (*itMisclass).second[bbLabel] = 1U;
                }
            }
            else {
                std::map<int, unsigned int> targetFreq;

                for (size_t index = 0; index < target.size(); ++index) {
                    if (target(index) >= 0)
                        ++targetFreq[target(index)];
                }
                
                const std::map<int, unsigned int>::const_iterator itFreq
                    = std::max_element(targetFreq.begin(), targetFreq.end(),
                                    Utils::PairSecondPred<int, unsigned int>());

                const int globalTarget = (*itFreq).first;
#pragma omp atomic
                confusionMatrix(globalTarget, bbLabel) += 1ULL;

                std::map<unsigned int, std::vector<unsigned int> >
                    ::iterator itMisclass;
                std::tie(itMisclass, std::ignore)
                    = misclass.insert(std::make_pair(globalTarget,
                        std::vector<unsigned int>(nbTargets, 0U)));
                (*itMisclass).second[bbLabel] = 1U;
            }

#pragma omp critical(TargetAggregate__processEstimatedLabels)
            misclassified[id].swap(misclass);
        }
    }
}

void N2D2::TargetAggregate::process(Database::StimuliSet set)
{
    Target::process(set);
    processEstimatedLabels(set);
}

void N2D2::TargetAggregate::log(const std::string& fileName,
                           Database::StimuliSet set)
{
    logConfusionMatrix(fileName, set);
    logMisclassified(fileName, set);
}

void N2D2::TargetAggregate::clear(Database::StimuliSet set)
{
    Target::clear(set);
    clearConfusionMatrix(set);
    clearMisclassified(set);
}
