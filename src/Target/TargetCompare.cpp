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

#include "Target/TargetCompare.hpp"

N2D2::Registrar<N2D2::Target>
N2D2::TargetCompare::mRegistrar("TargetCompare", N2D2::TargetCompare::create);

const char* N2D2::TargetCompare::Type = "TargetCompare";

N2D2::TargetCompare::TargetCompare(const std::string& name,
                         const std::shared_ptr<Cell>& cell,
                         const std::shared_ptr<StimuliProvider>& sp,
                         double targetValue,
                         double defaultValue,
                         unsigned int targetTopN,
                         const std::string& labelsMapping)
    : TargetScore(
          name, cell, sp, targetValue, defaultValue, targetTopN, labelsMapping),
      mDataPath(this, "DataPath", ""),
      mMatching(this, "Matching", "*.dat")
{
    // ctor
    mPopulateTargets = false;
}

void N2D2::TargetCompare::process(Database::StimuliSet set)
{
    if (mDataPath->empty()) {
        throw std::runtime_error("TargetCompare::process(): missing "
                                 "(empty) DataPath");
    }

    const Tensor4d<int>& labels = mStimuliProvider->getLabelsData();

    if (mTargets.empty()) {
        mTargets.resize(mCell->getOutputsWidth(),
                        mCell->getOutputsHeight(),
                        1,
                        labels.dimB());
        mEstimatedLabels.resize(mCell->getOutputsWidth(),
                                mCell->getOutputsHeight(),
                                mTargetTopN,
                                labels.dimB());
        mEstimatedLabelsValue.resize(mCell->getOutputsWidth(),
                                     mCell->getOutputsHeight(),
                                     mTargetTopN,
                                     labels.dimB());
    }

    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);
    const Tensor4d<Float_T>& values = targetCell->getOutputs();

    std::vector<double> meanSquareErrors(values.dimB());

#pragma omp parallel for if (values.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)values.dimB(); ++batchPos) {
        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of
            // the set)
            continue;
        }

        const std::string imgFile
            = mStimuliProvider->getDatabase().getStimulusName(id);
        const std::string baseName = Utils::baseName(imgFile);
        std::string fileBaseName = Utils::fileBaseName(baseName);
        fileBaseName = Utils::searchAndReplace(mMatching, "*", fileBaseName);
        const std::string dataFileName = Utils::expandEnvVars(mDataPath)
            + "/" + fileBaseName;

        std::ifstream dataFile(dataFileName);

        if (!dataFile.good()) {
            throw std::runtime_error("Could not open target data file: "
                                     + dataFileName);
        }

        Tensor3d<int> target = mTargets[batchPos];
        const Tensor3d<Float_T> value = values[batchPos];
        Tensor3d<Float_T> targetValues(value.dimX(),
                                       value.dimY(),
                                       value.dimZ());

        if (!(dataFile >> targetValues.data()))
            throw std::runtime_error("Unreadable data file: " + dataFileName);

        dataFile.close();

        const unsigned int nbOutputs = value.dimZ();

        double meanSquareError = 0.0;

        for (unsigned int oy = 0; oy < value.dimY(); ++oy) {
            for (unsigned int ox = 0; ox < value.dimX(); ++ox) {
                std::vector<std::pair<Float_T, size_t> >
                sortedLabelsValues;
                sortedLabelsValues.reserve(nbOutputs);

                for (unsigned int output = 0; output < nbOutputs; ++output) {
                    const double error = targetValues(ox, oy, output)
                                        - value(ox, oy, output);
                    meanSquareError += error * error;

                    sortedLabelsValues.push_back(std::make_pair(
                        targetValues(ox, oy, output), output));
                }

                // Top-n accuracy sorting
                std::partial_sort(
                    sortedLabelsValues.begin(),
                    sortedLabelsValues.begin() + 1,
                    sortedLabelsValues.end(),
                    std::greater<std::pair<Float_T, size_t> >());

                target(ox, oy, 0) = sortedLabelsValues[0].second;
            }
        }

        meanSquareError /= value.size();
        meanSquareErrors[batchPos] = meanSquareError;
    }

    for (int batchPos = 0; batchPos < (int)values.dimB(); ++batchPos) {
        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of
            // the set)
            continue;
        }

        std::cout << "MSE #" << id << " = " << meanSquareErrors[batchPos]
            << std::endl;
    }

    TargetScore::process(set);
}

N2D2::TargetCompare::~TargetCompare()
{
    // dtor
}
