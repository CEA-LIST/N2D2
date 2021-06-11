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

#include "Target/Target.hpp"
#include "N2D2.hpp"
#include "StimuliProvider.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/Cell_CSpike_Top.hpp"
#ifdef CUDA
#include "Target/Target_CUDA_kernels.hpp"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

N2D2::Registrar<N2D2::Target> N2D2::Target::mRegistrar("Target",
                                                       N2D2::Target::create);

const char* N2D2::Target::Type = "Target";

N2D2::Target::Target(const std::string& name,
                     const std::shared_ptr<Cell>& cell,
                     const std::shared_ptr<StimuliProvider>& sp,
                     double targetValue,
                     double defaultValue,
                     unsigned int targetTopN,
                     const std::string& labelsMapping_,
                     bool createMissingLabels)
    : mDataAsTarget(this, "DataAsTarget", false),
      mNoDisplayLabel(this, "NoDisplayLabel", -1),
      mLabelsHueOffset(this, "LabelsHueOffset", 0),
      mEstimatedLabelsValueDisplay(this, "EstimatedLabelsValueDisplay", true),
      mMaskedLabel(this, "MaskedLabel", -1),
      mMaskedLabelValue(this, "MaskedLabelValue", false),
      mBinaryThreshold(this, "BinaryThreshold", 0.5),
      mValueThreshold(this, "ValueThreshold", 0.0),
      mImageLogFormat(this, "ImageLogFormat", "jpg"),
      mWeakTarget(this, "WeakTarget", -2),
      mName(name),
      mCell(cell),
      mStimuliProvider(sp),
      mTargetValue(targetValue),
      mDefaultValue(defaultValue),
      mTargetTopN(targetTopN),
      mDefaultTarget(-2)
{
    // ctor
    Utils::createDirectories(name);

    if (!labelsMapping_.empty())
        labelsMapping(labelsMapping_, createMissingLabels);
}

unsigned int N2D2::Target::getNbTargets() const
{
    return (mCell->getNbOutputs() > 1) ? mCell->getNbOutputs() : 2;
}

void N2D2::Target::labelsMapping(const std::string& fileName,
                                 bool createMissingLabels)
{
    mLabelsMapping.clear();

    if (fileName.empty())
        return;

    std::ifstream clsFile(fileName.c_str());

    if (!clsFile.good())
        throw std::runtime_error("Could not open class mapping file: "
                                 + fileName);

    std::string line;

    while (std::getline(clsFile, line)) {
        // Remove optional comments
        line.erase(std::find(line.begin(), line.end(), '#'), line.end());
        // Left trim & right trim (right trim necessary for extra "!value.eof()"
        // check later)
        line.erase(
            line.begin(),
            std::find_if(line.begin(),
                         line.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
        line.erase(std::find_if(line.rbegin(),
                                line.rend(),
                                std::not1(std::ptr_fun<int, int>(std::isspace)))
                       .base(),
                   line.end());

        if (line.empty())
            continue;

        std::string className;
        int output;

        std::stringstream value(line);
        std::stringstream classNameStr;

        int wordInString = std::count_if(line.begin(), line.end(), [](char ch) { return isspace(ch); });

        for(int idx = 0; idx < wordInString; ++idx)
        {
            std::string str;
            if (!(value >> Utils::quoted(str)))
                throw std::runtime_error("Unreadable class name: " + line + " in file "
                                         + fileName);
             if(idx > 0)
                classNameStr << " ";

             classNameStr << str;
        }

        className = classNameStr.str();

        //if (!(value >> Utils::quoted(className)) || !(value >> output)
        //    || (output < 0 && output != -1) || !value.eof())
        //    throw std::runtime_error("Unreadable value: " + line + " in file "
         //                            + fileName);

        if (!(value >> output) || (output < 0 && output != -1) || !value.eof())
            throw std::runtime_error("Unreadable value: " + line + " in file "
                                     + fileName);


        if (className == "default") {
            if (mDefaultTarget >= -1)
                throw std::runtime_error(
                    "Default mapping already exists in file " + fileName);

            mDefaultTarget = output;
        } else {
            std::vector<int> labels;

            if (className != "*") {
                if (mStimuliProvider->getDatabase().isLabel(className)) {
                    if (className.find_first_of("*?") != std::string::npos) {
                        throw std::runtime_error("Ambiguous use of wildcard: "
                            + line + ", because there is a label named \""
                            + className + "\" in the database, in file "
                            + fileName);
                    }

                    labels.push_back(mStimuliProvider->getDatabase()
                        .getLabelID(className));
                }
                else {
                    labels = mStimuliProvider->getDatabase()
                        .getMatchingLabelsIDs(className);
                }
            }
            else {
                if (mStimuliProvider->getDatabase().isLabel(className)) {
                    throw std::runtime_error("Ambiguous ignore wildcard *,"
                        " because there is a label named \"*\" in the database,"
                        " in file " + fileName);
                }

                labels.push_back(-1);
            }

            if (labels.empty() && createMissingLabels) {
                // Remove wildcard
                className = Utils::searchAndReplace(className, "*", "");
                className = Utils::searchAndReplace(className, "?", "_");

                labels.push_back(mStimuliProvider->getDatabase().addLabel(className));
            }

            if (!labels.empty()) {
                for (std::vector<int>::const_iterator it = labels.begin(),
                    itEnd = labels.end(); it != itEnd; ++it)
                {
                    bool newInsert;
                    std::tie(std::ignore, newInsert)
                        = mLabelsMapping.insert(std::make_pair(*it, output));

                    if (!newInsert) {
                        throw std::runtime_error(
                            "Mapping already exists for label: " + line
                            + " in file " + fileName);
                    }
                }
            }
            else {
                std::cout
                    << Utils::cwarning
                    << "No label exists in the database with the name: "
                    << className << " in file " << fileName
                    << Utils::cdef << std::endl;
            }
        }
    }
}

void N2D2::Target::setLabelTarget(int label, int output)
{
    mLabelsMapping[label] = output;
}

void N2D2::Target::setDefaultTarget(int output)
{
    mDefaultTarget = output;
}

int N2D2::Target::getLabelTarget(int label) const
{
    if (mLabelsMapping.empty())
        return label;
    else {
        const std::map<int, int>::const_iterator it
            = mLabelsMapping.find(label);

        if (it != mLabelsMapping.end())
            return (*it).second;
        else if (mDefaultTarget >= -1)
            return mDefaultTarget;
        else {
            #pragma omp critical
            {
                std::stringstream labelStr;
                labelStr << label;

                const std::string labelName
                    = mStimuliProvider->getDatabase().getLabelName(label);

                throw std::runtime_error(
                    "Incomplete class mapping: no output specified for label #"
                    + labelStr.str() + " (" + labelName + ")");
            }
            return 0;
        }
    }
}

int N2D2::Target::getDefaultTarget() const
{
    if (mDefaultTarget >= -1)
        return mDefaultTarget;
    else
        throw std::runtime_error("No default target mapping");
}

std::vector<int> N2D2::Target::getTargetLabels(int output) const
{
    if (mLabelsMapping.empty())
        return std::vector<int>(1, output);
    else {
        std::vector<int> labels;

        for (std::map<int, int>::const_iterator it = mLabelsMapping.begin(),
                                                itEnd = mLabelsMapping.end();
             it != itEnd;
             ++it) {
            if ((*it).second == output)
                labels.push_back((*it).first);
        }

        return labels;
    }
}

const std::vector<std::string>& N2D2::Target::getTargetLabelsName() const
{
    if (mLabelsName.empty()) {
        const unsigned int nbTargets = getNbTargets();
        mLabelsName.reserve(nbTargets);

        for (int target = 0; target < (int)nbTargets; ++target) {
            std::stringstream labelName;

            if (target == mDefaultTarget)
                labelName << "default";
            else {
                const std::vector<int> cls = getTargetLabels(target);
                const Database& db = mStimuliProvider->getDatabase();

                if (!cls.empty()) {
                    labelName << ((cls[0] >= 0 && cls[0] < (int)db.getNbLabels())
                                    ? db.getLabelName(cls[0]) :
                                (cls[0] >= 0)
                                    ? "" :
                                    "*");

                    if (cls.size() > 1)
                        labelName << "...";
                }
            }

            mLabelsName.push_back(labelName.str());
        }
    }

    return mLabelsName;
}

void N2D2::Target::logLabelsMapping(const std::string& fileName) const
{
    if (mDataAsTarget)
        return;

    const std::string dataFileName = mName + "/" + fileName + ".dat";
    std::ofstream labelsData(dataFileName);

    if (!labelsData.good())
        throw std::runtime_error("Could not save log class mapping data file: "
                                 + dataFileName);

    labelsData << "label name output\n";

    for (unsigned int label = 0,
                      size = mStimuliProvider->getDatabase().getNbLabels();
         label < size;
         ++label)
    {
        labelsData << label
            << " " << Utils::quoted(mStimuliProvider->getDatabase()
                                        .getLabelName(label))
            << " " << getLabelTarget(label) << "\n";
    }
}

void N2D2::Target::provideTargets(Database::StimuliSet set) // TODO debug here 
{
    std::shared_ptr<Cell_Frame_Top> targetCell 
        = std::dynamic_pointer_cast<Cell_Frame_Top>(mCell);

    if (mDataAsTarget) {
        if (set == Database::Learn && targetCell) {
            // Update target values from input data
            targetCell->setOutputTargets(mStimuliProvider->getTargetData());
        }

        return;
    }

    const unsigned int nbTargets = getNbTargets();
    const bool validDatabase
        = (mStimuliProvider->getDatabase().getNbStimuli() > 0);
    const Tensor<int>& labels = mStimuliProvider->getLabelsData();

    int dev = 0;
#ifdef CUDA
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    if (mTargetData.empty()) {
#pragma omp critical(Target__provideTargets)
        if (mTargetData.empty()) {
            int count = 1;
#ifdef CUDA
            CHECK_CUDA_STATUS(cudaGetDeviceCount(&count));
#endif
            mTargetData.resize(count);
        }
    }

    Tensor<int>& targets = mTargetData[dev].targets;

    if (targets.empty()) {
        targets.resize({mCell->getOutputsWidth(), mCell->getOutputsHeight(), 1,
            labels.dimB()});
    }

    if (validDatabase) {
        // Generate targets
        const size_t size = targets.dimB() * targets.dimY();

        if (targets.dimX() != labels.dimX()
            || targets.dimY() != labels.dimY()) {
            const double xRatio = labels.dimX() / (double)targets.dimX();
            const double yRatio = labels.dimY() / (double)targets.dimY();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (targets.dimB() > 4 && size > 16)
#endif
            for (int batchPos = 0; batchPos < (int)targets.dimB();
                 ++batchPos)
            {
                for (int y = 0; y < (int)targets.dimY(); ++y) {
#ifdef CUDA
                    CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

                    const int id = mStimuliProvider->getBatch()[batchPos];

                    if (id < 0)
                        continue;

                    for (int x = 0; x < (int)targets.dimX(); ++x) {
                        const unsigned int xl0 = std::floor(x * xRatio);
                        const unsigned int xl1 = std::max(xl0 + 1,
                            (unsigned int)std::floor((x + 1) * xRatio));
                        const unsigned int yl0 = std::floor(y * yRatio);
                        const unsigned int yl1 = std::max(yl0 + 1,
                            (unsigned int)std::floor((y + 1) * yRatio));

                        // +1 takes into account ignore target (-1)
                        std::vector<int> targetHist(nbTargets + 1, 0);

                        for (unsigned int yl = yl0; yl < yl1; ++yl) {
                            for (unsigned int xl = xl0; xl < xl1; ++xl) {
                                const int target = getLabelTarget(
                                    labels(xl, yl, 0, batchPos));

                                // Target range checking
                                if (target >= (int)nbTargets) {
#pragma omp critical(Target__provideTargets)
                                    {
                                        std::cout << Utils::cwarning
                                                  << "Stimulus #" << id
                                                  << " has target " << target
                                                  << " @ (" << xl << "," << yl
                                                  << ") but number of output "
                                                     "target is "
                                                  << nbTargets << Utils::cdef
                                                  << std::endl;

                                        throw std::runtime_error(
                                            "Target::process(): target out "
                                            "of range.");
                                    }
                                }

                                assert(target >= -1);
                                ++targetHist[target + 1];
                            }
                        }

                        if (mWeakTarget >= -1) {
                            // initialize original index locations
                            // first index is -1 (ignore)
                            std::vector<int> targetHistIdx(targetHist.size());
                            std::iota(targetHistIdx.begin(),
                                targetHistIdx.end(), -1); // -1 = ignore

                            // sort indexes based on comparing values in
                            // targetHist. Sort in descending order.
                            std::partial_sort(targetHistIdx.begin(),
                                targetHistIdx.begin() + 2, targetHistIdx.end(),
                                [&targetHist](int i1, int i2) {
                                    return targetHist[i1 + 1]
                                        > targetHist[i2 + 1];
                                });

                            targets(x, y, 0, batchPos)
                                = (targetHistIdx[0] == mWeakTarget
                                      && targetHistIdx[1] > 0)
                                ? targetHistIdx[1]
                                : targetHistIdx[0];
                        }
                        else {
                            std::vector<int>::iterator maxElem
                                = std::max_element(
                                    targetHist.begin(), targetHist.end());

                            targets(x, y, 0, batchPos)
                                = std::distance(targetHist.begin(),
                                      maxElem)
                                - 1; // -1 = ignore
                        }
                    }
                }
            }
        }
        else {
            // one-to-one mapping
#pragma omp parallel for if (targets.dimB() > 64 && size > 256)
            for (int batchPos = 0; batchPos < (int)targets.dimB();
                 ++batchPos)
            {
#ifdef CUDA
                CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

                const int id = mStimuliProvider->getBatch()[batchPos];

                if (id < 0)
                    continue;

                // target only has 1 channel, whereas label has as many
                // channels as environment channels
                const Tensor<int> label = labels[batchPos][0];
                Tensor<int> target = targets[batchPos][0];

                for (int index = 0; index < (int)label.size(); ++index) {
                    target(index) = getLabelTarget(label(index));

                    // Target range checking
                    if (target(index) >= (int)nbTargets) {
#pragma omp critical(Target__provideTargets)
                        {
                            std::cout << Utils::cwarning << "Stimulus #" << id
                                      << " has target " << target(index)
                                      << " @ (" << index
                                      << ") but "
                                         "number of output target is "
                                      << nbTargets << Utils::cdef << std::endl;

                            throw std::runtime_error(
                                "Target::process(): target out of "
                                "range.");
                        }
                    }
                }
            }
        }
    }

    //Set label associated to targets
    if (set == Database::Learn && targetCell) {
        // Set targets
        if (targets.dimX() == 1 && targets.dimY() == 1) {
            for (unsigned int batchPos = 0; batchPos < targets.dimB();
                    ++batchPos) {
                if (targets(0, batchPos) < 0) {
                    std::cout << Utils::cwarning
                                << "Target::setTargetsValue(): ignore label "
                                    "with 1D output for stimuli ID "
                                << mStimuliProvider->getBatch()[batchPos]
                                << Utils::cdef << std::endl;
                }
            }
        }

        targetCell->setOutputTarget(targets);
    }
}

void N2D2::Target::process(Database::StimuliSet set) // TODO debug here 
{
    std::shared_ptr<Cell_Frame_Top> targetCell 
        = std::dynamic_pointer_cast<Cell_Frame_Top>(mCell);

    int dev = 0;
#ifdef CUDA
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    if (mTargetData.empty() || mLoss.empty()) {
#pragma omp critical(Target__process_mTargetData)
        if (mTargetData.empty() || mLoss.empty()) {
            int count = 1;
#ifdef CUDA
            CHECK_CUDA_STATUS(cudaGetDeviceCount(&count));
#endif
            mTargetData.resize(count);
            mLoss.resize(count);
        }
    }

    std::vector<Float_T>& loss = mLoss[dev];

    if (mDataAsTarget) {
        //mLoss.push_back(targetCell->applyLoss());

        loss.push_back(targetCell->applyLossDistribWeighted(20, -1.0, 1.0));

        //Tensor<Float_T> kernel({7, 7, 1, 1}, 1.0 / (7.0 * 7.0));
        //kernel(0, 0, 0, 0) = 0.0;
        //kernel(1, 0, 0, 0) = 2.0 / 16.0;
        //kernel(2, 0, 0, 0) = 0.0;
        //kernel(0, 1, 0, 0) = 2.0 / 16.0;
        //kernel(1, 1, 0, 0) = 4.0 / 16.0;
        //kernel(2, 1, 0, 0) = 2.0 / 16.0;
        //kernel(0, 2, 0, 0) = 0.0;
        //kernel(1, 2, 0, 0) = 2.0 / 16.0;
        //kernel(2, 2, 0, 0) = 0.0;

        //mLoss.push_back(targetCell->applyLossThroughKernel(kernel,
        //    std::bind((double(Cell_Frame_Top::*)(unsigned int, double, double))
        //            &Cell_Frame_Top::applyLossDistribWeighted,
        //        targetCell.get(), 100U, -1.0, 1.0)));

        //mLoss.push_back(targetCell->applyLossThroughKernel(kernel,
        //    std::bind((double(Cell_Frame_Top::*)())
        //        &Cell_Frame_Top::applyLoss,
        //    targetCell.get())));
        return;
    }

    loss.push_back(targetCell->applyLoss(mTargetValue, mDefaultValue));

    const Tensor<int>& labels = mStimuliProvider->getLabelsData();
    TensorLabels_T& estimatedLabels = mTargetData[dev].estimatedLabels;
    TensorLabelsValue_T& estimatedLabelsValue = mTargetData[dev].estimatedLabelsValue;

    std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
        = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);

    BaseTensor& outputsBaseTensor = (targetCell)
        ? targetCell->getOutputs() : targetCellCSpike->getOutputsActivity();
    // Find batchSize to ignore invalid stimulus in batch (can occur for the 
    // last batch of the set)
    int batchSize = 0;

    if (mStimuliProvider->getBatch().back() >= 0)
        batchSize = (int)labels.dimB();
    else {
        for (; batchSize < (int)labels.dimB(); ++batchSize) {
            const int id = mStimuliProvider->getBatch()[batchSize];

            if (id < 0)
                break;
        }
    }

    // batchSize may be 0 in multi-GPU for some GPUs...
    if (batchSize == 0)
        return;

#ifdef CUDA
    CudaBaseTensor* outputsCudaBaseTensor 
            = dynamic_cast<CudaBaseTensor*>(&outputsBaseTensor);

    if (outputsCudaBaseTensor != NULL) {
        const unsigned int nbOutputs = outputsCudaBaseTensor->dimZ();

        if (mTargetTopN > nbOutputs) {
            throw std::runtime_error("Target::process_Frame_CUDA(): target 'TopN' "
                                    "parameter must be <= to the network "
                                    "output size");
        }
        std::shared_ptr<CudaDeviceTensor<Float_T> > value
            = cuda_device_tensor_cast<Float_T>(*outputsCudaBaseTensor);
        
        process_Frame_CUDA(value->getDevicePtr(), batchSize);
    }
    else {
#endif
        process_Frame(outputsBaseTensor, batchSize);
#ifdef CUDA

        estimatedLabels.hostBased() = true;
        estimatedLabelsValue.hostBased() = true;
    }
#endif

    if (estimatedLabelsValue.dimX() == 1
        && estimatedLabelsValue.dimY() == 1)
    {
        static bool display = true;

        if (set == Database::Test && display) {
            estimatedLabels.synchronizeDBasedToH();
            estimatedLabelsValue.synchronizeDBasedToH();

            /*std::cout << "[";

            for (int i = 0; i < (int)estimatedLabelsValue.dimZ(); ++i) {
                std::cout << estimatedLabels(0, 0, i, 0) << ":"
                    << std::setprecision(2)
                    << std::fixed
                    << estimatedLabelsValue(0, 0, i, 0) << " ";
            }

            std::cout << "]" << std::endl;*/
            display = false;
        }
    }
}

#ifdef CUDA
void N2D2::Target::process_Frame_CUDA(Float_T* values,
                                      const int batchSize)
{ 
    int dev;
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));

    const Tensor<int>& labels = mStimuliProvider->getLabelsData();
    TensorLabels_T& estimatedLabels = mTargetData[dev].estimatedLabels;
    TensorLabelsValue_T& estimatedLabelsValue = mTargetData[dev].estimatedLabelsValue;

    if (estimatedLabels.empty()) {
        estimatedLabels.resize({mCell->getOutputsWidth(),
            mCell->getOutputsHeight(), mTargetTopN, labels.dimB()});
        estimatedLabelsValue.resize({mCell->getOutputsWidth(),
            mCell->getOutputsHeight(), mTargetTopN, labels.dimB()});
    }

    cudaGetEstimatedTarget( mTargetTopN,
                            mCell->getNbOutputs(),
                            mCell->getOutputsHeight(),
                            mCell->getOutputsWidth(),
                            batchSize,
                            mBinaryThreshold,
                            values,
                            estimatedLabelsValue.getDevicePtr(),
                            estimatedLabels.getDevicePtr());
}
#endif

void N2D2::Target::process_Frame(BaseTensor& values,
                                 const int batchSize)
{
    const unsigned int nbOutputs = values.dimZ();

    if (mTargetTopN > nbOutputs) {
        throw std::runtime_error("Target::process_Frame(): target 'TopN' "
                                "parameter must be <= to the network "
                                "output size");
    }

    const Tensor<Float_T>& value = tensor_cast<Float_T>(values);
    const size_t size = value.dimY() * batchSize;

    std::vector<int> outputsIdx(nbOutputs);

    if (nbOutputs > 1 && mTargetTopN > 1)
        std::iota(outputsIdx.begin(), outputsIdx.end(), 0);

    const Tensor<int>& labels = mStimuliProvider->getLabelsData();
    TensorLabels_T& estimatedLabels = mTargetData[0].estimatedLabels;
    TensorLabelsValue_T& estimatedLabelsValue = mTargetData[0].estimatedLabelsValue;

    if (estimatedLabels.empty()) {
        estimatedLabels.resize({mCell->getOutputsWidth(),
            mCell->getOutputsHeight(), mTargetTopN, labels.dimB()});
        estimatedLabelsValue.resize({mCell->getOutputsWidth(),
            mCell->getOutputsHeight(), mTargetTopN, labels.dimB()});
    }

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16) schedule(dynamic)
#else
#pragma omp parallel for if (batchSize > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)batchSize; ++batchPos)
    {
        for (int oy = 0; oy < (int)value.dimY(); ++oy) {
            for (int ox = 0; ox < (int)value.dimX(); ++ox) {
                if (nbOutputs > 1 && mTargetTopN > 1) {
                    // initialize original index locations
                    std::vector<int> sortedLabelsIdx(outputsIdx.begin(),
                                                    outputsIdx.end());

                    // sort indexes based on comparing values
                    std::partial_sort(sortedLabelsIdx.begin(),
                        sortedLabelsIdx.begin() + mTargetTopN,
                        sortedLabelsIdx.end(),
                        [&value, &ox, &oy, &batchPos](int i1, int i2)
                            {return value(ox, oy, i1, batchPos)
                                        > value(ox, oy, i2, batchPos);});

                    for (unsigned int i = 0; i < mTargetTopN; ++i) {
                        estimatedLabels(ox, oy, i, batchPos)
                            = sortedLabelsIdx[i];
                        estimatedLabelsValue(ox, oy, i, batchPos)
                            = value(ox, oy, sortedLabelsIdx[i], batchPos);
                    }
                }
                else if (nbOutputs > 1) {
                    size_t maxIdx = 0;
                    Float_T maxVal = value(ox, oy, 0, batchPos);

                    for (size_t i = 1; i < nbOutputs; ++i) {
                        if (value(ox, oy, i, batchPos) > maxVal) {
                            maxIdx = i;
                            maxVal = value(ox, oy, i, batchPos);
                        }
                    }

                    estimatedLabels(ox, oy, 0, batchPos) = maxIdx;
                    estimatedLabelsValue(ox, oy, 0, batchPos) = maxVal;
                }
                else {
                    estimatedLabels(ox, oy, 0, batchPos)
                        = (value(ox, oy, 0, batchPos) > mBinaryThreshold);
                    estimatedLabelsValue(ox, oy, 0, batchPos)
                        = (estimatedLabels(ox, oy, 0, batchPos) == 1)
                                ? value(ox, oy, 0, batchPos)
                                : (1.0 - value(ox, oy, 0, batchPos));
                }

            }
        }
    }
}



void N2D2::Target::logEstimatedLabels(const std::string& dirName) const
{
    const std::string dirPath = mName + "/" + dirName;
    Utils::createDirectories(dirPath);

    const bool validDatabase
        = (mStimuliProvider->getDatabase().getNbStimuli() > 0);

    int dev = 0;
#ifdef CUDA
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    const Tensor<int>& targets = mTargetData[dev].targets;
    const TensorLabels_T& estimatedLabels = mTargetData[dev].estimatedLabels;
    const TensorLabelsValue_T& estimatedLabelsValue = mTargetData[dev].estimatedLabelsValue;

    if (targets.dimX() == 1 && targets.dimY() == 1) {
#if !defined(WIN32) && !defined(__CYGWIN__) && !defined(_WIN32)
        const int ret = symlink(N2D2_PATH("tools/roc.py"),
                                (dirPath + "_roc.py").c_str());
        if (ret < 0) {
        } // avoid ignoring return value warning
#endif

        std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
            <Cell_Frame_Top>(mCell);
        std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
            = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);

        BaseTensor& valuesBaseTensor = (targetCell)
            ? targetCell->getOutputs() : targetCellCSpike->getOutputsActivity();
        Tensor<Float_T> values;
        valuesBaseTensor.synchronizeToH(values);

        const unsigned int nbOutputs = values.dimZ();

        const std::string fileName = dirPath + "/classif.log";

        std::ofstream data(fileName, std::ofstream::app);

        if (!data.good()) {
            throw std::runtime_error("Could not save log classif data file: "
                                    + fileName);
        }

        estimatedLabels.synchronizeDBasedToH();
        estimatedLabelsValue.synchronizeDBasedToH();

        for (int batchPos = 0; batchPos < (int)targets.dimB(); ++batchPos) {
            const int id = mStimuliProvider->getBatch()[batchPos];

            if (id < 0) {
                // Invalid stimulus in batch (can occur for the last batch of the
                // set)
                continue;
            }

            const Tensor<Float_T> value = values[batchPos];
            const Tensor<int> target = targets[batchPos][0];
            const Tensor<int> estLabels = estimatedLabels[batchPos][0];
            const Tensor<Float_T> estLabelsValue
                = estimatedLabelsValue[batchPos][0];

            std::ostringstream imgFile;

            if (validDatabase)
                imgFile << mStimuliProvider->getDatabase().getStimulusName(id);
            else
                imgFile << std::setw(10) << std::setfill('0') << id;

            data << id
                << " " << Utils::quoted(imgFile.str())
                << " " << target(0)
                << " " << estLabels(0)
                << " " << estLabelsValue(0);

            for (int i = 0; i < (int)nbOutputs; ++i)
                data << " " << value(i);

            data << "\n";
        }

        return;
    }

    if (mDataAsTarget) {
        std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
            <Cell_Frame_Top>(mCell);
        std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
            = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);

        BaseTensor& valuesBaseTensor = (targetCell)
            ? targetCell->getOutputs() : targetCellCSpike->getOutputsActivity();
        Tensor<Float_T> values;
        valuesBaseTensor.synchronizeToH(values);

        const int size = mStimuliProvider->getBatch().size();
        const double alpha
            = (mStimuliProvider->getParameter<bool>("DataSignedMapping"))
                ? 128.0 : 255.0;
        const double beta
            = (mStimuliProvider->getParameter<bool>("DataSignedMapping"))
                ? 128.0 : 0.0;

#pragma omp parallel for if (size > 4)
        for (int batchPos = 0; batchPos < size; ++batchPos) {
#ifdef CUDA
            CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

            const int id = mStimuliProvider->getBatch()[batchPos];

            if (id < 0) {
                // Invalid stimulus in batch (can occur for the last batch of
                // the set)
                continue;
            }

            std::string fileName;

            if (validDatabase) {
                const std::string imgFile
                    = mStimuliProvider->getDatabase().getStimulusName(id);
                const std::string baseName = Utils::baseName(imgFile);
                const std::string fileBaseName = Utils::fileBaseName(baseName);

                // Input image
                if (values.dimX() == 1 && values.dimY() == 1) {
                    fileName = dirPath + "/" + fileBaseName + "_target.dat";

                    StimuliProvider::logData(fileName,
                        mStimuliProvider->getTargetData()[batchPos]);

                    fileName = dirPath + "/" + fileBaseName + "_estimated.dat";
                }
                else {
                    std::string fileExtension = Utils::fileExtension(baseName);

                    if (!((std::string)mImageLogFormat).empty()) {
                        // Keep "[x,y]" after file extension, appended by
                        // getStimulusName() in case of slicing
                        fileExtension.replace(0, fileExtension.find_first_of('['),
                                            mImageLogFormat);
                    }

                    cv::Mat inputImg = (cv::Mat)mStimuliProvider->getTargetDataChannel(0, batchPos);
                    cv::Mat inputImg8U;
                    inputImg.convertTo(inputImg8U, CV_8U, alpha, beta);

                    fileName = dirPath + "/" + fileBaseName + "_target."
                                    + fileExtension;

                    if (!cv::imwrite(fileName, inputImg8U)) {
#pragma omp critical(Target__logEstimatedLabels)
                        throw std::runtime_error("Unable to write image: " + fileName);
                    }

                    fileName = dirPath + "/" + fileBaseName + "_estimated."
                            + fileExtension;
                }
            }
            else {
                std::ostringstream imgFile;
                imgFile << std::setw(10) << std::setfill('0') << id;

                const std::string fileExtension
                    = (!((std::string)mImageLogFormat).empty())
                        ? (std::string)mImageLogFormat
                        : std::string("jpg");

                fileName = dirPath + "/" + imgFile.str() + "."
                                        + fileExtension;
            }

            // Output image
            if (values.dimX() == 1 && values.dimY() == 1) {
                StimuliProvider::logData(fileName, values[batchPos]);
            }
            else {
                const cv::Mat outputImg = (cv::Mat)values[batchPos][0];
                cv::Mat outputImg8U;
                outputImg.convertTo(outputImg8U, CV_8U, alpha, beta);

                if (!cv::imwrite(fileName, outputImg8U)) {
#pragma omp critical(Target__logEstimatedLabels)
                    throw std::runtime_error("Unable to write image: " + fileName);
                }
            }
        }

        return;
    }

#if !defined(WIN32) && !defined(__CYGWIN__) && !defined(_WIN32)
    const int ret = symlink(N2D2_PATH("tools/target_viewer.py"),
                            (dirPath + ".py").c_str());
    if (ret < 0) {
    } // avoid ignoring return value warning
#endif

    const unsigned int nbTargets = getNbTargets();

    estimatedLabels.synchronizeDBasedToH();
    estimatedLabelsValue.synchronizeDBasedToH();

#pragma omp parallel for if (targets.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)targets.dimB(); ++batchPos) {
#ifdef CUDA
        CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        const Tensor<int> target = targets[batchPos][0];
        const Tensor<int> estLabels = estimatedLabels[batchPos][0];
        const Tensor<Float_T> estLabelsValue
            = estimatedLabelsValue[batchPos][0];

        cv::Mat targetImgHsv;

        if (validDatabase) {
            targetImgHsv = cv::Mat(cv::Size(targets.dimX(), targets.dimY()),
                                CV_8UC3,
                                cv::Scalar(0, 0, 0));
        }

        cv::Mat estimatedImgHsv(cv::Size(targets.dimX(), targets.dimY()),
                                CV_8UC3,
                                cv::Scalar(0, 0, 0));

        const TensorLabels_T mask = (mMaskLabelTarget && mMaskedLabel >= 0)
            ? mMaskLabelTarget->getEstimatedLabels()[batchPos][0]
            : TensorLabels_T();

        if (!mask.empty() && mask.dims() != target.dims()) {
            std::ostringstream errorStr;
            errorStr << "Mask dims (" << mask.dims() << ") from MaskLabelTarget"
                " does not match target dims (" << target.dims() << ") for"
                " target \"" << mName << "\"";

#pragma omp critical(Target__logEstimatedLabels)
            throw std::runtime_error(errorStr.str());
        }

        for (unsigned int oy = 0; oy < targets.dimY(); ++oy) {
            for (unsigned int ox = 0; ox < targets.dimX(); ++ox) {
                if (validDatabase) {
                    const int targetHue = (180 * target(ox, oy) / nbTargets
                                        + mLabelsHueOffset) % 180;

                    targetImgHsv.at<cv::Vec3b>(oy, ox)
                        = (target(ox, oy) >= 0)
                            ? ((target(ox, oy) != mNoDisplayLabel)
                                ? cv::Vec3f(targetHue, 255, 255)
                                : cv::Vec3f(targetHue, 10, 127)) // no color
                            : cv::Vec3f(0, 0, 127); // ignore = no color
                }

                const int estimatedHue = (180 * estLabels(ox, oy)
                                          / nbTargets + mLabelsHueOffset) % 180;

                estimatedImgHsv.at<cv::Vec3b>(oy, ox)
                    = ((mask.empty() || mask(ox, oy) == mMaskedLabel)
                        && (!(mValueThreshold > 0.0)
                            || estLabelsValue(ox, oy) >= mValueThreshold))
                        ? ((estLabels(ox, oy) != mNoDisplayLabel)
                            ? cv::Vec3f(estimatedHue, 255,
                                       (mEstimatedLabelsValueDisplay)
                                           ? 255 * estLabelsValue(ox, oy)
                                           : 255)
                            : cv::Vec3f(estimatedHue, 10, 127)) // no color
                        : cv::Vec3f(0, 0, 127); // not masked = no color
            }
        }

        const double alpha = 0.75;

        // Input image
        cv::Mat inputImg = (cv::Mat)mStimuliProvider->getDataChannel(0, batchPos);
        cv::Mat inputImg8U;
        // inputImg.convertTo(inputImg8U, CV_8U, 255.0);

        // Normalize image
        cv::Mat inputImgNorm;
        cv::normalize(
            inputImg.reshape(1), inputImgNorm, 0, 255, cv::NORM_MINMAX);
        inputImg = inputImgNorm.reshape(inputImg.channels());
        inputImg.convertTo(inputImg8U, CV_8U);

        cv::Mat inputImgColor;
#if CV_MAJOR_VERSION >= 3
        cv::cvtColor(inputImg8U, inputImgColor, cv::COLOR_GRAY2BGR);
#else
        cv::cvtColor(inputImg8U, inputImgColor, CV_GRAY2BGR);
#endif

        std::string fileName;
        cv::Mat imgColor, imgSampled, imgBlended;

        if (validDatabase) {
            const std::string imgFile
                = mStimuliProvider->getDatabase().getStimulusName(id);
            const std::string baseName = Utils::baseName(imgFile);
            const std::string fileBaseName = Utils::fileBaseName(baseName);
            std::string fileExtension = Utils::fileExtension(baseName);

            if (!((std::string)mImageLogFormat).empty()) {
                // Keep "[x,y]" after file extension, appended by
                // getStimulusName() in case of slicing
                fileExtension.replace(0, fileExtension.find_first_of('['),
                                    mImageLogFormat);
            }

            // Target image
#if CV_MAJOR_VERSION >= 3
            cv::cvtColor(targetImgHsv, imgColor, cv::COLOR_HSV2BGR);
#else
            cv::cvtColor(targetImgHsv, imgColor, CV_HSV2BGR);
#endif

            cv::resize(imgColor,
                    imgSampled,
                    cv::Size(mStimuliProvider->getSizeX(),
                                mStimuliProvider->getSizeY()),
                    0.0,
                    0.0,
                    cv::INTER_NEAREST);

            cv::addWeighted(
                inputImgColor, alpha, imgSampled, 1 - alpha, 0.0, imgBlended);

            fileName = dirPath + "/" + fileBaseName + "_target."
                                + fileExtension;

            if (!cv::imwrite(fileName, imgBlended)) {
#pragma omp critical(Target__logEstimatedLabels)
                throw std::runtime_error("Unable to write image: " + fileName);
            }

            fileName = dirPath + "/" + fileBaseName + "_estimated."
                + fileExtension;
        }
        else {
            std::ostringstream imgFile;
            imgFile << std::setw(10) << std::setfill('0') << id;

            const std::string fileExtension
                = (!((std::string)mImageLogFormat).empty())
                    ? (std::string)mImageLogFormat
                    : std::string("jpg");

            fileName = dirPath + "/" + imgFile.str() + "." + fileExtension;
        }

        // Estimated image
#if CV_MAJOR_VERSION >= 3
        cv::cvtColor(estimatedImgHsv, imgColor, cv::COLOR_HSV2BGR);
#else
        cv::cvtColor(estimatedImgHsv, imgColor, CV_HSV2BGR);
#endif

        cv::resize(imgColor,
                   imgSampled,
                   cv::Size(mStimuliProvider->getSizeX(),
                            mStimuliProvider->getSizeY()),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);

        cv::addWeighted(
            inputImgColor, alpha, imgSampled, 1 - alpha, 0.0, imgBlended);

        if (!cv::imwrite(fileName, imgBlended)) {
#pragma omp critical(Target__logEstimatedLabels)
            throw std::runtime_error("Unable to write image: " + fileName);
        }
    }
}

void N2D2::Target::logEstimatedLabelsJSON(const std::string& dirName,
                                          const std::string& fileName,
                                          unsigned int xOffset,
                                          unsigned int yOffset,
                                          bool append) const
{
    const std::string dirPath = mName + "/" + dirName;
    Utils::createDirectories(dirPath);

    int dev = 0;
#ifdef CUDA
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    const Tensor<int>& targets = mTargetData[dev].targets;
    const TensorLabels_T& estimatedLabels = mTargetData[dev].estimatedLabels;
    const TensorLabelsValue_T& estimatedLabelsValue = mTargetData[dev].estimatedLabelsValue;

    if (targets.dimX() == 1 && targets.dimY() == 1)
        return;

    if (mDataAsTarget)
        return;

    const std::vector<std::string>& labelsName = getTargetLabelsName();
    const bool validDatabase
        = (mStimuliProvider->getDatabase().getNbStimuli() > 0);
    const double xRatio = mStimuliProvider->getSizeX()
        / (double)estimatedLabels.dimX();
    const double yRatio = mStimuliProvider->getSizeY()
        / (double)estimatedLabels.dimY();
    const int scale = xRatio;

    if (xRatio != yRatio || xRatio != scale) {
        std::cout << Utils::cwarning << "Target::logEstimatedLabelsJSON(): "
            "x-ratio (" << xRatio << ") and y-ratio (" << yRatio << ") do not "
            "match and/or are not integers" << Utils::cdef << std::endl;
    }

    //const bool signedMapping
    //    = mStimuliProvider->getParameter<bool>("DataSignedMapping");

    estimatedLabels.synchronizeDBasedToH();

    if (mValueThreshold > 0.0)
        estimatedLabelsValue.synchronizeDBasedToH();

    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);
    std::string time = std::asctime(localNow);
    time.pop_back(); // remove \n introduced by std::asctime()

#ifdef _OPENMP
    omp_lock_t appendLock;
    omp_init_lock(&appendLock);
#endif

#pragma omp parallel for if (targets.dimB() > 4) schedule(dynamic)
    for (int batchPos = 0; batchPos < (int)targets.dimB(); ++batchPos) {
#ifdef CUDA
        CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        const Tensor<int> target = targets[batchPos][0];
        const Tensor<int> estLabels = estimatedLabels[batchPos][0];
        const Tensor<Float_T> estLabelsValue
            = estimatedLabelsValue[batchPos][0];

        const TensorLabels_T mask = (mMaskLabelTarget && mMaskedLabel >= 0)
            ? mMaskLabelTarget->getEstimatedLabels()[batchPos][0]
            : TensorLabels_T();

        mask.synchronizeDBasedToH();

        if (!mask.empty() && mask.dims() != target.dims()) {
            std::ostringstream errorStr;
            errorStr << "Mask dims (" << mask.dims() << ") from MaskLabelTarget"
                " does not match target dims (" << target.dims() << ") for"
                " target \"" << mName << "\"";

#pragma omp critical(Target__logEstimatedLabelsJSON)
            throw std::runtime_error(errorStr.str());
        }

        std::map<int, Tensor<bool> > estimatedBitmaps;

        for (unsigned int oy = 0; oy < targets.dimY(); ++oy) {
            for (unsigned int ox = 0; ox < targets.dimX(); ++ox) {
                if (estLabels(ox, oy) != mNoDisplayLabel
                    && (mask.empty() || mask(ox, oy) == mMaskedLabel)
                    && (!(mValueThreshold > 0.0)
                        || estLabelsValue(ox, oy) >= mValueThreshold))
                {
                    std::map<int, Tensor<bool> >::iterator itBitmap;
                    std::tie(itBitmap, std::ignore) = estimatedBitmaps.insert(
                        std::make_pair(estLabels(ox, oy),
                                    Tensor<bool>(estLabels.dims(), false)));

                    (*itBitmap).second(ox, oy) = true;
                }
            }
        }

        if (estimatedBitmaps.empty())
            continue;

        std::string jsonName(fileName);

        if (jsonName.empty()) {
            std::ostringstream imgFile;

            if (validDatabase) {
                imgFile << mStimuliProvider->getDatabase()
                                        .getStimulusName(id, false);

                const std::string baseName = Utils::baseName(imgFile.str());
                const std::string fileBaseName = Utils::fileBaseName(baseName);
                std::string fileExtension = Utils::fileExtension(baseName);
/*
                if (!((std::string)mImageLogFormat).empty()) {
                    // Keep "[x,y]" after file extension, appended by
                    // getStimulusName() in case of slicing
                    fileExtension.replace(0, fileExtension.find_first_of('['),
                                        mImageLogFormat);
                }
*/
                jsonName = dirPath + "/" + fileBaseName + "." + fileExtension;
            }
            else {
                imgFile << std::setw(10) << std::setfill('0') << id;

                const std::string fileExtension
                    = (!((std::string)mImageLogFormat).empty())
                        ? (std::string)mImageLogFormat
                        : std::string("jpg");

                jsonName = dirPath + "/" + imgFile.str() + "." + fileExtension;

            }
        }
/*
        // Input image
        cv::Mat inputImg = (cv::Mat)mStimuliProvider->getData(0, batchPos);
        cv::Mat inputImg8U;
        inputImg.convertTo(inputImg8U, CV_8U, 255.0, (signedMapping) ? 127.5 : 0.0);

        // Normalize image
        //cv::Mat inputImgNorm;
        //cv::normalize(
        //    inputImg.reshape(1), inputImgNorm, 0, 255, cv::NORM_MINMAX);
        //inputImg = inputImgNorm.reshape(inputImg.channels());
        //inputImg.convertTo(inputImg8U, CV_8U);

        if (!cv::imwrite(fileName, inputImg8U)) {
#pragma omp critical(Target__logEstimatedLabelsJSON)
            throw std::runtime_error("Unable to write image: " + fileName);
        }
*/
        jsonName += ".json";

        std::ostringstream jsonDataBuffer;

        for (std::map<int, Tensor<bool> >::const_iterator it
            = estimatedBitmaps.begin(), itEnd = estimatedBitmaps.end();
            it != itEnd; ++it)
        {
            if (it != estimatedBitmaps.begin())
                jsonDataBuffer << ",";

            unsigned int xSliceOffset = 0;
            unsigned int ySliceOffset = 0;

            if (validDatabase) {
                const ROI* slice
                    = mStimuliProvider->getDatabase().getStimulusSlice(id);

                if (slice != NULL) {
                    const cv::Rect bbRect = slice->getBoundingRect();
                    xSliceOffset = bbRect.x;
                    ySliceOffset = bbRect.y;

                    // If there is multiple slices, append MUST be true
                    // because 2nd argument of getStimulusName() is true
                    append = true;
                }
            }

            jsonDataBuffer << "{\"class_id\": " << (*it).first << ","
                "\"class_name\": \"" << labelsName[(*it).first] << "\","
                "\"info\": [\"BITMAP_CLASS_" << (*it).first << "\","
                    "false,"
                    "{\"CreationDate\": \"" << time << "\","
                        "\"Source\": \"N2D2\"}"
                "],"
                "\"type\": \"pixelwise\","
                "\"origin\": [" << xSliceOffset + xOffset << ", "
                    << ySliceOffset + yOffset << "],"
                "\"scale\": " << ((scale > 1) ? (-scale) : scale) << ","
                "\"size\": [" << (*it).second.dimX() << ","
                    << (*it).second.dimY() << "],"
                "\"data\": [";

            unsigned int p = 0;
            unsigned int c0 = 0;
            unsigned int c255 = 0;

            while (p < (*it).second.size()) {
                if (p > 0)
                    jsonDataBuffer << ",";

                while (p < (*it).second.size() && !(*it).second(p)) {
                    ++p;
                    ++c0;
                }

                while (p < (*it).second.size() && (*it).second(p)) {
                    ++p;
                    ++c255;
                }

                jsonDataBuffer << c0 << "," << c255;

                c0 = 0;
                c255 = 0;
            }

            jsonDataBuffer << "]}";
        }

        jsonDataBuffer << "]}";

#ifdef _OPENMP
        if (append && omp_in_parallel())
            omp_set_lock(&appendLock);
#endif
        std::fstream jsonData;
        bool newFile = true;

        if (append) {
            newFile = false;
            jsonData.open(jsonName.c_str(),
                          std::ofstream::in | std::ofstream::out);
        }
        else
            jsonData.open(jsonName.c_str(), std::ofstream::out);

        if (append && !jsonData.good()) {
            newFile = true;
            jsonData.open(jsonName.c_str(),
                          std::ofstream::in | std::ofstream::out | std::ofstream::app);
        }

        //std::ofstream jsonData(jsonName.c_str(),
        //    (append) ? std::fstream::app
        //             : std::fstream::out);

        if (!jsonData.good()) {
#pragma omp critical(Target__logEstimatedLabelsJSON)
            throw std::runtime_error("Could not create JSON file: " + jsonName);
        }

        if (newFile)
            jsonData << "{\"annotations\": [" << jsonDataBuffer.str();
        else {
            jsonData.seekp(-2, jsonData.end); // Go before "]}"
            jsonData.write(",", sizeof(char));
            jsonData.write(jsonDataBuffer.str().c_str(),
                           sizeof(char) * jsonDataBuffer.str().size());
        }

#ifdef _OPENMP
        if (append && omp_in_parallel())
            omp_unset_lock(&appendLock);
#endif
    }

#ifdef _OPENMP
    omp_destroy_lock(&appendLock);
#endif
}

void N2D2::Target::logLabelsLegend(const std::string& fileName) const
{
    if (mDataAsTarget)
        return;

    if (mCell->getOutputsWidth() == 1 && mCell->getOutputsHeight() == 1)
        return;

    // Legend image
    const unsigned int margin = 5;
    const unsigned int labelWidth = 300;
    const unsigned int cellWidth = 50;
    const unsigned int cellHeight = 50;

    const unsigned int nbTargets = getNbTargets();
    const std::vector<std::string>& labelsName = getTargetLabelsName();

    cv::Mat legendImg(cv::Size(cellWidth + labelWidth, cellHeight * nbTargets),
                      CV_8UC3,
                      cv::Scalar(0, 0, 0));

    for (unsigned int target = 0; target < nbTargets; ++target) {
        cv::rectangle(
            legendImg,
            cv::Point(margin, target * cellHeight + margin),
            cv::Point(cellWidth - margin, (target + 1) * cellHeight - margin),
            cv::Scalar((180 * target / nbTargets + mLabelsHueOffset) % 180,
                        255, 255),
#if CV_MAJOR_VERSION >= 3
            cv::FILLED);
#else
            CV_FILLED);
#endif

        std::stringstream legendStr;
        legendStr << target << " " << labelsName[target];

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            legendStr.str(), cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        cv::putText(legendImg,
                    legendStr.str(),
                    cv::Point(cellWidth + margin,
                              (target + 1) * cellHeight
                              - (cellHeight - textSize.height) / 2.0),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 0, 255),
                    2);
    }

    cv::Mat imgColor;
#if CV_MAJOR_VERSION >= 3
    cv::cvtColor(legendImg, imgColor, cv::COLOR_HSV2BGR);
#else
    cv::cvtColor(legendImg, imgColor, CV_HSV2BGR);
#endif

    if (!cv::imwrite(mName + "/" + fileName, imgColor))
        throw std::runtime_error("Unable to write image: " + mName + "/"
                                 + fileName);
}

N2D2::Target::TensorLabelsValue_T
N2D2::Target::getEstimatedLabels(const std::shared_ptr<ROI>& roi,
                                 unsigned int batchPos,
#ifdef CUDA
                                Float_T* values) const
#else
                                Float_T* /*values*/) const
#endif
{
    const Tensor<int>& labels = mStimuliProvider->getLabelsData();
    const double xRatio = labels.dimX() / (double)mCell->getOutputsWidth();
    const double yRatio = labels.dimY() / (double)mCell->getOutputsHeight();

    const cv::Rect rect = roi->getBoundingRect();
    // We should get back the coordinates from TargetROIs::process()
    // ( (*it).j0), (*it).i0, (*it).j1 + 1, (*it).i1 + 1 )
    // If xRatio and yRatio are integer, no problem
    // Otherwise, we hope that round() will give the correct result.
    // It works on simple cases, but I don't have a general demonstration.
    // Alternatively, we could provide the *output* coordinates to 
    // getEstimatedLabel(), but that would break existing demo code.
    const unsigned int x0 = Utils::round(rect.tl().x / xRatio);
    const unsigned int y0 = Utils::round(rect.tl().y / yRatio);
    const unsigned int x1 = Utils::round(rect.br().x / xRatio);
    const unsigned int y1 = Utils::round(rect.br().y / yRatio);
    const unsigned int size = (x1 - x0) * (y1 - y0);

    if (size == 0) {
#pragma omp critical(Target__getEstimatedLabel)
        throw std::runtime_error(
            "Target::getEstimatedLabel(): bounding box is empty");
    }

    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);
    std::shared_ptr<Cell_CSpike_Top> targetCellCSpike
        = std::dynamic_pointer_cast<Cell_CSpike_Top>(mCell);

    BaseTensor& outputsBaseTensor = (targetCell)
        ? targetCell->getOutputs()
        : targetCellCSpike->getOutputsActivity();

    if (x1 > outputsBaseTensor.dimX() || y1 > outputsBaseTensor.dimY()) {
#pragma omp critical(Target__getEstimatedLabel)
        throw std::runtime_error(
            "Target::getEstimatedLabel(): bounding box out of range");
    }

    const unsigned int nbOutputs = outputsBaseTensor.dimZ();
    TensorLabelsValue_T bbLabels;
    bbLabels.resize({(nbOutputs > 1) ? nbOutputs : 2}, 0.0);

    const TensorLabels_T mask = (mMaskLabelTarget && mMaskedLabel >= 0)
        ? mMaskLabelTarget->getEstimatedLabels()[batchPos][0]
        : TensorLabels_T();
    const TensorLabelsValue_T maskValue = (mMaskLabelTarget && mMaskedLabel >= 0
                                            && mMaskedLabelValue)
        ? mMaskLabelTarget->getEstimatedLabelsValue()[batchPos][0]
        : TensorLabelsValue_T();

    if (!mask.empty() && (mask.dimX() != outputsBaseTensor.dimX()
                         || mask.dimY() != outputsBaseTensor.dimY()))
    {
        std::ostringstream errorStr;
        errorStr << "Mask dims (" << mask.dims() << ") from MaskLabelTarget"
            " does not match target dims (" << outputsBaseTensor.dims() << ")"
            " for target \"" << mName << "\"";

#pragma omp critical(Target__getEstimatedLabel)
        throw std::runtime_error(errorStr.str());
    }

#ifdef CUDA
    CudaBaseTensor* outputsCudaBaseTensor 
            = dynamic_cast<CudaBaseTensor*>(&outputsBaseTensor);

    if (outputsCudaBaseTensor != NULL) {
        std::shared_ptr<CudaDeviceTensor<Float_T> > value
            = cuda_device_tensor_cast_nocopy<Float_T>(
                *outputsCudaBaseTensor);

        cudaGetEstimatedLabel( CudaContext::getDeviceProp(),
                                values == NULL ? value->getDevicePtr() : values,
                                outputsBaseTensor.dimX(),
                                outputsBaseTensor.dimY(),
                                nbOutputs,
                                batchPos,
                                x0,
                                x1,
                                y0,
                                y1,
                                bbLabels.getDevicePtr(),
                                (!mask.empty()) ? mask.getDevicePtr() : NULL,
                                mMaskedLabel,
                                (!maskValue.empty()) ? maskValue.getDevicePtr()
                                                     : NULL);
        
        bbLabels.synchronizeDToH();
    }
    else {
#endif
        // Sync. already done in process(), cast also
        const Tensor<Float_T>& value
            = tensor_cast_nocopy<Float_T>(outputsBaseTensor);
        const unsigned int dimZ = (nbOutputs > 1) ? nbOutputs : 2;

        for (unsigned int oy = y0; oy < y1; ++oy) {
            for (unsigned int ox = x0; ox < x1; ++ox) {
                if (mask.empty() || mask(ox, oy) == mMaskedLabel) {
                    for (unsigned int z = 0; z < dimZ; ++z) {
                        float val = (nbOutputs > 1 || z > 0)
                            ? value(ox, oy, z * (nbOutputs > 1))
                            // nbOutputs == 1 && z == 0
                            : 1.0f - value(ox, oy, z * (nbOutputs > 1));

                        if (mMaskedLabelValue)
                            val *= maskValue(ox, oy);

                        bbLabels(z) += val;
                    }
                }
            }
        }
#ifdef CUDA
    }
#endif

    return bbLabels;
}

std::pair<int, N2D2::Float_T>
N2D2::Target::getEstimatedLabel(const std::shared_ptr<ROI>& roi,
                                unsigned int batchPos,
                                Float_T* values) const
{
    const TensorLabelsValue_T bbLabels = getEstimatedLabels(roi, batchPos, values);

    const std::vector<Float_T>::const_iterator it
        = std::max_element(bbLabels.begin(), bbLabels.end());
    return std::make_pair(it - bbLabels.begin(), (*it)/* / size*/);
}

std::vector<N2D2::Float_T> N2D2::Target::getLoss() const
{
    if (mLoss.size() == 1)
        return mLoss[0];
    else {
        std::vector<Float_T> loss;

        for (int dev = 0; dev < (int)mLoss.size(); ++dev) {
            if (!mLoss[dev].empty()) {
                if (loss.empty())
                    loss = mLoss[dev];
                else {
                    std::transform(mLoss[dev].begin(), mLoss[dev].end(),
                                    loss.begin(), loss.begin(),
                                    std::plus<Float_T>());
                }
            }
        }

        return loss;
    }
}

void N2D2::Target::clear(Database::StimuliSet /*set*/)
{
    for (int dev = 0; dev < (int)mLoss.size(); ++dev)
        mLoss[dev].clear();
}
