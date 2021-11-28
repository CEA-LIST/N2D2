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

#ifndef N2D2_DATABASE_H
#define N2D2_DATABASE_H

#include <algorithm>
#include <string>
#include <vector>

#include <dirent.h>
// For the Windows version of dirent.h (http://www.softagalleria.net/dirent.php)
#undef min
#undef max

#ifdef OPENCV_USE_OLD_HEADERS       //  before OpenCV 2.2.0
    #include "cv.h"
    #include "highgui.h"
#else
    #include "opencv2/core/version.hpp"
    #if CV_MAJOR_VERSION == 2
        #include "opencv2/core/core.hpp"
        #include "opencv2/imgproc/imgproc.hpp"
        #include "opencv2/highgui/highgui.hpp"
    #elif CV_MAJOR_VERSION >= 3
        #include "opencv2/core.hpp"
        #include "opencv2/imgproc.hpp"
        #include "opencv2/highgui.hpp"
    #endif
#endif

#include "Transformation/CompositeTransformation.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Utils.hpp"
#include "utils/Registrar.hpp"
#include "DataFile/DataFile.hpp"

namespace N2D2 {

class ROI;

/**
 * Database specifications:
 * - Genericity: load image and sound, 1D, 2D or 3D data
 * - Associate a label for each data point or global to the stimulus, 1D or 2D
 * labels
 * - ROIs handling:
 *   + Convert ROIs to data point labels
 *   + Extract one or multiple ROIs from an initial dataset to create as many
 * corresponding stimuli
*/
class Database : public Parameterizable, public std::enable_shared_from_this<Database> {
public:
    /// Each stimulus in the database has a unique StimulusID, like the ID field
    /// of a relational database.
    /// StimulusID should not be confused to the stimulus index in a dataset
    /// (StimuliSet), which is attributed during
    /// partitioning.
    typedef unsigned int StimulusID;

    /**
     * The Stimulus object contains the in-memory fields associated to each
     * stimulus.
    */
    struct Stimulus {
        /// Stimulus name
        std::string name;
        /// Label associated to the stimulus
        /// If the label is equal to -1, it means the stimulus is composed of
        /// several labels, described by ROIs
        int label;
        /// ROIs associated to the stimulus
        std::vector<ROI*> ROIs;
        ROI* slice;

        Stimulus(const std::string& name_,
                 int label_ = -1,
                 const std::vector<ROI*>& ROIs_ = std::vector<ROI*>(),
                 ROI* slice_ = NULL)
            : name(name_), label(label_), ROIs(ROIs_), slice(slice_)
        {
        }
    };

    enum StimuliSet {
        Learn,
        Validation,
        Test,
        Unpartitioned
    };
    enum StimuliSetMask {
        LearnOnly,
        ValidationOnly,
        TestOnly,
        NoLearn,
        NoValidation,
        NoTest,
        All
    };
    enum CompositeLabel {
        None,
        Auto,
        Default,
        Disjoint,
        Combine
    };

    /**
     * This objects contains the list of stimuli in each database stimuli set.
     * There are 3 different stimuli sets possible;
     * - learn: the set used for the learning
     * - validation: the validation set
     * - test: the set for the testing
     * A fourth set is the unpartitioned set, which should not be used outside
     * the Database class. When stimuli are loaded into
     * the database, they belong to the unpartitioned set by default, until
     * partitioning is done using one of the partitioning
     * functions.
    */
    struct StimuliSets {
        // Use std::vector instead of std::set for faster random access (O(1) vs
        // O(n))
        std::vector<StimulusID> learn;
        std::vector<StimulusID> validation;
        std::vector<StimulusID> test;
        std::vector<StimulusID> unpartitioned;

        /// Operator to get a set by type
        inline std::vector<StimulusID>& operator()(StimuliSet set);
        inline const std::vector<StimulusID>& operator()(StimuliSet set) const;
    };

    Database(bool loadDataInMemory = false);
    virtual void loadROIs(const std::string& fileName,
                          const std::string& relPath = "",
                          bool noImageSize = false);
    virtual void loadROIsDir(const std::string& dirName,
                             const std::vector<std::string>& fileExt
                                                = std::vector<std::string>(),
                             int depth = 0);
    virtual void saveROIs(const std::string& fileName,
                          const std::string& header = "") const;
    void logPartition(const std::string& dirName) const;
    void logStats(const std::string& sizeFileName,
                  const std::string& labelFileName,
                  StimuliSetMask setMask = All) const;
    void logROIsStats(const std::string& sizeFileName,
                      const std::string& labelFileName,
                      StimuliSetMask setMask = All) const;
    void logMultiChannelStats(const std::string& fileName,
                              StimuliSetMask setMask = All) const;

    /**
     * Extract all the ROIs in stimuli as new stimuli and remove stimuli with no
     *ROI.
     * The new stimuli labels inherit from the ROI labels.
     *
     * Output:
     * - All stimuli have a valid label
     * - All stimuli have one and only one ROI
    */
    virtual void extractROIs();

    /**
     * Remove some ROIs label and optionally remove the associated stimuli.
     *
     * @param labels            List of ROI labels to filter
     * @param filterKeep        If true, keep only the labels in the list,
     *                          else, remove all the labels in the list
     * @param removeStimuli     If true, remove stimuli without remaining label
    */
    virtual void filterROIs(const std::vector<int>& labels,
                            bool filterKeep = true,
                            bool removeStimuli = true);
    inline void filterROIs(const std::vector<std::string>& names,
                           bool filterKeep = true,
                           bool removeStimuli = true);

    /**
     * When each stimulus has one and only one ROI, set the stimulus label to
     *the ROI label
     *
     * @param removeROIs        Remove ROIs after label extraction (in this
     *case, getStimulusData() will extract the whole
     *                          image instead of only the ROI)
    */
    virtual void extractLabels(bool removeROIs = false);

    /**
     * Extract slices from stimuli as as many new stimuli.
    */
    virtual void extractSlices(unsigned int width,
                               unsigned int height,
                               unsigned int strideX,
                               unsigned int strideY,
                               StimuliSetMask setMask,
                               bool randomShuffle = false,
                               bool overlapping = false);

    virtual void load(const std::string& /*dataPath*/,
                      const std::string& labelPath = "",
                      bool /*extractROIs*/ = false);
    virtual void save(const std::string& dataPath,
                      StimuliSetMask setMask,
                      CompositeTransformation trans
                        = CompositeTransformation(),
                      bool subDirPerClass = true);
    void append(const Database& database);
    void partitionStimulus(StimulusID id, StimuliSet set);
    void partitionStimuli(unsigned int nbStimuli, StimuliSet set);
    void partitionStimuli(double learn, double validation, double test);
    void partitionStimuliPerLabel(unsigned int nbStimuliPerLabel,
                                  StimuliSet set);
    void partitionStimuliPerLabel(double learnPerLabel,
                                  double validationPerLabel,
                                  double testPerLabel,
                                  bool equiLabel = false);

    /**
     * Returns the total number of loaded stimuli.
     *
     * @return Number of stimuli
    */
    inline unsigned int getNbStimuli() const;
    bool empty() const;

    /**
     * Returns the number of stimuli in one stimuli set.
     *
     * @param set           Set of stimuli
     * @return Number of stimuli in the set
    */
    inline unsigned int getNbStimuli(StimuliSet set) const;
    unsigned int getNbStimuliWithLabel(int label,
                                       StimuliSetMask setMask = All) const;
    inline unsigned int getNbStimuliWithLabel(const std::string& labelName,
                                            StimuliSetMask setMask = All) const;
    inline unsigned int getNbLabels() const;
    void addStimulus(const std::string& name,
                     const std::string& labelName,
                     StimuliSet set = Unpartitioned);
    void addStimulus(const std::string& name,
                     int label = -1,
                     StimuliSet set = Unpartitioned);
    int addLabel(const std::string& labelName);
    void removeStimulus(StimulusID id);
    void removeStimuli(const std::vector<StimulusID>& ids);
    void removeLabel(int label);
    inline void removeLabel(const std::string& labelName);
    void removeLabels(const std::vector<int>& labels);
    // void mergeLabels(const std::vector<int>& labels, const std::string&
    // newName = "");
    // inline void mergeLabels(const std::vector<std::string>& names, const
    // std::string& newName = "");

    void sortAndDropLabels(unsigned int nbKeep,
                           bool ascending = true,
                           StimuliSetMask setMask = All);

    // Setters
    inline void setStimulusLabel(StimulusID id, const std::string& labelName);
    inline void setStimulusROIs(StimulusID id,
                                const std::vector<ROI*>& ROIs = std::vector
                                <ROI*>());

    // Getters
    bool getLoadDataInMemory(){
        return mLoadDataInMemory;
    };
    
    inline StimulusID getStimulusID(StimuliSet set, unsigned int index) const;
    std::string getStimulusName(StimulusID id, bool appendSlice = true) const;
    inline std::string getStimulusName(StimuliSet set,
                                       unsigned int index,
                                       bool appendSlice = true) const;
    inline int getStimulusLabel(StimulusID id) const;
    inline int getStimulusLabel(StimuliSet set, unsigned int index) const;
    StimuliSet getStimulusSet(StimulusID id) const;
    inline const ROI* getStimulusSlice(StimulusID id) const;
    inline const ROI* getStimulusSlice(StimuliSet set, unsigned int index)
        const;
    //inline const std::vector<ROI*>& getStimulusROIs(StimulusID id) const;
    //inline const std::vector<ROI*>& getStimulusROIs(StimuliSet set,
    //                                                unsigned int index) const;
    std::vector<std::shared_ptr<ROI> > getStimulusROIs(StimulusID id) const;
    inline std::vector<std::shared_ptr<ROI> > getStimulusROIs(StimuliSet set,
        unsigned int index) const;
    unsigned int getNbROIs() const;
    unsigned int getNbROIsWithLabel(int label,
                                    StimuliSetMask setMask = All) const;
    inline unsigned int getNbROIsWithLabel(const std::string& labelName,
                                           StimuliSetMask setMask = All) const;
    bool isLabel(const std::string& labelName) const;
    bool isMatchingLabel(const std::string& labelMask) const;
    int getLabelID(const std::string& labelName) const;
    int getDefaultLabelID() const;
    const std::vector<std::string>& getLabels() const
    {
        return mLabelsName;
    }
    inline std::vector<int> getLabelsIDs(const std::vector
                                         <std::string>& names) const;
    std::vector<int> getMatchingLabelsIDs(const std::string& labelMask) const;
    std::vector<int> getMatchingLabelsIDs(
        const std::vector<std::string>& labelMask) const;
    inline const std::string& getLabelName(int label) const;
    int getStimuliDepth();
    cv::Mat getStimulusData(StimulusID id);
    inline cv::Mat getStimulusData(StimuliSet set, unsigned int index);
    cv::Mat getStimulusLabelsData(StimulusID id);
    inline cv::Mat getStimulusLabelsData(StimuliSet set, unsigned int index);
    virtual cv::Mat getStimulusTargetData(StimulusID id,
                        const cv::Mat& frame = cv::Mat(),
                        const cv::Mat& labels = cv::Mat(),
                        const std::vector<std::shared_ptr<ROI> >& labelsROI
                            = std::vector<std::shared_ptr<ROI> >());
    inline cv::Mat getStimulusTargetData(StimuliSet set,
                        unsigned int index,
                        const cv::Mat& frame = cv::Mat(),
                        const cv::Mat& labels = cv::Mat(),
                        const std::vector<std::shared_ptr<ROI> >& labelsROI
                            = std::vector<std::shared_ptr<ROI> >());
    std::vector<StimuliSet> getStimuliSets(StimuliSetMask setMask) const;
    StimuliSetMask getStimuliSetMask(StimuliSet set) const;
    virtual cv::Mat readLabel(const StimulusID id) { 
        std::string fileExtension = Utils::fileExtension(mStimuli[id].name);
        if(mDataFileLabel && Registrar<DataFile>::exists(fileExtension)){
            std::shared_ptr<DataFile> dataFile = Registrar
            <DataFile>::create(fileExtension)();
            return(dataFile->readLabel(mStimuli[id].name));
        } else {
            return cv::Mat();
        }
    }; 

    virtual ~Database();

protected:
    static const std::locale csvLocale;

    std::map<std::string, StimulusID>
    getRelPathStimuli(const std::string& fileName, const std::string& relPath);
    int labelID(const std::string& labelName);
    cv::Mat loadStimulusData(StimulusID id);
    cv::Mat loadStimulusLabelsData(StimulusID id);
    cv::Mat loadStimulusTargetData(StimulusID id);
    cv::Mat loadData(StimulusID id, int depth, const std::string fileName) const;
    std::vector<unsigned int> getLabelStimuliSetIndexes(int label,
                                                        StimuliSet set) const;
    std::vector<std::vector<unsigned int> >
    getLabelsStimuliSetIndexes(StimuliSet set) const;
    void partitionIndexes(std::vector<unsigned int>& unpartitionedIndexes,
                          std::vector<unsigned int>& partitionedIndexes,
                          unsigned int nbStimuli,
                          StimuliSet set);
    void removeIndexesFromSet(std::vector<unsigned int>& indexes,
                              StimuliSet set);
    void plotStats(
        const std::string& sizeFileName,
        const std::string& labelFileName,
        unsigned int totalCount,
        unsigned int minWidth,
        unsigned int maxWidth,
        unsigned int minHeight,
        unsigned int maxHeight,
        const std::map<std::pair<unsigned int, unsigned int>,
                                                        unsigned int>& sizeStats,
        const std::map<int, unsigned int>& labelStats) const;

    /// Default label for composite image (for areas outside the ROIs). If
    /// empty, no default label is created and label ID is -1
    Parameter<std::string> mDefaultLabel;
    /// Margin around the ROIs, in pixels, with no label (label ID = -1)
    Parameter<unsigned int> mROIsMargin;
    Parameter<bool> mRandomPartitioning;
    // If true, load image label with DataFile::readLabel()
    Parameter<bool> mDataFileLabel;
    // A label is said to be composite when it is not a single labelID for the 
    // stimulus (the stimulus label is a matrix of size > 1).
    // For the same stimulus, different type of labels can be specified,
    // i.e. the labelID, pixel-wise data and/or ROIs.
    // The way these different label types are handled is configured with the
    // mCompositeLabel parameter:
    // - None: only the labelID is used, pixel-wise data are ignored and ROIs 
    //         are loaded but ignored as well by loadStimulusLabelsData().
    // - Auto: the label is only composite when pixel-wise data are present
    //         or the stimulus labelID is -1 (in which case the defaultLabel
    //         is used for the whole label matrix). If the label is composite
    //         ROIs, if present, are applied. Otherwise, a single ROI is
    //         allowed and is automatically extracted when fetching the 
    //         stimulus.
    // - Default: the label is always composite. The labelID is ignored.
    //            If there is no pixel-wise data, the defaultLabel is used.
    //            ROIs, if present, are applied.
    // - Disjoint: the label is always composite.
    //             If there is no pixel-wise data:
    //             - the labelID is used if there is no ROI;
    //             - the defaultLabel is used if there is any ROI.
    //             ROIs, if present, are applied.
    // - Combine: the label is always composite.
    //            If there is no pixel-wise data, the labelID is used.
    //            ROIs, if present, are applied.
    Parameter<CompositeLabel> mCompositeLabel;
    Parameter<std::string> mTargetDataPath;
    Parameter<std::string> mMultiChannelMatch;
    Parameter<std::vector<std::string> > mMultiChannelReplace;

    /**
     * TABLES
    */
    /// Stimuli
    std::vector<Stimulus> mStimuli;
    /// Labels name
    std::vector<std::string> mLabelsName;
    /// Stimuli data
    std::vector<cv::Mat> mStimuliData;
    /// Labels matrix associated to each stimulus
    std::vector<cv::Mat> mStimuliLabelsData;
    /// Stimuli target data
    std::vector<cv::Mat> mStimuliTargetData;
    /// Stimuli sets
    StimuliSets mStimuliSets;

    /// Put data in program memory
    bool mLoadDataInMemory;
    /// Stimuli depth
    int mStimuliDepth;
    /// Stimuli target depth
    int mStimuliTargetDepth;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::Database::StimuliSet>::data[]
    = {"Learn", "Validation", "Test", "Unpartitioned"};
template <>
const char* const EnumStrings<N2D2::Database::StimuliSetMask>::data[]
    = {"LearnOnly",    "ValidationOnly", "TestOnly", "NoLearn",
       "NoValidation", "NoTest",         "All"};
template <>
const char* const EnumStrings<N2D2::Database::CompositeLabel>::data[]
    = {"None", "Auto", "Default", "Disjoint", "Combine"};
}

std::vector<N2D2::Database::StimulusID>& N2D2::Database::StimuliSets::
operator()(StimuliSet set)
{
    return (set == Learn) ? learn : (set == Validation)
                                        ? validation
                                        : (set == Test) ? test : unpartitioned;
}

const std::vector<N2D2::Database::StimulusID>& N2D2::Database::StimuliSets::
operator()(StimuliSet set) const
{
    return (set == Learn) ? learn : (set == Validation)
                                        ? validation
                                        : (set == Test) ? test : unpartitioned;
}

void N2D2::Database::filterROIs(const std::vector<std::string>& names,
                                bool filterKeep,
                                bool removeStimuli)
{
    filterROIs(getLabelsIDs(names), filterKeep, removeStimuli);
}

unsigned int N2D2::Database::getNbStimuli() const
{
    return mStimuli.size();
}

unsigned int N2D2::Database::getNbStimuli(StimuliSet set) const
{
    return mStimuliSets(set).size();
}

unsigned int N2D2::Database::getNbStimuliWithLabel(
    const std::string& labelName,
    StimuliSetMask setMask) const
{
    return getNbStimuliWithLabel(getLabelID(labelName), setMask);
}

unsigned int N2D2::Database::getNbLabels() const
{
    return mLabelsName.size();
}

void N2D2::Database::removeLabel(const std::string& labelName)
{
    removeLabel(getLabelID(labelName));
}
/*
void N2D2::Database::mergeLabels(const std::vector<std::string>& names, const
std::string& newName) {
    mergeLabels(getLabelsIDs(names), newName);
}
*/

void N2D2::Database::setStimulusLabel(StimulusID id,
                                      const std::string& labelName)
{
    assert(id < mStimuli.size());
    mStimuli[id].label = getLabelID(labelName);
}

void N2D2::Database::setStimulusROIs(StimulusID id,
                                     const std::vector<ROI*>& ROIs)
{
    assert(id < mStimuli.size());
    mStimuli[id].ROIs = ROIs;
}

N2D2::Database::StimulusID
N2D2::Database::getStimulusID(StimuliSet set, unsigned int index) const
{
    assert(index < mStimuliSets(set).size());
    return mStimuliSets(set)[index];
}

std::string N2D2::Database::getStimulusName(StimuliSet set,
                                            unsigned int index,
                                            bool appendSlice) const
{
    return getStimulusName(getStimulusID(set, index), appendSlice);
}

int N2D2::Database::getStimulusLabel(StimulusID id) const
{
    assert(id < mStimuli.size());
    return mStimuli[id].label;
}

int N2D2::Database::getStimulusLabel(StimuliSet set, unsigned int index) const
{
    return getStimulusLabel(getStimulusID(set, index));
}

const N2D2::ROI* N2D2::Database::getStimulusSlice(StimulusID id) const {
    return mStimuli[id].slice;
}

const N2D2::ROI* N2D2::Database::getStimulusSlice(StimuliSet set,
                                                  unsigned int index) const
{
    return getStimulusSlice(getStimulusID(set, index));
}

/*
const std::vector<N2D2::ROI*>&
N2D2::Database::getStimulusROIs(StimulusID id) const
{
    return mStimuli[id].ROIs;
}

const std::vector<N2D2::ROI*>&
N2D2::Database::getStimulusROIs(StimuliSet set, unsigned int index) const
{
    return getStimulusROIs(getStimulusID(set, index));
}
*/
std::vector<std::shared_ptr<N2D2::ROI> >
N2D2::Database::getStimulusROIs(StimuliSet set, unsigned int index) const
{
    return getStimulusROIs(getStimulusID(set, index));
}

unsigned int N2D2::Database::getNbROIsWithLabel(
    const std::string& labelName,
    StimuliSetMask setMask) const
{
    return getNbROIsWithLabel(getLabelID(labelName), setMask);
}

std::vector<int> N2D2::Database::getLabelsIDs(const std::vector
                                              <std::string>& names) const
{
    std::vector<int> labels;
    std::transform(
        names.begin(),
        names.end(),
        std::back_inserter(labels),
        std::bind(&Database::getLabelID, this, std::placeholders::_1));
    return labels;
}

const std::string& N2D2::Database::getLabelName(int label) const
{
    if (label < 0 || label >= (int)mLabelsName.size()) {
        std::stringstream msgStr;
        msgStr << "Database::getLabelName(): label ID (" << label
               << ") out of range [";

        if (!mLabelsName.empty())
            msgStr << "0," << (mLabelsName.size() - 1);

        msgStr << "]";

        throw std::domain_error(msgStr.str());
    }

    return mLabelsName[label];
}

cv::Mat N2D2::Database::getStimulusData(StimuliSet set, unsigned int index)
{
    return getStimulusData(getStimulusID(set, index));
}

cv::Mat N2D2::Database::getStimulusLabelsData(StimuliSet set,
                                              unsigned int index)
{
    return getStimulusLabelsData(getStimulusID(set, index));
}

cv::Mat N2D2::Database::getStimulusTargetData(StimuliSet set,
    unsigned int index,
    const cv::Mat& frame,
    const cv::Mat& labels,
    const std::vector<std::shared_ptr<ROI> >& labelsROI)
{
    return getStimulusTargetData(getStimulusID(set, index),
                                 frame, labels, labelsROI);
}

#endif // N2D2_DATABASE_H
