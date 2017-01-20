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

#ifndef N2D2_STIMULIPROVIDER_H
#define N2D2_STIMULIPROVIDER_H

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Database/Database.hpp"
#include "Transformation/CompositeTransformation.hpp"
#include "containers/Tensor3d.hpp"
#include "containers/Tensor4d.hpp"
#include "utils/BinaryCvMat.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Random.hpp"

namespace N2D2 {
typedef float Float_T;

class StimuliProvider : virtual public Parameterizable {
public:
    struct Transformations {
        CompositeTransformation cacheable;
        CompositeTransformation onTheFly;
    };

    struct TransformationsSets {
        Transformations learn;
        Transformations validation;
        Transformations test;

        /// Operator to get a set by type
        inline Transformations& operator()(Database::StimuliSet set);
        inline const Transformations&
        operator()(Database::StimuliSet set) const;
    };

    StimuliProvider(Database& database,
                    unsigned int sizeX,
                    unsigned int sizeY = 1,
                    unsigned int nbChannels = 1,
                    unsigned int batchSize = 1,
                    bool compositeStimuli = false);
    virtual void addChannel(const CompositeTransformation& /*transformation*/);

    /// Add global CACHEABLE transformations, before applying any channel
    /// transformation
    void addTransformation(const CompositeTransformation& transformation,
                           Database::StimuliSetMask setMask = Database::All);

    /// Add global ON-THE-FLY transformations, before applying any channel
    /// transformation
    /// The order of transformations is:
    ///     global CACHEABLE,
    ///     then global ON-THE-FLY
    void
    addOnTheFlyTransformation(const CompositeTransformation& transformation,
                              Database::StimuliSetMask setMask = Database::All);

    /// Add a new channel with a CACHEABLE transformation.
    /// Typically, the transformation can be a ChannelExtractionTransformation
    /// or a FilterTransformation
    /// Note that if there is any global ON-THE-FLY transformation, CACHEABLE
    /// channels transformations will not be cached.
    void addChannelTransformation(const CompositeTransformation& transformation,
                                  Database::StimuliSetMask setMask
                                  = Database::All);

    /// Add a new channel with a ON-THE-FLY transformation
    /// The order of transformations is:
    ///     global CACHEABLE,
    ///     then global ON-THE-FLY,
    ///     then channels CACHEABLE,
    ///     then channels ON-THE-FLY
    void addChannelOnTheFlyTransformation(const CompositeTransformation
                                          & transformation,
                                          Database::StimuliSetMask setMask
                                          = Database::All);

    /// Add a CACHEABLE transformation to an existing channel,
    /// to be executed after the current CACHEABLE channel transformations
    void addChannelTransformation(unsigned int channel,
                                  const CompositeTransformation& transformation,
                                  Database::StimuliSetMask setMask
                                  = Database::All);

    /// Add a ON-THE-FLY transformation to an existing channel,
    /// to be executed after the current ON-THE-FLY channel transformations
    void addChannelOnTheFlyTransformation(unsigned int channel,
                                          const CompositeTransformation
                                          & transformation,
                                          Database::StimuliSetMask setMask
                                          = Database::All);

    /// Add a CACHEABLE transformation to ALL the existing channel,
    /// to be executed after the current CACHEABLE channel transformations for
    /// each channel
    void
    addChannelsTransformation(const CompositeTransformation& transformation,
                              Database::StimuliSetMask setMask = Database::All);

    /// Add a ON-THE-FLY transformation to ALL the existing channel,
    /// to be executed after the current ON-THE-FLY channel transformations for
    /// each channel
    void addChannelsOnTheFlyTransformation(const CompositeTransformation
                                           & transformation,
                                           Database::StimuliSetMask setMask
                                           = Database::All);

    void future();
    void synchronize();

    /// Return a random index from the StimuliSet @p set
    unsigned int getRandomIndex(Database::StimuliSet set);

    /// Return a random StimulusID from the StimuliSet @p set
    Database::StimulusID getRandomID(Database::StimuliSet set);

    /// Read a whole random batch from the StimuliSet @p set, apply all the
    /// transformations and put the results in
    /// mData and mLabelsData
    void readRandomBatch(Database::StimuliSet set);

    /// Read a single random stimulus from the StimuliSet @p set, apply all the
    /// transformations and put the results at batch
    /// position @p batchPos in mData and mLabelsData
    /// @return StimulusID of the randomly chosen stimulus
    Database::StimulusID readRandomStimulus(Database::StimuliSet set,
                                            unsigned int batchPos = 0);

    /// Read a whole batch from the StimuliSet @p set, apply all the
    /// transformations and put the results in
    /// mData and mLabelsData
    void readBatch(Database::StimuliSet set, unsigned int startIndex);

    /// Read the stimulus with StimulusID @p id, apply all the transformations
    /// and put the results at batch
    /// position @p batchPos in mData and mLabelsData
    void readStimulus(Database::StimulusID id,
                      Database::StimuliSet set,
                      unsigned int batchPos = 0);

    /// Read the stimulus with index @p index in StimuliSet @p set, apply all
    /// the transformations and put the results at batch
    /// position @p batchPos in mData and mLabelsData
    Database::StimulusID readStimulus(Database::StimuliSet set,
                                      unsigned int index,
                                      unsigned int batchPos = 0);
    void streamStimulus(const cv::Mat& mat,
                        Database::StimuliSet set,
                        unsigned int batchPos = 0);
    void reverseLabels(const cv::Mat& mat,
                       Database::StimuliSet set,
                       Tensor2d<int>& labels,
                       std::vector<std::shared_ptr<ROI> >& labelsROIs);

    Tensor3d<Float_T> readRawData(Database::StimulusID id) const;
    inline Tensor3d<Float_T> readRawData(Database::StimuliSet set,
                                         unsigned int index) const;

    void setBatchSize(unsigned int batchSize);
    void setCachePath(const std::string& path = "");

    // Getters
    Database& getDatabase()
    {
        return mDatabase;
    };
    const Database& getDatabase() const
    {
        return mDatabase;
    };
    unsigned int getSizeX() const
    {
        return mSizeX;
    };
    unsigned int getSizeY() const
    {
        return mSizeY;
    };
    unsigned int getBatchSize() const
    {
        return mBatchSize;
    };
    bool isCompositeStimuli() const
    {
        return mCompositeStimuli;
    };
    inline unsigned int getNbChannels() const;
    unsigned int getNbTransformations(Database::StimuliSet set) const;
    inline CompositeTransformation& getTransformation(Database::StimuliSet set);
    inline CompositeTransformation&
    getOnTheFlyTransformation(Database::StimuliSet set);
    inline CompositeTransformation&
    getChannelTransformation(unsigned int channel, Database::StimuliSet set);
    inline CompositeTransformation&
    getChannelOnTheFlyTransformation(unsigned int channel,
                                     Database::StimuliSet set);
    const std::vector<int>& getBatch()
    {
        return mBatch;
    };
    Tensor4d<Float_T>& getData()
    {
        return mData;
    };
    Tensor4d<int>& getLabelsData()
    {
        return mLabelsData;
    };
    const Tensor4d<Float_T>& getData() const
    {
        return mData;
    };
    const Tensor4d<int>& getLabelsData() const
    {
        return mLabelsData;
    };
    const std::vector<std::vector<std::shared_ptr<ROI> > >&
    getLabelsROIs() const
    {
        return mLabelsROI;
    };
    const Tensor2d<Float_T> getData(unsigned int channel,
                                    unsigned int batchPos = 0) const;
    const Tensor2d<int> getLabelsData(unsigned int channel,
                                      unsigned int batchPos = 0) const;
    const std::vector<std::shared_ptr<ROI> >&
    getLabelsROIs(unsigned int batchPos = 0) const
    {
        return mLabelsROI[batchPos];
    };
    const std::string& getCachePath() const
    {
        return mCachePath;
    };
    virtual ~StimuliProvider() {};

    static void logData(const std::string& fileName,
                        const Tensor2d<Float_T>& data);
    static void logData(const std::string& fileName,
                        const Tensor3d<Float_T>& data);

protected:
    std::vector<cv::Mat> loadDataCache(const std::string& fileName) const;
    void saveDataCache(const std::string& fileName,
                       const std::vector<cv::Mat>& data) const;

    // Internal variables
    Database& mDatabase;
    unsigned int mSizeX;
    unsigned int mSizeY;
    unsigned int mNbChannels;
    unsigned int mBatchSize;
    bool mCompositeStimuli;
    /// Disk cache path for pre-processed stimuli (no disk cache if empty)
    std::string mCachePath;
    /// Global transformations
    TransformationsSets mTransformations;
    /// Channel transformations
    std::vector<TransformationsSets> mChannelsTransformations;
    /// StimuliID of current batch
    std::vector<int> mBatch;
    std::vector<int> mFutureBatch;
    /// Tensor (x, y, channel, batch)
    Tensor4d<Float_T> mData;
    Tensor4d<Float_T> mFutureData;
    /// Tensor (x, y, channel, batch)
    Tensor4d<int> mLabelsData;
    Tensor4d<int> mFutureLabelsData;
    /// ROIs of current batch
    std::vector<std::vector<std::shared_ptr<ROI> > > mLabelsROI;
    std::vector<std::vector<std::shared_ptr<ROI> > > mFutureLabelsROI;
    bool mFuture;
};
}

N2D2::StimuliProvider::Transformations&
N2D2::StimuliProvider::TransformationsSets::
operator()(Database::StimuliSet set)
{
    return (set == Database::Learn) ? learn : (set == Database::Validation)
                                                  ? validation
                                                  : test;
}

const N2D2::StimuliProvider::Transformations&
N2D2::StimuliProvider::TransformationsSets::
operator()(Database::StimuliSet set) const
{
    return (set == Database::Learn) ? learn : (set == Database::Validation)
                                                  ? validation
                                                  : test;
}

unsigned int N2D2::StimuliProvider::getNbChannels() const
{
    return (mChannelsTransformations.empty()) ? mNbChannels
                                              : mChannelsTransformations.size();
}

N2D2::CompositeTransformation&
N2D2::StimuliProvider::getTransformation(Database::StimuliSet set)
{
    return mTransformations(set).cacheable;
}

N2D2::CompositeTransformation&
N2D2::StimuliProvider::getOnTheFlyTransformation(Database::StimuliSet set)
{
    return mTransformations(set).onTheFly;
}

N2D2::CompositeTransformation&
N2D2::StimuliProvider::getChannelTransformation(unsigned int channel,
                                                Database::StimuliSet set)
{
    return mChannelsTransformations.at(channel)(set).cacheable;
}

N2D2::CompositeTransformation&
N2D2::StimuliProvider::getChannelOnTheFlyTransformation(
    unsigned int channel, Database::StimuliSet set)
{
    return mChannelsTransformations.at(channel)(set).onTheFly;
}

N2D2::Tensor3d<N2D2::Float_T>
N2D2::StimuliProvider::readRawData(Database::StimuliSet set,
                                   unsigned int index) const
{
    return readRawData(mDatabase.getStimulusID(set, index));
}

#endif // N2D2_STIMULIPROVIDER_H
