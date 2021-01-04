/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
		            Johannes THIELE (johannes.thiele@cea.fr)

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
#ifdef CUDA
#include "containers/CudaTensor.hpp"
#else
#include "containers/Tensor.hpp"
#endif
#include "utils/Parameterizable.hpp"
#include "FloatT.hpp"

namespace N2D2 {

class StimuliProvider : virtual public Parameterizable, public std::enable_shared_from_this<StimuliProvider> {
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

#ifdef CUDA
    typedef CudaTensor<Float_T> TensorData_T;
#else
    typedef Tensor<Float_T> TensorData_T;
#endif

    StimuliProvider(Database& database,
                    const std::vector<size_t>& size,
                    unsigned int batchSize = 1,
                    bool compositeStimuli = false);

    StimuliProvider(const StimuliProvider& other) = delete;
    StimuliProvider(StimuliProvider&& other);

    /// Return a partial copy of the StimuliProvider. Only the parameters of the
    /// StimuliProvider are copied, the loaded stimuli data are zero-initialized.
    StimuliProvider cloneParameters() const;

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

    /// Add the transformation so that it is executed last.
    /// If any channel tranformation are present, the transformation will be
    /// added as an ON-THE-FLY transformation to ALL the existing channel.
    /// Otherwise it will be added as a global ON-THE-FLY.
    void addTopTransformation(const CompositeTransformation& transformation,
                              Database::StimuliSetMask setMask = Database::All);

    /// Normalize the stimuli in the [0.0;1.0] or [-1.0;1.0], depending on the signess,
    /// if the results of all the transformations of the StimuliProvider are integers,
    /// do nothing otherwise. The stimuli are normalized by adding extra transformations.
    /// 
    /// 'envCvDepth' is the OpenCV depth of the inputs coming from the environment.
    bool normalizeIntegersStimuli(int envCvDepth);

    void logTransformations(const std::string& fileName) const;


    void future();
    void synchronize();

    /// Return a random index from the StimuliSet @p set
    unsigned int getRandomIndex(Database::StimuliSet set);

    /// Return a random StimulusID from the StimuliSet @p set
    Database::StimulusID getRandomID(Database::StimuliSet set);
    Database::StimulusID getRandomIDWithLabel(Database::StimuliSet set, int label);

    /// Read a whole random batch from the StimuliSet @p set, apply all the
    /// transformations and put the results in
    /// mData and mLabelsData
    virtual void readRandomBatch(Database::StimuliSet set);

//TODO: Required for spiking neural network batch parallelization
/*
    /// Read a whole random batch from the StimuliSet @p set, apply all the
    /// transformations and put the results in
    /// mData and mLabelsData. Save the IDs of the stimuli
    void readRandomBatch(Database::StimuliSet set,
                            std::vector<Database::StimulusID>& Ids);
*/

    /// Read a single random stimulus from the StimuliSet @p set, apply all the
    /// transformations and put the results at batch
    /// position @p batchPos in mData and mLabelsData
    /// @return StimulusID of the randomly chosen stimulus
    Database::StimulusID readRandomStimulus(Database::StimuliSet set,
                                            unsigned int batchPos = 0);

    /// Read a whole batch from the StimuliSet @p set, apply all the
    /// transformations and put the results in
    /// mData and mLabelsData
    virtual void readBatch(Database::StimuliSet set, unsigned int startIndex);
    void streamBatch(int startIndex = -1);

//TODO: Required for spiking neural network batch parallelization
/*
    /// Read a whole batch from the StimuliSet @p set, apply all the
    /// transformations and put the results in
    /// mData and mLabelsData. Save the IDs of the stimuli
    void readBatch(Database::StimuliSet set, unsigned int startIndex,
                             std::vector<Database::StimulusID>& Ids);
*/

    void readStimulusBatch(Database::StimulusID id,
                           Database::StimuliSet set);

    /// Read the stimulus with StimulusID @p id, apply all the transformations
    /// and put the results at batch
    /// position @p batchPos in mData and mLabelsData
    virtual void readStimulus(Database::StimulusID id,
                      Database::StimuliSet set,
                      unsigned int batchPos = 0);

    Database::StimulusID readStimulusBatch(Database::StimuliSet set,
                                           unsigned int index);

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
                       Tensor<int>& labels,
                       std::vector<std::shared_ptr<ROI> >& labelsROIs);

    Tensor<Float_T> readRawData(Database::StimulusID id) const;
    inline Tensor<Float_T> readRawData(Database::StimuliSet set,
                                         unsigned int index) const;

    virtual void setBatchSize(unsigned int batchSize);
    void setTargetSize(const std::vector<size_t>& size);
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
    const std::vector<size_t>& getSize() const
    {
        return mSize;
    };
    size_t getSizeX() const
    {
        assert(mSize.size() > 0);
        return mSize[0];
    };
    size_t getSizeY() const
    {
        assert(mSize.size() > 1);
        return mSize[1];
    };
    size_t getSizeD() const
    {
        assert(mSize.size() > 2);
        return mSize[2];
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
    void iterTransformations(Database::StimuliSet set,
                        std::function<void(const Transformation&)> func) const;
    std::vector<unsigned int>& getDatabaseLearnIndex(const unsigned int epoch)
    {
        if(epoch > mDatabaseLearnIndexes.size()) {
            std::stringstream msg;
            msg << "StimuliProvider::getDatabaseIndexOnEpoch(): epochId (" << epoch
                << ") is higher than the number of epoch initialized ";

            throw std::runtime_error(msg.str());
        }

        return mDatabaseLearnIndexes[epoch];
    };
    const std::vector<int>& getBatch()
    {
        return mBatch;
    };
    TensorData_T& getData()
    {
        return mData;
    };
    TensorData_T& getTargetData()
    {
        return (!mTargetData.empty()) ? mTargetData : mData;
    };
    Tensor<int>& getLabelsData()
    {
        return mLabelsData;
    };
    const TensorData_T& getData() const
    {
        return mData;
    };
    const Tensor<int>& getLabelsData() const
    {
        return mLabelsData;
    };
    const TensorData_T& getTargetData() const
    {
        return (!mTargetData.empty()) ? mTargetData : mData;
    };
    const std::vector<std::vector<std::shared_ptr<ROI> > >&
    getLabelsROIs() const
    {
        return mLabelsROI;
    };
    const TensorData_T getData(unsigned int channel,
                                    unsigned int batchPos = 0) const;
    const Tensor<int> getLabelsData(unsigned int channel,
                                      unsigned int batchPos = 0) const;
    const TensorData_T getTargetData(unsigned int channel,
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
                        Tensor<Float_T> data);
    static void logData(const std::string& fileName,
                        Tensor<Float_T> data,
                        const double minValue,
                        const double maxValue);
    static void logDataMatrix(const std::string& fileName,
                        const Tensor<Float_T>& data,
                        const double minValue,
                        const double maxValue);
    //static void logRgbData(const std::string& fileName,
    //                    const Tensor4d<Float_T>& data);
    unsigned int SetStimuliIndexes( Database::StimuliSet set,   
                                        const unsigned int nbEpochs = 1,
                                        const bool randPermutation = false); 
protected:
    std::vector<cv::Mat> loadDataCache(const std::string& fileName) const;
    void saveDataCache(const std::string& fileName,
                       const std::vector<cv::Mat>& data) const;

protected:
    /// Map unsigned integer range to signed before convertion to Float_T
    /// 8 bits image pixels will be converted from [0,255] to [-0.5, 0.5]
    Parameter<bool> mDataSignedMapping;
    /// Quantization levels (0 = no quantization)
    Parameter<unsigned int> mQuantizationLevels;
    /// Min. value for quantization
    Parameter<Float_T> mQuantizationMin;
    /// Max. value for quantization
    Parameter<Float_T> mQuantizationMax;

    // Internal variables
    Database& mDatabase;
    std::vector<size_t> mSize;
    std::vector<size_t> mTargetSize;
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
    TensorData_T mData;
    TensorData_T mFutureData;
    /// Tensor (x, y, channel, batch)
    Tensor<int> mLabelsData;
    Tensor<int> mFutureLabelsData;
    /// Tensor (x, y, channel, batch)
    TensorData_T mTargetData;
    TensorData_T mFutureTargetData;
    /// ROIs of current batch
    std::vector<std::vector<std::shared_ptr<ROI> > > mLabelsROI;
    std::vector<std::vector<std::shared_ptr<ROI> > > mFutureLabelsROI;
    bool mFuture;

    std::vector<std::vector<unsigned int > > mDatabaseLearnIndexes;
    std::vector<std::vector<unsigned int > > mDatabaseValIndexes;
    std::vector<std::vector<unsigned int > > mDatabaseTestIndexes;

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
    return (mChannelsTransformations.empty()) ? mSize.back()
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

N2D2::Tensor<N2D2::Float_T>
N2D2::StimuliProvider::readRawData(Database::StimuliSet set,
                                   unsigned int index) const
{
    return readRawData(mDatabase.getStimulusID(set, index));
}

#endif // N2D2_STIMULIPROVIDER_H
