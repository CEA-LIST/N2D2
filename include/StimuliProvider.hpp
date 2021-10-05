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
#include <set>
#include <deque>

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

class Adversarial;

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

    struct ProvidedData {
#ifdef CUDA
        ProvidedData():
            // mData and mFutureData are host-based by default.
            // This can be changed with the hostBased() method if data is directly
            // supplied to mData's device pointer.
            data(true),
            targetData(true) {}
#else
        ProvidedData() {}
#endif

        ProvidedData(ProvidedData&& other);
        void swap(ProvidedData& other);

        /// StimuliID of current batch
        std::vector<int> batch;
        /// Tensor (x, y, channel, batch)
        TensorData_T data;
        /// Tensor (x, y, channel, batch)
        Tensor<int> labelsData;
        /// Tensor (x, y, channel, batch)
        TensorData_T targetData;
        /// ROIs of current batch
        std::vector<std::vector<std::shared_ptr<ROI> > > labelsROI;
    };

    struct DevicesInfo {
#ifdef CUDA
        /// State of the devices
        std::vector<N2D2::DeviceState> states;
#endif
        /// Current batch's number provided to the devices
        std::vector<int> numBatchs;
        /// Future batch's number provided to the devices
        std::vector<int> numFutureBatchs;
    };

    StimuliProvider(Database& database,
                    const std::vector<size_t>& size,
                    unsigned int batchSize = 1,
                    bool compositeStimuli = false);

    StimuliProvider(const StimuliProvider& other) = delete;
    StimuliProvider(StimuliProvider&& other);

    /// Return a partial copy of the StimuliProvider. Only the parameters of the
    /// StimuliProvider are copied, the loaded stimuli data are zero-initialized.
    StimuliProvider cloneParameters() const;

    void setDevices(const std::set<int>& devices = std::set<int>());

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

    void logTransformations(const std::string& fileName,
        Database::StimuliSetMask setMask = Database::All) const;

    /// Return the number of batches remaining 
    /// in the indexes queue of the set
    unsigned int nbBatchsRemaining(Database::StimuliSet set)
    {
        std::deque<Database::StimulusID>& indexes = 
                (set == Database::StimuliSet::Learn) ? mIndexesLearn :
                (set == Database::StimuliSet::Validation) ? mIndexesVal :
                mIndexesTest;
        return indexes.size();
    };

    /// Return true if all batches from the set have been read, false otherwise
    /// Must be used with StimuliProvider::readBatch(Database::StimuliSet set)
    bool allBatchsProvided(Database::StimuliSet set)
    {
        bool allProvided = true;
        if (nbBatchsRemaining(set) == 0) {
            for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
                if (mDevices.find(dev) != mDevices.end()) {
                    if (mDevicesInfo.numFutureBatchs[dev] != -1)
                        allProvided = false;
                }
            }
        } else
            allProvided = false;
        
        return allProvided;
    };

    /// Return true if there are enough batches left for a last run
    /// Must be used with StimuliProvider::readBatch(Database::StimuliSet set) 
    bool isLastBatch(Database::StimuliSet set)
    {
        bool last = false;
        if (nbBatchsRemaining(set) == 0) {
            for (int dev = 0; dev < (int)mProvidedData.size(); ++dev) {
                if (mDevices.find(dev) != mDevices.end()) {
                    if (mDevicesInfo.numFutureBatchs[dev] != -1)
                        last = true;
                }
            }
        }
        return last;
    };

#ifdef CUDA
    /// Put back the batches of banned devices in the index queue.
    /// Must be used with StimuliProvider::readBatch(Database::StimuliSet set) 
    void adjustBatchs(Database::StimuliSet set);
#endif

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
                                            unsigned int batchPos = 0,
                                            int dev = -1);

    /// Read a whole batch from the StimuliSet @p set, apply all the
    /// transformations and put the results in
    /// mData and mLabelsData
    virtual void readBatch(Database::StimuliSet set, unsigned int startIndex);
    void streamBatch(int startIndex = -1, int dev = -1);

//TODO: Required for spiking neural network batch parallelization
/*
    /// Read a whole batch from the StimuliSet @p set, apply all the
    /// transformations and put the results in
    /// mData and mLabelsData. Save the IDs of the stimuli
    void readBatch(Database::StimuliSet set, unsigned int startIndex,
                             std::vector<Database::StimulusID>& Ids);
*/

    void readStimulusBatch(Database::StimulusID id,
                           Database::StimuliSet set,
                           int dev = -1);

    /// Read the stimulus with StimulusID @p id, apply all the transformations
    /// and put the results at batch
    /// position @p batchPos in mData and mLabelsData
    virtual void readStimulus(Database::StimulusID id,
                      Database::StimuliSet set,
                      unsigned int batchPos = 0,
                      int dev = -1);

    Database::StimulusID readStimulusBatch(Database::StimuliSet set,
                                           unsigned int index,
                                           int dev = -1);

    /// Read the stimulus with index @p index in StimuliSet @p set, apply all
    /// the transformations and put the results at batch
    /// position @p batchPos in mData and mLabelsData
    Database::StimulusID readStimulus(Database::StimuliSet set,
                                      unsigned int index,
                                      unsigned int batchPos = 0,
                                      int dev = -1);

    /** Read the batchs from a set
     * 
     * Select a batch from the set for each device which is 
     * connected to the deepNet. 
     * Then each device reads the whole assigned batch with the function 
     * StimuliProvider::readStimulus(). 
     * This function cannot be used without having previously called 
     * StimuliProvider::setBatch()
     * 
     * @param set   StimuliSet
     */
    void readBatch(Database::StimuliSet set); 
    
    void streamStimulus(const cv::Mat& mat,
                        Database::StimuliSet set,
                        unsigned int batchPos = 0,
                        int dev = -1);
    void synchronizeToDevices();

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

    /** Set the batchs for reading
     * 
     * All stimulus from the set are placed in a vector. 
     * The vector may be shuffled depending the value of @p randShuffle. 
     * Then the indexes of each batch start are placed in a
     * double-ended queue to facilitate batch reading. 
     * This function has to be used before
     * StimuliProvider::readBatch(Database::StimuliSet set)
     * 
     * @param set           StimuliSet
     * @param randShuffle   Boolean to shuffle all stimulus among batchs
     * @param nbMax         Maximum number of stimulus used to set up the batchs
     *                      (0 = all stimulus from the set are used)
     */
    void setBatch(Database::StimuliSet set,
                    bool randShuffle,
                    unsigned int nbMax = 0);
    
#ifdef CUDA
    /** Set the state of each device
     * 
     * @param states    Vector of device states
     */
    void setStates(std::vector<N2D2::DeviceState> states)
    {
       mDevicesInfo.states.assign(states.begin(), states.end()); 
    };
#endif

    void setAdversarialAttack(const std::shared_ptr<Adversarial>& attack)
    {
        mAttackAdv = attack;
    };

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
    unsigned int getMultiBatchSize() const
    {
        return mBatchSize * mDevices.size();
    };
    const std::set<int>& getDevices() const
    {
        return mDevices;
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
#ifdef CUDA
    std::vector<N2D2::DeviceState>& getStates()
    {
        return mDevicesInfo.states;
    };
#endif
    const std::vector<int>& getBatch(int dev = -1)
    {
        return mProvidedData[getDevice(dev)].batch;
    };
    TensorData_T& getDataInput()
    {
        return mProvidedData[getDevice(-1)].data;
    };
    TensorData_T& getData(int dev = -1)
    {
        return mProvidedData[getDevice(dev)].data;
    };
    TensorData_T& getTargetData(int dev = -1)
    {
        return (!mProvidedData[getDevice(dev)].targetData.empty())
            ? mProvidedData[getDevice(dev)].targetData
            : mProvidedData[getDevice(dev)].data;
    };
    Tensor<int>& getLabelsData(int dev = -1)
    {
        return mProvidedData[getDevice(dev)].labelsData;
    };
    const TensorData_T& getData(int dev = -1) const
    {
        return mProvidedData[getDevice(dev)].data;
    };
    const Tensor<int>& getLabelsData(int dev = -1) const
    {
        return mProvidedData[getDevice(dev)].labelsData;
    };
    const TensorData_T& getTargetData(int dev = -1) const
    {
        return (!mProvidedData[getDevice(dev)].targetData.empty())
            ? mProvidedData[getDevice(dev)].targetData
            : mProvidedData[getDevice(dev)].data;
    };
    std::shared_ptr<Adversarial> getAdversarialAttack() const
    {
        return mAttackAdv;
    };
/*
    const std::vector<std::vector<std::shared_ptr<ROI> > >&
    getLabelsROIs(int dev = -1) const
    {
        return mProvidedData[getDevice(dev)].labelsROI;
    };
*/
    const TensorData_T getDataChannel(unsigned int channel,
                                      unsigned int batchPos = 0,
                                      int dev = -1) const;
    const Tensor<int> getLabelsDataChannel(unsigned int channel,
                                           unsigned int batchPos = 0,
                                           int dev = -1) const;
    const TensorData_T getTargetDataChannel(unsigned int channel,
                                            unsigned int batchPos = 0,
                                            int dev = -1) const;
    const std::vector<std::shared_ptr<ROI> >&
    getLabelsROIs(unsigned int batchPos = 0, int dev = -1) const
    {
        return mProvidedData[getDevice(dev)].labelsROI[batchPos];
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


protected:
    std::vector<cv::Mat> loadDataCache(const std::string& fileName) const;
    void saveDataCache(const std::string& fileName,
                       const std::vector<cv::Mat>& data) const;
    inline int getDevice(int dev) const;

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
    /// Provided data
    std::vector<ProvidedData> mProvidedData;
    /// Future provided data
    std::vector<ProvidedData> mFutureProvidedData;
    /// Devices information
    DevicesInfo mDevicesInfo;
    bool mFuture;

    /// Adversarial attack used against the deepNet
    std::shared_ptr<Adversarial> mAttackAdv;

    /// Set of Device IDs used by the deepNet
    std::set<int> mDevices;

    /// Vectors containing StimulusIDs from datasets
    std::vector<unsigned int> mBatchsLearnIndexes;
    std::vector<unsigned int> mBatchsValIndexes;
    std::vector<unsigned int> mBatchsTestIndexes;

    /// Queues containing indexes of the batchs
    std::deque<unsigned int> mIndexesLearn;
    std::deque<unsigned int> mIndexesVal;
    std::deque<unsigned int> mIndexesTest;
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

int N2D2::StimuliProvider::getDevice(int dev) const {
#ifdef CUDA
    if (dev == -1) {
        const cudaError_t status = cudaGetDevice(&dev);
        if (status != cudaSuccess)
            dev = 0;
    }

    return dev;
#else
    // unused argument
    (void)(dev);
    return 0;
#endif
}

#endif // N2D2_STIMULIPROVIDER_H
