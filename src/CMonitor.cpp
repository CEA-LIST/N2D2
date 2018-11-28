/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes Thiele (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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


#include "CMonitor.hpp"


N2D2::CMonitor::CMonitor(Network& net)
    : mNet(net),
    mTotalBatchActivity(0),
    mTotalBatchFiringRate(0),
    mNbEvaluations(0),
    mRelTimeIndex(0),
    mSuccessCounter(0),
    mInitialized(false)

{
    // ctor
}


void N2D2::CMonitor::add(StimuliProvider& sp)
{
    CEnvironment* cenvCSpike = dynamic_cast<CEnvironment*>(&sp);
    if (!cenvCSpike) {
          throw std::runtime_error(
            "CMonitor::add(): CMonitor models require CEnvironment");
    }

    mInputs.push_back(&(cenvCSpike->getTickOutputs()));
    mInputs.back().setValid();

}


void N2D2::CMonitor::add(Cell* cell)
{

    Cell_CSpike* cellCSpike = dynamic_cast<Cell_CSpike*>(cell);
    if (cellCSpike) {
        mInputs.push_back(&(cellCSpike->getOutputs()));
    }
    else {
        throw std::runtime_error(
            "CMonitor::add(): CMonitor requires Cell_CSpike");
    }
    mInputs.back().setValid();
}


void N2D2::CMonitor::initialize(unsigned int nbTimesteps,
                                    unsigned int nbClasses)
{
    // TODO: Refactor this considering the new Tensor implementation
    if (!mInitialized) {
        mNbTimesteps = nbTimesteps;

        mNbClasses = nbClasses;

        mActivitySize = mInputs[0].dimX()* mInputs[0].dimY()
                              *mInputs[0].dimZ();

        mMostActiveId.resize({1, 1, 1, mInputs.dimB()}, 0);
        mMostActiveRate.resize({1, 1, 1, mInputs.dimB()}, 0);
        mFirstEventTime.resize({1, 1, 1, mInputs.dimB()}, 0);
        mLastEventTime.resize({1, 1, 1, mInputs.dimB()}, 0);

        mLastExampleActivity.resize({mInputs[0].dimX(), mInputs[0].dimY(),
                                mInputs[0].dimZ(), mInputs.dimB()}, 0);

        mBatchActivity.resize({1, mInputs[0].dimX(), mInputs[0].dimY(),
                                mInputs[0].dimZ()}, 0);
        mTotalActivity.resize({1, 1, 1, mInputs.dimB()}, 0);

        mFiringRate.resize({mInputs[0].dimX(), mInputs[0].dimY(),
                                mInputs[0].dimZ(), mInputs.dimB()}, 0);
        mBatchFiringRate.resize({1, mInputs[0].dimX(), mInputs[0].dimY(),
                                mInputs[0].dimZ()}, 0);
        mTotalFiringRate.resize({1, 1, 1, mInputs.dimB()}, 0);

        for (unsigned int k=0; k<nbTimesteps; k++) {
            mActivity.push_back(new Tensor<char>(mInputs[0].dims()));
        }
        for (unsigned int k=0; k<nbClasses; k++) {
            mStats.push_back(new Tensor<unsigned int> (mInputs[0].dims()));

        }

        mMaxClassResponse.resize(mInputs[0].dims(), 0);

        mInitialized = true;
    }
}


// TODO: Parallelize with CUDA
bool N2D2::CMonitor::tick(Time_T timestamp)
{
    for (unsigned int batch=0; batch<mInputs.dimB(); ++batch) {
        for (unsigned int channel=0; channel<mInputs[0].dimZ(); ++channel) {
            for (unsigned int y=0; y<mInputs[0].dimY(); ++y) {
                for (unsigned int x=0; x<mInputs[0].dimX(); ++x) {

                    char activity = mInputs[0](x, y ,channel, batch);

                    if ((int)activity > 0) {
                        if (mFirstEventTime(batch) == 0) {
                            mFirstEventTime(batch) = timestamp;
                        }
                        mLastEventTime(batch) = timestamp;
                    }

                    mActivity[mRelTimeIndex](x,y,channel,batch) = activity;
                }
            }
        }
    }

    mTimeIndex.insert(std::make_pair(timestamp,mRelTimeIndex));
    mRelTimeIndex++;

    return false;
}

// This is called to gather statistics of a training instance
void N2D2::CMonitor::update(Time_T start, Time_T stop)
{
    calcTotalBatchActivity(start, stop, true);
}


/*
bool N2D2::CMonitor::classifyMajorityVote(unsigned int batch, unsigned int cls, bool update)
{
    if (mNbClasses == 0) {
        throw std::runtime_error("Error in "
            "N2D2::CMonitor::checkLearningResponse: "
            "Number of classes not specified in Monitor");
    }
    bool success = false;

    //std::cout << "mNbClasses: " << mNbClasses << std::endl;
    //std::cout << "mACtivitySize: " << mActivitySize << std::endl;

    std::vector<unsigned int> preferedClass(mActivitySize, 0);

    for (unsigned int node=0; node<mActivitySize; ++node) {
        unsigned int maxClass = 0;
        double maxResponse = 0;
        for(unsigned int c=0; c<mNbClasses; ++c) {
            if (mStats[c](node, batch) > maxResponse){
                maxClass = c;
                maxResponse = mStats[c](node, batch);
            }
        }
        if (maxResponse > 0) {
            preferedClass[node] = maxClass;
        }
        else {
            preferedClass[node] = mNbClasses;
        }
        //std::cout << "node: "  << node << " "<<  preferedClass[node] << std::endl;
    }

    if (mMostActiveRate(batch) > 0) {

        double maxResponse = 0;
        unsigned int maxClass = 0;
        for(unsigned int c=0; c<mNbClasses; ++c) {
            double classAverage = 0;
            unsigned int averageCounter = 0;
            for (unsigned int node=0; node<mActivitySize; ++node) {
                if (preferedClass[node] == c) {
                    classAverage += mLastExampleActivity(node, batch);
                    averageCounter++;
                }
                //std::cout << "c: " << mStats[c](node,batch) << std::endl;
            }
            if (averageCounter != 0){
                classAverage = classAverage/averageCounter;
            }
            else {
                classAverage = 0;
            }
            if (classAverage > maxResponse) {
                maxClass = c;
                maxResponse = classAverage;
            }
        }
        //std::cout << "maxResponse: " << maxResponse << std::endl;
        if (maxResponse > 0) {
            if (maxClass == cls) {
                //std::cout << "maxClass: " << maxClass << std::endl;
                success = true;
                mSuccessCounter++;
            }
        }

        if (update) {
        // Increase class counter for most active neuron
            mStats[cls](mMostActiveId(batch), batch)++;
            // If class counter becomes higher update the max response class
            if (mStats[cls](mMostActiveId(batch), batch) >
            mMaxClassResponse(mMostActiveId(batch), batch)) {
                  //std::cout << "update" << std::endl;
                  mMaxClassResponse(mMostActiveId(batch), batch) =
                  mStats[cls](mMostActiveId(batch), batch);
            }
        }
        // No response is not counted as failure
        mNbEvaluations++;
        mSuccess.push_back(success);
    }
    return success;
}



bool N2D2::CMonitor::classifyFirstSpike(unsigned int batch,
                                        unsigned int cls,
                                        unsigned int node,
                                        bool update)
{
    if (mNbClasses == 0) {
        throw std::runtime_error("Error in "
            "N2D2::CMonitor::checkLearningResponse: "
            "Number of classes not specified in Monitor");
    }
    bool success = false;

    //std::cout << "MostActiveId: " << mMostActiveId(batch) << std::endl;
    //std::cout << "MostActiveRate: " << mMostActiveRate(batch) << std::endl;

    // Check if this neuron actually spiked
    if (mLastExampleActivity(node, batch) > 0) {

        unsigned int maxClass = 0;
        double maxResponse = 0;
        for(unsigned int c=0; c<mNbClasses; ++c) {
            if (mStats[c](node, batch) > maxResponse){
                maxClass = c;
                maxResponse = mStats[c](node, batch);
            }
        }

        if (maxResponse > 0) {
            if(maxClass == cls) {
                success = true;
                mSuccessCounter++;
            }
        }

        if (update) {
        // Increase class counter for most active neuron
            mStats[cls](node, batch)++;
            // If class counter becomes higher update the max response class
            if (mStats[cls](node, batch) >
            mMaxClassResponse(node, batch)) {
                  //std::cout << "update" << std::endl;
                  mMaxClassResponse(node, batch) =
                  mStats[cls](node, batch);
            }
        }
        // No response is not counted as failure
        mNbEvaluations++;
        mSuccess.push_back(success);
    }

    return success;
}
*/

bool N2D2::CMonitor::classifyRateBased(unsigned int target,
                                        unsigned int batch,
                                        std::vector<float> integrations,
                                        bool update)
{
    if (mNbClasses == 0) {
        throw std::runtime_error("Error in "
            "N2D2::CMonitor::checkLearningResponse: "
            "Number of classes not specified in Monitor");
    }
    bool success = false;

    unsigned int maxNode = 0;
    double maxResponse = 0;
    for(unsigned int i=0; i<mNbClasses; ++i) {
        if (mLastExampleActivity(i,batch) > maxResponse){
            maxNode = i;
            maxResponse = mLastExampleActivity(i,batch);
        }
        // If same number of spikes compare integrations. Same integration is very unlikely
        // This also includes the case of no spikes in all neurons
        else if (mLastExampleActivity(i,batch) == maxResponse) {
            if (integrations[i] >= integrations[maxNode]) {
                maxNode = i;
            }
        }
    }
    if(maxNode == target) {
        success = true;
        if (update) {
            mSuccessCounter++;
        }
    }

    if (update) {
        mNbEvaluations++;
        mSuccess.push_back(success);
    }

    return success;
}


bool N2D2::CMonitor::classifyIntegrationBased(unsigned int target,
                                                unsigned int /*batch*/,
                                                std::vector<float> integrations,
                                                bool update)
{
    if (mNbClasses == 0) {
        throw std::runtime_error("Error in "
            "N2D2::CMonitor::checkLearningResponse: "
            "Number of classes not specified in Monitor");
    }
    bool success = false;

    unsigned int maxNode = 0;
    double maxResponse = 0;
    for(unsigned int i=0; i<mNbClasses; ++i) {
        //std::cout << integrations[i] << std::endl;
        if (integrations[i] > maxResponse){
            maxNode = i;
            maxResponse = integrations[i];
        }
    }
    if(maxNode == target) {
        success = true;
        if (update) {
            mSuccessCounter++;
        }
    }

    if (update) {
        mNbEvaluations++;
        mSuccess.push_back(success);
    }

    return success;
}


bool N2D2::CMonitor::checkLearningResponse(unsigned int batch, unsigned int cls,
                                          bool update)
{
    if (mNbClasses == 0) {
        throw std::runtime_error("Error in "
            "N2D2::CMonitor::checkLearningResponse: "
            "Number of classes not specified in Monitor");
    }
    bool success = false;

    if (mMostActiveRate(batch) > 0) {


        // Check if target class is maxResponse class for neuron
        // which was most active
        if(mMaxClassResponse(mMostActiveId(batch), batch) ==
        mStats[cls](mMostActiveId(batch), batch) &&
        mStats[cls](mMostActiveId(batch), batch) > 0){

            success = true;
            mSuccessCounter++;
        }

        if (update) {
        // Increase class counter for most active neuron
            mStats[cls](mMostActiveId(batch), batch)++;
            // If class counter becomes higher update the max response class
            if (mStats[cls](mMostActiveId(batch), batch) >
            mMaxClassResponse(mMostActiveId(batch), batch)) {
                  mMaxClassResponse(mMostActiveId(batch), batch) =
                  mStats[cls](mMostActiveId(batch), batch);
            }
        }
        // No response is not counted as failure
        mNbEvaluations++;
        mSuccess.push_back(success);
    }
    return success;
}


unsigned int N2D2::CMonitor::checkBatchLearningResponse(std::vector<unsigned int>& cls,
                                          bool update)
{

    unsigned int success = 0;

    for (unsigned int batch=0; batch<mInputs.dimB(); ++batch) {
        success += static_cast<unsigned int>(
                    checkLearningResponse(batch, cls[batch], update));
    }


    return success;
}

void N2D2::CMonitor::updateSuccess(bool success)
{
    mNbEvaluations++;
    mSuccess.push_back(success);
}



double N2D2::CMonitor::inferRelation(unsigned int batch)
{
    unsigned int outputSize = mLastExampleActivity.size();
    Tensor<Float_T> reconstruction({1, 1, outputSize});

    Float_T maxValue = 0;

    for (unsigned int i = 0; i < outputSize; ++i) {
        reconstruction(i) += mLastExampleActivity(i, batch);
        if (reconstruction(i) > maxValue) {
            maxValue = reconstruction(i);
        }
    }

    Float_T maxPeriodMean = 0;
    int prediction = 0;
    for (unsigned int mean = 0; mean < outputSize; ++mean){
        Float_T periodMean = 0;
        for (unsigned int x = 0; x < outputSize; ++x){
            int distance = (int)x - (int)mean;

            if (distance > (int)outputSize/2) {
                distance = outputSize - distance;
            }
            if (distance < -(int)outputSize/2) {
                distance = outputSize + distance;
            }

            Float_T value = reconstruction(x);
            periodMean += value*(outputSize - 2*std::abs(distance));
        }
        if (periodMean > maxPeriodMean){
            maxPeriodMean = periodMean;
            prediction = mean;
        }
    }
    return prediction/(Float_T)outputSize;

}


void N2D2::CMonitor::inferRelation(const std::string& fileName,
                                    Tensor<Float_T>& relationalTargets,
                                    unsigned int targetVariable,
                                    unsigned int batch,
                                    bool plot)
{
    Float_T prediction = inferRelation(batch);
    Float_T target = relationalTargets(targetVariable);

    Float_T error = 0.0;

    Float_T diff = target-prediction;
    if (std::fabs(diff) > 0.5) {
        Float_T higher = std::fmax(target, prediction);
        Float_T lower = std::fmin(target, prediction);
        diff = 1.0 + lower - higher;
    }
    error += diff*diff;

    Tensor<Float_T> inferredRelation({relationalTargets.size()+3});
    for (unsigned int i=0; i<relationalTargets.size(); ++i){
        inferredRelation(i) = relationalTargets(i);
    }
    inferredRelation(relationalTargets.size()) = target;
    inferredRelation(relationalTargets.size()+1) = prediction;
    inferredRelation(relationalTargets.size()+2) = error;

    relationalInferences.push_back(inferredRelation);

    if (plot) {

        std::ofstream dataFile(fileName.c_str());

        if (!dataFile.good())
            throw std::runtime_error("Could not create data rate log file: "
                                     + fileName);
        dataFile.precision(4);

        Float_T averageError = 0.0;

        for (unsigned int k=0; k<relationalInferences.size(); ++k) {

            for (unsigned int i=0; i<relationalInferences[k].size(); ++i){
                dataFile << relationalInferences[k](i) << " ";
            }
            dataFile << "\n";

            averageError +=
                relationalInferences[k](relationalInferences[k].size()-1);

        }
        dataFile.close();


        std::ostringstream label;
        label << "\"MSE: "
            << averageError/relationalInferences.size() << " RMSE: "
            << std::sqrt(averageError/relationalInferences.size());
        label << "\" at graph 0.5, graph 0.1 front";

        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setYlabel("y");
        gnuplot.setXlabel("x");

        gnuplot.set("label", label.str());
        gnuplot.saveToFile(fileName);

        std::ostringstream plotIndices;
        plotIndices << "using " << relationalTargets.size()+1
            << ":" << relationalTargets.size()+2 << " with points";
        gnuplot.plot(fileName, plotIndices.str());
    }
}


unsigned int N2D2::CMonitor::getActivity(unsigned int x,
                                           unsigned int y,
                                           unsigned int z,
                                           unsigned int batch,
                                           Time_T start,
                                           Time_T stop)
{
    if (start > stop) {
        throw std::runtime_error("Error in "
            "N2D2::CMonitor::getActivity: "
            "start > stop");
    }

    bool all = false;
    if (start == 0 && stop == 0) {
        all = true;
    }

    unsigned int firingRate = 0;
    if (all) {
        for (unsigned int k=0; k<mTimeIndex.size(); k++) {
            firingRate += (int)mActivity[k](x,y,z,batch);
            if ((int)mActivity[k](x,y,z,batch) > 1){
                std::cout << " Warning xyz: " << (int)mActivity[k](x,y,z,batch) << std::endl;
                exit(0);
            }
        }
    }
    else{
        if (mTimeIndex.find(start) == mTimeIndex.end() ||
        mTimeIndex.find(stop) == mTimeIndex.end()) {
            throw std::runtime_error("Error in "
            "N2D2::CMonitor::getActivity: "
            "start or stop outside of map range");
        }
        for (unsigned int k=mTimeIndex.at(start); k<=mTimeIndex.at(stop); k++) {
            firingRate += (int)mActivity[k](x,y,z,batch);
        }
    }

    return firingRate;
}

unsigned int N2D2::CMonitor::getActivity(unsigned int index,
                                               unsigned int batch,
                                               Time_T start,
                                               Time_T stop)
{
    if (start > stop) {
        throw std::runtime_error("Error in "
            "N2D2::CMonitor::getActivity: "
            "start > stop");
    }

    bool all = false;
    if (start == 0 && stop == 0) {
        all = true;

    }

    unsigned int firingRate = 0;
    if (all) {
        for (unsigned int k=0; k<mTimeIndex.size(); k++) {
            firingRate += (int)mActivity[k](index,batch);
            if ((int)mActivity[k](index,batch) > 1){
                std::cout << " Warning index: " << (int)mActivity[k](index,batch) << std::endl;
                exit(0);
            }
        }
    }
    else{
        if (mTimeIndex.find(start) == mTimeIndex.end() ||
        mTimeIndex.find(stop) == mTimeIndex.end()) {
            throw std::runtime_error("Error in "
            "N2D2::CMonitor::getActivity: "
            "start or stop outside of map range");
        }
        for (unsigned int k=mTimeIndex.at(start); k<=mTimeIndex.at(stop); k++) {
            firingRate += (int)mActivity[k](index,batch);
        }
    }

    return firingRate;
}

unsigned int N2D2::CMonitor::getBatchActivity(unsigned int x,
                                               unsigned int y,
                                               unsigned int z,
                                               Time_T start,
                                               Time_T stop)
{
    unsigned int batchActivity = 0;
    for (unsigned int batch=0; batch<mInputs.dimB(); ++batch) {
        batchActivity += getActivity(x, y, z, batch, start, stop);
    }

    return batchActivity;
}

unsigned int N2D2::CMonitor::getBatchActivity(unsigned int index,
                                               Time_T start,
                                               Time_T stop)
{
    unsigned int batchActivity = 0;
    for (unsigned int batch=0; batch<mInputs.dimB(); ++batch) {
        batchActivity += getActivity(index, batch, start, stop);
    }

    return batchActivity;
}

//TODO: Parallelize with CUDA
unsigned int N2D2::CMonitor::calcTotalActivity(unsigned int batch,
                                                Time_T start,
                                                Time_T stop,
                                                bool update)
{
    unsigned int totalActivity = 0;

    for (unsigned int channel=0; channel<mInputs[0].dimZ(); ++channel) {
        for (unsigned int y=0; y<mInputs[0].dimY(); ++y) {
            for (unsigned int x=0; x<mInputs[0].dimX(); ++x) {
                unsigned int activity =
                    getActivity(x, y, channel , batch, start, stop);
                if (update) {
                    if (activity > mMostActiveRate(batch)){
                        mMostActiveRate(batch) = activity;
                        unsigned int nodeId =
                            channel*mInputs[0].dimY()*mInputs[0].dimX() +
                            y*mInputs[0].dimX() + x;
                        mMostActiveId(batch) = nodeId;
                    }
                    mBatchActivity(0,x,y,channel) += activity;
                    mFiringRate(x,y,channel,batch) += activity;
                    // NOTE: This will lead to overlap in CUDA if parallelized over batches
                    mBatchFiringRate(0,x,y,channel) += activity;
                    mTotalFiringRate(batch) += activity;
                    // NOTE: This will lead to overlap in CUDA if parallelized over batches
                    mTotalBatchFiringRate += activity;
                }
                totalActivity += activity;
            }
        }
    }
    return totalActivity;
}



unsigned int N2D2::CMonitor::calcTotalBatchActivity(Time_T start,
                                                    Time_T stop,
                                                    bool update)
{
    unsigned int activitySum = 0;
    for (unsigned int batch=0; batch<mInputs.dimB(); ++batch) {
        unsigned int activity = calcTotalActivity(batch, start, stop, update);
        activitySum += activity;
        if (update) {
            mTotalActivity(batch) = activity;
        }
    }
    if (update){
        mTotalBatchActivity = activitySum;
    }
    return activitySum;
}




double N2D2::CMonitor::getSuccessRate(unsigned int avgWindow) const
{
    const unsigned int size = mSuccess.size();

    if (size > 0) {
        return (avgWindow > 0 && size > avgWindow)
                   ? std::accumulate(mSuccess.end() - avgWindow,
                                     mSuccess.end(),
                                     0.0) / avgWindow
                   : std::accumulate(mSuccess.begin(), mSuccess.end(), 0.0)
                     / size;
    }
    else
        return 0.0;
}


double N2D2::CMonitor::getFastSuccessRate() const
{
    if (mNbEvaluations > 0) {
        return static_cast<double>(mSuccessCounter)/mNbEvaluations;
    } else
        return 0.0;
}

void N2D2::CMonitor::logSuccessRate(const std::string& fileName,
                                       unsigned int avgWindow,
                                       bool plot) const
{
    logDataRate(mSuccess, fileName, avgWindow, plot);
}


void N2D2::CMonitor::logFiringRate(const std::string& fileName,
                                    bool plot,
                                    Time_T start,
                                    Time_T stop)
{

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create firing rate log file: "
                                 + fileName);

    unsigned int totalActivity = 0;

    if (start==0 && stop==0) {
        for (unsigned int channel=0; channel<mInputs[0].dimZ(); ++channel) {
            for (unsigned int y=0; y<mInputs[0].dimY(); ++y) {
                for (unsigned int x=0; x<mInputs[0].dimX(); ++x) {
                    unsigned int nodeId =
                        channel*mInputs[0].dimY()*mInputs[0].dimX() +
                        y*mInputs[0].dimX() + x;
                    unsigned int rate = 0;
                    for (unsigned int batch=0; batch<mInputs.dimB(); ++batch) {
                        rate += mFiringRate(x, y, channel, batch);
                    }
                    data << nodeId << " " << rate << "\n";
                    totalActivity += rate;
                }
            }
        }
    }
    else {
        for (unsigned int channel=0; channel<mInputs[0].dimZ(); ++channel) {
            for (unsigned int y=0; y<mInputs[0].dimY(); ++y) {
                for (unsigned int x=0; x<mInputs[0].dimX(); ++x) {
                    unsigned int nodeId =
                        channel*mInputs[0].dimY()*mInputs[0].dimX() +
                        y*mInputs[0].dimX() + x;
                    unsigned int rate = 0;
                    for (unsigned int batch=0; batch<mInputs.dimB(); ++batch) {
                        rate += getActivity(x,y,channel,batch,start,stop);
                    }
                    data << nodeId << " " << rate << "\n";
                    totalActivity += rate;
                }
            }
        }
    }


    data.close();


    if (totalActivity == 0)
        std::cout << "Notice: no firing rate recorded." << std::endl;
    else if (plot) {
        NodeId_T xmin = 0;
        NodeId_T xmax = mInputs[0].dimY()*mInputs[0].dimX()*mInputs[0].dimZ() -1;

        std::ostringstream label;
        label << "\"Total: " << totalActivity << "\"";
        label << " at graph 0.5, graph 0.95 front";

        Gnuplot gnuplot;
        std::stringstream cmdStr;
        //cmdStr << "n = " << (double)nbEventTypes;
        cmdStr << "n = 1";
        gnuplot << cmdStr.str();
        gnuplot << "box_width = 0.75";
        gnuplot << "gap_width = 0.1";
        gnuplot << "total_width = (gap_width + box_width)";
        gnuplot << "d_width = total_width/n";
        gnuplot << "offset = -total_width/2.0 + d_width/2.0";
        gnuplot.set("style data boxes").set("style fill solid noborder");
        gnuplot.set("boxwidth", "box_width/n relative");
        gnuplot.setXrange(xmin - 0.5, xmax + 0.5);
        gnuplot.set("yrange [0:]");
        gnuplot.setYlabel("Number of activations");
        gnuplot.setXlabel("Node ID");

        if (mFiringRate.size() < 100) {
            gnuplot.set("grid");
            gnuplot.set("xtics", "1 rotate by 90");
        }

        gnuplot.set("label", label.str());
        gnuplot.saveToFile(fileName);

        gnuplot.plot(fileName, "using 1:2 notitle");
    }


}

void N2D2::CMonitor::logActivity(const std::string& fileName,
                                    unsigned int batch,
                                    bool plot,
                                    Time_T start,
                                    Time_T stop) const
{

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create activity log file: "
                                 + fileName);

    // Use the full double precision to keep accuracy even on small scales
    data.precision(std::numeric_limits<double>::digits10 + 1);

    bool isEmpty = true;
    for (unsigned int channel=0; channel<mInputs[0].dimZ(); ++channel) {
        for (unsigned int y=0; y<mInputs[0].dimY(); ++y) {
            for (unsigned int x=0; x<mInputs[0].dimX(); ++x) {
                unsigned int nodeId = channel*mInputs[0].dimY()*mInputs[0].dimX() +
                                        y*mInputs[0].dimX() + x;
                bool hasData = false;
                for (std::map<Time_T, unsigned int>::const_iterator it=mTimeIndex.find(start);
                it!=mTimeIndex.find(stop); ++it) {
                    int activity = (int) mActivity[(*it).second](nodeId, batch);
                    if(isEmpty && activity > 0) {
                        isEmpty = false;
                    }
                    if (activity > 0){
                        data << nodeId << " " << (*it).first / ((double)TimeS)
                            << " " << activity << "\n";
                        hasData = true;
                    }

                }
                if(hasData){
                    data << "\n\n";
                }
            }
        }
    }

    data.close();


    if (isEmpty)
        std::cout << "Notice: no activity recorded." << std::endl;
    else if (plot) {

        NodeId_T ymin = 0;
        NodeId_T ymax = mInputs[0].dimY()*mInputs[0].dimX()*mInputs[0].dimZ()-1;

        const double xmin = ((start>0) ? start : mFirstEventTime(batch)) /((double)TimeS);
        const double xmax = ((stop>0) ? stop : mLastEventTime(batch)) /((double)TimeS);

        Gnuplot gnuplot;
        gnuplot.set("bars", 0);
        gnuplot.set("pointsize", 0.01);
        gnuplot.setXrange(xmin, xmax);
        gnuplot.setYrange(ymin, ymax + 1);
        gnuplot.setXlabel("Time (s)");
        gnuplot.saveToFile(fileName);

        if (ymax - ymin < 100) {
            gnuplot.set("grid");
            gnuplot.set("ytics", 1);
        }

        gnuplot.plot(fileName,
                         "using 2:1:($1+0.8):($1) notitle with yerrorbars");
    }

}

void N2D2::CMonitor::clearAll(unsigned int nbTimesteps)
{

    mTotalBatchActivity = 0;
    mTotalBatchFiringRate = 0;
    mNbEvaluations = 0;
    mRelTimeIndex = 0;
    mSuccessCounter = 0;

    mActivitySize = mInputs[0].dimX()* mInputs[0].dimY()
                          *mInputs[0].dimZ();

    mMostActiveId.assign({1, 1, 1, mInputs.dimB()}, 0);
    mMostActiveRate.assign({1, 1, 1, mInputs.dimB()}, 0);
    mFirstEventTime.assign({1, 1, 1, mInputs.dimB()}, 0);
    mLastEventTime.assign({1, 1, 1, mInputs.dimB()}, 0);


    mBatchActivity.assign({1, mInputs[0].dimX(), mInputs[0].dimY(),
                            mInputs[0].dimZ()}, 0);
    mTotalActivity.assign({1, 1, 1, mInputs.dimB()}, 0);

    mFiringRate.assign(mInputs[0].dims(), 0);
    mBatchFiringRate.assign({1, mInputs[0].dimX(), mInputs[0].dimY(),
                            mInputs[0].dimZ()}, 0);
    mTotalFiringRate.assign({1, 1, 1, mInputs.dimB()}, 0);

    clearActivity(nbTimesteps);

    mSuccess.clear();
    clearFastSuccess();
    relationalInferences.clear();

}

void N2D2::CMonitor::clearActivity(unsigned int nbTimesteps)
{
    //TODO: clear seems not properly defined
    //mActivity.clear();
    //for (unsigned int k=0; k<mNbTimesteps; k++) {
    //    mActivity.push_back(new CudaTensor4d<char>(mInputs[0].dimX(),
    //                mInputs[0].dimY(), mInputs[0].dimZ(), mInputs.dimB()));
    //}
    //std::cout << "CMonitor clear activity" << std::endl;
    unsigned int oldNbTimesteps = mNbTimesteps;

    if (nbTimesteps != 0) {
        mNbTimesteps = nbTimesteps;
    }

    for (unsigned int k=0; k<mNbTimesteps; k++) {
        if (k >= oldNbTimesteps) {
             mActivity.push_back(new Tensor<char>(mInputs[0].dims()));
        }
        mActivity[k].assign(mInputs[0].dims(),0);
    }


    mTimeIndex.clear();
    mRelTimeIndex=0;

}

void N2D2::CMonitor::clearFiringRate()
{
    mFiringRate.assign(mInputs[0].dims(), 0);
}

void N2D2::CMonitor::clearMostActive()
{
    mMostActiveId.assign({1, 1, 1, mInputs.dimB()}, 0);
    mMostActiveRate.assign({1, 1, 1, mInputs.dimB()}, 0);
}

void N2D2::CMonitor::clearSuccess()
{
    mSuccess.clear();
}

void N2D2::CMonitor::clearFastSuccess()
{
    mSuccessCounter = 0;
    mNbEvaluations = 0;
}



