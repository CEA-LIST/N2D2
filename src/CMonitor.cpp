/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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


N2D2::CMonitor::CMonitor()
    : mTotalBatchExampleFiringRate(0),
    mTotalBatchFiringRate(0),
    mTotalBatchOutputsActivity(0),
    mNbEvaluations(0),
    mRelTimeIndex(0),
    mSuccessCounter(0),
    mInitialized(false)

{
    // ctor
}




void N2D2::CMonitor::add(Tensor<float>& input)
{
    //TODO: Add quantization function between cell and monitor to treat different float types
#ifdef CUDA
    mInputs = dynamic_cast<CudaTensor<float>*>(&(input));
#else
    mInputs = &input;
#endif
}




void N2D2::CMonitor::initialize(unsigned int nbTimesteps,
                                    unsigned int nbClasses)
{
    // TODO: Refactor this considering the new Tensor implementation
    if (!mInitialized) {

        mNbTimesteps = nbTimesteps;

        mNbClasses = nbClasses;

        mActivitySize = (*mInputs).dimX()* (*mInputs).dimY()
                              *(*mInputs).dimZ();

         for (unsigned int k=0; k<nbTimesteps; k++) {
            mActivity.push_back(new Tensor<int>((*mInputs).dims()));

        }
        mBatchActivity.resize({1, (*mInputs).dimX(), (*mInputs).dimY(),
                                (*mInputs).dimZ()}, 0);

        mFiringRate.resize((*mInputs).dims(), 0);
        mTotalFiringRate.resize({1, 1, 1, (*mInputs).dimB()}, 0);
        mBatchFiringRate.resize({1, (*mInputs).dimX(), (*mInputs).dimY(),
                                (*mInputs).dimZ()}, 0);

        mExampleFiringRate.resize((*mInputs).dims(), 0);
        mTotalExampleFiringRate.resize({1, 1, 1, (*mInputs).dimB()}, 0);

        mOutputsActivity.resize((*mInputs).dims(), 0);
        mTotalOutputsActivity.resize({1, 1, 1, (*mInputs).dimB()}, 0);

        mMostActiveId.resize({1, 1, 1, (*mInputs).dimB()}, 0);
        mMostActiveRate.resize({1, 1, 1, (*mInputs).dimB()}, 0);
        mFirstEventTime.resize({1, 1, 1, (*mInputs).dimB()}, 0);
        mLastEventTime.resize({1, 1, 1, (*mInputs).dimB()}, 0);

        for (unsigned int k=0; k<nbClasses; k++) {
            mStats.push_back(new Tensor<unsigned int> ((*mInputs).dims()));
            mStats.back().synchronizeHToD();

        }

        mInitialized = true;
    }
}


bool N2D2::CMonitor::tick(Time_T timestamp)
{
    for (unsigned int batch=0; batch<(*mInputs).dimB(); ++batch) {
        for (unsigned int channel=0; channel<(*mInputs).dimZ(); ++channel) {
            for (unsigned int y=0; y<(*mInputs).dimY(); ++y) {
                for (unsigned int x=0; x<(*mInputs).dimX(); ++x) {

                    // Extract sign
                    int activity = 0;
                    if ((*mInputs)(x, y ,channel, batch) != 0) {
                        activity = 1;

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
            firingRate += mActivity[k](x,y,z,batch);
            if (mActivity[k](x,y,z,batch) > 1){
                std::cout << " Warning xyz: " << mActivity[k](x,y,z,batch) << std::endl;
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
            firingRate += mActivity[k](x,y,z,batch);
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
            firingRate += mActivity[k](index,batch);
            if (mActivity[k](index,batch) > 1){
                std::cout << " Warning index: " << mActivity[k](index,batch) << std::endl;
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
            firingRate += mActivity[k](index,batch);
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
    for (unsigned int batch=0; batch<(*mInputs).dimB(); ++batch) {
        batchActivity += getActivity(x, y, z, batch, start, stop);
    }

    return batchActivity;
}

unsigned int N2D2::CMonitor::getBatchActivity(unsigned int index,
                                               Time_T start,
                                               Time_T stop)
{
    unsigned int batchActivity = 0;
    for (unsigned int batch=0; batch<(*mInputs).dimB(); ++batch) {
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

    for (unsigned int channel=0; channel<(*mInputs).dimZ(); ++channel) {
        for (unsigned int y=0; y<(*mInputs).dimY(); ++y) {
            for (unsigned int x=0; x<(*mInputs).dimX(); ++x) {
                unsigned int activity =
                    getActivity(x, y, channel , batch, start, stop);
                if (update) {
                    if (activity > mMostActiveRate(batch)){
                        mMostActiveRate(batch) = activity;
                        unsigned int nodeId =
                            channel*(*mInputs).dimY()*(*mInputs).dimX() +
                            y*(*mInputs).dimX() + x;
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
    for (unsigned int batch=0; batch<(*mInputs).dimB(); ++batch) {
        unsigned int activity = calcTotalActivity(batch, start, stop, update);
        activitySum += activity;
        if (update) {
            mTotalExampleFiringRate(batch) = activity;
        }
    }
    if (update){
        mTotalBatchExampleFiringRate = activitySum;
    }
    return activitySum;
}




void N2D2::CMonitor::logExampleFiringRate(const std::string& fileName)
{

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create firing rate log file: "
                                 + fileName);

    for (unsigned int channel=0; channel<(*mInputs).dimZ(); ++channel) {
        for (unsigned int y=0; y<(*mInputs).dimY(); ++y) {
            for (unsigned int x=0; x<(*mInputs).dimX(); ++x) {
                unsigned int nodeId =
                    channel*(*mInputs).dimY()*(*mInputs).dimX() +
                    y*(*mInputs).dimX() + x;
                unsigned int rate = 0;
                for (unsigned int batch=0; batch<(*mInputs).dimB(); ++batch) {
                    rate += mExampleFiringRate(x, y, channel, batch);
                }
                data << nodeId << " " << rate << "\n";
            }
        }
    }


    data.close();

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
        for (unsigned int channel=0; channel<(*mInputs).dimZ(); ++channel) {
            for (unsigned int y=0; y<(*mInputs).dimY(); ++y) {
                for (unsigned int x=0; x<(*mInputs).dimX(); ++x) {
                    unsigned int nodeId =
                        channel*(*mInputs).dimY()*(*mInputs).dimX() +
                        y*(*mInputs).dimX() + x;
                    unsigned int rate = 0;
                    for (unsigned int batch=0; batch<(*mInputs).dimB(); ++batch) {
                        rate += mFiringRate(x, y, channel, batch);
                    }
                    data << nodeId << " " << rate << "\n";
                    totalActivity += rate;
                }
            }
        }
    }
    else {
        for (unsigned int channel=0; channel<(*mInputs).dimZ(); ++channel) {
            for (unsigned int y=0; y<(*mInputs).dimY(); ++y) {
                for (unsigned int x=0; x<(*mInputs).dimX(); ++x) {
                    unsigned int nodeId =
                        channel*(*mInputs).dimY()*(*mInputs).dimX() +
                        y*(*mInputs).dimX() + x;
                    unsigned int rate = 0;
                    for (unsigned int batch=0; batch<(*mInputs).dimB(); ++batch) {
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
        NodeId_T xmax = (*mInputs).dimY()*(*mInputs).dimX()*(*mInputs).dimZ() -1;

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
    for (unsigned int channel=0; channel<(*mInputs).dimZ(); ++channel) {
        for (unsigned int y=0; y<(*mInputs).dimY(); ++y) {
            for (unsigned int x=0; x<(*mInputs).dimX(); ++x) {
                unsigned int nodeId = channel*(*mInputs).dimY()*(*mInputs).dimX() +
                                        y*(*mInputs).dimX() + x;
                bool hasData = false;
                for (std::map<Time_T, unsigned int>::const_iterator it=mTimeIndex.find(start);
                it!=std::next(mTimeIndex.find(stop),1); ++it) {
                    int activity = mActivity[(*it).second](nodeId, batch);
                    if(isEmpty && activity != 0) {
                        isEmpty = false;
                    }
                    if (activity != 0){
                        data << nodeId << " " << (*it).first / ((double)TimeS)
                            << " " << "1" << "\n";
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
        NodeId_T ymax = (*mInputs).dimY()*(*mInputs).dimX()*(*mInputs).dimZ();

        const double xmin = ((start>0) ? start : mFirstEventTime(batch)) /((double)TimeS);
        const double xmax = ((stop>0) ? stop : mLastEventTime(batch)) /((double)TimeS);

        double xOffset = (stop-start)/((mNbTimesteps-1) * (double)TimeS);

        Gnuplot gnuplot;
        gnuplot.set("bars", 0);
        gnuplot.set("pointsize", 0.01);
        gnuplot.setXrange(xmin, xmax + xOffset);
        gnuplot.setYrange(ymin, ymax );
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

void N2D2::CMonitor::clearAll()
{

    mTotalBatchExampleFiringRate = 0;
    mTotalBatchFiringRate = 0;
    mNbEvaluations = 0;
    //mRelTimeIndex = 0;

    mBatchActivity.assign({1, (*mInputs).dimX(), (*mInputs).dimY(),
                            (*mInputs).dimZ()}, 0);

    mFiringRate.assign((*mInputs).dims(), 0);
    mTotalFiringRate.assign({1, 1, 1, (*mInputs).dimB()}, 0);
    mBatchFiringRate.assign({1, (*mInputs).dimX(), (*mInputs).dimY(),
                            (*mInputs).dimZ()}, 0);

    mExampleFiringRate.assign((*mInputs).dims(), 0);
    mTotalExampleFiringRate.assign({1, 1, 1, (*mInputs).dimB()}, 0);

    mOutputsActivity.assign((*mInputs).dims(), 0);
    mTotalOutputsActivity.assign({1, 1, 1, (*mInputs).dimB()}, 0);

    mMostActiveId.assign({1, 1, 1, (*mInputs).dimB()}, 0);
    mMostActiveRate.assign({1, 1, 1, (*mInputs).dimB()}, 0);
    mFirstEventTime.assign({1, 1, 1, (*mInputs).dimB()}, 0);
    mLastEventTime.assign({1, 1, 1, (*mInputs).dimB()}, 0);

    clearActivity();

}

void N2D2::CMonitor::clearActivity()
{
    for (unsigned int k=0; k<mNbTimesteps; k++) {
        mActivity[k].assign((*mInputs).dims(),0);
    }

    mTimeIndex.clear();
    mRelTimeIndex = 0;

}

void N2D2::CMonitor::reset(Time_T timestamp)
{
    clearActivity();
    clearMostActive();

    mExampleFiringRate.assign((*mInputs).dims(), 0);
    mTotalExampleFiringRate.assign({1, 1, 1, (*mInputs).dimB()}, 0);

    mOutputsActivity.assign((*mInputs).dims(), 0);
    mTotalOutputsActivity.assign({1, 1, 1, (*mInputs).dimB()}, 0);

    mTimeIndex.insert(std::make_pair(timestamp,mRelTimeIndex));
    //mRelTimeIndex++;
}

void N2D2::CMonitor::clearFiringRate()
{
    mFiringRate.assign((*mInputs).dims(), 0);
}

void N2D2::CMonitor::clearMostActive()
{
    mMostActiveId.assign({1, 1, 1, (*mInputs).dimB()}, 0);
    mMostActiveRate.assign({1, 1, 1, (*mInputs).dimB()}, 0);
}


