/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes Thiele (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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
    : mRelTimeIndex(0),
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



void N2D2::CMonitor::initialize(unsigned int nbTimesteps)
{
    if (!mInitialized) {

        mNbTimesteps = nbTimesteps;

        for (unsigned int k=0; k<mNbTimesteps; k++) {
            mActivity.push_back(new Tensor<int>((*mInputs).dims()));
        }
      
        mFiringRate.resize((*mInputs).dims(), 0);
        mTotalFiringRate.resize((*mInputs).dims(), 0);

        mOutputsActivity.resize((*mInputs).dims(), 0);
        mTotalOutputsActivity.resize((*mInputs).dims(), 0);

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
                    if (std::round((*mInputs)(x, y ,channel, batch)) != 0) {
                        activity = 1;
                    }

                    mActivity[mRelTimeIndex](x,y,channel,batch) = activity;
                    mFiringRate(x,y,channel,batch) += activity;
                    mTotalFiringRate(x,y,channel,batch) += activity;
                }
            }
        }
    }

    mTimeIndex.insert(std::make_pair(timestamp,mRelTimeIndex));
    mRelTimeIndex++;

    if (mRelTimeIndex > mNbTimesteps){
        throw std::runtime_error("Error: more ticks than timesteps");
    }

    return false;
}




void N2D2::CMonitor::logFiringRate(const std::string& fileName, bool plot)
{

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create firing rate log file: "
                                 + fileName);

    unsigned int totalActivity = 0;

#ifdef CUDA
    mFiringRate.synchronizeDToH();
#endif

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

    data.close();


    if (totalActivity == 0)
        std::cout << "Notice: no firing rate recorded." << std::endl;
    else if (plot){
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





void N2D2::CMonitor::logTotalFiringRate(const std::string& fileName, bool plot)
{

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create firing rate log file: "
                                 + fileName);

    unsigned int totalActivity = 0;
#ifdef CUDA
    mTotalFiringRate.synchronizeDToH();
#endif
    for (unsigned int channel=0; channel<(*mInputs).dimZ(); ++channel) {
        for (unsigned int y=0; y<(*mInputs).dimY(); ++y) {
            for (unsigned int x=0; x<(*mInputs).dimX(); ++x) {
                unsigned int nodeId =
                    channel*(*mInputs).dimY()*(*mInputs).dimX() +
                    y*(*mInputs).dimX() + x;
                unsigned int rate = 0;
                for (unsigned int batch=0; batch<(*mInputs).dimB(); ++batch) {
                    rate += mTotalFiringRate(x, y, channel, batch);
                }
                data << nodeId << " " << rate << "\n";
                totalActivity += rate;
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

        if (mTotalFiringRate.size() < 100) {
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
                                    bool plot) const
{

    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create activity log file: "
                                 + fileName);

    // Use the full double precision to keep accuracy even on small scales
    data.precision(std::numeric_limits<double>::digits10 + 1);

    bool isEmpty = true;

#ifdef CUDA
    for (unsigned int k=0; k<mActivity.size(); k++) {
        mActivity[k].synchronizeDToH();
    }
#endif

    for (unsigned int channel=0; channel<(*mInputs).dimZ(); ++channel) {
        for (unsigned int y=0; y<(*mInputs).dimY(); ++y) {
            for (unsigned int x=0; x<(*mInputs).dimX(); ++x) {
                unsigned int nodeId = channel*(*mInputs).dimY()*(*mInputs).dimX() +
                                        y*(*mInputs).dimX() + x;
                bool hasData = false;
                for (unsigned int k=0; k<mActivity.size(); k++) {
                    int activity = mActivity[k](nodeId, batch);
                    if(isEmpty && activity != 0) {
                        isEmpty = false;
                    }
                    if (activity != 0){
                        data << nodeId << " " << k / ((double)TimeS)
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

        const double xmin = 0/((double)TimeS);
        const double xmax = mNbTimesteps/((double)TimeS);

        Gnuplot gnuplot;
        gnuplot.set("bars", 0);
        gnuplot.set("pointsize", 0.01);
        gnuplot.setXrange(xmin, xmax);
        gnuplot.setYrange(ymin, ymax);
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
    clearAccumulators();
    clearActivity();
}

void N2D2::CMonitor::clearActivity()
{
    for (unsigned int k=0; k<mNbTimesteps; k++) {
        mActivity[k].assign(mActivity[k].dims(),0);
    }

    mTimeIndex.clear();
    mRelTimeIndex = 0;

}

void N2D2::CMonitor::clearAccumulators() 
{
    mFiringRate.assign(mFiringRate.dims(), 0);
    mTotalFiringRate.assign(mTotalFiringRate.dims(), 0);

    mOutputsActivity.assign(mOutputsActivity.dims(), 0);
    mTotalOutputsActivity.assign(mTotalOutputsActivity.dims(), 0);
}

// Resets variables for new example/stimulus
void N2D2::CMonitor::reset(Time_T /*timestamp*/)
{
   
    mFiringRate.assign(mFiringRate.dims(), 0);
    mOutputsActivity.assign(mOutputsActivity.dims(), 0);

    clearActivity();
    
}




