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
      mMatching(this, "Matching", "*.dat"),
      mDataTargetFormat(this, "TargetFormat", NCHW),
      mInputTranspose(this, "TargetTranspose", false),
      mLogDistrib(this, "LogDistrib", false),
      mLogError(this, "LogError", false),
      mBatchPacked(this, "BatchPacked", 1)
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
/*
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
*/
    const std::string dirPath = mCell->getName() + ".Target/Compare";

    if(mLogDistrib || mLogError)
        Utils::createDirectories(dirPath);

    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);

    const Tensor4d<Float_T>& values = targetCell->getOutputs();

    std::vector<double> meanSquareErrors(values.dimB());
    std::cout << "target " << mCell->getName()<< std::endl;
    std::cout << "target dimB: " << values.dimB() << std::endl;
    std::cout << "target dimZ: " << values.dimZ() << std::endl;
    std::cout << "target dimY: " << values.dimY() << std::endl;
    std::cout << "target dimX: " << values.dimX() << std::endl;

//#pragma omp parallel for if (values.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)values.dimB(); batchPos += mBatchPacked) {
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
        std::cout << "fileBaseName: " << fileBaseName << std::endl;
        std::ifstream dataFile(dataFileName);

        if (!dataFile.good()) {
            throw std::runtime_error("Could not open target data file: "
                                     + dataFileName);
        }

        //Tensor3d<int> target = mTargets[batchPos];
        const Tensor4d<Float_T> value = values;//[batchPos];
        unsigned int TargetDimX = value.dimX();
        unsigned int TargetDimY = value.dimY();
        unsigned int TargetDimZ = value.dimZ();
        unsigned int TargetDimB = mBatchPacked;

        Tensor4d<Float_T> errorValues(TargetDimX, TargetDimY, TargetDimZ, TargetDimB, 0.0);
        Tensor4d<Float_T> refValues(TargetDimX, TargetDimY, TargetDimZ, TargetDimB, 0.0);

        std::vector<double> refVect;
        std::vector<double> estVect;

        //if(mLogDistrib) {
        refVect.reserve(TargetDimX * TargetDimY * TargetDimZ * TargetDimB);
        estVect.reserve(TargetDimX * TargetDimY * TargetDimZ * TargetDimB);
        //}

        const unsigned int nbOutputs = value.dimZ();

        double meanSquareError = 0.0;

        if(mDataTargetFormat == NHWC) {
            TargetDimX = value.dimZ();
            TargetDimY = value.dimX();
            TargetDimZ = value.dimY();
        }
        Tensor4d<Float_T> targetValues(TargetDimX,
                                       TargetDimY,
                                       TargetDimZ,
                                       TargetDimB);
        if (!(dataFile >> targetValues.data()))
            throw std::runtime_error("Unreadable data file: " + dataFileName);

        dataFile.close();

        for(unsigned int batchPacked = 0; batchPacked < TargetDimB; ++ batchPacked) {
            for (unsigned int oy = 0; oy < value.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < value.dimX(); ++ox) {
                    //std::vector<std::pair<Float_T, size_t> >
                    //sortedLabelsValues;
                    //sortedLabelsValues.reserve(nbOutputs);

                    for (unsigned int output = 0; output < nbOutputs; ++output) {

                        double error = 0.0;

                        if(mDataTargetFormat == NCHW) {
                            if(!mInputTranspose)
                                error = targetValues(ox, oy, output, batchPacked)
                                          - value(ox, oy, output, batchPacked);
                            else
                                error = targetValues(ox, oy, output, batchPacked)
                                          - value(value.dimX() - ox - 1,
                                                  value.dimY() - oy - 1,
                                                  output,
                                                  batchPacked);
                            refValues(ox, oy, output, batchPacked) = targetValues(ox, oy, output, batchPacked);

                            //sortedLabelsValues.push_back(std::make_pair(
                            //    targetValues(ox, oy, output), output));
                        }
                        else if (mDataTargetFormat == NHWC) {
                            if(!mInputTranspose)
                                error = targetValues(output, ox, oy, batchPacked)
                                          - value(ox,
                                                  oy,
                                                  output,
                                                  batchPacked);

                            else
                                error = targetValues(output, ox, oy, batchPacked)
                                          - value(value.dimX() - ox - 1,
                                                  value.dimY() - oy - 1,
                                                  output,
                                                  batchPacked);

                            refValues(ox, oy, output, batchPacked) = targetValues(output, ox, oy, batchPacked);

                            //sortedLabelsValues.push_back(std::make_pair(
                            //    targetValues(output, ox, oy), output));
                        }
                        errorValues(ox, oy, output, batchPacked) = error;
                        meanSquareError += error * error;

                        refVect.push_back(refValues(ox, oy, output, batchPacked));
                        estVect.push_back(value(ox, oy, output, batchPacked));

                    }

                    // Top-n accuracy sorting
                    //std::partial_sort(
                    //    sortedLabelsValues.begin(),
                    //    sortedLabelsValues.begin() + 1,
                    //    sortedLabelsValues.end(),
                    //    std::greater<std::pair<Float_T, size_t> >());

                    //target(ox, oy, 0) = sortedLabelsValues[0].second;
                }
            }
            //std::cout << "meanSquareError: " << meanSquareError << std::endl;
        }
        meanSquareError /= value.size();
        meanSquareErrors[batchPos] = meanSquareError;

        std::sort(estVect.begin(), estVect.end());
        const std::pair<double, double> meanStdDevEst = Utils::meanStdDev(estVect);

        std::sort(refVect.begin(), refVect.end());
        const std::pair<double, double> meanStdDevRef = Utils::meanStdDev(refVect);

        if(std::abs(meanStdDevEst.first - meanStdDevRef.first) > 0.0001)
            std::cout << "TargetCompare: " << mCell->getName() << Utils::cwarning << std::setprecision(10)
                      << "Warning: The average value of the estimated target (" << meanStdDevEst.first << ")"
                      << "is different than the average value of the reference target (" << meanStdDevRef.first << ")"
                      << " for cell " << mCell->getName()
                      << Utils::cdef << std::endl;

        if(std::abs(meanStdDevEst.second - meanStdDevRef.second) > 0.0001)
            std::cout << "TargetCompare: " << mCell->getName() << Utils::cwarning << std::setprecision(10)
                      << "Warning: The Std Dev of the estimated target (" << meanStdDevEst.second << ")"
                      << "is different than the Std Dev of the reference target (" << meanStdDevRef.second << ")"
                      << " for cell " << mCell->getName()
                      << Utils::cdef << std::endl;
        for(unsigned int batchPacked = 0; batchPacked < TargetDimB; ++ batchPacked) {  
            //unsigned int batchPacked = 0;    
            if(mLogError) {
                std::stringstream fileErr;
                fileErr << dirPath << "/compare_" << batchPacked << "_";

                StimuliProvider::logData(fileErr.str() + "error.dat",
                                        errorValues[batchPacked]);
                StimuliProvider::logData(fileErr.str() + "est.dat",
                                        value[batchPacked]);
                StimuliProvider::logData(fileErr.str() + "ref.dat",
                                        refValues[batchPacked]);
            }
        }

        if(mLogDistrib) {
            std::stringstream fileDistrib;
            fileDistrib << dirPath << "/distrib_" << batchPos << "_";

            logTargetDistrib(meanStdDevEst, estVect,
                             fileDistrib.str() + "est.dat");
            logTargetDistrib(meanStdDevRef, refVect,
                             fileDistrib.str() + "ref.dat");
        }

    }

    for (int batchPos = 0; batchPos < (int)values.dimB(); batchPos += mBatchPacked) {
        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of
            // the set)
            continue;
        }

        std::cout << "MSE #" << id << " = " << meanSquareErrors[batchPos]
                  << " Cell: " << mCell->getName() << std::endl;

        if(meanSquareErrors[batchPos] > 0.0001)
            std::cout << Utils::cwarning << std::setprecision(10)
                      << "Warning: Mean square error computed in TargetCompare is superior "
                      << "than 0.0001 for cell " << mCell->getName() << "\n"
                      << Utils::cdef << std::endl;
    }

    TargetScore::process(set);
}

void N2D2::TargetCompare::logTargetDistrib(const std::pair<double, double> meanStdDev,
                                           std::vector<double> target,
                                           const std::string& fileName)
{
    // Write data file
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not save target distrib file.");

    std::copy(target.begin(),
          target.end(),
          std::ostream_iterator<double>(data, "\n"));
    data.close();

    std::ostringstream label;
    label << "\"Average: " << meanStdDev.first << "\\n";
    label << "Std. dev.: " << meanStdDev.second << "\"";
    label << " at graph 0.7, graph 0.8 front";

    // Plot results
    Gnuplot gnuplot;
    gnuplot.set("grid front").set("key off");
    gnuplot << "binwidth=0.01";
    gnuplot << "bin(x,width)=width*floor(x/width+0.5)";
    gnuplot.set("boxwidth", "binwidth");
    gnuplot.set("style data boxes").set("style fill solid noborder");
    gnuplot.set("xtics", "0.2");
    gnuplot.set("mxtics", "2");
    gnuplot.set("grid", "mxtics");
    gnuplot.set("label", label.str());
    gnuplot.set("yrange", "[0:]");

    gnuplot.set("style rect fc lt -1 fs solid 0.15 noborder behind");
    gnuplot.set("obj rect from graph 0, graph 0 to -1, graph 1");
    gnuplot.set("obj rect from 1, graph 0 to graph 1, graph 1");

    const double minVal = (target.front() < -1.0) ? target.front() : -1.0;
    const double maxVal = (target.back() > 1.0) ? target.back() : 1.0;
    gnuplot.setXrange(minVal - 0.05, maxVal + 0.05);

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName,
             "using (bin($1,binwidth)):(1.0) smooth freq with boxes");


}
N2D2::TargetCompare::~TargetCompare()
{
    // dtor
}
