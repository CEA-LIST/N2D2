/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_POOLCELL_TRANSCODE_H
#define N2D2_POOLCELL_TRANSCODE_H

#include "PoolCell_Frame.hpp"

#ifdef CUDA
#include "PoolCell_Frame_CUDA.hpp"
#endif

#include "Cell/NodeOut.hpp"
#include "DeepNet.hpp"
#include "PoolCell_Spike.hpp"
#include "utils/Gnuplot.hpp"

namespace N2D2 {
template <class FRAME = PoolCell_Frame<Float_T>, class SPIKE = PoolCell_Spike>
class PoolCell_Transcode : public FRAME, public SPIKE {
public:
    enum TranscodeMode {
        Frame,
        Spike
    };

    PoolCell_Transcode(Network& net, const DeepNet& deepNet, 
                       const std::string& name,
                       const std::vector<unsigned int>& poolDims,
                       unsigned int nbOutputs,
                       const std::vector<unsigned int>& strideDims
                          = std::vector<unsigned int>(2, 1U),
                       const std::vector<unsigned int>& paddingDims
                          = std::vector<unsigned int>(2, 0),
                       PoolCell::Pooling pooling = PoolCell::Max,
                       const std::shared_ptr<Activation>& activation
                       = std::shared_ptr<Activation>());
    static std::shared_ptr<PoolCell>
    create(Network& net, const DeepNet& deepNet, 
           const std::string& name,
           const std::vector<unsigned int>& poolDims,
           unsigned int nbOutputs,
           const std::vector<unsigned int>& strideDims
              = std::vector<unsigned int>(2, 1U),
           const std::vector<unsigned int>& paddingDims
              = std::vector<unsigned int>(2, 0),
           PoolCell::Pooling pooling = PoolCell::Max,
           const std::shared_ptr<Activation>& activation
           = std::shared_ptr<Activation>())
    {
        return std::make_shared<PoolCell_Transcode>(net, deepNet,
                                                    name,
                                                    poolDims,
                                                    nbOutputs,
                                                    strideDims,
                                                    paddingDims,
                                                    pooling,
                                                    activation);
    }

    void addInput(StimuliProvider& sp,
                  unsigned int channel,
                  unsigned int x0,
                  unsigned int y0,
                  unsigned int width,
                  unsigned int height,
                  const Tensor<bool>& mapping = Tensor<bool>());
    void addInput(StimuliProvider& sp,
                  unsigned int x0 = 0,
                  unsigned int y0 = 0,
                  unsigned int width = 0,
                  unsigned int height = 0,
                  const Tensor<bool>& mapping = Tensor<bool>());
    void addInput(Cell* cell, const Tensor<bool>& mapping = Tensor<bool>());
    void addInput(Cell* cell,
                  unsigned int x0,
                  unsigned int y0,
                  unsigned int width = 0,
                  unsigned int height = 0);
    
    void clearInputs();

    void setExtendedPadding(const std::vector<int>& paddingDims);
    void initialize();
    void spikeCodingCompare(const std::string& fileName) const;
    virtual ~PoolCell_Transcode() {};

protected:
    TranscodeMode mTranscodeMode;

private:
    static Registrar<PoolCell> mRegistrar;
};
}

template <class FRAME, class SPIKE>
N2D2::PoolCell_Transcode
    <FRAME, SPIKE>::PoolCell_Transcode(Network& net,
           const DeepNet& deepNet, 
           const std::string& name,
           const std::vector<unsigned int>& poolDims,
           unsigned int nbOutputs,
           const std::vector<unsigned int>& strideDims,
           const std::vector<unsigned int>& paddingDims,
           PoolCell::Pooling pooling,
           const std::shared_ptr<Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      PoolCell(deepNet, name,
               poolDims,
               nbOutputs,
               strideDims,
               paddingDims,
               pooling),
      FRAME(deepNet, name,
            poolDims,
            nbOutputs,
            strideDims,
            paddingDims,
            pooling,
            activation),
      SPIKE(net, deepNet, 
            name,
            poolDims,
            nbOutputs,
            strideDims,
            paddingDims,
            pooling),
      mTranscodeMode(Frame)
{
    // ctor
}

template <class FRAME, class SPIKE>
void N2D2::PoolCell_Transcode
    <FRAME, SPIKE>::addInput(StimuliProvider& sp,
                             unsigned int channel,
                             unsigned int x0,
                             unsigned int y0,
                             unsigned int width,
                             unsigned int height,
                             const Tensor<bool>& mapping)
{
    FRAME::addInput(sp, channel, x0, y0, width, height, mapping);
    const std::vector<size_t> inputsDims = FRAME::mInputsDims;
    const Tensor<bool> frameMapping = FRAME::mMapping.clone();

    FRAME::mInputsDims.clear();
    FRAME::mMapping.clear();
    SPIKE::addInput(sp, channel, x0, y0, width, height, mapping);

    assert(inputsDims == SPIKE::mInputsDims);
    assert(frameMapping.data() == SPIKE::mMapping.data());
}

template <class FRAME, class SPIKE>
void N2D2::PoolCell_Transcode
    <FRAME, SPIKE>::addInput(StimuliProvider& sp,
                             unsigned int x0,
                             unsigned int y0,
                             unsigned int width,
                             unsigned int height,
                             const Tensor<bool>& mapping)
{
    FRAME::addInput(sp, x0, y0, width, height, mapping);
    const std::vector<size_t> inputsDims = FRAME::mInputsDims;
    const Tensor<bool> frameMapping = FRAME::mMapping.clone();

    FRAME::mInputsDims.clear();
    FRAME::mMapping.clear();
    SPIKE::addInput(sp, x0, y0, width, height, mapping);

    assert(inputsDims == SPIKE::mInputsDims);
    assert(frameMapping.data() == SPIKE::mMapping.data());
}

template <class FRAME, class SPIKE>
void N2D2::PoolCell_Transcode
    <FRAME, SPIKE>::addInput(Cell* cell, const Tensor<bool>& mapping)
{
    FRAME::addInput(cell, mapping);
    const std::vector<size_t> inputsDims = FRAME::mInputsDims;
    const Tensor<bool> frameMapping = FRAME::mMapping.clone();

    FRAME::mInputsDims.clear();
    FRAME::mMapping.clear();
    SPIKE::addInput(cell, mapping);

    assert(inputsDims == SPIKE::mInputsDims);
    assert(frameMapping.data() == SPIKE::mMapping.data());
}

template <class FRAME, class SPIKE>
void N2D2::PoolCell_Transcode<FRAME, SPIKE>::addInput(Cell* cell,
                                                      unsigned int x0,
                                                      unsigned int y0,
                                                      unsigned int width,
                                                      unsigned int height)
{
    FRAME::addInput(cell, x0, y0, width, height);
    const std::vector<size_t> inputsDims = FRAME::mInputsDims;
    const Tensor<bool> frameMapping = FRAME::mMapping.clone();

    FRAME::mInputsDims.clear();
    FRAME::mMapping.clear();
    SPIKE::addInput(cell, x0, y0, width, height);

    assert(inputsDims == SPIKE::mInputsDims);
    assert(frameMapping.data() == SPIKE::mMapping.data());
}

template <class FRAME, class SPIKE>
void N2D2::PoolCell_Transcode<FRAME, SPIKE>::clearInputs() {
    throw std::runtime_error("PoolCell_Transcode::clearInputs(): not supported.");
}

template <class FRAME, class SPIKE>
void N2D2::PoolCell_Transcode<FRAME, SPIKE>::setExtendedPadding(
    const std::vector<int>& paddingDims)
{
    FRAME::setExtendedPadding(paddingDims);
    SPIKE::setExtendedPadding(paddingDims);
}

template <class FRAME, class SPIKE>
void N2D2::PoolCell_Transcode<FRAME, SPIKE>::initialize()
{
    FRAME::initialize();
    SPIKE::initialize();
}

template <class FRAME, class SPIKE>
void N2D2::PoolCell_Transcode
    <FRAME, SPIKE>::spikeCodingCompare(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error(
            "Could not save spike coding compare data file.");

    const unsigned int oxSize
        = (unsigned int)((FRAME::mInputsDims[0] - FRAME::mPoolDims[0]
                          + FRAME::mStrideDims[0]) / (double)FRAME::mStrideDims[0]);
    const unsigned int oySize
        = (unsigned int)((FRAME::mInputsDims[1] - FRAME::mPoolDims[1]
                          + FRAME::mStrideDims[1]) / (double)FRAME::mStrideDims[1]);

    FRAME::getOutputs().synchronizeDToH();
    const Tensor<Float_T>& outputs = tensor_cast<Float_T>(FRAME::getOutputs());
    std::vector<Float_T> minVal(FRAME::getNbOutputs());
    std::vector<Float_T> maxVal(FRAME::getNbOutputs());

    Float_T avgSignal = 0.0;
    int avgActivity = 0;

    for (unsigned int output = 0; output < FRAME::getNbOutputs(); ++output) {
        minVal[output] = outputs(0, 0, output, 0);
        maxVal[output] = outputs(0, 0, output, 0);

        for (unsigned int oy = 0; oy < std::max(2U, oySize); ++oy) {
            for (unsigned int ox = 0; ox < std::max(2U, oxSize); ++ox) {
                if (ox < oxSize && oy < oySize) {
                    const int activity
                        = (int)SPIKE::mOutputs(ox, oy, output, 0)->getActivity(
                              0, 0, 0) - (int)SPIKE::mOutputs(ox, oy, output, 0)
                                             ->getActivity(0, 0, 1);

                    minVal[output]
                        = std::min(minVal[output], outputs(ox, oy, output, 0));
                    maxVal[output]
                        = std::max(maxVal[output], outputs(ox, oy, output, 0));

                    avgSignal += outputs(ox, oy, output, 0);
                    avgActivity += activity;

                    data << output << " " << ox << " " << oy << " "
                         << outputs(ox, oy, output, 0) << " " << activity
                         << "\n";
                } else {
                    // Dummy data for gnuplot
                    data << output << " " << ox << " " << oy << " 0 0\n";
                }
            }
        }

        data << "\n\n";
    }

    data.close();

    const double scalingRatio = avgActivity / avgSignal;

    // Plot results
    Gnuplot gnuplot;

    std::stringstream scalingStr;
    scalingStr << "scalingRatio=" << scalingRatio;

    gnuplot << scalingStr.str();
    gnuplot.set("key off");
    gnuplot.setXrange(-0.5, oxSize - 0.5);
    gnuplot.setYrange(oySize - 0.5, -0.5);

    for (unsigned int output = 0; output < FRAME::getNbOutputs(); ++output) {
        std::stringstream cbRangeStr, paletteStr;
        cbRangeStr << "cbrange [";
        paletteStr << "palette defined (";

        if (minVal[output] < -1.0) {
            cbRangeStr << minVal[output];
            paletteStr << minVal[output] << " \"blue\", -1 \"cyan\", ";
        } else if (minVal[output] < 0.0) {
            cbRangeStr << -1.0;
            paletteStr << "-1 \"cyan\", ";
        } else
            cbRangeStr << 0.0;

        cbRangeStr << ":";
        paletteStr << "0 \"black\"";

        if (maxVal[output] > 1.0) {
            cbRangeStr << maxVal[output];
            paletteStr << ", 1 \"white\", " << maxVal[output] << " \"red\"";
        } else if (maxVal[output] > 0.0 || !(minVal[output] < 0)) {
            cbRangeStr << 1.0;
            paletteStr << ", 1 \"white\"";
        } else
            cbRangeStr << 0.0;

        cbRangeStr << "]";
        paletteStr << ")";

        gnuplot.set(paletteStr.str());
        gnuplot.set(cbRangeStr.str());

        std::stringstream plotStr;
        plotStr << output;

        gnuplot.saveToFile(fileName, "-" + plotStr.str());
        plotStr.str(std::string());
        plotStr << "index " << output << " using 2:3:4 with image,"
                                         " \"\" index " << output
                << " using 2:3:(abs($5) < 1 ? \"\" : sprintf(\"%d\",$5)) with "
                   "labels";
        gnuplot.plot(fileName, plotStr.str());

        plotStr.str(std::string());
        plotStr << output;

        gnuplot.saveToFile(fileName, "-" + plotStr.str() + "-diff");
        plotStr.str(std::string());
        plotStr << "index " << output << " using 2:3:4 with image,"
                                         " \"\" index " << output
                << " using 2:3:(($4*scalingRatio-$5) < 1 ? \"\" : "
                   "sprintf(\"%d\",$4*scalingRatio-$5)) with labels";
        gnuplot.plot(fileName, plotStr.str());
    }
}

#endif // N2D2_POOLCELL_TRANSCODE_H
