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

#ifndef N2D2_CONVCELL_TRANSCODE_H
#define N2D2_CONVCELL_TRANSCODE_H

#include "ConvCell_Frame.hpp"
#include "ConvCell_Spike.hpp"

#ifdef CUDA
#include "ConvCell_Frame_CUDA.hpp"
#endif

namespace N2D2 {
template <class FRAME = ConvCell_Frame, class SPIKE = ConvCell_Spike>
class ConvCell_Transcode : public FRAME, public SPIKE {
public:
    enum TranscodeMode {
        Frame,
        Spike
    };

    ConvCell_Transcode(Network& net,
                       const std::string& name,
                       unsigned int kernelWidth,
                       unsigned int kernelHeight,
                       unsigned int nbOutputs,
                       unsigned int subSampleX = 1,
                       unsigned int subSampleY = 1,
                       unsigned int strideX = 1,
                       unsigned int strideY = 1,
                       int paddingX = 0,
                       int paddingY = 0,
                       const std::shared_ptr<Activation<Float_T> >& activation
                       = std::make_shared<TanhActivation_Frame<Float_T> >());
    static std::shared_ptr<ConvCell> create(Network& net,
                                            const std::string& name,
                                            unsigned int kernelWidth,
                                            unsigned int kernelHeight,
                                            unsigned int nbOutputs,
                                            unsigned int subSampleX = 1,
                                            unsigned int subSampleY = 1,
                                            unsigned int strideX = 1,
                                            unsigned int strideY = 1,
                                            int paddingX = 0,
                                            int paddingY = 0,
                                            const std::shared_ptr
                                            <Activation<Float_T> >& activation
                                            = std::make_shared
                                            <TanhActivation_Frame<Float_T> >())
    {
        return std::make_shared<ConvCell_Transcode>(net,
                                                    name,
                                                    kernelWidth,
                                                    kernelHeight,
                                                    nbOutputs,
                                                    subSampleX,
                                                    subSampleY,
                                                    strideX,
                                                    strideY,
                                                    paddingX,
                                                    paddingY,
                                                    activation);
    }

    inline unsigned int getNbChannels() const;
    inline bool isConnection(unsigned int channel, unsigned int output) const;
    void addInput(StimuliProvider& sp,
                  unsigned int channel,
                  unsigned int x0,
                  unsigned int y0,
                  unsigned int width,
                  unsigned int height,
                  const std::vector<bool>& mapping = std::vector<bool>());
    void addInput(StimuliProvider& sp,
                  unsigned int x0 = 0,
                  unsigned int y0 = 0,
                  unsigned int width = 0,
                  unsigned int height = 0,
                  const Matrix<bool>& mapping = Matrix<bool>());
    void addInput(Cell* cell, const Matrix<bool>& mapping = Matrix<bool>());
    void addInput(Cell* cell,
                  unsigned int x0,
                  unsigned int y0,
                  unsigned int width = 0,
                  unsigned int height = 0);
    void initialize();
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    void spikeCodingCompare(const std::string& fileName) const;
    virtual ~ConvCell_Transcode() {};

private:
    inline void setWeight(unsigned int output,
                          unsigned int channel,
                          unsigned int sx,
                          unsigned int sy,
                          Float_T value);
    inline Float_T getWeight(unsigned int output,
                             unsigned int channel,
                             unsigned int sx,
                             unsigned int sy) const;
    inline void setBias(unsigned int output, Float_T value);
    inline Float_T getBias(unsigned int output) const;

    TranscodeMode mTranscodeMode;

private:
    static Registrar<ConvCell> mRegistrar;
};
}

template <class FRAME, class SPIKE>
unsigned int N2D2::ConvCell_Transcode<FRAME, SPIKE>::getNbChannels() const
{
    return (mTranscodeMode == Frame) ? FRAME::getNbChannels()
                                     : SPIKE::getNbChannels();
}

template <class FRAME, class SPIKE>
bool N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::isConnection(unsigned int channel,
                                 unsigned int output) const
{
    return (mTranscodeMode == Frame) ? FRAME::isConnection(channel, output)
                                     : SPIKE::isConnection(channel, output);
}

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode<FRAME, SPIKE>::setWeight(unsigned int output,
                                                       unsigned int channel,
                                                       unsigned int sx,
                                                       unsigned int sy,
                                                       Float_T value)
{
    FRAME::setWeight(output, channel, sx, sy, value);
    SPIKE::setWeight(output, channel, sx, sy, value);
}

template <class FRAME, class SPIKE>
N2D2::Float_T N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::getWeight(unsigned int output,
                              unsigned int channel,
                              unsigned int sx,
                              unsigned int sy) const
{
    return (mTranscodeMode == Frame)
               ? FRAME::getWeight(output, channel, sx, sy)
               : SPIKE::getWeight(output, channel, sx, sy);
}

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::setBias(unsigned int output, Float_T value)
{
    FRAME::setBias(output, value);
    SPIKE::setBias(output, value);
}

template <class FRAME, class SPIKE>
N2D2::Float_T N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::getBias(unsigned int output) const
{
    return (mTranscodeMode == Frame) ? FRAME::getBias(output)
                                     : SPIKE::getBias(output);
}

template <class FRAME, class SPIKE>
N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::ConvCell_Transcode(Network& net,
                                       const std::string& name,
                                       unsigned int kernelWidth,
                                       unsigned int kernelHeight,
                                       unsigned int nbOutputs,
                                       unsigned int subSampleX,
                                       unsigned int subSampleY,
                                       unsigned int strideX,
                                       unsigned int strideY,
                                       int paddingX,
                                       int paddingY,
                                       const std::shared_ptr
                                       <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      ConvCell(name,
               kernelWidth,
               kernelHeight,
               nbOutputs,
               subSampleX,
               subSampleY,
               strideX,
               strideY,
               paddingX,
               paddingY),
      FRAME(name,
            kernelWidth,
            kernelHeight,
            nbOutputs,
            subSampleX,
            subSampleY,
            strideX,
            strideY,
            paddingX,
            paddingY,
            activation),
      SPIKE(net,
            name,
            kernelWidth,
            kernelHeight,
            nbOutputs,
            subSampleX,
            subSampleY,
            strideX,
            strideY,
            paddingX,
            paddingY),
      mTranscodeMode(Frame)
{
    // ctor
}

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::addInput(StimuliProvider& sp,
                             unsigned int channel,
                             unsigned int x0,
                             unsigned int y0,
                             unsigned int width,
                             unsigned int height,
                             const std::vector<bool>& mapping)
{
    FRAME::addInput(sp, channel, x0, y0, width, height, mapping);
    SPIKE::addInput(sp, channel, x0, y0, width, height, mapping);
}

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::addInput(StimuliProvider& sp,
                             unsigned int x0,
                             unsigned int y0,
                             unsigned int width,
                             unsigned int height,
                             const Matrix<bool>& mapping)
{
    FRAME::addInput(sp, x0, y0, width, height, mapping);
    SPIKE::addInput(sp, x0, y0, width, height, mapping);
}

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::addInput(Cell* cell, const Matrix<bool>& mapping)
{
    FRAME::addInput(cell, mapping);
    SPIKE::addInput(cell, mapping);
}

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode<FRAME, SPIKE>::addInput(Cell* cell,
                                                      unsigned int x0,
                                                      unsigned int y0,
                                                      unsigned int width,
                                                      unsigned int height)
{
    FRAME::addInput(cell, x0, y0, width, height);
    SPIKE::addInput(cell, x0, y0, width, height);
}

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode<FRAME, SPIKE>::initialize()
{
    FRAME::initialize();
    SPIKE::initialize();
}

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::spikeCodingCompare(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error(
            "Could not save spike coding compare data file.");

    const unsigned int oxSize
        = (unsigned int)((ConvCell::mChannelsWidth + 2 * ConvCell::mPaddingX
                          - ConvCell::mKernelWidth + ConvCell::mStrideX)
                         / (double)ConvCell::mStrideX);
    const unsigned int oySize
        = (unsigned int)((ConvCell::mChannelsHeight + 2 * ConvCell::mPaddingY
                          - ConvCell::mKernelHeight + ConvCell::mStrideY)
                         / (double)ConvCell::mStrideY);

    const Tensor4d<Float_T>& outputs = FRAME::getOutputs();
    std::vector<Float_T> minVal(ConvCell::mNbOutputs);
    std::vector<Float_T> maxVal(ConvCell::mNbOutputs);

    Float_T avgSignal = 0.0;
    int avgActivity = 0;

    for (unsigned int output = 0; output < ConvCell::mNbOutputs; ++output) {
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
                    data << output << " " << ox << " " << oy << " 0 0 0\n";
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

    for (unsigned int output = 0; output < ConvCell::mNbOutputs; ++output) {
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

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::saveFreeParameters(const std::string& fileName) const
{
    FRAME::saveFreeParameters(fileName);
    SPIKE::saveFreeParameters(fileName);
}

template <class FRAME, class SPIKE>
void N2D2::ConvCell_Transcode
    <FRAME, SPIKE>::loadFreeParameters(const std::string& fileName,
                                       bool ignoreNotExists)
{
    FRAME::loadFreeParameters(fileName, ignoreNotExists);
    SPIKE::loadFreeParameters(fileName, ignoreNotExists);
}

#endif // N2D2_CONVCELL_TRANSCODE_H
