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

#ifndef N2D2_FCCELL_TRANSCODE_H
#define N2D2_FCCELL_TRANSCODE_H

#include "FcCell_Frame.hpp"
#include "FcCell_Spike.hpp"

#ifdef CUDA
#include "FcCell_Frame_CUDA.hpp"
#endif

namespace N2D2 {
template <class FRAME = FcCell_Frame, class SPIKE = FcCell_Spike>
class FcCell_Transcode : public FRAME, public SPIKE {
public:
    enum TranscodeMode {
        Frame,
        Spike
    };

    FcCell_Transcode(Network& net,
                     const std::string& name,
                     unsigned int nbOutputs,
                     const std::shared_ptr<Activation<Float_T> >& activation
                     = std::make_shared<TanhActivation_Frame<Float_T> >());
    static std::shared_ptr<FcCell> create(Network& net,
                                          const std::string& name,
                                          unsigned int nbOutputs,
                                          const std::shared_ptr
                                          <Activation<Float_T> >& activation
                                          = std::make_shared
                                          <TanhActivation_Frame<Float_T> >())
    {
        return std::make_shared
            <FcCell_Transcode>(net, name, nbOutputs, activation);
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
    virtual ~FcCell_Transcode() {};

protected:
    inline void
    setWeight(unsigned int output, unsigned int channel, Float_T value);
    inline Float_T getWeight(unsigned int output, unsigned int channel) const;
    inline void setBias(unsigned int output, Float_T value);
    inline Float_T getBias(unsigned int output) const;

    TranscodeMode mTranscodeMode;

private:
    static Registrar<FcCell> mRegistrar;
};
}

template <class FRAME, class SPIKE>
unsigned int N2D2::FcCell_Transcode<FRAME, SPIKE>::getNbChannels() const
{
    return (mTranscodeMode == Frame) ? FRAME::getNbChannels()
                                     : SPIKE::getNbChannels();
}

template <class FRAME, class SPIKE>
bool N2D2::FcCell_Transcode
    <FRAME, SPIKE>::isConnection(unsigned int channel,
                                 unsigned int output) const
{
    return (mTranscodeMode == Frame) ? FRAME::isConnection(channel, output)
                                     : SPIKE::isConnection(channel, output);
}

template <class FRAME, class SPIKE>
void N2D2::FcCell_Transcode<FRAME, SPIKE>::setWeight(unsigned int output,
                                                     unsigned int channel,
                                                     Float_T value)
{
    FRAME::setWeight(output, channel, value);
    SPIKE::setWeight(output, channel, value);
}

template <class FRAME, class SPIKE>
N2D2::Float_T N2D2::FcCell_Transcode
    <FRAME, SPIKE>::getWeight(unsigned int output, unsigned int channel) const
{
    return (mTranscodeMode == Frame) ? FRAME::getWeight(output, channel)
                                     : SPIKE::getWeight(output, channel);
}

template <class FRAME, class SPIKE>
void N2D2::FcCell_Transcode
    <FRAME, SPIKE>::setBias(unsigned int output, Float_T value)
{
    FRAME::setBias(output, value);
    SPIKE::setBias(output, value);
}

template <class FRAME, class SPIKE>
N2D2::Float_T N2D2::FcCell_Transcode
    <FRAME, SPIKE>::getBias(unsigned int output) const
{
    return (mTranscodeMode == Frame) ? FRAME::getBias(output)
                                     : SPIKE::getBias(output);
}

template <class FRAME, class SPIKE>
N2D2::FcCell_Transcode
    <FRAME, SPIKE>::FcCell_Transcode(Network& net,
                                     const std::string& name,
                                     unsigned int nbOutputs,
                                     const std::shared_ptr
                                     <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      FcCell(name, nbOutputs),
      FRAME(name, nbOutputs, activation),
      SPIKE(net, name, nbOutputs),
      mTranscodeMode(Frame)
{
    // ctor
}

template <class FRAME, class SPIKE>
void N2D2::FcCell_Transcode
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
void N2D2::FcCell_Transcode<FRAME, SPIKE>::addInput(StimuliProvider& sp,
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
void N2D2::FcCell_Transcode
    <FRAME, SPIKE>::addInput(Cell* cell, const Matrix<bool>& mapping)
{
    FRAME::addInput(cell, mapping);
    SPIKE::addInput(cell, mapping);
}

template <class FRAME, class SPIKE>
void N2D2::FcCell_Transcode<FRAME, SPIKE>::addInput(Cell* cell,
                                                    unsigned int x0,
                                                    unsigned int y0,
                                                    unsigned int width,
                                                    unsigned int height)
{
    FRAME::addInput(cell, x0, y0, width, height);
    SPIKE::addInput(cell, x0, y0, width, height);
}

template <class FRAME, class SPIKE>
void N2D2::FcCell_Transcode<FRAME, SPIKE>::initialize()
{
    FRAME::initialize();
    SPIKE::initialize();
}

template <class FRAME, class SPIKE>
void N2D2::FcCell_Transcode
    <FRAME, SPIKE>::spikeCodingCompare(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error(
            "Could not save spike coding compare data file.");

    const Tensor4d<Float_T>& outputs = FRAME::getOutputs();
    Float_T minSignal = 0.0;
    Float_T maxSignal = 1.0;
    int minActivity = 0;
    int maxActivity = 1;

    Float_T avgSignal = 0.0;
    int avgActivity = 0;

    for (unsigned int output = 0; output < FcCell::mNbOutputs; ++output) {
        const int activity
            = (int)SPIKE::mOutputs(output, 0)->getActivity(0, 0, 0)
              - (int)SPIKE::mOutputs(output, 0)->getActivity(0, 0, 1);

        minSignal = std::min(outputs(output, 0), minSignal);
        maxSignal = std::max(outputs(output, 0), maxSignal);
        minActivity = std::min(activity, minActivity);
        maxActivity = std::max(activity, maxActivity);

        avgSignal += std::abs(outputs(output, 0));
        avgActivity += std::abs(activity);

        data << output << " " << outputs(output, 0) << " " << activity << "\n";
    }

    data.close();

    const double scalingRatio = avgSignal / avgActivity;

    // Plot results
    Gnuplot gnuplot;
    gnuplot.set("key off")
        .set("style data boxes"); //.set("style fill solid 0.25 noborder");
    gnuplot.set("grid");
    gnuplot.set("boxwidth", 0.9);
    gnuplot.setXrange(-0.5, FcCell::mNbOutputs - 0.5);
    gnuplot.setYrange(
        (minSignal <= scalingRatio * minActivity) ? minSignal : scalingRatio
                                                                * minActivity,
        (maxSignal >= scalingRatio * maxActivity) ? maxSignal : scalingRatio
                                                                * maxActivity);
    gnuplot.setY2range(
        (minSignal <= scalingRatio * minActivity) ? minSignal / scalingRatio
                                                  : minActivity,
        (maxSignal >= scalingRatio * maxActivity) ? maxSignal / scalingRatio
                                                  : maxActivity);
    gnuplot.set("y2tics");
    gnuplot.set("ytics textcolor lt 0");
    gnuplot.set("ylabel textcolor lt 0");
    gnuplot.set("y2tics textcolor lt 1");
    gnuplot.set("y2label textcolor lt 1");
    gnuplot.setYlabel("Static signal");
    gnuplot.setY2label("Spiking activity");

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName,
                 "using 1:2 with boxes fill solid lt 0, \"\" using "
                 "1:3 with boxes lt 1 axes x1y2");
}

template <class FRAME, class SPIKE>
void N2D2::FcCell_Transcode
    <FRAME, SPIKE>::saveFreeParameters(const std::string& fileName) const
{
    FRAME::saveFreeParameters(fileName);
    SPIKE::saveFreeParameters(fileName);
}

template <class FRAME, class SPIKE>
void N2D2::FcCell_Transcode
    <FRAME, SPIKE>::loadFreeParameters(const std::string& fileName,
                                       bool ignoreNotExists)
{
    FRAME::loadFreeParameters(fileName, ignoreNotExists);
    SPIKE::loadFreeParameters(fileName, ignoreNotExists);
}

#endif // N2D2_FCCELL_TRANSCODE_H
