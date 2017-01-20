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

#include "Cell/FMPCell.hpp"

const char* N2D2::FMPCell::Type = "FMP";

N2D2::FMPCell::FMPCell(const std::string& name,
                       double scalingRatio,
                       unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      mOverlapping(this, "Overlapping", true),
      mPseudoRandom(this, "PseudoRandom", true),
      mScalingRatio(scalingRatio),
      mPoolNbChannels(nbOutputs, 0)
{
    // ctor
}

void N2D2::FMPCell::initialize()
{
    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel)
            mPoolNbChannels[output] += isConnection(channel, output);
    }
}

unsigned long long int N2D2::FMPCell::getNbConnections() const
{
    double nbConnections = 0;

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int oy = 0; oy < mOutputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < mOutputsWidth; ++ox) {
                for (unsigned int channel = 0; channel < getNbChannels();
                     ++channel) {
                    if (isConnection(channel, output))
                        nbConnections += (mOverlapping)
                                             ? (mScalingRatio + 1)
                                               * (mScalingRatio + 1)
                                             : mScalingRatio * mScalingRatio;
                }
            }
        }
    }

    return (unsigned long long int)nbConnections;
}

void N2D2::FMPCell::writeMap(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not save map file.");

    Gnuplot::setDefaultOutput("png", "size 800,600 tiny", "png");

    Gnuplot gnuplot;
    gnuplot.set("key off").unset("colorbox");
    gnuplot.setXlabel("Output maps");
    gnuplot.setYlabel("Input maps");
    gnuplot.set("palette", "defined  (0 'white', 1 'black')");
    gnuplot.set("cbrange", "[0:1]");
    gnuplot.set("yrange", "[] reverse");
    gnuplot.set("grid", "front xtics ytics lc rgb 'grey'");

    gnuplot.saveToFile(fileName);

    std::stringstream ytics;
    ytics << "(";

    std::stringstream plotCmd;

    for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
        for (unsigned int output = 0; output < mNbOutputs; ++output) {
            data << isConnection(channel, output) << " ";
            plotCmd << isConnection(channel, output) << " ";
        }

        if (mNbOutputs == 1)
            plotCmd << "0 ";

        plotCmd << "\n";

        if (channel > 0)
            ytics << ", ";

        data << "\n";
        ytics << "\"" << channel << "\" " << channel;
    }

    if (getNbChannels() == 1) {
        for (unsigned int output = 0; output < mNbOutputs; ++output)
            plotCmd << "0 ";

        plotCmd << "\n";
    }

    ytics << ")";
    data.close();

    std::stringstream xtics;
    xtics << "(";

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        if (output > 0)
            xtics << ", ";

        xtics << "\"" << getName() << "(" << output << ")\" " << output;
    }

    xtics << ")";

    gnuplot.set("xtics rotate by 90", xtics.str());
    gnuplot.set("ytics", ytics.str());

    gnuplot.plot("-", "matrix with image");
    gnuplot << plotCmd.str();
    gnuplot << "e";

    Gnuplot::setDefaultOutput();
}

void N2D2::FMPCell::getStats(Stats& stats) const
{
    stats.nbNodes += getNbOutputs() * getOutputsWidth() * getOutputsHeight();
    stats.nbConnections += getNbConnections();
}

void N2D2::FMPCell::setOutputsSize()
{
    mOutputsWidth = (unsigned int)Utils::round(mChannelsWidth / mScalingRatio);
    mOutputsHeight
        = (unsigned int)Utils::round(mChannelsHeight / mScalingRatio);
}
