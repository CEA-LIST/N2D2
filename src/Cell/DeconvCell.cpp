/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "Cell/DeconvCell.hpp"

const char* N2D2::DeconvCell::Type = "Deconv";

N2D2::DeconvCell::DeconvCell(const std::string& name,
                             unsigned int kernelWidth,
                             unsigned int kernelHeight,
                             unsigned int nbOutputs,
                             unsigned int strideX,
                             unsigned int strideY,
                             int paddingX,
                             int paddingY)
    : Cell(name, nbOutputs),
      mNoBias(this, "NoBias", true),
      mBackPropagate(this, "BackPropagate", true),
      mKernelWidth(kernelWidth),
      mKernelHeight(kernelHeight),
      mStrideX(strideX),
      mStrideY(strideY),
      mPaddingX(paddingX),
      mPaddingY(paddingY),
      mWeightsFiller(new NormalFiller<Float_T>(0.0, 0.05)),
      mBiasFiller(new NormalFiller<Float_T>(0.0, 0.05))
{
    // ctor
}

void N2D2::DeconvCell::logFreeParameters(const std::string& fileName,
                                         unsigned int output,
                                         unsigned int channel) const
{
    if (output >= mNbOutputs)
        throw std::domain_error(
            "DeconvCell::logFreeParameters(): output not within range.");

    if (channel >= getNbChannels())
        throw std::domain_error(
            "DeconvCell::logFreeParameters(): channel not within range.");

    if (!isConnection(channel, output)) {
        std::cout << Utils::cnotice << "Notice: channel #" << channel
                  << " not connected to output #" << output << "."
                  << Utils::cdef << std::endl;
        return;
    }

    Tensor2d<Float_T> weights(mKernelWidth, mKernelHeight);

    for (unsigned int y = 0; y < mKernelHeight; ++y) {
        for (unsigned int x = 0; x < mKernelWidth; ++x)
            weights(x, y) = getWeight(output, channel, x, y);
    }

    StimuliProvider::logData(fileName, weights);
}

void N2D2::DeconvCell::logFreeParameters(const std::string& fileName,
                                         unsigned int output) const
{
    if (output >= mNbOutputs)
        throw std::domain_error(
            "DeconvCell::logFreeParameters(): output not within range.");

    Tensor3d<Float_T> weights(mKernelWidth, mKernelHeight, getNbChannels());

    for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
        for (unsigned int y = 0; y < mKernelHeight; ++y) {
            for (unsigned int x = 0; x < mKernelWidth; ++x)
                weights(x, y, channel) = (isConnection(channel, output))
                                             ? getWeight(output, channel, x, y)
                                             : 0.0;
        }
    }

    StimuliProvider::logData(fileName, weights);
}

void N2D2::DeconvCell::logFreeParameters(const std::string& dirName) const
{
    Utils::createDirectories(dirName);

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        std::ostringstream fileName;
        fileName << dirName << "/cell-" << output << ".dat";

        logFreeParameters(fileName.str(), output);
    }

    std::stringstream termStr;
    termStr << "set term png size " << 50 * getNbChannels() << ","
            << 50 * mNbOutputs << " enhanced";

    Gnuplot multiplot;
    multiplot.saveToFile(dirName + ".dat");
    multiplot << termStr.str();
    multiplot.setMultiplot(mNbOutputs, getNbChannels());
    multiplot.set("lmargin 0.1");
    multiplot.set("tmargin 0.1");
    multiplot.set("rmargin 0.1");
    multiplot.set("bmargin 0.1");
    multiplot.unset("xtics");
    multiplot.unset("ytics");
    multiplot.set("format x \"\"");
    multiplot.set("format y \"\"");
    multiplot.unset("colorbox");

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        std::ostringstream fileName;
        fileName << dirName << "/cell-" << output << ".dat";

        multiplot.readCmd(fileName.str() + ".gnu");
    }
}

unsigned long long int N2D2::DeconvCell::getNbSharedSynapses() const
{
    unsigned long long int nbSharedSynapses = 0;

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (isConnection(channel, output))
                nbSharedSynapses += mKernelWidth * mKernelHeight;
        }
    }

    if (!mNoBias)
        nbSharedSynapses += mNbOutputs;

    return nbSharedSynapses;
}

unsigned long long int N2D2::DeconvCell::getNbVirtualSynapses() const
{
    unsigned long long int nbVirtualSynapses = 0;

    for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
        for (unsigned int iy = 0; iy < mChannelsHeight; ++iy) {
            for (unsigned int ix = 0; ix < mChannelsWidth; ++ix) {
                const unsigned int sxMin = (unsigned int)std::max(
                    (int)mPaddingX - (int)(ix * mStrideX), 0);
                const unsigned int syMin = (unsigned int)std::max(
                    (int)mPaddingY - (int)(iy * mStrideY), 0);
                const unsigned int sxMax = Utils::clamp<int>(
                    mOutputsWidth + mPaddingX - ix * mStrideX, 0, mKernelWidth);
                const unsigned int syMax = Utils::clamp
                    <int>(mOutputsHeight + mPaddingY - iy * mStrideY,
                          0,
                          mKernelHeight);

                for (unsigned int output = 0; output < mNbOutputs; ++output) {
                    if (isConnection(channel, output))
                        nbVirtualSynapses += (sxMax - sxMin) * (syMax - syMin);
                }
            }
        }

        if (!mNoBias)
            ++nbVirtualSynapses;
    }

    return nbVirtualSynapses;
}

void N2D2::DeconvCell::setKernel(unsigned int output,
                                 unsigned int channel,
                                 const Matrix<double>& value,
                                 bool normalize)
{
    if (output >= mNbOutputs)
        throw std::domain_error(
            "DeconvCell::setKernel(): output not within range.");

    if (channel >= getNbChannels())
        throw std::domain_error(
            "DeconvCell::setKernel(): channel not within range.");

    if (value.cols() != mKernelWidth || value.rows() != mKernelHeight)
        throw std::runtime_error("DeconvCell::setKernel(): wrong kernel size");

    if (!isConnection(channel, output))
        throw std::domain_error(
            "DeconvCell::setKernel(): channel not connected to this output.");

    double valueMin = value(0);
    double valueMax = value(0);

    if (normalize) {
        for (Matrix<double>::const_iterator it = value.begin(),
                                            itEnd = value.end();
             it != itEnd;
             ++it) {
            if ((*it) > valueMax)
                valueMax = (*it);
            if ((*it) < valueMin)
                valueMin = (*it);
        }
    }

    for (unsigned int y = 0; y < mKernelHeight; ++y) {
        for (unsigned int x = 0; x < mKernelWidth; ++x) {
            const double relWeight = (normalize) ? 2.0
                                                   * (value(y, x) - valueMin)
                                                   / (valueMax - valueMin) - 1.0
                                                 : value(y, x);
            setWeight(output, channel, x, y, relWeight);
        }
    }
}

void N2D2::DeconvCell::exportFreeParameters(const std::string& fileName) const
{
    std::ofstream syn(fileName.c_str());

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file: " + fileName);

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            for (unsigned int sy = 0; sy < mKernelHeight; ++sy) {
                for (unsigned int sx = 0; sx < mKernelWidth; ++sx)
                    syn << getWeight(output, channel, sx, sy) << " ";
            }
        }

        if (!mNoBias)
            syn << getBias(output) << " ";

        syn << "\n";
    }
}

void N2D2::DeconvCell::importFreeParameters(const std::string& fileName,
                                            bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str());

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << fileName
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + fileName);
    }

    double weight;

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            for (unsigned int sy = 0; sy < mKernelHeight; ++sy) {
                for (unsigned int sx = 0; sx < mKernelWidth; ++sx) {
                    if (!(syn >> weight))
                        throw std::runtime_error(
                            "Error while reading synaptic file: " + fileName);

                    setWeight(output, channel, sx, sy, weight);
                }
            }
        }

        if (!mNoBias) {
            if (!(syn >> weight))
                throw std::runtime_error("Error while reading synaptic file: "
                                         + fileName);

            setBias(output, weight);
        }
    }

    // Discard trailing whitespaces
    while (std::isspace(syn.peek()))
        syn.ignore();

    if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Synaptic file size larger than expected: "
                                 + fileName);
}

void N2D2::DeconvCell::logFreeParametersDistrib(const std::string
                                                & fileName) const
{
    // Append all weights
    std::vector<double> weights;
    weights.reserve(mNbOutputs * getNbChannels());

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            for (unsigned int sy = 0; sy < mKernelHeight; ++sy) {
                for (unsigned int sx = 0; sx < mKernelWidth; ++sx)
                    weights.push_back(getWeight(output, channel, sx, sy));
            }
        }

        if (!mNoBias)
            weights.push_back(getBias(output));
    }

    std::sort(weights.begin(), weights.end());

    // Write data file
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not save weights distrib file.");

    std::copy(weights.begin(),
              weights.end(),
              std::ostream_iterator<double>(data, "\n"));
    data.close();

    const std::pair<double, double> meanStdDev = Utils::meanStdDev(weights);

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

    const double minVal = (weights.front() < -1.0) ? weights.front() : -1.0;
    const double maxVal = (weights.back() > 1.0) ? weights.back() : 1.0;
    gnuplot.setXrange(minVal - 0.05, maxVal + 0.05);

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName,
                 "using (bin($1,binwidth)):(1.0) smooth freq with boxes");
}

void N2D2::DeconvCell::writeMap(const std::string& fileName) const
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
        /*
                NodeOut* const parent =
           dynamic_cast<NodeOut*>(mInputs[channel](0)->getParent());

                if (parent != NULL) {
                    data << " # " << parent->getCell().getName() << "\n";
                    ytics << "\"" << channel << " (" <<
           parent->getCell().getName() << ")\" " << channel;
                }
                else {
                    data << "\n";
                    ytics << "\"" << channel << " (env)\" " << channel;
                }
        */
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

void N2D2::DeconvCell::discretizeFreeParameters(unsigned int nbLevels)
{
#pragma omp parallel for if (mNbOutputs > 16)
    for (int output = 0; output < (int)mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            for (unsigned int sy = 0; sy < mKernelHeight; ++sy) {
                for (unsigned int sx = 0; sx < mKernelWidth; ++sx) {
                    double weight = getWeight(output, channel, sx, sy);
                    weight = Utils::round((nbLevels - 1) * weight)
                             / (nbLevels - 1);

                    setWeight(output, channel, sx, sy, weight);
                }
            }
        }

        if (!mNoBias) {
            double bias = getBias(output);
            bias = Utils::round((nbLevels - 1) * bias) / (nbLevels - 1);

            setBias(output, bias);
        }
    }
}

void N2D2::DeconvCell::randomizeFreeParameters(double stdDev)
{
    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            for (unsigned int sy = 0; sy < mKernelHeight; ++sy) {
                for (unsigned int sx = 0; sx < mKernelWidth; ++sx) {
                    double weight = getWeight(output, channel, sx, sy);
                    weight = Utils::clamp(
                        Random::randNormal(weight, stdDev), -1.0, 1.0);

                    setWeight(output, channel, sx, sy, weight);
                }
            }
        }

        if (!mNoBias) {
            double bias = getBias(output);
            bias = Utils::clamp(Random::randNormal(bias, stdDev), -1.0, 1.0);

            setBias(output, bias);
        }
    }
}

void N2D2::DeconvCell::getStats(Stats& stats) const
{
    const unsigned long long int nbVirtualSynapses = getNbVirtualSynapses();

    stats.nbNeurons += getNbOutputs() * getOutputsWidth() * getOutputsHeight();
    stats.nbNodes += getNbOutputs() * getOutputsWidth() * getOutputsHeight();
    stats.nbSynapses += getNbSharedSynapses();
    stats.nbVirtualSynapses += nbVirtualSynapses;
    stats.nbConnections += nbVirtualSynapses;
}

void N2D2::DeconvCell::setOutputsSize()
{
    mOutputsWidth = mChannelsWidth * mStrideX + mKernelWidth - 2 * mPaddingX
                    - mStrideX;
    mOutputsHeight = mChannelsHeight * mStrideY + mKernelHeight - 2 * mPaddingY
                     - mStrideY;
}
