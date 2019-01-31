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

#include "Activation/Activation.hpp"
#include "Cell/ConvCell.hpp"
#include "containers/Matrix.hpp"
#include "controler/Interface.hpp"
#include "Solver/Solver.hpp"
#include "StimuliProvider.hpp"

const char* N2D2::ConvCell::Type = "Conv";

N2D2::ConvCell::ConvCell(const std::string& name,
                         const std::vector<unsigned int>& kernelDims,
                         unsigned int nbOutputs,
                         const std::vector<unsigned int>& subSampleDims,
                         const std::vector<unsigned int>& strideDims,
                         const std::vector<int>& paddingDims,
                         const std::vector<unsigned int>& dilationDims)
    : Cell(name, nbOutputs),
      mNoBias(this, "NoBias", false),
      mBackPropagate(this, "BackPropagate", true),
      mWeightsExportFormat(this, "WeightsExportFormat", OCHW),
      mWeightsExportFlip(this, "WeightsExportFlip", false),
      mOutputsRemap(this, "OutputsRemap", ""),
      mKernelDims(kernelDims),
      mSubSampleDims(subSampleDims),
      mStrideDims(strideDims),
      mPaddingDims(paddingDims),
      mDilationDims(dilationDims)
{
    // ctor
}

void N2D2::ConvCell::logFreeParameters(const std::string& fileName,
                                       unsigned int output,
                                       unsigned int channel) const
{
    if (output >= getNbOutputs())
        throw std::domain_error(
            "ConvCell::logFreeParameters(): output not within range.");

    if (channel >= getNbChannels())
        throw std::domain_error(
            "ConvCell::logFreeParameters(): channel not within range.");

    if (!isConnection(channel, output)) {
        std::cout << Utils::cnotice << "Notice: channel #" << channel
                  << " not connected to output #" << output << "."
                  << Utils::cdef << std::endl;
        return;
    }

    Tensor<Float_T> kernel;
    getWeight(output, channel, kernel);
    StimuliProvider::logData(fileName, kernel);
}

void N2D2::ConvCell::logFreeParameters(const std::string& fileName,
                                       unsigned int output) const
{
    if (output >= getNbOutputs())
        throw std::domain_error(
            "ConvCell::logFreeParameters(): output not within range.");

    Tensor<Float_T> weights;

    for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
        Tensor<Float_T> kernel;

        if (isConnection(channel, output))
            getWeight(output, channel, kernel);
        else
            kernel.resize(std::vector<size_t>(mKernelDims.begin(),
                                              mKernelDims.end()), 0.0);

        weights.push_back(kernel);
    }

    StimuliProvider::logData(fileName, weights);
}

void N2D2::ConvCell::logFreeParameters(const std::string& dirName) const
{
    Utils::createDirectories(dirName);

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        std::ostringstream fileName;
        fileName << dirName << "/cell-" << output << ".dat";

        logFreeParameters(fileName.str(), output);
    }

    std::stringstream termStr;
    termStr << "set term png size " << 50 * getNbChannels() << ","
            << 50 * getNbOutputs() << " enhanced";

    Gnuplot multiplot;
    multiplot.saveToFile(dirName + ".dat");
    multiplot << termStr.str();
    multiplot.setMultiplot(getNbOutputs(), getNbChannels());
    multiplot.set("lmargin 0.1");
    multiplot.set("tmargin 0.1");
    multiplot.set("rmargin 0.1");
    multiplot.set("bmargin 0.1");
    multiplot.unset("xtics");
    multiplot.unset("ytics");
    multiplot.set("format x \"\"");
    multiplot.set("format y \"\"");
    multiplot.unset("colorbox");

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        std::ostringstream fileName;
        fileName << dirName << "/cell-" << output << ".dat";

        multiplot.readCmd(fileName.str() + ".gnu");
    }
}

unsigned long long int N2D2::ConvCell::getNbSharedSynapses() const
{
    const unsigned int kernelSize = (!mKernelDims.empty())
        ? std::accumulate(mKernelDims.begin(), mKernelDims.end(),
                          1U, std::multiplies<unsigned int>())
        : 0U;

    unsigned long long int nbSharedSynapses = 0;

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (isConnection(channel, output))
                nbSharedSynapses += kernelSize;
        }
    }

    if (!mNoBias)
        nbSharedSynapses += getNbOutputs();

    return nbSharedSynapses;
}

unsigned long long int N2D2::ConvCell::getNbVirtualSynapses() const
{
    std::vector<size_t> oSizes;

    for (unsigned int dim = 0; dim < mKernelDims.size(); ++dim) {
        const int kernelExtent
            = mDilationDims[dim] * (mKernelDims[dim] - 1) + 1;

        oSizes.push_back((unsigned int)((mInputsDims[dim]
                            + 2 * mPaddingDims[dim] - kernelExtent
                            + mStrideDims[dim]) / (double)mStrideDims[dim]));
    }

    const size_t oSize = (!oSizes.empty())
        ? std::accumulate(oSizes.begin(), oSizes.end(),
                          1U, std::multiplies<size_t>())
        : 0U;

    unsigned long long int nbSynapsesPerConnection = 0;
    std::vector<size_t> oIndex(oSizes.size(), 0);

    for (size_t o = 0; o < oSize; ++o) {
        unsigned long long int nbSynapsesO = 1;
        bool stopIndex = false;

        for (int dim = oSizes.size() - 1; dim >= 0; --dim) {
            if (!stopIndex) {
                if (++oIndex[dim] < oSizes[dim])
                    stopIndex = true;
                else
                    oIndex[dim] = 0;
            }

            const int kernelExtent
                = mDilationDims[dim] * (mKernelDims[dim] - 1) + 1;
            const int sMin = (int)std::max(
                (int)mPaddingDims[dim] - (int)(oIndex[dim] * mStrideDims[dim]),
                0);
            const int sMax = Utils::clamp
                <int>(mInputsDims[dim] + mPaddingDims[dim] - oIndex[dim]
                                                            * mStrideDims[dim],
                      0,
                      kernelExtent);

            nbSynapsesO *= (unsigned long long int)(std::floor((sMax - sMin - 1)
                                             / (double)mDilationDims[dim]) + 1);
        }

        nbSynapsesPerConnection += nbSynapsesO;
    }

    unsigned long long int nbVirtualSynapses = 0;

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (isConnection(channel, output))
                nbVirtualSynapses += nbSynapsesPerConnection;
        }

        if (!mNoBias)
            ++nbVirtualSynapses;
    }

    return nbVirtualSynapses;
}

void N2D2::ConvCell::setKernel(unsigned int output,
                               unsigned int channel,
                               const Matrix<double>& value,
                               bool normalize)
{
    if (output >= getNbOutputs())
        throw std::domain_error(
            "ConvCell::setKernel(): output not within range.");

    if (channel >= getNbChannels())
        throw std::domain_error(
            "ConvCell::setKernel(): channel not within range.");

    if (mKernelDims.size() != 2
        || value.cols() != mKernelDims[0]
        || value.rows() != mKernelDims[1])
    {
        throw std::runtime_error("ConvCell::setKernel(): wrong kernel size");
    }

    if (!isConnection(channel, output))
        throw std::domain_error(
            "ConvCell::setKernel(): channel not connected to this output.");

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

    Tensor<Float_T> kernel(mKernelDims);

    for (unsigned int y = 0; y < mKernelDims[1]; ++y) {
        for (unsigned int x = 0; x < mKernelDims[0]; ++x) {
            kernel(x, y) = (normalize) ? 2.0 * (value(y, x) - valueMin)
                                                / (valueMax - valueMin) - 1.0
                                       : value(y, x);
        }
    }

    setWeight(output, channel, kernel);
}

void N2D2::ConvCell::exportFreeParameters(const std::string& fileName) const
{
    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string weightsFile = fileBase + "_weights" + fileExt;
    const std::string biasesFile = fileBase + "_biases" + fileExt;

    std::ofstream weights(weightsFile.c_str());

    if (!weights.good())
        throw std::runtime_error("Could not create synaptic file: "
                                 + weightsFile);

    const std::map<unsigned int, unsigned int> outputsMap = outputsRemap();

    if (mWeightsExportFormat == OCHW) {
        for (unsigned int output = 0; output < getNbOutputs(); ++output) {
            const unsigned int outputRemap = (!outputsMap.empty())
                ? outputsMap.find(output)->second : output;

            for (unsigned int channel = 0; channel < getNbChannels(); ++channel)
            {
                if (!isConnection(channel, outputRemap))
                    continue;

                Tensor<Float_T> kernel;
                getWeight(outputRemap, channel, kernel);

                for (unsigned int index = 0, size = kernel.size(); index < size;
                    ++index)
                {
                    const Float_T weight = (mWeightsExportFlip)
                        ? kernel(size - 1 - index)
                        : kernel(index);

                    weights << weight << " ";
                }
            }

            weights << "\n";
        }
    }
    else if (mWeightsExportFormat == HWCO) {
        const unsigned int kernelSize = (!mKernelDims.empty())
            ? std::accumulate(mKernelDims.begin(), mKernelDims.end(),
                              1U, std::multiplies<unsigned int>())
            : 0U;

        for (unsigned int index = 0; index < kernelSize; ++index) {
            for (unsigned int channel = 0; channel < getNbChannels();
                ++channel)
            {
                for (unsigned int output = 0; output < getNbOutputs(); ++output)
                {
                    const unsigned int outputRemap = (!outputsMap.empty())
                        ? outputsMap.find(output)->second : output;

                    if (!isConnection(channel, outputRemap))
                        continue;

                    Tensor<Float_T> kernel;
                    getWeight(outputRemap, channel, kernel);

                    const Float_T weight = (mWeightsExportFlip)
                        ? kernel(kernelSize - 1 - index)
                        : kernel(index);

                    weights << weight << " ";
                }
            }

            weights << "\n";
        }
    }
    else
        throw std::runtime_error("Unsupported weights export format");

    if (!mNoBias) {
        std::ofstream biases(biasesFile.c_str());

        if (!biases.good())
            throw std::runtime_error("Could not create synaptic file: "
                                      + biasesFile);

        for (unsigned int output = 0; output < getNbOutputs(); ++output) {
            const unsigned int outputRemap = (!outputsMap.empty())
                ? outputsMap.find(output)->second : output;

            Tensor<Float_T> bias;
            getBias(outputRemap, bias);
            biases << bias(0) << "\n";
        }
    }
}

void N2D2::ConvCell::importFreeParameters(const std::string& fileName,
                                          bool ignoreNotExists)
{
    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const bool singleFile = (std::ifstream(fileName.c_str()).good());
    const std::string weightsFile = (singleFile) ? fileName
        : fileBase + "_weights" + fileExt;
    const std::string biasesFile = (singleFile) ? fileName
        : fileBase + "_biases" + fileExt;

    std::ifstream weights(weightsFile.c_str());

    if (!weights.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << weightsFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + weightsFile);
    }

    std::ifstream biases_;

    if (!singleFile && !mNoBias) {
        biases_.open(biasesFile.c_str());

        if (!biases_.good())
            throw std::runtime_error("Could not open synaptic file: "
                                     + biasesFile);
    }

    std::ifstream& biases = (!singleFile && !mNoBias) ? biases_ : weights;

    double weight;

    const std::map<unsigned int, unsigned int> outputsMap = outputsRemap();

    if (mWeightsExportFormat == OCHW) {
        for (unsigned int output = 0; output < getNbOutputs(); ++output) {
            const unsigned int outputRemap = (!outputsMap.empty())
                ? outputsMap.find(output)->second : output;

            for (unsigned int channel = 0; channel < getNbChannels(); ++channel)
            {
                if (!isConnection(channel, outputRemap))
                    continue;

                Tensor<Float_T> kernel(mKernelDims);

                for (unsigned int index = 0, size = kernel.size(); index < size;
                    ++index)
                {
                    if (!(weights >> weight))
                        throw std::runtime_error(
                            "Error while reading synaptic file: "
                            + weightsFile);

                    if (mWeightsExportFlip)
                        kernel(size - 1 - index) = weight;
                    else
                        kernel(index) = weight;
                }

                setWeight(outputRemap, channel, kernel);
            }

            if (!mNoBias) {
                if (!(biases >> weight))
                    throw std::runtime_error("Error while reading synaptic "
                                             "file: " + biasesFile);

                Tensor<Float_T> bias({1}, weight);
                setBias(outputRemap, bias);
            }
        }
    }
    else if (mWeightsExportFormat == HWCO) {
        std::vector<size_t> kernelsDims(mKernelDims.begin(), mKernelDims.end());
        kernelsDims.push_back(getNbChannels());
        kernelsDims.push_back(getNbOutputs());

        Tensor<Float_T> kernels(kernelsDims);

        const unsigned int kernelSize = (!mKernelDims.empty())
            ? std::accumulate(mKernelDims.begin(), mKernelDims.end(),
                              1U, std::multiplies<unsigned int>())
            : 0U;

        for (unsigned int index = 0; index < kernelSize; ++index) {
            for (unsigned int channel = 0; channel < getNbChannels();
                ++channel)
            {
                for (unsigned int output = 0; output < getNbOutputs(); ++output)
                {
                    const unsigned int outputRemap = (!outputsMap.empty())
                        ? outputsMap.find(output)->second : output;

                    if (!isConnection(channel, outputRemap))
                        continue;

                    if (!(weights >> weight))
                        throw std::runtime_error(
                            "Error while reading synaptic file: "
                            + weightsFile);

                    if (mWeightsExportFlip) {
                        kernels[outputRemap][channel](kernelSize - 1 - index)
                                                                    = weight;
                    }
                    else
                        kernels[outputRemap][channel](index) = weight;
                }
            }
        }

        for (unsigned int channel = 0; channel < getNbChannels();
            ++channel)
        {
            for (unsigned int output = 0; output < getNbOutputs(); ++output)
            {
                setWeight(output, channel, kernels[output][channel]);
            }
        }

        if (!mNoBias) {
            for (unsigned int output = 0; output < getNbOutputs(); ++output) {
                const unsigned int outputRemap = (!outputsMap.empty())
                    ? outputsMap.find(output)->second : output;

                if (!(biases >> weight))
                    throw std::runtime_error("Error while reading "
                                             "synaptic file: " + biasesFile);

                Tensor<Float_T> bias({1}, weight);
                setBias(outputRemap, bias);
            }
        }
    }
    else
        throw std::runtime_error("Unsupported weights export format");

    // Discard trailing whitespaces
    while (std::isspace(weights.peek()))
        weights.ignore();

    if (weights.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Synaptic file size larger than expected: "
                                 + weightsFile);

    if (!singleFile && !mNoBias) {
        // Discard trailing whitespaces
        while (std::isspace(biases.peek()))
            biases.ignore();

        if (biases.get() != std::fstream::traits_type::eof())
            throw std::runtime_error("Synaptic file size larger than expected: "
                                     + biasesFile);
    }
}

void N2D2::ConvCell::logFreeParametersDistrib(const std::string& fileName) const
{
    // Append all weights
    const unsigned int kernelSize = (!mKernelDims.empty())
        ? std::accumulate(mKernelDims.begin(), mKernelDims.end(),
                          1U, std::multiplies<unsigned int>())
        : 0U;

    std::vector<double> weights;
    weights.reserve(getNbOutputs() * getNbChannels() * kernelSize);

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            Tensor<double> kernel;
            getWeight(output, channel, kernel);
            weights.insert(weights.end(), kernel.begin(), kernel.end());
        }

        if (!mNoBias) {
            Tensor<double> bias;
            getBias(output, bias);
            weights.push_back(bias(0));
        }
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
    gnuplot << "binwidth=0.0078";   // < 1/128
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

void N2D2::ConvCell::writeMap(const std::string& fileName) const
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
        for (unsigned int output = 0; output < getNbOutputs(); ++output) {
            data << isConnection(channel, output) << " ";
            plotCmd << isConnection(channel, output) << " ";
        }

        if (getNbOutputs() == 1)
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
        for (unsigned int output = 0; output < getNbOutputs(); ++output)
            plotCmd << "0 ";

        plotCmd << "\n";
    }

    ytics << ")";
    data.close();

    std::stringstream xtics;
    xtics << "(";

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
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

void N2D2::ConvCell::discretizeFreeParameters(unsigned int nbLevels)
{
#pragma omp parallel for if (getNbOutputs() > 16)
    for (int output = 0; output < (int)getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            Tensor<Float_T> kernel;
            getWeight(output, channel, kernel);

            for (unsigned int index = 0; index < kernel.size(); ++index) {
                kernel(index) = Utils::round((nbLevels - 1) * kernel(index))
                         / (nbLevels - 1);
            }

            setWeight(output, channel, kernel);
        }

        if (!mNoBias) {
            Tensor<Float_T> bias;
            getBias(output, bias);
            bias(0) = Utils::round((nbLevels - 1) * bias(0)) / (nbLevels - 1);

            setBias(output, bias);
        }
    }
}

std::pair<N2D2::Float_T, N2D2::Float_T> N2D2::ConvCell::getFreeParametersRange()
    const
{
    Float_T wMin = 0.0;
    Float_T wMax = 0.0;

    for (int output = 0; output < (int)getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            Tensor<Float_T> kernel;
            getWeight(output, channel, kernel);

            for (unsigned int index = 0; index < kernel.size(); ++index) {
                const Float_T weight = kernel(index);

                if (weight < wMin)  wMin = weight;
                if (weight > wMax)  wMax = weight;
            }
        }

        if (!mNoBias) {
            Tensor<Float_T> bias;
            getBias(output, bias);

            if (bias(0) < wMin)  wMin = bias(0);
            if (bias(0) > wMax)  wMax = bias(0);
        }
    }

    return std::make_pair(wMin, wMax);
}

void N2D2::ConvCell::randomizeFreeParameters(double stdDev)
{
    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            Tensor<Float_T> kernel;
            getWeight(output, channel, kernel);

            for (unsigned int index = 0; index < kernel.size(); ++index) {
                kernel(index) = Utils::clamp(
                    Random::randNormal(kernel(index), stdDev), -1.0, 1.0);
            }

            setWeight(output, channel, kernel);
        }

        if (!mNoBias) {
            Tensor<Float_T> bias;
            getBias(output, bias);
            bias(0) = Utils::clamp(Random::randNormal(bias(0), stdDev),
                                   -1.0, 1.0);

            setBias(output, bias);
        }
    }
}

void N2D2::ConvCell::processFreeParameters(const std::function
                                           <double(const double&)>& func)
{
    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (!isConnection(channel, output))
                continue;

            Tensor<Float_T> kernel;
            getWeight(output, channel, kernel);

            for (unsigned int index = 0; index < kernel.size(); ++index)
                kernel(index) = func(kernel(index));

            setWeight(output, channel, kernel);
        }

        if (!mNoBias) {
            Tensor<Float_T> bias;
            getBias(output, bias);
            bias(0) = func(bias(0));

            setBias(output, bias);
        }
    }
}

void N2D2::ConvCell::getStats(Stats& stats) const
{
    const unsigned long long int nbVirtualSynapses = getNbVirtualSynapses();

    stats.nbNeurons += getOutputsSize();
    stats.nbNodes += getOutputsSize();
    stats.nbSynapses += getNbSharedSynapses();
    stats.nbVirtualSynapses += nbVirtualSynapses;
    stats.nbConnections += nbVirtualSynapses;
}

std::vector<unsigned int> N2D2::ConvCell::getReceptiveField(
    const std::vector<unsigned int>& outputField) const
{
    std::vector<unsigned int> receptiveField(outputField);
    receptiveField.resize(mKernelDims.size(), 1);

    for (unsigned int dim = 0; dim < mKernelDims.size(); ++dim) {
        const int kernelExtent
            = mDilationDims[dim] * (mKernelDims[dim] - 1) + 1;

        receptiveField[dim] = mSubSampleDims[dim] * (kernelExtent
                                + (receptiveField[dim] - 1) * mStrideDims[dim]);
    }

    return receptiveField;
}

void N2D2::ConvCell::setOutputsDims()
{
    if (mKernelDims.size() != mInputsDims.size() - 1) {
        std::stringstream msgStr;
        msgStr << "ConvCell::setOutputsDims(): the number of dimensions of the"
            " kernel (" << mKernelDims << ") must be equal to the number of"
            " dimensions of the inputs (" << mInputsDims << ") minus one"
            << std::endl;
        throw std::runtime_error(msgStr.str());
    }

    // Keep the last dimension of mOutputsDims
    mOutputsDims.resize(mInputsDims.size(), mOutputsDims.back());

    for (unsigned int dim = 0; dim < mKernelDims.size(); ++dim) {
        const int kernelExtent
            = mDilationDims[dim] * (mKernelDims[dim] - 1) + 1;

        mOutputsDims[dim] = (unsigned int)std::ceil(
            std::floor((mInputsDims[dim] + 2 * mPaddingDims[dim]
                        - kernelExtent + mStrideDims[dim])
                               / (double)mStrideDims[dim])
                                    / (double)mSubSampleDims[dim]);
    }
}

std::map<unsigned int, unsigned int> N2D2::ConvCell::outputsRemap() const
{
    const std::vector<std::string> mapping = Utils::split(mOutputsRemap,
                                                          ",", true);

    unsigned int index = 0;
    std::map<unsigned int, unsigned int> outputRemap;

    for (std::vector<std::string>::const_iterator it = mapping.begin(),
        itEnd = mapping.end(); it != itEnd; ++it)
    {
        unsigned int offset;
        int step;

        std::stringstream offsetStepStr(*it);
        offsetStepStr.imbue(std::locale(std::locale(),
                            new N2D2::Utils::streamIgnore(": \t")));

        if (!(Utils::signChecked<unsigned int>(offsetStepStr) >> offset)
            || !(offsetStepStr >> step)
            || !offsetStepStr.eof())
        {
            throw std::runtime_error(
                "ConvCell::outputsRemap(): unable to read mapping: "
                + (std::string)mOutputsRemap);
        }

        for (int k = offset; k >= 0 && k < (int)getNbOutputs(); k+= step) {
            outputRemap[k] = index;
            ++index;
        }
    }

    if (!outputRemap.empty()) {
        std::cout << "ConvCell::outputsRemap(): " << mName << std::endl;

        for (std::map<unsigned int, unsigned int>::const_iterator
            it = outputRemap.begin(), itEnd = outputRemap.end(); it != itEnd;
            ++it)
        {
            std::cout << "  " << (*it).first << " -> " << (*it).second
            << std::endl;
        }
    }

    return outputRemap;
}
