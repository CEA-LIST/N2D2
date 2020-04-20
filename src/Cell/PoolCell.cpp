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

#include "Cell/PoolCell.hpp"
#include "DeepNet.hpp"
#include "utils/Gnuplot.hpp"

const char* N2D2::PoolCell::Type = "Pool";

N2D2::PoolCell::PoolCell(const DeepNet& deepNet, const std::string& name,
                         const std::vector<unsigned int>& poolDims,
                         unsigned int nbOutputs,
                         const std::vector<unsigned int>& strideDims,
                         const std::vector<unsigned int>& paddingDims,
                         Pooling pooling)
    : Cell(deepNet, name, nbOutputs),
      mPoolDims(poolDims),
      mStrideDims(strideDims),
      mPaddingDims(paddingDims),
      mPooling(pooling),
      mExtPaddingDims(2 * poolDims.size(), 0)
{
    // ctor
}

void N2D2::PoolCell::setExtendedPadding(const std::vector<int>& paddingDims)
{
    if (paddingDims.size() != mExtPaddingDims.size()) {
        throw std::domain_error("PoolCell: the number of dimensions"
                                " of padding must match the number of"
                                " dimensions of the kernel.");
    }

    mExtPaddingDims = paddingDims;
}

unsigned long long int N2D2::PoolCell::getNbConnections() const
{
    const size_t oSize = (!mOutputsDims.empty())
        ? std::accumulate(mOutputsDims.begin(),
                          mOutputsDims.begin() + mPoolDims.size(),
                          1U, std::multiplies<size_t>())
        : 0U;

    unsigned long long int nbConnectionsPerConnection = 0;
    std::vector<size_t> oIndex(mPoolDims.size(), 0);

    for (size_t o = 0; o < oSize; ++o) {
        unsigned long long int nbConnectionsO = 1;
        bool stopIndex = false;

        for (int dim = mPoolDims.size() - 1; dim >= 0; --dim) {
            if (!stopIndex) {
                if (++oIndex[dim] < mOutputsDims[dim])
                    stopIndex = true;
                else
                    oIndex[dim] = 0;
            }

            const unsigned int sMax = std::min<size_t>(mInputsDims[dim]
                            - oIndex[dim] * mStrideDims[dim], mPoolDims[dim]);

            nbConnectionsO *= sMax;
        }

        nbConnectionsPerConnection += nbConnectionsO;
    }

    unsigned long long int nbConnections = 0;

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel) {
            if (isConnection(channel, output))
                nbConnections += nbConnectionsPerConnection;
        }
    }

    return nbConnections;
}

void N2D2::PoolCell::writeMap(const std::string& fileName) const
{
    const std::string dirName = Utils::dirName(fileName);

    if (!dirName.empty())
        Utils::createDirectories(dirName);

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

                if (channel > 0)
                    ytics << ", ";

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

void N2D2::PoolCell::getStats(Stats& stats) const
{
    stats.nbNodes += getOutputsSize();
    stats.nbConnections += getNbConnections();
}

std::vector<unsigned int> N2D2::PoolCell::getReceptiveField(
    const std::vector<unsigned int>& outputField) const
{
    std::vector<unsigned int> receptiveField(outputField);
    receptiveField.resize(mPoolDims.size(), 1);

    for (unsigned int dim = 0; dim < mPoolDims.size(); ++dim) {
        receptiveField[dim] = mPoolDims[dim]
                                 + (receptiveField[dim] - 1) * mStrideDims[dim];
    }

    return receptiveField;
}

void N2D2::PoolCell::setOutputsDims()
{
    if (mPoolDims.size() != mInputsDims.size() - 1) {
        std::stringstream msgStr;
        msgStr << "PoolCell::setOutputsDims(): the number of dimensions of the"
            " pooling (" << mPoolDims << ") must be equal to the number of"
            " dimensions of the inputs (" << mInputsDims << ") minus one"
            << std::endl;
        throw std::runtime_error(msgStr.str());
    }

    // Keep the last dimension of mOutputsDims
    mOutputsDims.resize(mInputsDims.size(), mOutputsDims.back());

    for (unsigned int dim = 0; dim < mPoolDims.size(); ++dim) {
        mOutputsDims[dim] = (unsigned int)((mInputsDims[dim]
            + 2 * mPaddingDims[dim] + mExtPaddingDims[dim]
            + mExtPaddingDims[mPoolDims.size() + dim]
            - mPoolDims[dim] + mStrideDims[dim])
                    / (double)mStrideDims[dim]);
    }
}

std::pair<double, double> N2D2::PoolCell::getOutputsRange() const {
    return Cell::getOutputsRangeParents();
}
