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

#include "Cell/UnpoolCell.hpp"
#include "controler/Interface.hpp"
#include "DeepNet.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/Utils.hpp"

const char* N2D2::UnpoolCell::Type = "Unpool";

N2D2::UnpoolCell::UnpoolCell(const DeepNet& deepNet, const std::string& name,
                         const std::vector<unsigned int>& poolDims,
                         unsigned int nbOutputs,
                         const std::vector<unsigned int>& strideDims,
                         const std::vector<unsigned int>& paddingDims,
                         Pooling pooling)
    : Cell(deepNet, name, nbOutputs),
      mPoolDims(poolDims),
      mStrideDims(strideDims),
      mPaddingDims(paddingDims),
      mPooling(pooling)
{
    // ctor
}

unsigned long long int N2D2::UnpoolCell::getNbConnections() const
{
    const size_t iSize = (!mInputsDims.empty())
        ? std::accumulate(mInputsDims.begin(),
                          mInputsDims.begin() + mPoolDims.size(),
                          1U, std::multiplies<size_t>())
        : 0U;

    unsigned long long int nbConnectionsPerConnection = 0;
    std::vector<size_t> iIndex(mPoolDims.size(), 0);

    for (size_t i = 0; i < iSize; ++i) {
        unsigned long long int nbSynapsesI = 1;
        bool stopIndex = false;

        for (int dim = mPoolDims.size() - 1; dim >= 0; --dim) {
            if (!stopIndex) {
                if (++iIndex[dim] < mInputsDims[dim])
                    stopIndex = true;
                else
                    iIndex[dim] = 0;
            }

            const unsigned int sMin = (unsigned int)std::max(
                (int)mPaddingDims[dim] - (int)(iIndex[dim] * mStrideDims[dim]),
                0);
            const unsigned int sMax = Utils::clamp
                <int>(mOutputsDims[dim] + mPaddingDims[dim] - iIndex[dim]
                                                            * mStrideDims[dim],
                      0,
                      mPoolDims[dim]);

            nbSynapsesI *= (sMax - sMin);
        }

        nbConnectionsPerConnection += nbSynapsesI;
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

void N2D2::UnpoolCell::writeMap(const std::string& fileName) const
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

void N2D2::UnpoolCell::getStats(Stats& stats) const
{
    stats.nbNodes += getOutputsSize();
    stats.nbConnections += getNbConnections();
}

void N2D2::UnpoolCell::setOutputsDims()
{
    if (mPoolDims.size() != mInputsDims.size() - 1) {
        std::stringstream msgStr;
        msgStr << "UnpoolCell::setOutputsDims(): the number of dimensions of the"
            " pooling (" << mPoolDims << ") must be equal to the number of"
            " dimensions of the inputs (" << mInputsDims << ") minus one"
            << std::endl;
        throw std::runtime_error(msgStr.str());
    }

    // Keep the last dimension of mOutputsDims
    mOutputsDims.resize(mInputsDims.size(), mOutputsDims.back());

    for (unsigned int dim = 0; dim < mPoolDims.size(); ++dim) {
        mOutputsDims[dim] = mInputsDims[dim] * mStrideDims[dim]
            + mPoolDims[dim] - 2 * mPaddingDims[dim] - mStrideDims[dim];
    }
}

std::pair<double, double> N2D2::UnpoolCell::getOutputsRange() const {
    return Cell::getOutputsRangeParents();
}
