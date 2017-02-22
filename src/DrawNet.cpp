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

#include "DrawNet.hpp"

void N2D2::DrawNet::draw(DeepNet& deepNet, const std::string& fileName)
{
    // Constants
    const unsigned int svgHeight = 600;
    const unsigned int svgMarginHeight = 20;
    const unsigned int bidimWidth = 150;
    const unsigned int bidimHeight = 70;
    const unsigned int bidimPersp = 15;
    const unsigned int unidimWidth = 20;
    const unsigned int unidimPersp = 5;

    // File creation
    std::ofstream svg(fileName.c_str());

    if (!svg.good())
        throw std::runtime_error("Could not save SVG network file.");

    // Pre-calculating
    unsigned int nbUnidimOutputsMax = 1;
    unsigned int nbBidimOutputsMax
        = deepNet.getStimuliProvider()->getNbChannels();
    unsigned int svgWidth = 200;

    const std::vector<std::vector<std::string> >& layers = deepNet.getLayers();

    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        bool bidim = false;

        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);

            if (cell->getOutputsWidth() != 1 || cell->getOutputsHeight() != 1) {
                nbBidimOutputsMax
                    = std::max(nbBidimOutputsMax, cell->getNbOutputs());
                bidim = true;
            } else
                nbUnidimOutputsMax
                    = std::max(nbUnidimOutputsMax, cell->getNbOutputs());
        }

        svgWidth += (bidim) ? 200 : 100;
    }

    svg << "<?xml version=\"1.0\" standalone=\"yes\"?>"
           "<svg version=\"1.1\" baseProfile=\"full\" "
           "xmlns=\"http://www.w3.org/2000/svg\""
           " xmlns:xlink=\"http://www.w3.org/1999/xlink\""
           " xmlns:ev=\"http://www.w3.org/2001/xml-events\""
           " width=\"" << svgWidth << "px\" height=\"" << svgHeight
        << "px\""
           " viewBox=\"0 0 " << svgWidth << " " << svgHeight
        << "\" style=\"background-color: white\">"
           "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

    double offsetX = 0.0;
    double offsetY = svgMarginHeight;
    const double centerX = offsetX + 100.0;

    svg << "<text x=\"" << centerX << "\" y=\"" << offsetY
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\""
           " font-size=\"20px\" fill=\"black\">env</text>\n";
    offsetY += 25.0;

    svg << "<text x=\"" << centerX << "\" y=\"" << offsetY
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\""
           " font-size=\"16px\" fill=\"black\">"
        << deepNet.getStimuliProvider()->getSizeX() << "x"
        << deepNet.getStimuliProvider()->getSizeY() << "</text>\n";
    offsetY += svgMarginHeight;

    const unsigned int nbChannels
        = deepNet.getStimuliProvider()->getNbChannels();
    const double vertSpace = (svgHeight - offsetY - bidimHeight
                              - svgMarginHeight) / (double)nbBidimOutputsMax;

    offsetY += nbChannels * vertSpace;

    for (unsigned int channel = 0; channel < nbChannels; ++channel) {
        const double width = bidimWidth;
        offsetY -= vertSpace;

        svg << "<polygon points=\"" << centerX - width / 2.0 + bidimPersp << ","
            << offsetY << " " << centerX + width / 2.0 + bidimPersp << ","
            << offsetY << " " << centerX + width / 2.0 - bidimPersp << ","
            << offsetY + bidimHeight << " "
            << centerX - width / 2.0 - bidimPersp << ","
            << offsetY + bidimHeight << "\""
                                        " fill=\"white\" stroke=\"black\"/>\n";
    }

    offsetX += 200.0;

    // Actual rendering
    for (std::vector<std::vector<std::string> >::const_iterator itLayer
         = layers.begin() + 1,
         itLayerBegin = layers.begin(),
         itLayerEnd = layers.end();
         itLayer != itLayerEnd;
         ++itLayer) {
        const unsigned int nbCellPerLayer = (*itLayer).size();
        bool bidim = false;
        offsetY = 0.0;

        for (std::vector<std::string>::const_iterator it = (*itLayer).begin(),
                                                      itBegin
                                                      = (*itLayer).begin(),
                                                      itEnd = (*itLayer).end();
             it != itEnd;
             ++it) {
            const std::shared_ptr<Cell> cell = deepNet.getCell(*it);
            const unsigned int nbOutputs = cell->getNbOutputs();
            const unsigned int outputsWidth = cell->getOutputsWidth();
            const unsigned int outputsHeight = cell->getOutputsHeight();
            offsetY += svgMarginHeight;

            if (outputsWidth != 1 || outputsHeight != 1) {
                const double centerX = offsetX + 100.0;

                svg << "<text x=\"" << centerX << "\" y=\"" << offsetY
                    << "\" text-anchor=\"middle\" font-family=\"sans-serif\""
                       " font-size=\"20px\" fill=\"black\">" << *it
                    << "</text>\n";
                offsetY += 25.0;

                svg << "<text x=\"" << centerX << "\" y=\"" << offsetY
                    << "\" text-anchor=\"middle\" font-family=\"sans-serif\""
                       " font-size=\"16px\" fill=\"black\">" << nbOutputs
                    << " (" << outputsWidth << "x" << outputsHeight << ")";

                if (cell->getType() == PoolCell::Type) {
                    svg << " <tspan font-size=\"12px\" font-style=\"italic\">"
                        << std::dynamic_pointer_cast
                           <PoolCell>(cell)->getPooling() << "</tspan>";
                }

                svg << "</text>\n";

                offsetY += svgMarginHeight;

                const double vertSpace
                    = std::max((svgHeight / (double)nbCellPerLayer - 25.0
                                - 3.0 * svgMarginHeight - bidimHeight)
                               / (double)nbBidimOutputsMax,
                               0.0);

                offsetY += nbOutputs * vertSpace;

                for (unsigned int output = 0; output < nbOutputs; ++output) {
                    offsetY -= vertSpace;

                    svg << "<polygon id=\"l" << (itLayer - itLayerBegin) << "_"
                        << output << "\" points=\""
                        << centerX - bidimWidth / 2.0 + bidimPersp << ","
                        << offsetY << " " << centerX + bidimWidth / 2.0
                                             + bidimPersp << "," << offsetY
                        << " " << centerX + bidimWidth / 2.0 - bidimPersp << ","
                        << offsetY + bidimHeight << " "
                        << centerX - bidimWidth / 2.0 - bidimPersp << ","
                        << offsetY + bidimHeight
                        << "\""
                           " fill=\"white\" stroke=\"black\"/>\n";
                }

                const double xSpace = bidimWidth / (double)outputsWidth;
                const double ySpace = bidimHeight / (double)outputsHeight;

                if (outputsWidth * outputsHeight < 1000) {
                    // If output map size is too big, skip the drawing of the
                    // grid
                    for (unsigned int x = 0; x < outputsWidth; ++x) {
                        for (unsigned int y = 0; y < outputsHeight; ++y) {
                            const double baseX
                                = centerX - bidimWidth / 2.0 + x * xSpace
                                  + bidimPersp
                                    * (1.0 - 2.0 * (y / (double)outputsHeight));

                            svg << "<polygon points=\"" << baseX << ","
                                << offsetY + y* ySpace << " " << baseX + xSpace
                                << "," << offsetY + y* ySpace << " "
                                << baseX + xSpace - 2.0 * bidimPersp
                                                    / outputsHeight << ","
                                << offsetY + (y + 1) * ySpace << " "
                                << baseX - 2.0 * bidimPersp / outputsHeight
                                << "," << offsetY + (y + 1) * ySpace
                                << "\""
                                   " fill=\"white\" stroke=\"black\"/>\n";
                        }
                    }
                }

                bidim = true;

                if (itLayer + 1 == itLayerEnd)
                    continue;

                for (std::vector<std::string>::const_iterator itNext
                     = (*(itLayer + 1)).begin(),
                     itNextEnd = (*(itLayer + 1)).end();
                     itNext != itNextEnd;
                     ++itNext) {
                    const std::shared_ptr<Cell> nextCell
                        = deepNet.getCell(*itNext);
                    const std::vector<std::shared_ptr<Cell> >& parentCells
                        = deepNet.getParentCells(nextCell->getName());

                    if (std::find(parentCells.begin(), parentCells.end(), cell)
                        != parentCells.end()) {
                        if (nextCell->getType()
                            == ConvCell::Type /*|| nextCell->getType() ==
                                                 Cell::Lc*/
                            || nextCell->getType() == PoolCell::Type) {
                            unsigned int maskWidth, maskHeight, strideX,
                                strideY;

                            if (nextCell->getType() == ConvCell::Type) {
                                const std::shared_ptr<ConvCell> convCell
                                    = std::dynamic_pointer_cast
                                    <ConvCell>(nextCell);
                                maskWidth = convCell->getKernelWidth();
                                maskHeight = convCell->getKernelHeight();
                                strideX = convCell->getStrideX();
                                strideY = convCell->getStrideY();
                            }
                            /*
                            else if (nextCell->getType() == Cell::Lc) {
                                const std::shared_ptr<LcCell> lcCell =
                            std::dynamic_pointer_cast<LcCell>(nextCell);
                                maskWidth = lcCell->getKernelWidth();
                                maskHeight = lcCell->getKernelHeight();
                                strideX = lcCell->getStrideX();
                                strideY = lcCell->getStrideY();
                            }
                            */
                            else {
                                const std::shared_ptr<PoolCell> poolCell
                                    = std::dynamic_pointer_cast
                                    <PoolCell>(nextCell);
                                maskWidth = poolCell->getPoolWidth();
                                maskHeight = poolCell->getPoolHeight();
                                strideX = poolCell->getStrideX();
                                strideY = poolCell->getStrideY();
                            }

                            // Stride X
                            double baseX1 = centerX - bidimWidth / 2.0
                                            + strideX * xSpace + bidimPersp;
                            double baseX2 = baseX1
                                            - 2.0 * bidimPersp
                                              * ((maskHeight - 1)
                                                 / (double)outputsHeight);

                            svg << "<polygon points=\"" << baseX1 << ","
                                << offsetY << " " << baseX1 + maskWidth* xSpace
                                << "," << offsetY << " "
                                << baseX2 + maskWidth* xSpace
                                   - 2.0 * bidimPersp / outputsHeight << ","
                                << offsetY + maskHeight* ySpace << " "
                                << baseX2 - 2.0 * bidimPersp / outputsHeight
                                << "," << offsetY + maskHeight* ySpace
                                << "\""
                                   " fill-opacity=\"0\" stroke=\"cyan\" "
                                   "stroke-dasharray=\"2,2\" "
                                   "stroke-width=\"2\"/>\n";

                            // Stride Y
                            baseX1
                                = centerX - bidimWidth / 2.0
                                  + bidimPersp
                                    * (1.0 - 2.0 * (strideY
                                                    / (double)outputsHeight));
                            baseX2
                                = centerX - bidimWidth / 2.0
                                  + bidimPersp
                                    * (1.0 - 2.0 * ((maskHeight + strideY - 1)
                                                    / (double)outputsHeight));

                            svg << "<polygon points=\"" << baseX1 << ","
                                << offsetY + strideY* ySpace << " "
                                << baseX1 + maskWidth* xSpace << ","
                                << offsetY + strideY* ySpace << " "
                                << baseX2 + maskWidth* xSpace
                                   - 2.0 * bidimPersp / outputsHeight << ","
                                << offsetY + (maskHeight + strideY) * ySpace
                                << " " << baseX2 - 2.0 * bidimPersp
                                                   / outputsHeight << ","
                                << offsetY + (maskHeight + strideY) * ySpace
                                << "\""
                                   " fill-opacity=\"0\" stroke=\"cyan\" "
                                   "stroke-dasharray=\"2,2\" "
                                   "stroke-width=\"2\"/>\n";

                            // Initial pos
                            baseX1 = centerX - bidimWidth / 2.0 + bidimPersp;
                            baseX2 = baseX1 - 2.0 * bidimPersp
                                              * ((maskHeight - 1)
                                                 / (double)outputsHeight);

                            svg << "<polygon points=\"" << baseX1 << ","
                                << offsetY << " " << baseX1 + maskWidth* xSpace
                                << "," << offsetY << " "
                                << baseX2 + maskWidth* xSpace
                                   - 2.0 * bidimPersp / outputsHeight << ","
                                << offsetY + maskHeight* ySpace << " "
                                << baseX2 - 2.0 * bidimPersp / outputsHeight
                                << "," << offsetY + maskHeight* ySpace
                                << "\""
                                   " fill-opacity=\"0\" stroke=\"blue\" "
                                   "stroke-width=\"2\"/>\n";
                        }
                    }
                }
            } else {
                const double centerX = offsetX + 50.0;

                svg << "<text x=\"" << centerX << "\" y=\"" << offsetY
                    << "\" text-anchor=\"middle\" font-family=\"sans-serif\""
                       " font-size=\"20px\" fill=\"black\">" << *it
                    << "</text>\n";
                offsetY += 25.0;

                svg << "<text x=\"" << centerX << "\" y=\"" << offsetY
                    << "\" text-anchor=\"middle\" font-family=\"sans-serif\""
                       " font-size=\"16px\" fill=\"black\">" << nbOutputs
                    << "</text>\n";
                offsetY += svgMarginHeight;

                const double height = std::max(
                    nbOutputs * (svgHeight / (double)nbCellPerLayer - 25.0
                                 - 3.0 * svgMarginHeight - unidimPersp)
                    / (double)nbUnidimOutputsMax,
                    0.0);

                svg << "<polygon id=\"l" << (itLayer - itLayerBegin)
                    << "\" points=\"" << centerX - unidimWidth / 2.0 << ","
                    << offsetY + unidimPersp << " "
                    << centerX + unidimWidth / 2.0 << ","
                    << offsetY - unidimPersp << " "
                    << centerX + unidimWidth / 2.0 << ","
                    << offsetY + height - unidimPersp << " "
                    << centerX - unidimWidth / 2.0 << ","
                    << offsetY + height + unidimPersp
                    << "\""
                       " fill=\"white\" stroke=\"black\"/>\n";
            }

            offsetY = (it - itBegin + 1) * (svgHeight / (double)nbCellPerLayer);
        }

        offsetX += (bidim) ? 200.0 : 100.0;
    }

    svg << "</svg>";
}
