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

#include "Cell/Cell.hpp"

unsigned int N2D2::Cell::mIdCnt = 1;

N2D2::Cell::Cell(const std::string& name, unsigned int nbOutputs)
    : mId(mIdCnt++),
      mName(name),
      mChannelsWidth(0),
      mChannelsHeight(0),
      mNbOutputs(nbOutputs),
      mOutputsWidth(1),
      mOutputsHeight(1)
{
    // ctor
}

void N2D2::Cell::addMultiscaleInput(HeteroStimuliProvider& sp,
                                    unsigned int x0,
                                    unsigned int y0,
                                    unsigned int width,
                                    unsigned int height,
                                    const Matrix<bool>& mapping)
{
    const unsigned int nbMaps = sp.size();

    if (!mapping.empty()) {
        unsigned int nbChannels = 0;

        for (unsigned int map = 0; map < nbMaps; ++map)
            nbChannels += sp[map]->getNbChannels();

        if (mapping.rows() != nbChannels) {
            throw std::runtime_error("Cell::addMultiscaleInput(): number of "
                                     "mapping rows must be equal to the total "
                                     "number of"
                                     " input filters");
        }
    }

    if (width == 0)
        width = sp[0]->getSizeX() - x0;
    if (height == 0)
        height = sp[0]->getSizeY() - y0;

    unsigned int mappingOffset = 0;

    for (unsigned int map = 0; map < nbMaps; ++map) {
        const unsigned int nbChannels = sp[map]->getNbChannels();
        const double scale = sp[map]->getSizeX()
                             / (double)sp[0]->getSizeX(); // TODO: unsafe legacy
        // code adaptation
        const unsigned int scaledX0 = (unsigned int)(scale * x0);
        const unsigned int scaledY0 = (unsigned int)(scale * y0);
        const unsigned int scaledWidth = (unsigned int)(scale * width);
        const unsigned int scaledHeight = (unsigned int)(scale * height);

        addInput(
            *(sp[map]),
            scaledX0,
            scaledY0,
            scaledWidth,
            scaledHeight,
            (mapping.empty())
                ? Matrix<bool>()
                : mapping.block(mappingOffset, 0, nbChannels, mapping.cols()));
        mappingOffset += nbChannels;
    }
}

void N2D2::Cell::save(const std::string& dirName) const
{
    // Save parameters
    std::ostringstream fileName;
    fileName << dirName << "/cell_" << mId << "_" << getType() << ".cfg";
    saveParameters(fileName.str());

    // Save free parameters
    fileName.str(std::string());
    fileName << dirName << "/cell_" << mId << "_" << getType() << ".syn";
    saveFreeParameters(fileName.str());
}

void N2D2::Cell::load(const std::string& dirName)
{
    // Load parameters
    std::ostringstream fileName;
    fileName << dirName << "/cell_" << mId << "_" << getType() << ".cfg";
    loadParameters(fileName.str());

    // Load free parameters
    fileName.str(std::string());
    fileName << dirName << "/cell_" << mId << "_" << getType() << ".syn";
    loadFreeParameters(fileName.str());
}

void N2D2::Cell::setInputsSize(unsigned int width, unsigned int height)
{
    if (mChannelsWidth == 0 && mChannelsHeight == 0) {
        mChannelsWidth = width;
        mChannelsHeight = height;
    } else if (mChannelsWidth != width || mChannelsHeight != height)
        throw std::runtime_error("Cell::setInputsSize(): an input area is "
                                 "already connected with a different size");
}
