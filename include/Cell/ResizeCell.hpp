/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_RESIZECELL_H
#define N2D2_RESIZECELL_H

#include <string>
#include <vector>

#include "Cell.hpp"
#include "utils/Registrar.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {

class DeepNet;

class ResizeCell : public virtual Cell {
public:
    enum ResizeMode {
        Bilinear,   // Compatible with OpenCV resize() [INTER_LINEAR] function
        BilinearTF,  // Compatible with TensorFlow crop_and_resize() function
        NearestNeighbor,  // Compatible with OpenCV resize() [INTER_NEAREST] function
    };

    typedef std::function
        <std::shared_ptr<ResizeCell>(const DeepNet& , const std::string&,
                                         unsigned int,
                                         unsigned int,
                                         unsigned int,
                                         ResizeMode)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    ResizeCell(const DeepNet& deepNet, const std::string& name,
                   unsigned int outputsWidth,
                   unsigned int outputsHeight,
                   unsigned int nbOutputs,
                   ResizeMode resizeMode);
    const char* getType() const
    {
        return Type;
    };
    ResizeMode getMode() const
    {
        return mResizeMode;
    };
    bool isAlignedCorner() const
    {
        return mAlignCorners;
    };

    void getStats(Stats& stats) const;
    virtual ~ResizeCell() {};

protected:
    //virtual void setInputsDims(const std::vector<size_t>& dims);
    virtual void setOutputsDims();

    Parameter<bool> mAlignCorners;

    // resize type
    const ResizeMode mResizeMode;
    unsigned int mResizeOutputWidth;
    unsigned int mResizeOutputHeight;

};
}

namespace {
template <>
const char* const EnumStrings<N2D2::ResizeCell::ResizeMode>::data[]
    = {"Bilinear", "BilinearTF", "NearestNeighbor"};
}

#endif // N2D2_RESIZECELL_H
