/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_RPCELL_H
#define N2D2_RPCELL_H

#include <string>
#include <vector>

#include "Cell.hpp"
#include "utils/Registrar.hpp"

#ifdef WIN32
// For static library
/*
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@RPCell_Frame_CUDA@N2D2@@0U?$Registrar@VRPCell@N2D2@@@2@A")
#endif
*/
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@RPCell_Frame@N2D2@@0U?$Registrar@VRPCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class RPCell : public virtual Cell {
public:
    typedef std::function
        <std::shared_ptr<RPCell>(const std::string&,
                                 unsigned int,
                                 unsigned int,
                                 unsigned int,
                                 unsigned int)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    RPCell(const std::string& name,
           unsigned int nbAnchors,
           unsigned int nbProposals,
           unsigned int scoreIndex = 0,
           unsigned int IoUIndex = 5);
    const char* getType() const
    {
        return Type;
    };
    unsigned int getNbAnchors() const
    {
        return mNbAnchors;
    };
    unsigned int getNbProposals() const
    {
        return mNbProposals;
    };
    double getMinWidth() const { return (double) mMinWidth; };
    double getMinHeight() const { return (double) mMinHeight; };
    double getNMSParam() const { return (double) mNMS_IoU_Threshold; };
    unsigned int getScoreIndex() const
    {
        return (unsigned int) mScoreIndex;
    };
    unsigned int getIoUIndex() const
    {
        return (unsigned int) mIoUIndex;
    };

    unsigned int getPreNMSParam() const
    {
        return (unsigned int) mPre_NMS_TopN;
    };
    const std::vector<Tensor4d<int>::Index>& getAnchors() const
    {
        return mAnchors;
    }
    void getStats(Stats& stats) const;
    virtual ~RPCell() {};

protected:
    virtual void setOutputsSize();

    Parameter<double> mMinWidth;
    Parameter<double> mMinHeight;
    Parameter<double> mNMS_IoU_Threshold;
    Parameter<unsigned int> mPre_NMS_TopN;
    Parameter<double> mForegroundRate;
    Parameter<double> mForegroundMinIoU;
    Parameter<double> mBackgroundMaxIoU;
    Parameter<double> mBackgroundMinIoU;

    unsigned int mNbAnchors;
    unsigned int mNbProposals;
    unsigned int mScoreIndex;
    unsigned int mIoUIndex;
    std::vector<Tensor4d<int>::Index> mAnchors;
};
}

#endif // N2D2_RPCELL_H
