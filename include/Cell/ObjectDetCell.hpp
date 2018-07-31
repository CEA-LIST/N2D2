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

#ifndef N2D2_OBJECTDETCELL_H
#define N2D2_OBJECTDETCELL_H

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
    "/include:?mRegistrar@ObjectDetCell_Frame_CUDA@N2D2@@0U?$Registrar@VObjectDetCell@N2D2@@@2@A")
#endif
*/
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ObjectDetCell_Frame@N2D2@@0U?$Registrar@VObjectDetCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class ObjectDetCell : public virtual Cell {
public:
    typedef std::function
        <std::shared_ptr<ObjectDetCell>(const std::string&,
                                        StimuliProvider&,
                                        const unsigned int,
                                        unsigned int,
                                        unsigned int,
                                        unsigned int,
                                        Float_T,
                                        Float_T)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    ObjectDetCell(const std::string& name,
                 StimuliProvider& sp,
                 const unsigned int nbOutputs,
                 unsigned int nbAnchors,
                 unsigned int nbProposals,
                 unsigned int nbClass,
                 Float_T nmsThreshold,
                 Float_T scoreThreshold);

    const char* getType() const
    {
        return Type;
    };

    unsigned int getNbProposals() const
    {
        return mNbProposals;
    };

    unsigned int getNbAnchors() const
    {
        return mNbAnchors;
    };

    Float_T getNMSParam() const { return (double) mNMS_IoU_Threshold; };
    Float_T getScoreThreshold() const { return (double) mScoreThreshold; };

    unsigned int getNbClass() const { return mNbClass; };

    void getStats(Stats& stats) const;

    virtual ~ObjectDetCell() {};

protected:
    virtual void setOutputsDims();

    StimuliProvider& mStimuliProvider;

    unsigned int mNbAnchors;
    unsigned int mNbProposals;
    unsigned int mNbClass;

    Float_T mNMS_IoU_Threshold;
    Float_T mScoreThreshold;


};
}

#endif // N2D2_OBJECTDETCELL_H
