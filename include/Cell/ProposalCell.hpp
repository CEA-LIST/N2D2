/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_PROPOSALCELL_H
#define N2D2_PROPOSALCELL_H

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
    "/include:?mRegistrar@ProposalCell_Frame_CUDA@N2D2@@0U?$Registrar@VProposalCell@N2D2@@@2@A")
#endif
*/
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@ProposalCell_Frame@N2D2@@0U?$Registrar@VProposalCell@N2D2@@@2@A")
#endif

namespace N2D2 {
class ProposalCell : public virtual Cell {
public:
    typedef std::function
        <std::shared_ptr<ProposalCell>(const std::string&,
                                        StimuliProvider&,
                                        unsigned int,
                                        unsigned int,
                                        unsigned int,
                                        bool,
                                        std::vector<double>,
                                        std::vector<double>)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    ProposalCell(const std::string& name,
                 StimuliProvider& sp,
                 unsigned int nbProposals,
                 unsigned int scoreIndex = 0,
                 unsigned int IoUIndex = 5,
                 bool isNms = false,
                 std::vector<double> meansFactor = { 0.0, 0.0, 0.0, 0.0},
                 std::vector<double> stdFactor = {1.0, 1.0, 1.0, 1.0});

    const char* getType() const
    {
        return Type;
    };

    unsigned int getNbProposals() const
    {
        return mNbProposals;
    };

    double getNMSParam() const { return (double) mNMS_IoU_Threshold; };
    double getScoreThreshold() const { return (double) mScoreThreshold; };

    unsigned int getScoreIndex() const
    {
        return (unsigned int) mScoreIndex;
    };
    unsigned int getIoUIndex() const
    {
        return (unsigned int) mIoUIndex;
    };

    bool getIsNMS() const { return mApplyNMS; };
    std::vector<double> getMeanFactor() const { return mMeanFactor; };
    std::vector<double> getStdFactor() const { return mStdFactor; };
    
    void getStats(Stats& stats) const;

    virtual ~ProposalCell() {};

protected:
    virtual void setOutputsSize();

    Parameter<double> mNMS_IoU_Threshold;
    Parameter<double> mScoreThreshold;
    Parameter<bool> mKeepMax;
    StimuliProvider& mStimuliProvider;

    unsigned int mNbProposals;
    unsigned int mScoreIndex;
    unsigned int mIoUIndex;
    bool mApplyNMS;
    std::vector<double> mMeanFactor;
    std::vector<double> mStdFactor;
    unsigned int mNbClass;

    
};
}

#endif // N2D2_PROPOSALCELL_H
