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

#ifndef N2D2_PADDINGCELL_H
#define N2D2_PADDINGCELL_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "utils/Registrar.hpp"
#include "Cell.hpp"

#ifdef WIN32
// For static library
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@PaddingCell_Frame_CUDA@N2D2@@0U?$Registrar@VPaddingCell@N2D2@@@2@A")
#endif
#endif

namespace N2D2 {
class PaddingCell : public virtual Cell {
public:
    typedef std::function
        <std::shared_ptr<PaddingCell>(const std::string&, 
                                      unsigned int,
                                      int,
                                      int,
                                      int,
                                      int)>
    RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }
    static const char* Type;

    PaddingCell(const std::string& name, 
                unsigned int nbOutputs,
                int topPad,
                int botPad,
                int leftPad,
                int rightPad);
    
    const char* getType() const
    {
        return Type;
    };
    unsigned long long int getNbConnections() const;
    
    int getTopPad() const
    {
        return mTopPad;
    };
    int getBotPad() const
    {
        return mBotPad;
    };
    int getLeftPad() const
    {
        return mLeftPad;
    };
    int getRightPad() const
    {
        return mRightPad;
    };
    void discretizeFreeParameters(unsigned int /*nbLevels*/) {}; // no free
    
    void getStats(Stats& stats) const;
    virtual ~PaddingCell() {};

protected:
    virtual void setOutputsSize();
    
    //void setTopPad (int pad) { mTopPad = pad; };
    //void setBotPad (int pad) { mBotPad = pad; };
    //void setLeftPad (int pad) { mLeftPad = pad; };
    //void setRightPad (int pad) { mRightPad = pad; };

    //PaddingCell can be useful to implement asymetric padding///
    /*
    Example for a padding set with 
        Y axis padding: {mTopPad = 1; mBotPad = 2}
        X axis padding: {mLeftPad = 2; mRightPad = 3}
            00000000000000000000000
            00111111111111111111000
            00111111111111111111000
            00111111111111111111000
            00111111111111111111000
            00111111111111111111000
            00111111111111111111000
            00000000000000000000000
            00000000000000000000000
    */

    const int mTopPad;
    const int mBotPad;
    const int mLeftPad;
    const int mRightPad;

};
}

#endif // N2D2_PADDINGCELL_H
