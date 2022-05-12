/**
 * (C) Copyright 2020 CEA LIST. All Rights Reserved.
 *  Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
 *                  David BRIAND (david.briand@cea.fr)
 *                  Inna KUCHER (inna.kucher@cea.fr)
 *                  Olivier BICHLER (olivier.bichler@cea.fr)
 *                  Vincent TEMPLIER (vincent.templier@cea.fr)
 * 
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 * 
 */

#ifndef N2D2_LSQQUANTIZERACTIVATION_H
#define N2D2_LSQQUANTIZERACTIVATION_H

#include "Quantizer/QAT/Activation/QuantizerActivation.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {

class LSQQuantizerActivation : virtual public QuantizerActivation {
public:
    typedef std::function<std::shared_ptr<LSQQuantizerActivation>()> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static const char* Type;

    LSQQuantizerActivation();


    void setOptInitStepSize(bool flag) 
    {
        mSetOptInitStepSize = flag;
    };
    void setStepSizeValue(float stepSize)
    {
        mStepSizeParameter=stepSize;
    };

    float getStepSizeValue() 
    {
        return mStepSizeParameter;
    };
    virtual const char* getType() const {
        return Type;
    };
 
    virtual void exportParameters(const std::string& /*dirName*/, const std::string& /*cellName*/) const {};
    virtual void importParameters(const std::string& /*dirName*/, const std::string& /*cellName*/, bool /*ignoreNotExists*/) {};

    virtual ~LSQQuantizerActivation() {};

protected:
    // Other parameters
    Parameter<float> mStepSizeParameter;
    Parameter<bool> mSetOptInitStepSize;

    std::pair<int, int> mBitRanges;
private:
};

}

#endif // N2D2_LSQQUANTIZERACTIVATION_H
