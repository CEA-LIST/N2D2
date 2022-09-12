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

#ifndef N2D2_SATQUANTIZERACTIVATION_H
#define N2D2_SATQUANTIZERACTIVATION_H

#include "Quantizer/QAT/Activation/QuantizerActivation.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {

class SATQuantizerActivation : virtual public QuantizerActivation {
public:
    typedef std::function<std::shared_ptr<SATQuantizerActivation>()> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    static const char* Type;

    SATQuantizerActivation();
    virtual const char* getType() const
    {
        return Type;
    }

    virtual void setSolver(const std::shared_ptr<Solver>& solver) = 0;
    virtual std::shared_ptr<Solver> getSolver() = 0;

    void setAlpha(float alpha)
    {
        mAlphaParameter=alpha;
        mSyncAlpha = false;
    };
    float getAlphaParameter()
    {
        return mAlphaParameter;
    };

    virtual void exportParameters(const std::string& /*dirName*/, const std::string& /*cellName*/) const {};
    virtual void importParameters(const std::string& /*dirName*/, const std::string& /*cellName*/, bool /*ignoreNotExists*/) {};

    virtual ~SATQuantizerActivation() {};

protected:
    Parameter<float> mAlphaParameter;
    bool mSyncAlpha = false;

private:
};

}

#endif  // N2D2_SATQUANTIZERACTIVATION_H
