/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): Vincent TEMPLIER (vincent.templier@cea.fr)

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

#ifndef N2D2_ADVERSARIAL_H
#define N2D2_ADVERSARIAL_H

#include "DeepNet.hpp"
#include "utils/Parameterizable.hpp"

namespace N2D2 {

class Adversarial : public Parameterizable, public std::enable_shared_from_this<Adversarial> {
public:
    enum Attack_T {
        None,
        Vanilla,
        GN,
        FGSM,
        PGD
    };

    Adversarial(const Attack_T attackName = None);

    void attackLauncher(std::shared_ptr<DeepNet>& deepNet);

    void singleTestAdv(std::shared_ptr<DeepNet>& deepNet, 
                       std::string dirName);
    
    void multiTestAdv(std::shared_ptr<DeepNet>& deepNet, 
                      std::string dirName);

    // Setters

    void setAttackName(const Attack_T attackName)
    {
        mName = attackName;
    };
    void setEps(const float eps)
    {
        mEps = eps;
    };
    void setNbIterations(const unsigned int nbIter)
    {
        mNbIterations = nbIter;
    };
    void setRandomStart(const bool random)
    {
        mRandomStart = random;
    };
    void setTargeted(const bool targeted)
    {
        mTargeted = targeted;
    };

    // Getters

    Attack_T getAttackName()
    {
        return mName;
    };
    float getEps()
    {
        return mEps;
    };
    unsigned int getNbIterations()
    {
        return mNbIterations;
    };
    bool getRandomStart()
    {
        return mRandomStart;
    };
    bool getTargeted()
    {
        return mTargeted;
    };

    virtual ~Adversarial() {};

private:
    /// Adversarial attack name
    Attack_T mName;
    /// Degradation rate
    float mEps;
    /// Number of attacks on each image
    unsigned int mNbIterations;
    /// Random start
    bool mRandomStart;
    /// Targeted attack
    bool mTargeted;

};


// ----------------------------------------------------------------------------
// --------------------------- Adversarial attacks ----------------------------
// ----------------------------------------------------------------------------

// These attacks aim to modify the images provided by the StimuliProvider
// After calling the following functions, the images contained 
// in StimuliProvider are modified

void Vanilla_attack();

void GN_attack(std::shared_ptr<DeepNet>& deepNet, 
               const float eps);

void FGSM_attack(std::shared_ptr<DeepNet>& deepNet,
                 const float eps, 
                 const bool targeted = false);

void FFGSM_attack(std::shared_ptr<DeepNet>& deepNet,
                  const float eps, 
                  const float alpha,
                  const bool targeted = false);

void PGD_attack(std::shared_ptr<DeepNet>& deepNet,
                const float eps, 
                const unsigned int nbIter,
                const float alpha,
                const bool targeted = false,
                const bool random_start = false);

// Not implemented
/*
void CW_attack(std::shared_ptr<DeepNet>& deepNet,
               const float c,
               const float kappa,
               const unsigned int nbIter,
               const float lr);
*/

}

namespace {
template <>
const char* const EnumStrings<N2D2::Adversarial::Attack_T>::data[]
    = {"None", "Vanilla", "GN", "FGSM", "PGD"};
}

#endif // N2D2_ADVERSARIAL_H