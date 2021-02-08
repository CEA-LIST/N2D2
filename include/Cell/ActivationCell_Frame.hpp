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

#ifndef N2D2_ACTIVATIONCELL_FRAME_H
#define N2D2_ACTIVATIONCELL_FRAME_H

#include "Cell_Frame.hpp"
#include "ActivationCell.hpp"
#include "DeepNet.hpp"
#include "Activation/TanhActivation_Frame.hpp"

namespace N2D2 {
template <class T>
class ActivationCell_Frame : public virtual ActivationCell, public Cell_Frame<T> {
public:
    using Cell_Frame<T>::mInputs;
    using Cell_Frame<T>::mOutputs;
    using Cell_Frame<T>::mDiffInputs;
    using Cell_Frame<T>::mDiffOutputs;
    using Cell_Frame<T>::mActivation;

    ActivationCell_Frame(const DeepNet& deepNet, const std::string& name,
                   unsigned int nbOutputs,
                   const std::shared_ptr<Activation>& activation
                        = std::make_shared<TanhActivation_Frame<T> >());
    static std::shared_ptr<ActivationCell> create(const DeepNet& deepNet, 
             const std::string& name,
             unsigned int nbOutputs,
             const std::shared_ptr<Activation>& activation
                    = std::make_shared<TanhActivation_Frame<T> >())
    {
        return std::make_shared<ActivationCell_Frame<T> >(deepNet, 
                                                    name,
                                                    nbOutputs,
                                                    activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    virtual ~ActivationCell_Frame();

private:
    Tensor<T> mWorkspaceCPU;
    static Registrar<ActivationCell> mRegistrar;
};
}

#endif // N2D2_ACTIVATIONCELL_FRAME_H
