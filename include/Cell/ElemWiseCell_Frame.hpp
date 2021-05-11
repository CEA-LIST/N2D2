/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_ELEMWISECELL_FRAME_H
#define N2D2_ELEMWISECELL_FRAME_H

#include "Cell_Frame.hpp"
#include "DeepNet.hpp"
#include "ElemWiseCell.hpp"

namespace N2D2 {
class ElemWiseCell_Frame : public virtual ElemWiseCell, public Cell_Frame<Float_T> {
public:
    ElemWiseCell_Frame(const DeepNet& deepNet, const std::string& name,
                   unsigned int nbOutputs,
                   Operation operation = Sum,
                   CoeffMode mode = PerLayer,
                   const std::vector<Float_T>& weights = std::vector<Float_T>(),
                   const std::vector<Float_T>& shifts = std::vector<Float_T>(),
                   const std::shared_ptr<Activation>& activation
                   = std::shared_ptr<Activation>());
    static std::shared_ptr<ElemWiseCell> create(Network& /*net*/, const DeepNet& deepNet, 
                                            const std::string& name,
                                            unsigned int nbOutputs,
                                            Operation operation = Sum,
                                            CoeffMode mode = PerLayer,
                                            const std::vector<Float_T>& weights
                                            = std::vector<Float_T>(),
                                            const std::vector<Float_T>& shifts
                                            = std::vector<Float_T>(),
                                            const std::shared_ptr
                                            <Activation>& activation
                                            = std::shared_ptr
                                            <Activation>())
    {
        return std::make_shared<ElemWiseCell_Frame>(deepNet, name,
                                                nbOutputs,
                                                operation,
                                                mode,
                                                weights,
                                                shifts,
                                                activation);
    }

    virtual void initialize();
    virtual void initializeDataDependent();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);

    std::pair<double, double> getOutputsRange() const;

    virtual ~ElemWiseCell_Frame();

protected:
    Tensor<Float_T> mInterTerm;
    Tensor<unsigned int> mArgMax;

private:
    static Registrar<ElemWiseCell> mRegistrar;
};
}

#endif // N2D2_ELEMWISECELL_FRAME_H
