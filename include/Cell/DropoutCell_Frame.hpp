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

#ifndef N2D2_DROPOUTCELL_FRAME_H
#define N2D2_DROPOUTCELL_FRAME_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell_Frame.hpp"
#include "DropoutCell.hpp"
#include "containers/Tensor4d.hpp"

namespace N2D2 {
class DropoutCell_Frame : public virtual DropoutCell,
                               public Cell_Frame {
public:
    DropoutCell_Frame(const std::string& name, unsigned int nbOutputs);
    static std::shared_ptr<DropoutCell> create(const std::string& name,
                                               unsigned int nbOutputs)
    {
        return std::make_shared<DropoutCell_Frame>(name, nbOutputs);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double /*epsilon*/ = 1.0e-4,
                       double /*maxError*/ = 1.0e-6) {};
    virtual ~DropoutCell_Frame();

protected:
    Tensor4d<char> mMask;

private:
    static Registrar<DropoutCell> mRegistrar;
};
}

#endif // N2D2_DROPOUTCELL_FRAME_H
