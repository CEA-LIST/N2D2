/*
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): N2D2 Team (n2d2-contact@cea.fr)

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

#ifndef N2D2_PRUNING_H
#define N2D2_PRUNING_H

#include "Cell/Cell.hpp"
#include "DeepNet.hpp"
#include "utils/Parameterizable.hpp"


namespace N2D2 {

// Class Pruning used to apply pruning methods to a DeepNet object
// For Cells, used directly the pruning methods
class Pruning : public Parameterizable, public std::enable_shared_from_this<Pruning> {
public:
    enum Prune_T {
        None,
        Random,
    };

    Pruning(const Prune_T pruneName = None);
    Pruning(const std::string pruneStr);

    void apply(std::shared_ptr<DeepNet>& deepNet, 
               std::vector<float> opts);

    virtual ~Pruning() {};

private:
    /// Pruning method name
    Prune_T mName;

};

// ----------------------------------------------------------------------------
// ----------------------------- Pruning methods ------------------------------
// ----------------------------------------------------------------------------

void prune_random(std::shared_ptr<DeepNet>& deepNet, 
                  const float threshold);

void prune_random(std::shared_ptr<Cell>& cell, 
                  const float threshold);


}   // N2D2

namespace {
template <>
const char* const EnumStrings<N2D2::Pruning::Prune_T>::data[]
    = {"None", "Random"};
}

#endif  // N2D2_PRUNING_H