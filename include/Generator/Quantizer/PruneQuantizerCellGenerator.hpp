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

#ifndef N2D2_PRUNEQUANTIZERCELLGENERATOR_H
#define N2D2_PRUNEQUANTIZERCELLGENERATOR_H

#include "Quantizer/QAT/Cell/Prune/PruneQuantizerCell.hpp"
#include "Generator/Quantizer/QuantizerCellGenerator.hpp"
#include "third_party/half.hpp"

namespace N2D2 {
class PruneQuantizerCellGenerator : public QuantizerCellGenerator {
public:
    static std::shared_ptr<PruneQuantizerCell>
    generate(IniParser& iniConfig,
             const std::string& section,
             const std::string& model,
             const DataType& dataType,
             const std::string& name);

private:
    static Registrar<QuantizerCellGenerator> mRegistrar;
};
}

#endif // N2D2_PRUNEQUANTIZERCELLGENERATOR_H