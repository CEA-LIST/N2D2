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

#include "N2D2.hpp"

#include "Activation/Activation.hpp"
#include "Activation/Activation_Kernels.hpp"
#include "Xnet/Network.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(Activation,
             log2Round,
             (double value, double rate, double power, double expectedValue),
             // no rounding expected:
             std::make_tuple(1.254, 0.0, 1.0, 1.254),
             std::make_tuple(1.0, 1.0, 1.0, 1.0),
             std::make_tuple(1.0, 1.0, 0.0, 1.0),
             std::make_tuple(2.0, 1.0, 1.0, 2.0),
             std::make_tuple(2.0, 1.0, 0.0, 2.0),
             std::make_tuple(0.5, 1.0, 1.0, 0.5),
             std::make_tuple(0.5, 1.0, 0.0, 0.5),
             // limit cases:
             std::make_tuple(std::pow(2.0, 1.5), 1.0, 1.0, std::pow(2.0, 1.5)),
             std::make_tuple(std::pow(2.0, 1.5), 1.0, 0.0, 4.0),
             std::make_tuple(std::pow(2.0, 1.49), 1.0, 0.0, 2.0),
             std::make_tuple(std::pow(2.0, -0.5), 1.0, 1.0, std::pow(2.0, -0.5)),
             std::make_tuple(std::pow(2.0, -0.5), 1.0, 0.0, 1.0),
             std::make_tuple(std::pow(2.0, -0.51), 1.0, 0.0, 0.5),
             // rounding:
             std::make_tuple(2.5, 1.0, 1.0, 2.45082066365619),
             std::make_tuple(3.0, 1.0, 1.0, 3.036888230582165),
             std::make_tuple(3.5, 1.0, 1.0, 3.572564673766028))
{
    const double result = log2Round(value, rate, power);

    ASSERT_EQUALS_DELTA(result, expectedValue, 1.0e-12);
}

RUN_TESTS()
