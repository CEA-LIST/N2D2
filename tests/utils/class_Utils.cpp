/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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

#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;


TEST(Utils, saturate_cast_signed) {
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(1024u), 127);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(1024), 127);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(128u), 127);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(128), 127);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(127u), 127);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(127), 127);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(126u), 126);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(126), 126);

    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(0), 0);

    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(-127), -127);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(-128), -128);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(-129), -128);
    ASSERT_EQUALS(Utils::saturate_cast<std::int8_t>(-1024), -128);
}

TEST(Utils, saturate_cast_unsigned) {
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(1024u), 255);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(1024), 255);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(256u), 255);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(256), 255);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(255u), 255);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(255), 255);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(254u), 254);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(254), 254);

    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(0), 0);

    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(-1), 0);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(-255), 0);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(-256), 0);
    ASSERT_EQUALS(Utils::saturate_cast<std::uint8_t>(-1024), 0);
}

RUN_TESTS()
