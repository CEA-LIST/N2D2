/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <cstdlib>

#include "N2D2.hpp"

#include "Export/C/C_DeepNetExport.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST(C_DeepNetExport, generateParamsHeader)
{
    C_DeepNetExport::generateParamsHeader("params.h");

    ASSERT_EQUALS(system("g++ -Wall -Wextra -pedantic params.h"), 0);
}

RUN_TESTS()
