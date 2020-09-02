/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <cstdlib>

#include "N2D2.hpp"

#include "Environment.hpp"
#include "Export/C/C_DeepNetExport.hpp"
#include "Export/C/C_FcCellExport.hpp"
#include "Cell/FcCell_Frame.hpp"
#include "DeepNet.hpp"
#include "Network.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST(C_FcCellExport, generate)
{
    Network net;
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {24, 24, 1});

    FcCell_Frame<Float_T> fcCell(net, "fc_C", 10);
    fcCell.addInput(env);
    fcCell.initialize();

    CellExport::mPrecision = static_cast<CellExport::Precision>(-32);


    C_FcCellExport::generate(fcCell, ".");
    C_DeepNetExport::generateParamsHeader("include/params.h");

    std::string cmd = "g++ -Wall -Wextra -pedantic";
    cmd += " -I./include/ -I" + std::string(N2D2_PATH("export/C/include"));
    cmd += " include/fc_C.h";

    ASSERT_EQUALS(system(cmd.c_str()), 0);
}

RUN_TESTS()
