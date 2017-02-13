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

#include "utils/Parameterizable.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

class TestParameterizable : public Parameterizable {
public:
    TestParameterizable()
        : mParam1(this, "Param1", 1.0),
          mParam2(this, "Param2", 3U),
          mParam3(this, "Param3", true),
          mParam4(this, "Param4", 10.0, 1.0),
          mParam5(this, "Param5", 5.0, Percent(15)),
          mParam6(this, "Param6", std::vector<int>(2, 1))
    {
    }

    Parameter<double> mParam1;
    Parameter<unsigned int> mParam2;
    Parameter<bool> mParam3;
    ParameterWithSpread<double> mParam4;
    ParameterWithSpread<double> mParam5;
    Parameter<std::vector<int> > mParam6;
};

TEST(Parameterizable, isParameter)
{
    TestParameterizable param;

    ASSERT_TRUE(param.isParameter("Param1"));
    ASSERT_TRUE(param.isParameter("Param2"));
    ASSERT_TRUE(param.isParameter("Param3"));
    ASSERT_TRUE(param.isParameter("Param4"));
    ASSERT_TRUE(param.isParameter("Param5"));
    ASSERT_TRUE(!param.isParameter("ParamX"));
}

TEST(Parameterizable, getParameter)
{
    TestParameterizable param;

    ASSERT_EQUALS(param.getParameter("Param1"), "1.00000");
    ASSERT_EQUALS(param.getParameter<double>("Param1"), 1.0);
    ASSERT_EQUALS(param.getParameter("Param2"), "3");
    ASSERT_EQUALS(param.getParameter<unsigned int>("Param2"), 3U);
    ASSERT_EQUALS(param.getParameter("Param3"), "1");
    ASSERT_EQUALS(param.getParameter<bool>("Param3"), true);
    ASSERT_EQUALS(param.getParameter("Param4"), "10.0000; 10.0000; 1.00000");
    ASSERT_EQUALS(param.getParameter<Spread<double> >("Param4"),
                  Spread<double>(10.0, 1.0));
    ASSERT_EQUALS(param.getParameter("Param5"), "5.00000; 5.00000; 15.0000%");
    ASSERT_EQUALS(param.getParameter<Spread<double> >("Param5"),
                  Spread<double>(5.0, Percent(15)));

    std::vector<int> param6 = param.getParameter<std::vector<int> >("Param6");
    ASSERT_EQUALS(param6.size(), 2U);
    ASSERT_EQUALS(param6[0], 1);
    ASSERT_EQUALS(param6[1], 1);
}

TEST(Parameterizable, setParameter)
{
    TestParameterizable param;

    std::vector<int> param6_set;
    param6_set.push_back(1);
    param6_set.push_back(2);
    param6_set.push_back(3);
    param6_set.push_back(4);

    param.setParameter("Param1", 2.3);
    param.setParameter("Param2", 236U);
    param.setParameter("Param3", false);
    param.setParameter("Param4", 6.0, Percent(20));
    param.setParameter("Param5", 11.0, 2.0);
    param.setParameter("Param6", param6_set);

    ASSERT_EQUALS(param.getParameter<double>("Param1"), 2.3);
    ASSERT_EQUALS(param.getParameter<unsigned int>("Param2"), 236U);
    ASSERT_EQUALS(param.getParameter<bool>("Param3"), false);
    ASSERT_EQUALS(param.getParameter<Spread<double> >("Param4"),
                  Spread<double>(6.0, Percent(20)));
    ASSERT_EQUALS(param.getParameter<Spread<double> >("Param5"),
                  Spread<double>(11.0, 2.0));

    ASSERT_THROW(param.setParameter("Param2", -10), std::runtime_error);

    std::vector<int> param6_get
        = param.getParameter<std::vector<int> >("Param6");
    ASSERT_EQUALS(param6_get.size(), param6_set.size());
    ASSERT_EQUALS(param6_get[0], param6_set[0]);
    ASSERT_EQUALS(param6_get[1], param6_set[1]);
    ASSERT_EQUALS(param6_get[2], param6_set[2]);
    ASSERT_EQUALS(param6_get[3], param6_set[3]);
}

TEST(Parameterizable, conversion_operator)
{
    TestParameterizable param;

    ASSERT_EQUALS(param.mParam1, 1.0);
    ASSERT_EQUALS(param.mParam2, 3U);
    ASSERT_EQUALS(param.mParam3, true);

    param.mParam1 = 12.0;
    param.mParam2 = 36U;
    param.mParam3 = false;

    ASSERT_EQUALS(param.mParam1, 12.0);
    ASSERT_EQUALS(param.mParam2, 36U);
    ASSERT_EQUALS(param.getParameter<double>("Param1"), 12.0);
    ASSERT_EQUALS(param.getParameter<unsigned int>("Param2"), 36U);
    ASSERT_EQUALS(param.getParameter<bool>("Param3"), false);
}

TEST(Parameterizable, saveParameters)
{
    const std::string fileName("Parameterizable_saveParameters.out");
    UnitTest::FileRemove(fileName);

    TestParameterizable param;
    param.saveParameters(fileName);

    ASSERT_TRUE(UnitTest::FileExists(fileName));

    ASSERT_EQUALS(UnitTest::FileReadContent(fileName, 0, 1).at(0),
                  '#'); // First line starts with #
    ASSERT_EQUALS(UnitTest::FileReadContent(fileName, 1),
                  "Param1 = 1.00000\n"
                  "Param2 = 3\n"
                  "Param3 = 1\n"
                  "Param4 = 10.0000; 10.0000; 1.00000\n"
                  "Param5 = 5.00000; 5.00000; 15.0000%\n"
                  "Param6 = 1 1 \n");
}

TEST_DATASET(Parameterizable,
             loadParameters,
             (std::string data),
             std::make_tuple( // No comment
                 "Param1 = 3.25000\n"
                 "Param2 = 590000\n"
                 "Param3 = 0\n"
                 "Param4 = 2.0; 3.0; 1.0\n"
                 "Param5 = 1.0;1.0;5%\n"
                 "Param6 = 1 2 3 4 \n"),
             std::make_tuple( // Comment at the end of a line
                 "Param1=3.25000\n"
                 "Param2 = 590000 # comment\n"
                 "Param3 = 0  \n"
                 "Param4 = 2.0; 3.0; 1.0\n"
                 "Param5 = 1.0; 1.0; 5% \n"
                 "Param6 = 1 2 3 4 \n"),
             std::make_tuple( // Comment at the beginning & thousands separator
                 "# comment\n"
                 "Param1 = 3.25\n"
                 "Param2 = 590,000\n"
                 "Param3 =     0\n"
                 "Param4 = 2.0; 1.0\n"
                 "Param5 = 1.0; 5%\n"
                 "Param6 = 1 2 3 4 \n"))
{
    const std::string fileName("Parameterizable_loadParameters.in");
    UnitTest::FileWriteContent(fileName, data);

    TestParameterizable param;
    param.loadParameters(fileName);

    ASSERT_EQUALS(param.getParameter<double>("Param1"), 3.25);
    ASSERT_EQUALS(param.getParameter<unsigned int>("Param2"), 590000U);
    ASSERT_EQUALS(param.getParameter<bool>("Param3"), false);
    ASSERT_EQUALS(param.getParameter<Spread<double> >("Param4"),
                  Spread<double>(2.0, 1.0));
    ASSERT_EQUALS(param.getParameter<Spread<double> >("Param5"),
                  Spread<double>(1.0, Percent(5)));

    std::vector<int> param6 = param.getParameter<std::vector<int> >("Param6");
    ASSERT_EQUALS(param6.size(), 4U);
    ASSERT_EQUALS(param6[0], 1);
    ASSERT_EQUALS(param6[1], 2);
    ASSERT_EQUALS(param6[2], 3);
    ASSERT_EQUALS(param6[3], 4);
}

TEST(Parameterizable, loadParameters__missing)
{
    const std::string fileName("Parameterizable_loadParameters__missing.in");
    UnitTest::FileWriteContent(fileName,
                               "Param1 = 3.25000\n"
                               "Param3 = 0\n");

    TestParameterizable param;
    param.loadParameters(fileName);

    ASSERT_EQUALS(param.getParameter<double>("Param1"), 3.25);
    ASSERT_EQUALS(param.getParameter<unsigned int>("Param2"), 3U);
    ASSERT_EQUALS(param.getParameter<bool>("Param3"), false);
}

TEST_DATASET(Parameterizable,
             loadParameters__throw,
             (std::string data),
             std::make_tuple( // Extra parameter
                 "Param1 = 3.25000\n"
                 "Param2 = 590000\n"
                 "Param3 = 0\n"
                 "ParamX = 102\n"),
             std::make_tuple( // Invalid parameter
                 "Param1 = 3.25000\n"
                 "Param2 = abc\n"
                 "Param3 = 0\n"),
             std::make_tuple( // Negative number for unsigned parameter
                 "Param1 = 3.25000\n"
                 "Param2 = -590000\n"
                 "Param3 = 0\n"),
             std::make_tuple( // Invalid syntax
                 "Param1 = 3.25000\n"
                 "Param2 : 590000\n"
                 "Param3 = 0\n"))
{
    const std::string fileName("Parameterizable_loadParameters__throw.in");
    UnitTest::FileWriteContent(fileName, data);

    TestParameterizable param;

    ASSERT_THROW(param.loadParameters(fileName), std::runtime_error);
}

RUN_TESTS()
