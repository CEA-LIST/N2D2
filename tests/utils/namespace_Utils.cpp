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

#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST(Utils, round)
{
    ASSERT_EQUALS(Utils::round(23.5, Utils::HalfUp), 24.0);
    ASSERT_EQUALS(Utils::round(-23.5, Utils::HalfUp), -23.0);
    ASSERT_EQUALS(Utils::round(23.5, Utils::HalfDown), 23.0);
    ASSERT_EQUALS(Utils::round(-23.5, Utils::HalfDown), -24.0);
    ASSERT_EQUALS(Utils::round(23.5, Utils::HalfAwayFromZero), 24.0);
    ASSERT_EQUALS(Utils::round(-23.5, Utils::HalfAwayFromZero), -24.0);
    ASSERT_EQUALS(Utils::round(23.5, Utils::HalfTowardsZero), 23.0);
    ASSERT_EQUALS(Utils::round(-23.5, Utils::HalfTowardsZero), -23.0);
}

TEST_DATASET(
    Utils,
    normalizedAngle,
    (double rad, double radMinusPiToPi, double radZeroToTwoPi),
    std::make_tuple(-M_PI, -M_PI, M_PI),
    std::make_tuple(-M_PI / 2.0, -M_PI / 2.0, 3.0 * M_PI / 2.0),
    std::make_tuple(0.0, 0.0, 0.0),
    std::make_tuple(M_PI, -M_PI, M_PI),
    std::make_tuple(3.0 * M_PI / 2.0, -M_PI / 2.0, 3.0 * M_PI / 2.0),
    std::make_tuple(2.0 * M_PI - 1.0e-12, -1.0e-12, 2.0 * M_PI - 1.0e-12),
    std::make_tuple(2.0 * M_PI + 1.0e-12, 1.0e-12, 1.0e-12),
    std::make_tuple(3.0 * M_PI - 1.0e-12, M_PI - 1.0e-12, M_PI - 1.0e-12),
    std::make_tuple(3.0 * M_PI + 1.0e-12, -M_PI + 1.0e-12, M_PI + 1.0e-12))
{
    ASSERT_EQUALS_DELTA(Utils::normalizedAngle(rad, Utils::MinusPiToPi),
                        radMinusPiToPi,
                        1.0e-12);
    ASSERT_EQUALS_DELTA(Utils::normalizedAngle(rad, Utils::ZeroToTwoPi),
                        radZeroToTwoPi,
                        1.0e-12);
}

TEST_DATASET(Utils,
             degToRad,
             (double deg, double rad),
             std::make_tuple(0.0, 0.0),
             std::make_tuple(180.0, M_PI),
             std::make_tuple(360.0, 2.0 * M_PI))
{
    ASSERT_EQUALS(Utils::degToRad(deg), rad);
}

TEST_DATASET(Utils,
             radToDeg,
             (double deg, double rad),
             std::make_tuple(0.0, 0.0),
             std::make_tuple(180.0, M_PI),
             std::make_tuple(360.0, 2.0 * M_PI))
{
    ASSERT_EQUALS(Utils::radToDeg(rad), deg);
}

TEST_DATASET(Utils,
             searchAndReplace,
             (std::string value, std::string expected),
             std::make_tuple("token", "nekot"),
             std::make_tuple(" token ", " nekot "),
             std::make_tuple("blablatokenblabla", "blablanekotblabla"),
             std::make_tuple("blablatokenblabla token",
                             "blablanekotblabla nekot"),
             std::make_tuple("tokenblablatokenblabla",
                             "nekotblablanekotblabla"))
{
    ASSERT_EQUALS(Utils::searchAndReplace(value, "token", "nekot"), expected);
}

TEST_DATASET(Utils,
             match,
             (std::string first, std::string second, bool match),
             std::make_tuple("test", "test", true),
             std::make_tuple("*test", "test", true),
             std::make_tuple("test*", "test", true),
             std::make_tuple("*test", "qsdgtest", true),
             std::make_tuple("test*", "testqsdg", true),
             std::make_tuple("test*aa", "testqsdgaa", true),
             std::make_tuple("aa*test", "aaqsdgtest", true),
             std::make_tuple("aa*test*", "aaqsdgtest", true),
             std::make_tuple("aa*test*", "aaqsdgtestqsf", true),
             std::make_tuple("test", "tet", false),
             std::make_tuple("*test", "dest", false),
             std::make_tuple("test*", "tes", false),
             std::make_tuple("aa*test*", "aaqsdgtst", false),
             std::make_tuple("aa*test*", "aqsdgtestqsf", false))
{
    ASSERT_EQUALS(Utils::match(first, second), match);
}

TEST_DATASET(
    Utils,
    fileName,
    (std::string filePath,
     std::string dirName,
     std::string baseName,
     std::string fileBaseName,
     std::string fileExtension),
    std::make_tuple("test.dat", ".", "test.dat", "test", "dat"),
    std::make_tuple(
        "/local/test.dat", "/local/", "test.dat", "/local/test", "dat"),
    std::make_tuple(
        "/lo.cal/test.dat", "/lo.cal/", "test.dat", "/lo.cal/test", "dat"),
    std::make_tuple("/lo.cal/test", "/lo.cal/", "test", "/lo.cal/test", ""),
    std::make_tuple("./lo.cal/test", "./lo.cal/", "test", "./lo.cal/test", ""),
    std::make_tuple("C:\\local\\test.dat",
                    "C:\\local\\",
                    "test.dat",
                    "C:\\local\\test",
                    "dat"),
    std::make_tuple("/test.dat", "/", "test.dat", "/test", "dat"),
    std::make_tuple("./test.dat", "./", "test.dat", "./test", "dat"),
    std::make_tuple("./test", "./", "test", "./test", ""),
    std::make_tuple("C:\\test.dat", "C:\\", "test.dat", "C:\\test", "dat"),
    std::make_tuple(".gitignore", ".", ".gitignore", ".gitignore", ""),
    std::make_tuple(
        "/local/.gitignore", "/local/", ".gitignore", "/local/.gitignore", ""),
    std::make_tuple("C:\\local\\.gitignore",
                    "C:\\local\\",
                    ".gitignore",
                    "C:\\local\\.gitignore",
                    ""),
    std::make_tuple("/local/a.gitignore",
                    "/local/",
                    "a.gitignore",
                    "/local/a",
                    "gitignore"),
    std::make_tuple("C:\\local\\a.gitignore",
                    "C:\\local\\",
                    "a.gitignore",
                    "C:\\local\\a",
                    "gitignore"),
    std::make_tuple("test.dat.back", ".", "test.dat.back", "test.dat", "back"),
    std::make_tuple("/local/test.dat.back",
                    "/local/",
                    "test.dat.back",
                    "/local/test.dat",
                    "back"),
    std::make_tuple("C:\\local\\test.dat.back",
                    "C:\\local\\",
                    "test.dat.back",
                    "C:\\local\\test.dat",
                    "back"),
    // dirname and basename additional behavior check, different from GNU or
    // POSIX behaviors
    std::make_tuple("/", "/", "", "/", ""),
    std::make_tuple("/a/b/", "/a/b/", "", "/a/b/", ""),
    std::make_tuple("/a", "/", "a", "/a", ""),
    std::make_tuple("/a/b", "/a/", "b", "/a/b", ""),
    std::make_tuple("a/b", "a/", "b", "a/b", ""),
    std::make_tuple("C:\\", "C:\\", "", "C:\\", ""),
    std::make_tuple("C:\\a\\b\\", "C:\\a\\b\\", "", "C:\\a\\b\\", ""),
    std::make_tuple("C:\\a", "C:\\", "a", "C:\\a", ""),
    std::make_tuple("C:\\a\\b", "C:\\a\\", "b", "C:\\a\\b", ""),
    std::make_tuple("a\\b", "a\\", "b", "a\\b", ""))
{
    ASSERT_EQUALS(Utils::dirName(filePath), dirName);
    ASSERT_EQUALS(Utils::baseName(filePath), baseName);
    ASSERT_EQUALS(Utils::fileBaseName(filePath), fileBaseName);
    ASSERT_EQUALS(Utils::fileExtension(filePath), fileExtension);
}

TEST_DATASET(Utils,
             left_shift_operator,
             (unsigned int size, std::string values),
             std::make_tuple(3U, "1 2 3"),
             std::make_tuple(9U, "1 2 3 4 5 6 7 8 9"),
             std::make_tuple(6U, "1 2 3 4 5 6"))
{
    std::vector<int> vec;
    vec << values;

    ASSERT_EQUALS(vec.size(), size);
    ASSERT_EQUALS(vec[0], 1);
    ASSERT_EQUALS(vec[1], 2);
}

TEST(Utils, left_shift_operator__throw)
{
    std::vector<int> vec(4);
    vec << "1 2 3 4 ";
    ASSERT_EQUALS(vec.size(), 4U);
    vec << " 1 2 3 4";
    ASSERT_EQUALS(vec.size(), 4U);
    vec << " 1   2   3   4";
    ASSERT_EQUALS(vec.size(), 4U);

    ASSERT_THROW(vec << "1 a 3 4", std::runtime_error);
    ASSERT_THROW(vec << "1 2 3 b", std::runtime_error);
}

TEST_DATASET(Utils,
             mean,
             (std::string values, double mean),
             std::make_tuple("1", 1.0),
             std::make_tuple("1 2", 1.5),
             std::make_tuple("3 0 1 2", 1.5),
             std::make_tuple("4 0 1 2 3", 2.0),
             std::make_tuple("3 2 4 1 3 5", 3.0),
             std::make_tuple("3 2 1 6 4 5 3", 3.428571428571428),
             std::make_tuple("3 2 4 1 7 5", 3.666666666666667),
             std::make_tuple("7 2 1 6 4 5 3", 4.0))
{
    std::vector<double> vec;
    vec << values;

    ASSERT_EQUALS_DELTA(Utils::mean(vec), mean, 1e-12);
}

TEST_DATASET(Utils,
             median,
             (std::string values, double median),
             std::make_tuple("1", 1.0),
             std::make_tuple("1 2", 1.5),
             std::make_tuple("3 0 1 2", 1.5),
             std::make_tuple("4 0 1 2 3", 2.0),
             std::make_tuple("3 2 4 1 3 5", 3.0),
             std::make_tuple("3 2 1 6 4 5 3", 3.0),
             std::make_tuple("3 2 4 1 7 5", 3.5),
             std::make_tuple("7 2 1 6 4 5 3", 4.0))
{
    std::vector<double> vec;
    vec << values;

    ASSERT_EQUALS(Utils::median(vec), median);
}

RUN_TESTS()
