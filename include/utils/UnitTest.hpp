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

#ifndef N2D2_UNITTEST_H
#define N2D2_UNITTEST_H

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <iterator>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <vector>

#ifndef WIN32
#include <fenv.h>
#endif

#if !defined(WIN32) && !defined(__APPLE__)
#include <csignal>
#include <cstring>
#include <execinfo.h>
#include <unistd.h>
#endif

#ifdef CUDA
#include "CudaUtils.hpp"
#endif

namespace N2D2 {
#if !defined(WIN32) && !defined(__APPLE__)
void UnitTest_exceptionHandler(int sig);
#endif

class UnitTest_Test;

class UnitTest {
public:
    enum TestResult {
        Success,
        Error,
        Failure,
        Skipped
    };

    static void addTest(UnitTest_Test* test);
    static int runTests();

    static bool FileExists(const std::string& fileName);
    static bool DirExists(const std::string& dirName);
    static bool CudaDeviceExists(int minMajor = 1, int minMinor = 0);
    static std::string FileReadContent(const std::string& fileName,
                                       unsigned int firstLine = 0,
                                       unsigned int nbLines = 0);
    static void FileWriteContent(const std::string& fileName,
                                 const std::string& content);
    static bool FileRemove(const std::string& fileName);

private:
    static UnitTest& instance();
    UnitTest() {}; // This is a static class, it cannot be instantiated

    std::map<std::string, std::vector<UnitTest_Test*> > mTests;
};

class UnitTest_Test {
public:
    typedef std::tuple
        <std::string, unsigned int, std::string, UnitTest::TestResult> Report_T;

    UnitTest_Test(const std::string& testCaseName, const std::string& testName);
    virtual void run() = 0;
    void addReport(const std::string& file,
                   unsigned int line,
                   const std::string& condition,
                   UnitTest::TestResult result);
    void skip(bool skip)
    {
        mSkip = skip;
    }
    const std::string& getCaseName()
    {
        return mTestCaseName;
    }
    const std::string& getName()
    {
        return mTestName;
    }
    unsigned int getNbSuccesses()
    {
        return mNbSuccesses;
    }
    unsigned int getNbFailures()
    {
        return mNbFailures;
    }
    unsigned int getNbErrors()
    {
        return mNbErrors;
    }
    unsigned int getNbSkipped()
    {
        return mNbSkipped;
    }
    const std::vector<Report_T>& getReport()
    {
        return mReport;
    }

private:
    const std::string mTestCaseName;
    const std::string mTestName;
    bool mSkip;
    unsigned int mNbSuccesses;
    unsigned int mNbFailures;
    unsigned int mNbErrors;
    unsigned int mNbSkipped;
    std::vector<Report_T> mReport;
};

#ifndef WIN32
namespace expandTupleImpl {
    template <typename Obj,
              typename F,
              typename Tuple,
              bool Done,
              int Total,
              int... N>
    struct expandTupleImpl {
        static void expandTuple(Obj* obj, F f, Tuple&& t)
        {
            expandTupleImpl
                <Obj,
                 F,
                 Tuple,
                 Total == 1 + sizeof...(N),
                 Total,
                 N...,
                 sizeof...(N)>::expandTuple(obj, f, std::forward<Tuple>(t));
        }
    };

    template <typename Obj, typename F, typename Tuple, int Total, int... N>
    struct expandTupleImpl<Obj, F, Tuple, true, Total, N...> {
        static void expandTuple(Obj* obj, F f, Tuple&& t)
        {
            (obj->*f)(std::get<N>(std::forward<Tuple>(t))...);
        }
    };
}

// user invokes this
template <typename Obj, typename F, typename Tuple>
void expandTuple(Obj* obj, F f, Tuple&& t)
{
    typedef typename std::decay<Tuple>::type ttype;
    expandTupleImpl::expandTupleImpl
        <Obj,
         F,
         Tuple,
         0 == std::tuple_size<ttype>::value,
         std::tuple_size<ttype>::value>::expandTuple(obj,
                                                     f,
                                                     std::forward<Tuple>(t));
}
#else
template <class Obj, class Func, class T0>
static void expandTuple(Obj* obj, Func func, const std::tuple<T0>& t)
{
    (obj->*func)(std::get<0>(t));
}

template <class Obj, class Func, class T0, class T1>
static void expandTuple(Obj* obj, Func func, const std::tuple<T0, T1>& t)
{
    (obj->*func)(std::get<0>(t), std::get<1>(t));
}

template <class Obj, class Func, class T0, class T1, class T2>
static void expandTuple(Obj* obj, Func func, const std::tuple<T0, T1, T2>& t)
{
    (obj->*func)(std::get<0>(t), std::get<1>(t), std::get<2>(t));
}

template <class Obj, class Func, class T0, class T1, class T2, class T3>
static void
expandTuple(Obj* obj, Func func, const std::tuple<T0, T1, T2, T3>& t)
{
    (obj
     ->*func)(std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t));
}

template
    <class Obj, class Func, class T0, class T1, class T2, class T3, class T4>
static void
expandTuple(Obj* obj, Func func, const std::tuple<T0, T1, T2, T3, T4>& t)
{
    (obj->*func)(std::get<0>(t),
                 std::get<1>(t),
                 std::get<2>(t),
                 std::get<3>(t),
                 std::get<4>(t));
}

template <class Obj,
          class Func,
          class T0,
          class T1,
          class T2,
          class T3,
          class T4,
          class T5>
static void
expandTuple(Obj* obj, Func func, const std::tuple<T0, T1, T2, T3, T4, T5>& t)
{
    (obj->*func)(std::get<0>(t),
                 std::get<1>(t),
                 std::get<2>(t),
                 std::get<3>(t),
                 std::get<4>(t),
                 std::get<5>(t));
}

template <class Obj,
          class Func,
          class T0,
          class T1,
          class T2,
          class T3,
          class T4,
          class T5,
          class T6>
static void expandTuple(Obj* obj,
                        Func func,
                        const std::tuple<T0, T1, T2, T3, T4, T5, T6>& t)
{
    (obj->*func)(std::get<0>(t),
                 std::get<1>(t),
                 std::get<2>(t),
                 std::get<3>(t),
                 std::get<4>(t),
                 std::get<5>(t),
                 std::get<6>(t));
}

template <class Obj,
          class Func,
          class T0,
          class T1,
          class T2,
          class T3,
          class T4,
          class T5,
          class T6,
          class T7>
static void expandTuple(Obj* obj,
                        Func func,
                        const std::tuple<T0, T1, T2, T3, T4, T5, T6, T7>& t)
{
    (obj->*func)(std::get<0>(t),
                 std::get<1>(t),
                 std::get<2>(t),
                 std::get<3>(t),
                 std::get<4>(t),
                 std::get<5>(t),
                 std::get<6>(t),
                 std::get<7>(t));
}

template <class Obj,
          class Func,
          class T0,
          class T1,
          class T2,
          class T3,
          class T4,
          class T5,
          class T6,
          class T7,
          class T8>
static void expandTuple(Obj* obj,
                        Func func,
                        const std::tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8>& t)
{
    (obj->*func)(std::get<0>(t),
                 std::get<1>(t),
                 std::get<2>(t),
                 std::get<3>(t),
                 std::get<4>(t),
                 std::get<5>(t),
                 std::get<6>(t),
                 std::get<7>(t),
                 std::get<8>(t));
}

template <class Obj,
          class Func,
          class T0,
          class T1,
          class T2,
          class T3,
          class T4,
          class T5,
          class T6,
          class T7,
          class T8,
          class T9>
static void expandTuple(Obj* obj,
                        Func func,
                        const std::tuple
                        <T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& t)
{
    (obj->*func)(std::get<0>(t),
                 std::get<1>(t),
                 std::get<2>(t),
                 std::get<3>(t),
                 std::get<4>(t),
                 std::get<5>(t),
                 std::get<6>(t),
                 std::get<7>(t),
                 std::get<8>(t),
                 std::get<9>(t));
}
#endif
}

#define TEST(testCaseName, testName)                                           \
    class UnitTest_##testCaseName##_##testName : public UnitTest_Test {        \
    public:                                                                    \
        UnitTest_##testCaseName##_##testName()                                 \
            : UnitTest_Test(#testCaseName, #testName)                          \
        {                                                                      \
        }                                                                      \
        void run();                                                            \
    } UnitTest_##testCaseName##_##testName##_Instance;                         \
    void UnitTest_##testCaseName##_##testName::run()

#define FIRST(...) FIRST_HELPER(__VA_ARGS__, throwaway)
#define FIRST_HELPER(first, ...) first

#define TEST_DATASET(testCaseName, testName, params, ...)                      \
    class UnitTest_##testCaseName##_##testName : public UnitTest_Test {        \
    public:                                                                    \
        UnitTest_##testCaseName##_##testName()                                 \
            : UnitTest_Test(#testCaseName, #testName)                          \
        {                                                                      \
        }                                                                      \
        void run()                                                             \
        {                                                                      \
            static const decltype(FIRST(__VA_ARGS__)) args[] = {__VA_ARGS__};  \
            for (unsigned int i = 0, size = sizeof(args) / sizeof(args[0]);    \
                 i < size;                                                     \
                 ++i)                                                          \
                expandTuple(this,                                              \
                            &UnitTest_##testCaseName##_##testName::runData,    \
                            args[i]);                                          \
        }                                                                      \
                                                                               \
    private:                                                                   \
        void runData params;                                                   \
    } UnitTest_##testCaseName##_##testName##_Instance;                         \
    void UnitTest_##testCaseName##_##testName::runData params

#define REQUIRED(condition)                                                    \
    {                                                                          \
        const bool conditionVal = condition;                                   \
        if (!conditionVal)                                                     \
            skip(true);                                                        \
    }

#define NOREQUIRE()                                                            \
    {                                                                          \
        skip(false);                                                           \
    }

#define ASSERT_EQUALS(actual, expected)                                        \
    {                                                                          \
        const auto actualVal = actual;                                         \
        const auto expectedVal = expected;                                     \
        if (actualVal == expectedVal)                                          \
            addReport(__FILE__,                                                \
                      __LINE__,                                                \
                      #actual " == " #expected,                                \
                      UnitTest::Success);                                      \
        else {                                                                 \
            std::stringstream values;                                          \
            values << std::setprecision(std::numeric_limits<double>::digits10  \
                                        + 1) << actualVal                      \
                   << " != " << expectedVal;                                   \
            addReport(__FILE__,                                                \
                      __LINE__,                                                \
                      #actual " != " #expected " with values " + values.str(), \
                      UnitTest::Failure);                                      \
            return;                                                            \
        }                                                                      \
    }

#define ASSERT_EQUALS_DELTA(actual, expected, delta)                           \
    {                                                                          \
        const auto actualVal = actual;                                         \
        const auto expectedVal = expected;                                     \
        if ((actualVal >= expectedVal && (actualVal - expectedVal) <= delta)   \
            || (expectedVal >= actualVal && (expectedVal - actualVal)          \
                                            <= delta))                         \
            addReport(__FILE__,                                                \
                      __LINE__,                                                \
                      #actual " ~= " #expected " with delta = " #delta,        \
                      UnitTest::Success);                                      \
        else {                                                                 \
            std::stringstream values;                                          \
            values << std::setprecision(std::numeric_limits<double>::digits10  \
                                        + 1) << actualVal                      \
                   << " != " << expectedVal;                                   \
            addReport(__FILE__,                                                \
                      __LINE__,                                                \
                      #actual " != " #expected " with values " + values.str()  \
                      + " with delta = " #delta,                               \
                      UnitTest::Failure);                                      \
            return;                                                            \
        }                                                                      \
    }

#define ASSERT_TRUE(condition)                                                 \
    {                                                                          \
        if (condition)                                                         \
            addReport(__FILE__, __LINE__, #condition, UnitTest::Success);      \
        else {                                                                 \
            addReport(__FILE__, __LINE__, #condition, UnitTest::Failure);      \
            return;                                                            \
        }                                                                      \
    }

#define ASSERT_THROW(condition, error)                                         \
    {                                                                          \
        try                                                                    \
        {                                                                      \
            condition;                                                         \
            addReport(__FILE__,                                                \
                      __LINE__,                                                \
                      #condition " did not throw " #error "!",                 \
                      UnitTest::Error);                                        \
        }                                                                      \
        catch (const error&)                                                   \
        {                                                                      \
            addReport(__FILE__, __LINE__, #condition, UnitTest::Success);      \
        }                                                                      \
        catch (...)                                                            \
        {                                                                      \
            throw;                                                             \
        }                                                                      \
    }

#define ASSERT_NOTHROW(condition, error)                                       \
    {                                                                          \
        try                                                                    \
        {                                                                      \
            condition;                                                         \
            addReport(__FILE__, __LINE__, #condition, UnitTest::Success);      \
        }                                                                      \
        catch (const error&)                                                   \
        {                                                                      \
            addReport(__FILE__,                                                \
                      __LINE__,                                                \
                      #condition " did throw " #error "!",                     \
                      UnitTest::Error);                                        \
        }                                                                      \
        catch (...)                                                            \
        {                                                                      \
            throw;                                                             \
        }                                                                      \
    }

#define ASSERT_THROW_ANY(condition)                                            \
    {                                                                          \
        try                                                                    \
        {                                                                      \
            condition;                                                         \
            addReport(__FILE__,                                                \
                      __LINE__,                                                \
                      #condition " did not throw!",                            \
                      UnitTest::Error);                                        \
        }                                                                      \
        catch (...)                                                            \
        {                                                                      \
            addReport(__FILE__, __LINE__, #condition, UnitTest::Success);      \
        }                                                                      \
    }

#define ASSERT_NOTHROW_ANY(condition)                                          \
    {                                                                          \
        try                                                                    \
        {                                                                      \
            condition;                                                         \
            addReport(__FILE__, __LINE__, #condition, UnitTest::Success);      \
        }                                                                      \
        catch (...)                                                            \
        {                                                                      \
            addReport(__FILE__,                                                \
                      __LINE__,                                                \
                      #condition " did throw!",                                \
                      UnitTest::Error);                                        \
        }                                                                      \
    }

#define RUN_TESTS()                                                            \
    int main()                                                                 \
    {                                                                          \
        return UnitTest::runTests();                                           \
    }

#endif // N2D2_UNITTEST_H
