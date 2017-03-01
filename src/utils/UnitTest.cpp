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

#if !defined(WIN32) && !defined(__APPLE__)
namespace N2D2 {
void UnitTest_exceptionHandler(int sig, siginfo_t* info, void* /*data*/)
{
    void* array[50];
    const unsigned int size
        = backtrace(array, sizeof(array) / sizeof(array[0]));

    std::cerr << strsignal(sig);

    if (info->si_signo == SIGFPE) {
        std::cerr
            << (info->si_code == FPE_INTDIV
                    ? " [integer divide by zero]"
                    : info->si_code == FPE_INTOVF
                          ? " [integer overflow]"
                          : info->si_code == FPE_FLTDIV
                                ? " [floating point divide by zero]"
                                : info->si_code == FPE_FLTOVF
                                      ? " [floating point overflow]"
                                      : info->si_code == FPE_FLTUND
                                            ? " [floating point underflow]"
                                            : info->si_code == FPE_FLTRES
                                                  ? " [floating point inexact "
                                                    "result]"
                                                  : info->si_code == FPE_FLTINV
                                                        ? " [floating point "
                                                          "invalid operation]"
                                                        : info->si_code
                                                              == FPE_FLTSUB
                                                              ? " [subscript "
                                                                "out of range]"
                                                              : " [unknown]");
    }

    std::cerr << std::endl;
    std::cerr << "backtrace() returned " << size << " addresses" << std::endl;
    backtrace_symbols_fd(array, size, STDERR_FILENO);

    std::exit(EXIT_FAILURE);
}
}
#endif

N2D2::UnitTest_Test::UnitTest_Test(const std::string& testCaseName,
                                   const std::string& testName)
    : mTestCaseName(testCaseName),
      mTestName(testName),
      mSkip(false),
      mNbSuccesses(0),
      mNbFailures(0),
      mNbErrors(0),
      mNbSkipped(0)
{
    UnitTest::addTest(this);
}

void N2D2::UnitTest_Test::addReport(const std::string& file,
                                    unsigned int line,
                                    const std::string& condition,
                                    UnitTest::TestResult result)
{
    mReport.push_back(std::make_tuple(
        file, line, condition, (mSkip) ? UnitTest::Skipped : result));

    if (mSkip)
        ++mNbSkipped;
    else if (result == UnitTest::Failure)
        ++mNbFailures;
    else if (result == UnitTest::Error)
        ++mNbErrors;
    else
        ++mNbSuccesses;
}

void N2D2::UnitTest::addTest(UnitTest_Test* test)
{
    std::map<std::string, std::vector<UnitTest_Test*> >::iterator it;
    std::tie(it, std::ignore) = instance().mTests.insert(
        std::make_pair(test->getCaseName(), std::vector<UnitTest_Test*>()));
    (*it).second.push_back(test);
}

int N2D2::UnitTest::runTests()
{
#if !defined(WIN32) && !defined(__APPLE__)
    // Additional check on floating point operations.
    {
        struct sigaction action;
        action.sa_sigaction = UnitTest_exceptionHandler;
        sigemptyset(&action.sa_mask);
        action.sa_flags = SA_SIGINFO;

        sigaction(SIGFPE, &action, NULL);
        sigaction(SIGSEGV, &action, NULL);
    }

    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW /*| FE_UNDERFLOW*/);
#endif

    unsigned int nbTestCases = 0;
    unsigned int nbSuccesses = 0;
    unsigned int nbFailures = 0;
    unsigned int nbErrors = 0;
    unsigned int nbSkipped = 0;

    for (std::map<std::string, std::vector<UnitTest_Test*> >::const_iterator it
         = instance().mTests.begin(),
         itEnd = instance().mTests.end();
         it != itEnd;
         ++it) {
        for (std::vector<UnitTest_Test*>::const_iterator itTest
             = (*it).second.begin(),
             itTestEnd = (*it).second.end();
             itTest != itTestEnd;
             ++itTest) {
            try
            {
                (*itTest)->run();
            }
            catch (const std::exception& e)
            {
                (*itTest)->addReport("", 0, e.what(), Error);
            }
            catch (...)
            {
                (*itTest)->addReport("", 0, "Unexpected error occured", Error);
            }

            nbSuccesses += (*itTest)->getNbSuccesses();
            nbFailures += (*itTest)->getNbFailures();
            nbErrors += (*itTest)->getNbErrors();
            nbSkipped += (*itTest)->getNbSkipped();
        }

        ++nbTestCases;
    }

    for (std::map<std::string, std::vector<UnitTest_Test*> >::const_iterator it
         = instance().mTests.begin(),
         itEnd = instance().mTests.end();
         it != itEnd;
         ++it) {
        unsigned int nbTestSuccesses = 0;
        unsigned int nbTestFailures = 0;
        unsigned int nbTestErrors = 0;
        unsigned int nbTestSkipped = 0;

        for (std::vector<UnitTest_Test*>::const_iterator itTest
             = (*it).second.begin(),
             itTestEnd = (*it).second.end();
             itTest != itTestEnd;
             ++itTest) {
            nbTestSuccesses += (*itTest)->getNbSuccesses();
            nbTestFailures += (*itTest)->getNbFailures();
            nbTestErrors += (*itTest)->getNbErrors();
            nbTestSkipped += (*itTest)->getNbSkipped();
        }

        const std::string msg
            = (nbTestFailures > 0 || nbTestErrors > 0)
                  ? "FAILED"
                  : ((nbTestSkipped > 0) ? "SKIPPED" : "SUCCEEDED");

        std::cout << "Test case \"" << (*it).first << "\" " << msg << " with "
                  << nbTestErrors << " error(s), " << nbTestFailures
                  << " failure(s), " << nbTestSuccesses << " success(es)";

        if (nbTestSkipped > 0)
            std::cout << " and " << nbTestSkipped << " skipped";

        std::cout << ":" << std::endl;

        for (std::vector<UnitTest_Test*>::const_iterator itTest
             = (*it).second.begin(),
             itTestEnd = (*it).second.end();
             itTest != itTestEnd;
             ++itTest) {
            unsigned int nbReports = 0;
            unsigned int nbReportPassed = 0;
            unsigned int nbReportSkipped = 0;

            for (std::vector<UnitTest_Test::Report_T>::const_iterator itReport
                 = (*itTest)->getReport().begin(),
                 itReportEnd = (*itTest)->getReport().end();
                 itReport != itReportEnd;
                 ++itReport) {
                TestResult result;
                std::tie(std::ignore, std::ignore, std::ignore, result)
                    = (*itReport);

                if (result == Success)
                    ++nbReportPassed;
                else if (result == Skipped)
                    ++nbReportSkipped;

                ++nbReports;
            }

            if (nbReports == nbReportPassed + nbReportSkipped) {
                const std::string msg = (nbReportSkipped > 0) ? "SKIPPED"
                                                              : "PASSED";

                std::cout << "  Test \"" << (*itTest)->getName() << "\" " << msg
                          << " (" << nbReportPassed << "/" << nbReports << ")."
                          << std::endl;
            } else {
                std::string file;
                unsigned int line;
                std::string condition;
                TestResult result;

                std::cout << "  Test \"" << (*itTest)->getName()
                          << "\" FAILED (" << nbReportPassed << "/" << nbReports
                          << "):" << std::endl;

                for (std::vector<UnitTest_Test::Report_T>::const_iterator
                         itReport = (*itTest)->getReport().begin(),
                         itReportEnd = (*itTest)->getReport().end();
                     itReport != itReportEnd;
                     ++itReport) {
                    std::tie(file, line, condition, result) = (*itReport);

                    if (result == Failure)
                        std::cout << "    Failure: \"" << condition
                                  << "\" line " << line << " in file " << file
                                  << std::endl;
                    else if (result == Error)
                        std::cout << "    Error: \"" << condition << "\""
                                  << std::endl;
                }
            }
        }

        std::cout << std::endl;
    }

    std::cout << "Test summary: "
              << ((nbFailures > 0 || nbErrors > 0) ? "FAIL" : "SUCCESS")
              << "\n"
                 "  Number of test cases ran: " << nbTestCases
              << "\n"
                 "  Tests that succeeded: " << nbSuccesses
              << "\n"
                 "  Tests with errors: " << nbErrors
              << "\n"
                 "  Tests that failed: " << nbFailures
              << "\n"
                 "  Tests skipped: " << nbSkipped << std::endl;

    return (nbFailures > 0 || nbErrors > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}

bool N2D2::UnitTest::FileExists(const std::string& fileName)
{
    return std::ifstream(fileName.c_str()).good();
}

bool N2D2::UnitTest::DirExists(const std::string& dirName)
{
    struct stat fileStat;
    return (stat(dirName.c_str(), &fileStat) == 0 && fileStat.st_mode
                                                     & S_IFDIR);
}

#ifdef CUDA
bool N2D2::UnitTest::CudaDeviceExists(int minMajor, int minMinor) {
    int deviceCount = 0;
    cudaError status = cudaGetDeviceCount(&deviceCount);

    if (status == cudaSuccess && deviceCount > 0) {
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            if (prop.major > minMajor
                || (prop.major == minMajor && prop.minor >= minMinor))
            {
                std::cout << "Found CUDA compute capability "
                    << prop.major << "." << prop.minor << " for device #"
                    << i << std::endl;
                return true;
            }
        }
    }
    else {
        std::cout << "Cuda failure: " << cudaGetErrorString(status) << " ("
            << (int)status << ")" << std::endl;
    }

    return false;
}
#else
bool N2D2::UnitTest::CudaDeviceExists(int /*minMajor*/, int /*minMinor*/) {
    return false;
}
#endif

std::string N2D2::UnitTest::FileReadContent(const std::string& fileName,
                                            unsigned int firstLine,
                                            unsigned int nbLines)
{
    std::ifstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error(
            "UnitTest::FileReadContent(): File does not exist: " + fileName);

    if (firstLine > 0 || nbLines > 0) {
        std::string line;
        unsigned int numLine = 0;
        std::string content;

        while (std::getline(data, line)) {
            if (numLine >= firstLine) {
                content += line;
                content.push_back('\n');
            }

            ++numLine;

            if (nbLines > 0 && numLine >= firstLine + nbLines)
                break;
        }

        return content;
    } else
        return std::string(std::istreambuf_iterator<char>(data),
                           std::istreambuf_iterator<char>());
}

void N2D2::UnitTest::FileWriteContent(const std::string& fileName,
                                      const std::string& content)
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error(
            "UnitTest::FileWriteContent(): Unable to create file: " + fileName);

    data.write(content.c_str(), sizeof(char) * content.size());

    if (!data.good())
        throw std::runtime_error(
            "UnitTest::FileWriteContent(): error writing file" + fileName);
}

bool N2D2::UnitTest::FileRemove(const std::string& fileName)
{
    const bool removed = (std::remove(fileName.c_str()) == 0);

    if (std::ifstream(fileName.c_str()).good())
        throw std::runtime_error(
            "UnitTest::FileRemove(): File cannot be removed: " + fileName);

    return removed;
}

N2D2::UnitTest& N2D2::UnitTest::instance()
{
    static UnitTest unitTest;
    return unitTest;
}
