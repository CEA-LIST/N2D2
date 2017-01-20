/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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

#include "Network.hpp"

#include "NodeNeuron.hpp"
#include "SpikeEvent.hpp"
#include "Xcell.hpp"

namespace N2D2 {
const Time_T TimeFs = 1;
const Time_T TimePs = 1000 * TimeFs;
const Time_T TimeNs = 1000 * TimePs;
const Time_T TimeUs = 1000 * TimeNs;
const Time_T TimeMs = 1000 * TimeUs;
const Time_T TimeS = 1000 * TimeMs;

#if !defined(WIN32) && !defined(__APPLE__)
void exceptionHandler(int sig, siginfo_t* info, void* /*data*/)
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
                                                          "invalid "
                                                          "operation]"
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
#endif
}

N2D2::NetworkObserver::NetworkObserver(Network& net) : mNet(net)
{
    net.addObserver(this);
}

N2D2::NetworkObserver::~NetworkObserver()
{
    mNet.removeObserver(this);
}

N2D2::Network::Network(unsigned int seed)
    : mInitialized(false),
      mFirstEvent(0),
      mLastEvent(0),
      mStop(0),
      mDiscard(false),
      mStartTime(std::chrono::high_resolution_clock::now())
{
// ctor
#if !defined(WIN32) && !defined(__APPLE__)
    // Additional check on floating point operations.
    {
        struct sigaction action;
        action.sa_sigaction = exceptionHandler;
        sigemptyset(&action.sa_mask);
        action.sa_flags = SA_SIGINFO;

        sigaction(SIGFPE, &action, NULL);
        sigaction(SIGSEGV, &action, NULL);
    }

    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW /*| FE_UNDERFLOW*/);
#endif

#if defined(_GLIBCXX_PARALLEL)
    // Compatibility with OpenCV 2.0.0 (for cv::imwrite()). Disable parallel
    // mode at runtime, as it has issues.
    __gnu_parallel::_Settings s;
    s.algorithm_strategy = __gnu_parallel::force_sequential;
    __gnu_parallel::_Settings::set(s);
#endif

    if (seed == 0)
        seed = mStartTime.time_since_epoch().count();

    Random::mtSeed(seed);

    std::ofstream seedFile("seed.dat");

    if (!seedFile.good())
        throw std::runtime_error("Could not create seed file.");

    seedFile << seed;
    seedFile.close();
}

bool N2D2::Network::run(Time_T stop, bool clearActivity)
{
    if (clearActivity)
        mSpikeRecording.clear();

    // Auto-initialization the first time run() is lauched
    if (!mInitialized) {
        std::for_each(mObservers.begin(),
                      mObservers.end(),
                      std::bind(&NetworkObserver::notify,
                                std::placeholders::_1,
                                mLastEvent,
                                NetworkObserver::Initialize));
        mInitialized = true;
    }

    SpikeEvent* event;
    bool stopped = false;

    if (!mEvents.empty())
        mFirstEvent = mEvents.top()->getTimestamp();

    mStop = stop;
    mDiscard = false;

    while (!mEvents.empty()) {
        event = mEvents.top();

        if (event->isDiscarded()) {
            mEvents.pop();
            mEventsPool.push(event);
            continue;
        }

        // Safety check
        if (event->getTimestamp() < mLastEvent) {
            std::ostringstream errorMsg;
            errorMsg
                << "Cannot go back in time! I want to deal with event at time "
                << event->getTimestamp() << " whereas last event was at "
                << mLastEvent << ", type is " << event->getType();
            throw std::runtime_error(errorMsg.str());
        }

        if (mStop > 0 && event->getTimestamp() >= mStop) {
            // In this case, the event should not be released. It must be kept
            // in the priority queue.
            // mLastEvent should not be changed either, because if one loads new
            // events starting from mStop afterward, and the
            // timestamp of this event if > mStop, we have a completely
            // legitimate "cannot go back in time" error!
            stopped = true;
            break;
        }

        // On supprime d'abord l'évènement de la priority_queue avant de le
        // traiter. Dans le cas limite où la fonction release()
        // crérait un nouvel évènement au même timestamp que l'évènement
        // courant, celui-ci pourrait se retrouver en haut de la
        // queue si bien que si on faisait dans ce cas le pop() après le
        // release(), on risque de supprimer le mauvais évènement.
        mEvents.pop();
        mLastEvent = event->release();
        mEventsPool.push(event);
    }

    if (mDiscard) {
        while (!mEvents.empty()) {
            mEventsPool.push(mEvents.top());
            mEvents.pop();
        }
    }

    std::for_each(mObservers.begin(),
                  mObservers.end(),
                  std::bind(&NetworkObserver::notify,
                            std::placeholders::_1,
                            mLastEvent,
                            NetworkObserver::Finalize));

    return stopped;
}

void N2D2::Network::reset(Time_T timestamp)
{
    mFirstEvent = timestamp;
    mLastEvent = timestamp;

    std::for_each(mObservers.begin(),
                  mObservers.end(),
                  std::bind(&NetworkObserver::notify,
                            std::placeholders::_1,
                            mLastEvent,
                            NetworkObserver::Reset));
}

void N2D2::Network::save(const std::string& dirName)
{
    Utils::createDirectories(dirName);
    mLoadSavePath = dirName;

    std::for_each(mObservers.begin(),
                  mObservers.end(),
                  std::bind(&NetworkObserver::notify,
                            std::placeholders::_1,
                            mLastEvent,
                            NetworkObserver::Save));
}

void N2D2::Network::load(const std::string& dirName)
{
    mLoadSavePath = dirName;

    std::for_each(mObservers.begin(),
                  mObservers.end(),
                  std::bind(&NetworkObserver::notify,
                            std::placeholders::_1,
                            mLastEvent,
                            NetworkObserver::Load));
}

void N2D2::Network::addObserver(NetworkObserver* obs)
{
    mObservers.insert(obs);
}

void N2D2::Network::removeObserver(NetworkObserver* obs)
{
    mObservers.erase(obs);
}

N2D2::SpikeEvent* N2D2::Network::newEvent(Node* origin,
                                          Node* destination,
                                          Time_T timestamp,
                                          EventType_T type)
{
    SpikeEvent* event;

    if (mEventsPool.empty())
        event = new SpikeEvent(origin, destination, timestamp, type);
    else {
        event = mEventsPool.top();
        mEventsPool.pop();
        event->initialize(origin, destination, timestamp, type);
    }

    mEvents.push(event);
    return event;
}

N2D2::Network::~Network()
{
    // dtor
    while (!mEventsPool.empty()) {
        delete mEventsPool.top();
        mEventsPool.pop();
    }

    const double timeElapsed
        = std::chrono::duration_cast<std::chrono::duration<double> >(
            std::chrono::high_resolution_clock::now() - mStartTime).count();

    std::cout << "Time elapsed: " << timeElapsed << " s" << std::endl;
}

unsigned int N2D2::Network::readSeed(const std::string& fileName)
{
    std::ifstream seedFile(fileName);

    if (!seedFile.good())
        throw std::runtime_error("Could not read seed file: " + fileName);

    unsigned int seed;

    if (!(Utils::signChecked<unsigned int>(seedFile) >> seed))
        throw std::runtime_error("Could not read seed value in file: "
                                 + fileName);

    if (seedFile.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Extra data in seed file: " + fileName);

    return seed;
}
