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

#include "Export/MemoryManager.hpp"
#include "utils/Gnuplot.hpp"

std::shared_ptr<N2D2::MemoryManager::MemorySpace> N2D2::MemoryManager::reserve(
    unsigned int size,
    const std::vector<std::shared_ptr<Cell> >& dependencies)
{
    const unsigned int offset = onStack(size);

    std::shared_ptr<MemorySpace> memSpace
        = std::make_shared<MemorySpace>(mClock, offset, size, dependencies);
    mMemSpaces.push_back(memSpace);
    return memSpace;
}

void N2D2::MemoryManager::expand(
    std::shared_ptr<MemorySpace> memSpace,
    unsigned int requiredSize)
{
    assert(std::find(mMemSpaces.begin(), mMemSpaces.end(), memSpace)
            != mMemSpaces.end());

    memSpace->size = std::max(memSpace->size, requiredSize);

    // Rebuild the stack from the beginning, taking into account the new size.
    // Everything else stay the same.
    mMemStack.clear();

    for (Clock_T clock = 0; clock <= mClock; ++clock) {
        for (std::vector<std::shared_ptr<MemorySpace> >::iterator
            it = mMemSpaces.begin(), itEnd = mMemSpaces.end(); it != itEnd;
            ++it)
        {
            if ((*it)->allocated == clock)
                (*it)->offset = onStack((*it)->size);
        }

        // MemorySpace released at clock are still valid until the next tick;
        // make sure offStack() only append after all onStack() are done.
        for (std::vector<std::shared_ptr<MemorySpace> >::iterator
            it = mMemSpaces.begin(), itEnd = mMemSpaces.end(); it != itEnd;
            ++it)
        {
            if ((*it)->released == clock && (*it)->dependencies.empty())
                offStack((*it)->offset);
        }
    }
}

N2D2::MemoryManager::MemoryPlane N2D2::MemoryManager::allocate(
    unsigned int size,
    const std::vector<std::shared_ptr<Cell> >& dependencies,
    unsigned int stride,
    unsigned int length,
    unsigned int count)
{
    const unsigned int fullSize = std::max(size, stride) * length * count;
    return MemoryPlane(reserve(fullSize, dependencies),
                       mClock, 0, size, stride, length, count);
}

unsigned int N2D2::MemoryManager::allocate(
    const std::shared_ptr<Cell>& cell,
    unsigned int size,
    const std::vector<std::shared_ptr<Cell> >& dependencies,
    unsigned int stride,
    unsigned int length,
    unsigned int count)
{
    std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >::iterator it;
    std::tie(it, std::ignore) = mMemPlanes.insert(std::make_pair(cell,
                                                std::vector<MemoryPlane>()));

    (*it).second.push_back(allocate(size, dependencies, stride, length, count));
    return ((*it).second.size() - 1);
}

bool N2D2::MemoryManager::isWrapAround(
    std::shared_ptr<MemorySpace> memSpace,
    unsigned int offset,
    unsigned int size,
    unsigned int stride,
    unsigned int length,
    unsigned int count) const
{
    const unsigned int fullSize = std::max(size, stride) * length * count;
    return (offset + fullSize > memSpace->size);
}

N2D2::MemoryManager::MemoryPlane N2D2::MemoryManager::reallocate(
    std::shared_ptr<MemorySpace> memSpace,
    unsigned int offset,
    unsigned int size,
    bool wrapAround,
    unsigned int extraSize,
    const std::vector<std::shared_ptr<Cell> >& additionalDependencies,
    unsigned int stride,
    unsigned int length,
    unsigned int count)
{
    const unsigned int fullSize = std::max(size, stride) * length * count;
    unsigned int requiredSize = offset + fullSize;

    if (wrapAround) {
        requiredSize = fullSize + extraSize;

        if (count > 1) {
            // (requiredSize - offset) must be a multiple of (stride * length)
            requiredSize = offset
                + std::ceil((requiredSize - offset)
                    / (double)(std::max(size, stride) * length))
                        * (std::max(size, stride) * length);
        }
    }

    if (requiredSize > memSpace->size || memSpace->released >= 0) {
        // Expand in size and/or duration.
        // If memSpace was already released, put it back on the stack
        memSpace->released = -1;
        expand(memSpace, requiredSize);
    }

    memSpace->dependencies.insert(memSpace->dependencies.end(),
                                  additionalDependencies.begin(),
                                  additionalDependencies.end());

    return MemoryPlane(memSpace, mClock, offset, size, stride, length, count);
}

N2D2::MemoryManager::MemoryPlane N2D2::MemoryManager::reallocate(
    const MemoryPlane& memPlane,
    unsigned int extraOffset,
    unsigned int size,
    bool wrapAround,
    unsigned int extraSize,
    const std::vector<std::shared_ptr<Cell> >& additionalDependencies,
    unsigned int stride,
    unsigned int length,
    unsigned int count)
{
    const unsigned int initialOffset = memPlane.getFinalOffset()
        - memPlane.memSpace->offset + extraOffset;
    const unsigned int fullSize = std::max(size, stride) * length * count;
    unsigned int requiredSize = initialOffset + fullSize;

    if (wrapAround) {
        requiredSize = fullSize + extraSize;

        if (count > 1) {
            // (requiredSize - offset) must be a multiple of (stride * length)
            requiredSize = initialOffset
                + std::ceil((requiredSize - initialOffset)
                    / (double)(std::max(size, stride) * length))
                        * (std::max(size, stride) * length);
        }
    }

    if (requiredSize > memPlane.memSpace->size
        || memPlane.memSpace->released >= 0)
    {
        // Expand in size and/or duration.
        // If memSpace was already released, put it back on the stack
        memPlane.memSpace->released = -1;
        expand(memPlane.memSpace, requiredSize);
    }

    memPlane.memSpace->dependencies.insert(
        memPlane.memSpace->dependencies.end(),
        additionalDependencies.begin(),
        additionalDependencies.end());

    const unsigned int finalOffset = memPlane.getFinalOffset()
        - memPlane.memSpace->offset + extraOffset;

    return MemoryPlane(memPlane.memSpace, mClock,
                       finalOffset, size, stride, length, count);
}

unsigned int N2D2::MemoryManager::reallocate(
    const MemoryPlane& memPlane,
    const std::shared_ptr<Cell>& cell,
    unsigned int extraOffset,
    unsigned int size,
    bool wrapAround,
    unsigned int extraSize,
    const std::vector<std::shared_ptr<Cell> >& additionalDependencies,
    unsigned int stride,
    unsigned int length,
    unsigned int count)
{
    std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >::iterator it;
    std::tie(it, std::ignore) = mMemPlanes.insert(std::make_pair(cell,
                                                std::vector<MemoryPlane>()));

    (*it).second.push_back(reallocate(memPlane, extraOffset, size, wrapAround,
                                      extraSize, additionalDependencies,
                                      stride, length, count));
    return ((*it).second.size() - 1);
}

unsigned int N2D2::MemoryManager::reallocate(
    std::shared_ptr<MemorySpace> memSpace,
    const std::shared_ptr<Cell>& cell,
    unsigned int offset,
    unsigned int size,
    bool wrapAround,
    unsigned int extraSize,
    const std::vector<std::shared_ptr<Cell> >& additionalDependencies,
    unsigned int stride,
    unsigned int length,
    unsigned int count)
{
    std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >::iterator it;
    std::tie(it, std::ignore) = mMemPlanes.insert(std::make_pair(cell,
                                                std::vector<MemoryPlane>()));

    (*it).second.push_back(reallocate(memSpace, offset, size, wrapAround,
                                      extraSize, additionalDependencies,
                                      stride, length, count));
    return ((*it).second.size() - 1);
}

unsigned int N2D2::MemoryManager::release(std::shared_ptr<MemorySpace> memSpace)
{
    if (memSpace->released == -1) {
        memSpace->released = mClock;

        if (memSpace->dependencies.empty())
            return offStack(memSpace->offset);
    }

    return 0;
}

unsigned int N2D2::MemoryManager::release(const std::shared_ptr<Cell>& cell)
{
    const std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::iterator it = mMemPlanes.find(cell);

    if (it == mMemPlanes.end()) {
        std::cout << Utils::cwarning << "release(): there is no allocated "
            "memory for cell \"" << ((cell) ? cell->getName() : "env")
            << "\"" << Utils::cdef << std::endl;
        return 0;
    }

    unsigned int releasedMemSize = 0;

    for (std::vector<MemoryPlane>::iterator itPlanes = (*it).second.begin(),
        itPlanesEnd = (*it).second.end(); itPlanes != itPlanesEnd; ++itPlanes)
    {
        releasedMemSize += release((*itPlanes).memSpace);
    }

    // Remove dependencies
    releasedMemSize += releaseDependencies(cell);

    return releasedMemSize;
}

unsigned int N2D2::MemoryManager::releaseDependencies(
    const std::shared_ptr<Cell>& cell)
{
    unsigned int releasedMemSize = 0;

    for (std::vector<std::shared_ptr<MemorySpace> >::iterator
        it = mMemSpaces.begin(), itEnd = mMemSpaces.end(); it != itEnd;
        ++it)
    {
        if (!(*it)->dependencies.empty()) {
            (*it)->dependencies.erase(std::remove((*it)->dependencies.begin(),
                                                  (*it)->dependencies.end(),
                                                  cell),
                                      (*it)->dependencies.end());

            if ((*it)->released <= mClock
                && (*it)->dependencies.empty())
            {
                (*it)->released = mClock;
                releasedMemSize += offStack((*it)->offset);
            }
        }
    }

    return releasedMemSize;
}

bool N2D2::MemoryManager::MaxLifetimeMinSizeFirst::operator()(
    const std::shared_ptr<MemorySpace>& p0,
    const std::shared_ptr<MemorySpace>& p1)
{
    const Clock_T lifetime0
        = ((p0->released >= 0) ? p0->released : maxLifetime) - p0->allocated;
    const Clock_T lifetime1
        = ((p1->released >= 0) ? p1->released : maxLifetime) - p1->allocated;

    return (lifetime0 > lifetime1
            || (lifetime0 == lifetime1 && p0->size < p1->size));
}

bool N2D2::MemoryManager::MaxLifetimeMaxSizeFirst::operator()(
    const std::shared_ptr<MemorySpace>& p0,
    const std::shared_ptr<MemorySpace>& p1)
{
    const Clock_T lifetime0
        = ((p0->released >= 0) ? p0->released : maxLifetime) - p0->allocated;
    const Clock_T lifetime1
        = ((p1->released >= 0) ? p1->released : maxLifetime) - p1->allocated;

    return (lifetime0 > lifetime1
            || (lifetime0 == lifetime1 && p0->size > p1->size));
}

bool N2D2::MemoryManager::MaxHoleMaxLifetimeFirst::operator()(
    const std::shared_ptr<MemorySpace>& p0,
    const std::shared_ptr<MemorySpace>& p1)
{
    const Clock_T lifetime0
        = ((p0->released >= 0) ? p0->released : maxLifetime) - p0->allocated;
    const Clock_T lifetime1
        = ((p1->released >= 0) ? p1->released : maxLifetime) - p1->allocated;

    const std::pair<Clock_T, unsigned int> maxHole0 = inst->getMaxHole(p0);
    const std::pair<Clock_T, unsigned int> maxHole1 = inst->getMaxHole(p1);

    return (maxHole0.second > maxHole1.second
            || (maxHole0.second == maxHole1.second && lifetime0 > lifetime1));
}

void N2D2::MemoryManager::optimize(OptimizeStrategy strategy) {
    if (strategy == None)
        return;

    const unsigned int maxLifetime = getMaxLifetime();

    if (strategy == OptimizeMaxLifetimeMinSizeFirst) {
        std::stable_sort(mMemSpaces.begin(), mMemSpaces.end(),
                        MemoryManager::MaxLifetimeMinSizeFirst(maxLifetime));
    }
    else if (strategy == OptimizeMaxLifetimeMaxSizeFirst) {
        std::stable_sort(mMemSpaces.begin(), mMemSpaces.end(),
                        MemoryManager::MaxLifetimeMaxSizeFirst(maxLifetime));
    }
    else if (strategy == OptimizeMaxHoleMaxLifetimeFirst) {
        std::stable_sort(mMemSpaces.begin(), mMemSpaces.end(),
                        MemoryManager::MaxHoleMaxLifetimeFirst(maxLifetime, this));
    }

    std::vector<std::map<unsigned int, unsigned int> > stacks(maxLifetime + 1,
                                        std::map<unsigned int, unsigned int>());

    for (std::vector<std::shared_ptr<MemorySpace> >::const_iterator
        it = mMemSpaces.begin(), itEnd = mMemSpaces.end(); it != itEnd; ++it)
    {
        const Clock_T maxT = ((*it)->released >= 0
                                && (*it)->dependencies.empty())
                                    ? (*it)->released : maxLifetime;

        // Merge stacks over memSpace lifetime
        std::map<unsigned int, unsigned int> mergedStacks;

        for (Clock_T t = (*it)->allocated; t <= maxT; ++t) {
            for (std::map<unsigned int, unsigned int>::iterator itMem
                = stacks[t].begin(), itMemEnd = stacks[t].end();
                itMem != itMemEnd; ++itMem)
            {
                bool newInsert;
                std::map<unsigned int, unsigned int>::iterator itMergedMem;
                std::tie(itMergedMem, newInsert) = mergedStacks.insert(
                    std::make_pair((*itMem).first, (*itMem).second));

                if (!newInsert) {
                    (*itMergedMem).second = std::max((*itMergedMem).second,
                                                     (*itMem).second);
                }
            }
        }

        std::map<unsigned int, unsigned int> mergedStack;

        if (!mergedStacks.empty()) {
            std::map<unsigned int, unsigned int>::iterator itMem
                = mergedStacks.begin();

            mergedStack.insert(*itMem);
            ++itMem;

            while (itMem != mergedStacks.end()) {
                std::map<unsigned int, unsigned int>::reverse_iterator
                    itMergedMem = mergedStack.rbegin();
                const unsigned int nextOffset = (*itMergedMem).first
                                                + (*itMergedMem).second;

                if ((*itMem).first <= nextOffset) {
                    (*itMergedMem).second
                        = std::max((*itMem).first + (*itMem).second, nextOffset)
                            - (*itMergedMem).first;
                }
                else
                    mergedStack.insert(*itMem);

                ++itMem;
            }
        }

        // Allocate in merged stack
        unsigned int offset = 0;
        std::map<unsigned int, unsigned int>::iterator itMem
            = mergedStack.begin();

        while (true) {
            if (itMem == mergedStack.end()
                || (*itMem).first - offset >= (*it)->size)
            {
                mergedStack.insert(std::make_pair(offset, (*it)->size));
                break;
            }
            else {
                offset = (*itMem).first + (*itMem).second;
                ++itMem;
            }
        }

        (*it)->offset = offset;

        for (Clock_T t = (*it)->allocated; t <= maxT; ++t) {
            const std::map<unsigned int, unsigned int> stack
                = getStack((*it), t);
            stacks[t].insert(stack.begin(), stack.end());

            //stacks[t].insert(std::make_pair(offset, (*it)->size));
        }
    }
}

unsigned int N2D2::MemoryManager::getOffset(const std::shared_ptr<Cell>& cell,
                                            unsigned int plane) const
{
    const std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator it = mMemPlanes.find(cell);

    if (it == mMemPlanes.end()) {
        throw std::runtime_error("getOffset(): no memory allocated for cell "
                                 "name " + ((cell) ? cell->getName() : "env"));
    }

    if (plane >= (*it).second.size()) {
        throw std::runtime_error("getOffset(): plane out of range for cell "
                                 "name " + ((cell) ? cell->getName() : "env"));
    }

    return ((*it).second[plane].memSpace->offset + (*it).second[plane].offset);
}

unsigned int N2D2::MemoryManager::getSize(const std::shared_ptr<Cell>& cell,
                                          unsigned int plane) const
{
    const std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator it = mMemPlanes.find(cell);

    if (it == mMemPlanes.end()) {
        throw std::runtime_error("getSize(): no memory allocated for cell "
                                 "name " + ((cell) ? cell->getName() : "env"));
    }

    if (plane >= (*it).second.size()) {
        throw std::runtime_error("getSize(): plane out of range for cell "
                                 "name " + ((cell) ? cell->getName() : "env"));
    }

    return (*it).second[plane].getSize();
}

unsigned int N2D2::MemoryManager::getSize(const std::shared_ptr<Cell>& cell)
    const
{
    const std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator it = mMemPlanes.find(cell);

    if (it == mMemPlanes.end()) {
        throw std::runtime_error("getSize(): no memory allocated for cell "
                                 "name " + ((cell) ? cell->getName() : "env"));
    }

    unsigned int size = 0;

    for (std::vector<MemoryPlane>::const_iterator itPlanes
        = (*it).second.begin(), itPlanesEnd = (*it).second.end();
        itPlanes != itPlanesEnd; ++itPlanes)
    {
        size += (*itPlanes).getSize();
    }

    return size;
}

unsigned int N2D2::MemoryManager::getNbPlanes(const std::shared_ptr<Cell>& cell)
    const
{
    const std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator it = mMemPlanes.find(cell);

    if (it == mMemPlanes.end()) {
        throw std::runtime_error("getSize(): no memory allocated for cell "
                                 "name " + ((cell) ? cell->getName() : "env"));
    }

    return (*it).second.size();
}

unsigned int N2D2::MemoryManager::getPeakUsage() const {
    unsigned int peakUsage = 0;

    for (std::vector<std::shared_ptr<MemorySpace> >::const_iterator
        it = mMemSpaces.begin(), itEnd = mMemSpaces.end(); it != itEnd; ++it)
    {
        peakUsage = std::max(peakUsage,
                             (*it)->offset + (*it)->size);
    }

    return peakUsage;
}

N2D2::MemoryManager::Clock_T N2D2::MemoryManager::getMaxLifetime() const {
    Clock_T maxLifetime = 0;

    for (std::vector<std::shared_ptr<MemorySpace> >::const_iterator
        it = mMemSpaces.begin(), itEnd = mMemSpaces.end(); it != itEnd; ++it)
    {
        maxLifetime = std::max(maxLifetime,
            std::max((*it)->allocated, (*it)->released));
    }

    return maxLifetime;
}

const std::vector<N2D2::MemoryManager::MemoryPlane>&
N2D2::MemoryManager::getPlanes(const std::shared_ptr<Cell>& cell) const
{
    const std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator it = mMemPlanes.find(cell);

    if (it == mMemPlanes.end()) {
        throw std::runtime_error("getSize(): no memory allocated for cell "
                                 "name " + ((cell) ? cell->getName() : "env"));
    }

    return (*it).second;
}

std::map<std::shared_ptr<N2D2::Cell>, std::vector<N2D2::MemoryManager::MemoryPlane> >
N2D2::MemoryManager::getPlanes(std::shared_ptr<MemorySpace> memSpace) const
{
    std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> > planes;

    for (std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator itCell = mMemPlanes.begin(),
        itCellEnd = mMemPlanes.end(); itCell != itCellEnd; ++itCell)
    {
        for (std::vector<MemoryPlane>::const_iterator itPlane
             = (*itCell).second.begin(), itPlaneEnd = (*itCell).second.end();
             itPlane != itPlaneEnd; ++itPlane)
        {
            if ((*itPlane).memSpace == memSpace) {
                std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
                    ::iterator it;
                std::tie(it, std::ignore) = planes.insert(
                    std::make_pair((*itCell).first,
                                   std::vector<MemoryPlane>()));

                (*it).second.push_back((*itPlane));
            }
        }
    }

    return planes;
}

unsigned int N2D2::MemoryManager::getNbPlanes(
    std::shared_ptr<MemorySpace> memSpace) const
{
    unsigned int count = 0;

    for (std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator itCell = mMemPlanes.begin(),
        itCellEnd = mMemPlanes.end(); itCell != itCellEnd; ++itCell)
    {
        for (std::vector<MemoryPlane>::const_iterator itPlane
             = (*itCell).second.begin(), itPlaneEnd = (*itCell).second.end();
             itPlane != itPlaneEnd; ++itPlane)
        {
            if ((*itPlane).memSpace == memSpace)
                ++count;
        }
    }

    return count;
}

void N2D2::MemoryManager::tick(bool autoRelease)
{
    ++mClock;

    if (autoRelease) {
        for (std::vector<std::shared_ptr<MemorySpace> >::const_iterator
            it = mMemSpaces.begin(), itEnd = mMemSpaces.end(); it != itEnd;
            ++it)
        {
            if ((*it)->allocated <= mClock - 1)
                release((*it));
        }
    }
}

void N2D2::MemoryManager::log(const std::string& fileName) const
{
    std::ofstream memData(fileName.c_str());

    if (!memData.good()) {
        throw std::runtime_error("Could not create memory layout log file: "
                                 + fileName);
    }

    const Clock_T maxLifetime = getMaxLifetime();
    const unsigned int peakUsage = getPeakUsage();

    Gnuplot::setDefaultOutput("png", "size 1280,768", "png");

    Gnuplot gnuplot("gnuplot.dat");
    gnuplot.setXrange(0, maxLifetime + 1);
    gnuplot.setYrange(0, 1.05 * (peakUsage / 1024.0));
    gnuplot.setXlabel("Time");
    gnuplot.setYlabel("Memory usage (KB)");
    gnuplot.set("style fill solid");
    gnuplot.set("grid");
    gnuplot.set("xtics", 1);
    gnuplot.unset("key");
    gnuplot.set("grid");
    gnuplot.set("style rectangle fs noborder");

    unsigned int objectId = 1;
    unsigned int labelId = 1;

    memData << std::setfill('0');

    std::map<std::shared_ptr<MemorySpace>,
             std::pair<unsigned int, unsigned int> > memSpaceCount;

    for (std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator it = mMemPlanes.begin(), itEnd = mMemPlanes.end();
        it != itEnd; ++it)
    {
        const std::string name = ((*it).first) ? (*it).first->getName() : "env";
        memData << name << ":\n";

        double minX = -1;
        unsigned int maxY = 0;
        bool odd = false;

        for (std::vector<MemoryPlane>::const_iterator itPlanes
             = (*it).second.begin(), itPlanesBegin = (*it).second.begin(),
            itPlanesEnd = (*it).second.end(); itPlanes != itPlanesEnd;
            ++itPlanes)
        {
            const unsigned int contiguousOffset
                = (*itPlanes).getContiguousOffset();
            const unsigned int contiguousSize = (*itPlanes).getContiguousSize();
            const unsigned int wrappedOffset = (*itPlanes).getWrappedOffset();
            const unsigned int wrappedSize = (*itPlanes).getWrappedSize();

            const Clock_T allocated = (*itPlanes).allocated;
            const Clock_T released = (*itPlanes).memSpace->released;
            const bool isReleased = (released >= 0
                                && (*itPlanes).memSpace->dependencies.empty());

            memData << "  " << (itPlanes - itPlanesBegin) << " "
                << contiguousOffset
                << " (0x" << std::hex << std::setw(8) << contiguousOffset << "U)"
                << std::dec << " -> "
                << (contiguousOffset + contiguousSize)
                << " (0x" << std::hex << std::setw(8)
                << (contiguousOffset + contiguousSize) << "U)";

            if (wrappedSize > 0) {
                memData << " + "
                    << wrappedOffset
                    << " (0x" << std::hex << std::setw(8)
                    << wrappedOffset << "U)"
                    << std::dec << " -> "
                    << (wrappedOffset + wrappedSize)
                    << " (0x" << std::hex << std::setw(8)
                    << (wrappedOffset + wrappedSize) << "U)";
            }

            memData << std::dec << " [" << (*itPlanes).getSize() << "] @ "
                << allocated;

            if (isReleased)
                memData << " to " << released;

            memData << "\n";

            // Gnuplot
            const double startX = allocated;

            minX = (minX > 0.0) ? std::min(minX, startX) : startX;
            maxY = std::max(maxY, contiguousOffset + contiguousSize);

            std::stringstream setStr;

            if ((*itPlanes).size != (*itPlanes).stride) {
                for (unsigned int offset = contiguousOffset;
                    offset < contiguousOffset + contiguousSize;
                    offset += (*itPlanes).stride)
                {
                    setStr.str(std::string());
                    setStr << "set object " << (allocated * 100 + objectId)
                        << " rectangle from "
                        << startX << ","
                        << (offset / 1024.0)
                        << " to " << (((isReleased) ? released : maxLifetime) + 1)
                        << "," << (std::min((offset + (*itPlanes).size),
                                        contiguousOffset + contiguousSize) / 1024.0)
                        << " fc lt " << labelId;

                    if (odd)
                        setStr << " fill transparent solid 0.66";

                    gnuplot << setStr.str();
                    ++objectId;
                }
            }
            else {
                setStr.str(std::string());
                setStr << "set object " << (allocated * 100 + objectId)
                    << " rectangle from "
                    << startX << ","
                    << (contiguousOffset / 1024.0)
                    << " to " << (((isReleased) ? released : maxLifetime) + 1)
                    << "," << ((contiguousOffset + contiguousSize) / 1024.0)
                    << " fc lt " << labelId;

                if (odd)
                    setStr << " fill transparent solid 0.66";

                gnuplot << setStr.str();
                ++objectId;
            }

            if (wrappedSize > 0) {
                setStr.str(std::string());
                setStr << "set object " << (allocated * 100 + objectId)
                    << " rectangle from "
                    << startX << ","
                    << (wrappedOffset / 1024.0)
                    << " to " << (((isReleased) ? released : maxLifetime) + 1)
                    << "," << ((wrappedOffset + wrappedSize) / 1024.0)
                    << " fc lt " << labelId;

                if (odd)
                    setStr << " fill transparent solid 0.66";

                gnuplot << setStr.str();
                ++objectId;

                setStr.str(std::string());
                setStr << "set arrow from "
                    << startX << ","
                    << (contiguousOffset / 1024.0)
                    << " to " << (startX + 0.1)
                    << "," << (contiguousOffset / 1024.0)
                    << " nohead";
                gnuplot << setStr.str();

                setStr.str(std::string());
                setStr << "set arrow from "
                    << (startX + 0.05) << ","
                    << ((contiguousOffset + contiguousSize) / 1024.0)
                    << " to " << (startX + 0.05)
                    << "," << (wrappedOffset / 1024.0);
                gnuplot << setStr.str();
            }

            odd = !odd;
        }

        std::stringstream setStr;
        setStr << "set label " << labelId << " '" << name << "' at "
            << minX << "," << (maxY / 1024.0) << " rotate by 30 font \",8\""
            " offset char 0.5,0.5";
        gnuplot << setStr.str();
        ++labelId;

        memData << "\n";
    }

    std::stringstream setStr;
    setStr << "set arrow from 0," << (peakUsage / 1024.0)
        << " to " << (maxLifetime + 1) << "," << (peakUsage / 1024.0)
        << " nohead lt 1";
    gnuplot << setStr.str();

    setStr.str(std::string());
    setStr << "set label " << labelId << " 'Peak usage = "
        << (peakUsage / 1024.0) << " KB' at 0," << (peakUsage / 1024.0)
        << " textcolor lt 1 offset char 0.5,0.5";
    gnuplot << setStr.str();

    gnuplot.saveToFile(fileName);
    gnuplot << "plot 0";

    Gnuplot::setDefaultOutput();
}

unsigned int N2D2::MemoryManager::onStack(unsigned int size)
{
    unsigned int offset = 0;
    std::map<unsigned int, unsigned int>::iterator itMem = mMemStack.begin();

    while (true) {
        if (itMem == mMemStack.end()
            || (*itMem).first - offset >= size)
        {
            mMemStack.insert(std::make_pair(offset, size));
            break;
        }
        else {
            offset = (*itMem).first + (*itMem).second;
            ++itMem;
        }
    }

    return offset;
}

unsigned int N2D2::MemoryManager::offStack(unsigned int offset)
{
    std::map<unsigned int, unsigned int>::iterator itMem
        = mMemStack.find(offset);

    if (itMem == mMemStack.end()) {
        throw std::runtime_error("offStack(): offset not found in stack");
    }
    else {
        const unsigned int size = (*itMem).second;
        mMemStack.erase(offset);
        return size;
    }
}

std::map<unsigned int, unsigned int> N2D2::MemoryManager::getStack(
    std::shared_ptr<MemorySpace> memSpace,
    Clock_T clock) const
{
    // Find all planes associated to memSpace and index them by their allocated
    // value in a map
    std::map<Clock_T, std::vector<MemoryPlane> > planes;

    for (std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator itCell = mMemPlanes.begin(),
        itCellEnd = mMemPlanes.end(); itCell != itCellEnd; ++itCell)
    {
        for (std::vector<MemoryPlane>::const_iterator itPlane
             = (*itCell).second.begin(), itPlaneEnd = (*itCell).second.end();
             itPlane != itPlaneEnd; ++itPlane)
        {
            if ((*itPlane).memSpace == memSpace) {
                std::map<Clock_T, std::vector<MemoryPlane> >::iterator it;
                std::tie(it, std::ignore) = planes.insert(
                    std::make_pair((*itPlane).allocated,
                                   std::vector<MemoryPlane>()));

                (*it).second.push_back((*itPlane));
            }
        }
    }

    // Find the planes allocated at time clock or the one just before
    // => obtain all the planes that are considered valid at the time clock
    Clock_T c = clock;
    std::map<Clock_T, std::vector<MemoryPlane> >::iterator itPlanes;

    do
        itPlanes = planes.find(c);
    while (itPlanes == planes.end() && (c--) > 0);

    assert(itPlanes != planes.end());

    // Fill the stack at time clock
    std::map<unsigned int, unsigned int> stack;

    for (std::vector<MemoryPlane>::const_iterator
        it = (*itPlanes).second.begin(), itEnd = (*itPlanes).second.end();
        it != itEnd; ++it)
    {
        stack.insert(std::make_pair((*it).getContiguousOffset(),
                                    (*it).getContiguousSize()));

        if ((*it).getWrappedSize() > 0) {
            stack.insert(std::make_pair((*it).getWrappedOffset(),
                                        (*it).getWrappedSize()));
        }
    }

    return stack;
}

std::pair<N2D2::MemoryManager::Clock_T, unsigned int>
N2D2::MemoryManager::getMaxHole(std::shared_ptr<MemorySpace> memSpace) const
{
    std::map<Clock_T, unsigned int> holesSize;

    for (std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        ::const_iterator itCell = mMemPlanes.begin(),
        itCellEnd = mMemPlanes.end(); itCell != itCellEnd; ++itCell)
    {
        for (std::vector<MemoryPlane>::const_iterator itPlane
             = (*itCell).second.begin(), itPlaneEnd = (*itCell).second.end();
             itPlane != itPlaneEnd; ++itPlane)
        {
            if ((*itPlane).memSpace == memSpace) {
                const unsigned int holeSize = memSpace->size
                    - (*itPlane).getContiguousSize()
                    - (*itPlane).getWrappedSize();

                std::map<Clock_T, unsigned int>::iterator it;
                bool newInsert;
                std::tie(it, newInsert) = holesSize.insert(
                    std::make_pair((*itPlane).allocated, holeSize));

                if (!newInsert) {
                    // Another plane exists at the same time, one must substract
                    // the size of this other plane from the hole size
                    (*it).second = std::max(0, (int)(*it).second
                        - (int)(*itPlane).getContiguousSize()
                        - (int)(*itPlane).getWrappedSize());
                }
            }
        }
    }

    return *std::max_element(holesSize.begin(),
                             holesSize.end(),
                             Utils::PairSecondPred<Clock_T, unsigned int>());
}
