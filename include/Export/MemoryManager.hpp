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

#ifndef N2D2_RCAR_MEMORY_MANAGER_H
#define N2D2_RCAR_MEMORY_MANAGER_H

#include <memory>
#include <vector>
#include <map>

#include "Cell/Cell.hpp"

namespace N2D2 {
class MemoryManager {
public:
    typedef int Clock_T;

    enum OptimizeStrategy {
        None,
        OptimizeMaxLifetimeMinSizeFirst,
        OptimizeMaxLifetimeMaxSizeFirst,
        OptimizeMaxHoleMaxLifetimeFirst
    };

    // MemorySpace are contiguous, non-overlapping memory blocks, that can be
    // re-arranged freely.
    struct MemorySpace {
        MemorySpace(Clock_T clock_,
                    unsigned int offset_,
                    unsigned int size_,
                    std::vector<std::shared_ptr<Cell> > dependencies_
                        = std::vector<std::shared_ptr<Cell> >()
        ):
            offset(offset_),
            size(size_),
            dependencies(dependencies_),
            allocated(clock_),
            released(-1) {}

        unsigned int offset;
        unsigned int size;
        std::vector<std::shared_ptr<Cell> > dependencies;
        Clock_T allocated;
        Clock_T released;
    };

    // MemoryPlane belongs to a MemorySpace. Any number of potentially
    // overlapping planes can be associated to a MemorySpace.
    // MemoryPlane can be non-contiguous (in case of stride, or wrapping, when
    // offset + size > memSpace.size).
    // MemoryPlane cannot be re-arranged inside a MemorySpace.
    struct MemoryPlane {
        MemoryPlane(std::shared_ptr<MemorySpace> memSpace_,
                    Clock_T clock_,
                    unsigned int offset_,
                    unsigned int size_,
                    unsigned int stride_ = 0,
                    unsigned int length_ = 1,
                    unsigned int count_ = 1
        ):
            memSpace(memSpace_),
            allocated(clock_),
            offset(offset_),
            // limit must be a multiple of (stride * length)
            // uses floor() to stay below memSpace_->size
            limit((count_ > 1)
                ? std::floor((memSpace_->size - offset)
                        / (double)(std::max(size_, stride_) * length_))
                    * (std::max(size_, stride_) * length_)
                : memSpace_->size - offset_),
            size(size_),
            stride(std::max(size_, stride_)),
            length(length_),
            count(count_) {}

        inline unsigned int getSize() const {
            return stride * length * count;
        }

        inline unsigned int getUsefulSize() const {
            return size * length * count;
        }

        inline unsigned int getContiguousOffset() const {
            return memSpace->offset + offset;
        }

        inline unsigned int getContiguousSize() const {
            return std::min(getSize(), getLimit());
        }

        inline unsigned int getWrappedOffset() const {
            return memSpace->offset;
        }

        inline unsigned int getWrappedSize() const {
            return getSize() - getContiguousSize();
        }

        inline unsigned int getFinalOffset() const {
            return (getWrappedSize() > 0)
                ? getWrappedOffset() + getWrappedSize()
                : getContiguousOffset() + getContiguousSize();
        }

        inline unsigned int getUpperOffset() const {
            return (getContiguousOffset() + getContiguousSize());
        }

        std::shared_ptr<MemorySpace> memSpace;
        Clock_T allocated;
        unsigned int offset;
        unsigned int limit;
        unsigned int size;
        unsigned int stride;
        unsigned int length;
        unsigned int count;

    private:
        inline unsigned int getLimit() const {
            return (count > 1)
                ? std::floor((memSpace->size - offset)
                        / (double)(stride * length)) * (stride * length)
                : memSpace->size - offset;
        }
    };

    struct MaxLifetimeMinSizeFirst {
        MaxLifetimeMinSizeFirst(unsigned int maxLifetime_)
            : maxLifetime(maxLifetime_) {}
        const unsigned int maxLifetime;

        bool operator()(const std::shared_ptr<MemorySpace>& p0,
                        const std::shared_ptr<MemorySpace>& p1);
    };

    struct MaxLifetimeMaxSizeFirst {
        MaxLifetimeMaxSizeFirst(unsigned int maxLifetime_)
            : maxLifetime(maxLifetime_) {}
        const unsigned int maxLifetime;

        bool operator()(const std::shared_ptr<MemorySpace>& p0,
                        const std::shared_ptr<MemorySpace>& p1);
    };

    struct MaxHoleMaxLifetimeFirst {
        MaxHoleMaxLifetimeFirst(unsigned int maxLifetime_, MemoryManager* inst_)
            : maxLifetime(maxLifetime_),
              inst(inst_) {}
        const unsigned int maxLifetime;
        MemoryManager* inst;

        bool operator()(const std::shared_ptr<MemorySpace>& p0,
                        const std::shared_ptr<MemorySpace>& p1);
    };

    MemoryManager(): mClock(0) {}
    /// Generates a new MemorySpace
    std::shared_ptr<MemorySpace> reserve(unsigned int size,
                                    const std::vector<std::shared_ptr<Cell> >&
                          dependencies = std::vector<std::shared_ptr<Cell> >());
    /// Expand an existing MemorySpace, without affecting its MemoryPlane
    /// This function rebuild the memory stack mMemStack
    void expand(std::shared_ptr<MemorySpace> memSpace,
                unsigned int requiredSize,
                const std::vector<std::shared_ptr<Cell> >&
                additionalDependencies = std::vector<std::shared_ptr<Cell> >());
    /// Generates a MemoryPlane in a new MemorySpace
    MemoryPlane allocate(unsigned int size,
                         const std::vector<std::shared_ptr<Cell> >&
                          dependencies = std::vector<std::shared_ptr<Cell> >(),
                         unsigned int stride = 0,
                         unsigned int length = 1,
                         unsigned int count = 1);
    /// Generates a MemoryPlane in a new MemorySpace, associated to a Cell
    unsigned int allocate(const std::shared_ptr<Cell>& cell,
                          unsigned int size,
                          const std::vector<std::shared_ptr<Cell> >&
                          dependencies = std::vector<std::shared_ptr<Cell> >(),
                          unsigned int stride = 0,
                          unsigned int length = 1,
                          unsigned int count = 1);
    bool isWrapAround(std::shared_ptr<MemorySpace> memSpace,
                      unsigned int offset,
                      unsigned int size,
                      unsigned int stride = 0,
                      unsigned int length = 1,
                      unsigned int count = 1) const;
    /// Generate a new MemoryPlane in an existing MemorySpace
    MemoryPlane reallocate(std::shared_ptr<MemorySpace> memSpace,
                           unsigned int offset,
                           unsigned int size,
                           bool wrapAround,
                           unsigned int extraSize = 0,
                           const std::vector<std::shared_ptr<Cell> >&
                additionalDependencies = std::vector<std::shared_ptr<Cell> >(),
                           unsigned int stride = 0,
                           unsigned int length = 1,
                           unsigned int count = 1);
    /// Generate a new MemoryPlane directly following an existing MemoryPlane
    /// memPlane with an additionnal offset extraOffset
    MemoryPlane reallocate(const MemoryPlane& memPlane,
                           unsigned int extraOffset,
                           unsigned int size,
                           bool wrapAround,
                           unsigned int extraSize = 0,
                           const std::vector<std::shared_ptr<Cell> >&
                additionalDependencies = std::vector<std::shared_ptr<Cell> >(),
                           unsigned int stride = 0,
                           unsigned int length = 1,
                           unsigned int count = 1);
    /// Generate a new MemoryPlane in an existing MemorySpace, associated to a 
    /// Cell
    unsigned int reallocate(std::shared_ptr<MemorySpace> memSpace,
                            const std::shared_ptr<Cell>& cell,
                            unsigned int offset,
                            unsigned int size,
                            bool wrapAround,
                            unsigned int extraSize = 0,
                            const std::vector<std::shared_ptr<Cell> >&
                additionalDependencies = std::vector<std::shared_ptr<Cell> >(),
                            unsigned int stride = 0,
                            unsigned int length = 1,
                            unsigned int count = 1);
    /// Generate a new MemoryPlane directly following an existing MemoryPlane
    /// memPlane with an additionnal offset extraOffset
    unsigned int reallocate(const MemoryPlane& memPlane,
                            const std::shared_ptr<Cell>& cell,
                            unsigned int extraOffset,
                            unsigned int size,
                            bool wrapAround,
                            unsigned int extraSize = 0,
                            const std::vector<std::shared_ptr<Cell> >&
                additionalDependencies = std::vector<std::shared_ptr<Cell> >(),
                            unsigned int stride = 0,
                            unsigned int length = 1,
                            unsigned int count = 1);

    unsigned int release(std::shared_ptr<MemorySpace> memSpace);
    unsigned int release(const std::shared_ptr<Cell>& cell);
    unsigned int releaseDependencies(const std::shared_ptr<Cell>& cell);
    void optimize(OptimizeStrategy strategy);
    unsigned int getOffset(const std::shared_ptr<Cell>& cell,
                           unsigned int plane = 0) const;
    unsigned int getSize(const std::shared_ptr<Cell>& cell,
                         unsigned int plane) const;
    unsigned int getSize(const std::shared_ptr<Cell>& cell) const;
    unsigned int getNbPlanes(const std::shared_ptr<Cell>& cell) const;
    unsigned int getPeakUsage() const;
    Clock_T getMaxLifetime() const;
    const std::vector<MemoryPlane>& getPlanes(const std::shared_ptr<Cell>& cell)
        const;
    const std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >&
        getPlanes() const { return mMemPlanes; }
    std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> >
        getPlanes(std::shared_ptr<MemorySpace> memSpace) const;
    unsigned int getNbPlanes(std::shared_ptr<MemorySpace> memSpace) const;
    void tick(bool autoRelease = true);
    void log(const std::string& fileName) const;

private:
    /// Find a valid offset in the memory stack that can fit a contiguous chunk
    /// of memory of size @size
    unsigned int onStack(unsigned int size);
    unsigned int offStack(unsigned int offset);
    std::map<unsigned int, unsigned int> getStack(
        std::shared_ptr<MemorySpace> memSpace,
        Clock_T clock) const;
    std::pair<Clock_T, unsigned int> getMaxHole(
        std::shared_ptr<MemorySpace> memSpace) const;

    std::map<unsigned int, unsigned int> mMemStack;
    std::vector<std::shared_ptr<MemorySpace> > mMemSpaces;
    std::map<std::shared_ptr<Cell>, std::vector<MemoryPlane> > mMemPlanes;
    Clock_T mClock;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::MemoryManager::OptimizeStrategy>::data[]
    = {"None",
       "OptimizeMaxLifetimeMinSizeFirst",
       "OptimizeMaxLifetimeMaxSizeFirst",
       "OptimizeMaxHoleMaxLifetimeFirst"};
}

#endif // N2D2_RCAR_MEMORY_MANAGER_H
