/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

    This file is not part of the open source version of N2D2 and is NOT under
    the CeCILL-C license. This code is the property of the CEA. It can not be
    copied or disseminated without its authorization.
*/

#include <cstdlib>

#include "DeepNet.hpp"
#include "Cell/FcCell_Frame.hpp"
#include "Export/MemoryManager.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(MemoryManager, allocate1) {
    Network net;
    DeepNet deepNet(net);

    std::shared_ptr<Cell> cell1
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell1", 1);
    std::shared_ptr<Cell> cell2
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell2", 1);
    std::shared_ptr<Cell> cell3
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell3", 1);
    std::shared_ptr<Cell> cell4
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell4", 1);

    MemoryManager memManager;
    memManager.allocate(cell1, 1024, {cell2});

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell2}));
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().limit, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().stride, 1024);

    memManager.releaseDependencies(cell1);

    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell2}));
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, -1);

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, 1);

    memManager.allocate(cell2, 2048, {cell3});

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024 + 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->offset, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().stride, 2048);

    memManager.releaseDependencies(cell2);

    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, -1);
    ASSERT_TRUE(memManager.getPlanes(cell1).back().memSpace->dependencies.empty());

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, 2);

    const std::vector<MemoryManager::MemoryPlane>& memPlanes
        = memManager.getPlanes(cell2);

    ASSERT_EQUALS(memPlanes.size(), 1);

    memManager.reallocate(memPlanes.back().memSpace,
                          cell3, 512, 2048, false, 0, {cell4});

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024 + 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->offset, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->size, 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().offset, 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().stride, 2048);

    memManager.releaseDependencies(cell3);
    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, 3);

    memManager.allocate(cell4, 1024);

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024 + 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell4).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->allocated, 3);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().limit, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().stride, 1024);

    memManager.releaseDependencies(cell4);
    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->allocated, 3);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->released, 4);

    memManager.log("MemoryManager_allocate1.log");
}

TEST(MemoryManager, allocate2) {
    Network net;
    DeepNet deepNet(net);

    std::shared_ptr<Cell> cell1
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell1", 1);
    std::shared_ptr<Cell> cell2
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell2", 1);
    std::shared_ptr<Cell> cell3
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell3", 1);
    std::shared_ptr<Cell> cell4
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell4", 1);

    MemoryManager memManager;
    memManager.allocate(cell1, 1024, {cell2});

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell2}));
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().limit, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().stride, 1024);

    memManager.releaseDependencies(cell1);

    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell2}));
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, -1);

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, 1);

    memManager.allocate(cell2, 2048, {cell3});

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024 + 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->offset, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().stride, 2048);

    memManager.releaseDependencies(cell2);

    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, -1);
    ASSERT_TRUE(memManager.getPlanes(cell1).back().memSpace->dependencies.empty());

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, 2);

    const std::vector<MemoryManager::MemoryPlane>& memPlanes
        = memManager.getPlanes(cell1);

    ASSERT_EQUALS(memPlanes.size(), 1);

    memManager.reallocate(memPlanes.back().memSpace,
                          cell3, 512, 2048, false, 0, {cell4});

    ASSERT_EQUALS(memManager.getPeakUsage(), 2048 + 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->size, 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell4}));
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().offset, 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().stride, 2048);

    // cell2 memSpace should have moved
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, 2);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->offset, 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().stride, 2048);

    memManager.releaseDependencies(cell3);

    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell4}));
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, -1);
    ASSERT_TRUE(memManager.getPlanes(cell2).back().memSpace->dependencies.empty());

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, 3);

    memManager.allocate(cell4, 1024);

    ASSERT_EQUALS(memManager.getPeakUsage(), 2048 + 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell4).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->allocated, 3);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->offset, 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().limit, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().stride, 1024);

    memManager.releaseDependencies(cell4);
    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->allocated, 3);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->released, 4);

    memManager.log("MemoryManager_allocate2.log");
}

TEST(MemoryManager, allocate3) {
    Network net;
    DeepNet deepNet(net);

    std::shared_ptr<Cell> cell1
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell1", 1);
    std::shared_ptr<Cell> cell2
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell2", 1);
    std::shared_ptr<Cell> cell3
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell3", 1);
    std::shared_ptr<Cell> cell4
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell4", 1);

    MemoryManager memManager;
    memManager.allocate(cell1, 1024, {cell2});

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell2}));
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().limit, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().stride, 1024);

    memManager.releaseDependencies(cell1);

    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell2}));
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, -1);

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, 1);

    memManager.allocate(cell2, 2048, {cell3});

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024 + 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->offset, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().stride, 2048);

    memManager.releaseDependencies(cell2);

    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, -1);
    ASSERT_TRUE(memManager.getPlanes(cell1).back().memSpace->dependencies.empty());

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, 2);

    const std::vector<MemoryManager::MemoryPlane>& memPlanes
        = memManager.getPlanes(cell1);

    ASSERT_EQUALS(memPlanes.size(), 1);

    memManager.reallocate(memPlanes.back().memSpace,
                          cell3, 512, 2048, false);

    ASSERT_EQUALS(memManager.getPeakUsage(), 2048 + 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->size, 2048 + 512);
    ASSERT_TRUE(memManager.getPlanes(cell3).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().offset, 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().stride, 2048);

    // cell2 memSpace should have moved
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, 2);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->offset, 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().stride, 2048);

    memManager.releaseDependencies(cell3);

    ASSERT_TRUE(memManager.getPlanes(cell3).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, -1);
    ASSERT_TRUE(memManager.getPlanes(cell2).back().memSpace->dependencies.empty());

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, 3);

    memManager.reallocate(memPlanes.back().memSpace,
                          cell4, 256, 1024, false);

    ASSERT_EQUALS(memManager.getPeakUsage(), 2048 + 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell4).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->size, 2048 + 512);
    ASSERT_TRUE(memManager.getPlanes(cell4).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().offset, 256);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().limit, 2048 + 256);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().stride, 1024);

    // cell2 memSpace should not have moved
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, 2);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->offset, 2048 + 512);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().stride, 2048);

    memManager.releaseDependencies(cell4);
    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->released, 4);

    memManager.log("MemoryManager_allocate3.log");
}

TEST(MemoryManager, allocate3_wrapAround) {
    Network net;
    DeepNet deepNet(net);

    std::shared_ptr<Cell> cell1
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell1", 1);
    std::shared_ptr<Cell> cell2
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell2", 1);
    std::shared_ptr<Cell> cell3
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell3", 1);
    std::shared_ptr<Cell> cell4
        = std::make_shared<FcCell_Frame<Float_T> >(deepNet, "cell4", 1);

    MemoryManager memManager;
    memManager.allocate(cell1, 1024, {cell2});

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell2}));
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().size, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().limit, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().stride, 1024);

    memManager.releaseDependencies(cell1);

    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell2}));
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, -1);

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell1).back().memSpace->released, 1);

    memManager.allocate(cell2, 2048, {cell3});

    ASSERT_EQUALS(memManager.getPeakUsage(), 1024 + 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->offset, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().stride, 2048);

    memManager.releaseDependencies(cell2);

    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, -1);
    ASSERT_TRUE(memManager.getPlanes(cell1).back().memSpace->dependencies.empty());

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, 2);

    const std::vector<MemoryManager::MemoryPlane>& memPlanes
        = memManager.getPlanes(cell1);

    ASSERT_EQUALS(memPlanes.size(), 1);

    memManager.reallocate(memPlanes.back().memSpace,
                          cell3, 512, 2048, true);

    ASSERT_EQUALS(memManager.getPeakUsage(), 2048 + 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell3).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->size, 2048);
    ASSERT_TRUE(memManager.getPlanes(cell3).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().offset, 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().limit, 2048 - 512);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().stride, 2048);

    // cell2 memSpace should have moved
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, 2);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->offset, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->dependencies,
        std::vector<std::shared_ptr<Cell> >({cell3}));
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().stride, 2048);

    memManager.releaseDependencies(cell3);

    ASSERT_TRUE(memManager.getPlanes(cell3).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, -1);
    ASSERT_TRUE(memManager.getPlanes(cell2).back().memSpace->dependencies.empty());

    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell3).back().memSpace->released, 3);

    memManager.reallocate(memPlanes.back().memSpace,
                          cell4, 1024, 1792, true);

    ASSERT_EQUALS(memManager.getPeakUsage(), 2048 + 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell4).size(), 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->released, -1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->size, 2048);
    ASSERT_TRUE(memManager.getPlanes(cell4).back().memSpace->dependencies.empty());
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().offset, 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().size, 1792);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().limit, 2048 - 1024);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().stride, 1792);

    // cell2 memSpace should not have moved
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->allocated, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->released, 2);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->offset, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().memSpace->size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().offset, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().size, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().limit, 2048);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().count, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().length, 1);
    ASSERT_EQUALS(memManager.getPlanes(cell2).back().stride, 2048);

    memManager.releaseDependencies(cell4);
    memManager.tick();

    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->allocated, 0);
    ASSERT_EQUALS(memManager.getPlanes(cell4).back().memSpace->released, 4);

    memManager.log("MemoryManager_allocate3_wrapAround.log");
}

RUN_TESTS()
