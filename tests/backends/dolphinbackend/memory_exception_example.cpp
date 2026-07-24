#include <gtest/gtest.h>
#include "dolphinbackend/Exceptions.h"
#include <cstdlib>

TEST(MemoryExceptionTest, LargeAllocationThrows) {
    size_t hugeSize = SIZE_MAX / 2;
#ifdef __APPLE__
    GTEST_SKIP() << "macOS overcommits memory — malloc succeeds for huge sizes";
#else
    EXPECT_THROW(
        {
            void* ptr = malloc(hugeSize);
            if (!ptr) {
                throw dolphin::backend::MemoryException("test", "malloc", hugeSize);
            }
            free(ptr);
        },
        dolphin::backend::MemoryException
    );
#endif
}

TEST(MemoryExceptionTest, ExceptionMessage) {
    try {
        throw dolphin::backend::MemoryException("allocation failed", "test_backend", 999999, "allocate");
    } catch (const dolphin::backend::MemoryException& e) {
        std::string msg = e.what();
        EXPECT_FALSE(msg.empty());
        EXPECT_EQ(e.getBackendType(), "test_backend");
    }
}

TEST(MemoryExceptionTest, BackendExceptionType) {
    dolphin::backend::BackendException ex("backend error", "cpu", "operation");
    EXPECT_EQ(ex.getBackendType(), "cpu");
    EXPECT_EQ(ex.getOperation(), "operation");
    EXPECT_FALSE(ex.getDetailedMessage().empty());
}
