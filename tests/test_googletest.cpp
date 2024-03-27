#include <gtest/gtest.h>
#include "net.h"

int One() {
    return 1;
};

TEST(SuiteName, TestingTest) {
    EXPECT_EQ(One(), 1);
    EXPECT_EQ(One() - 1, 1);
}
