#include "gtest/gtest.h"
#include "loss_functions/quadratic_loss.h"

TEST(testLoss, losstest) {
	QuadraticLoss qloss;

	EXPECT_EQ(0.5, qloss.loss(1.0, 2.0));
}
