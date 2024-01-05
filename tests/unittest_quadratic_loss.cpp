#include "gtest/gtest.h"
#include "loss_functions/quadratic_loss.h"
#include "linalg.h"

TEST(testLoss, quadloss) {
	QuadraticLoss qloss;

	EXPECT_EQ(0.5, qloss.loss(1.0, 2.0));
	EXPECT_EQ(-1.0, qloss.first_derivative(1.0, 2.0));
	EXPECT_EQ(1.0, qloss.second_derivative(1.0, 2.0));

	std::vector<double> vals = {1.0,2.0,3.0};
	Matrix testmat = Matrix(vals);

	Matrix minimizer = qloss.minimizer_matrix(testmat);

	//quadratic loss minimizer is the mean of the matrix
	std::vector<std::vector<double>> target = {
		{2.0},
		{2.0},
		{2.0}
	};

	Matrix comparemat = Matrix(target);

	EXPECT_EQ(comparemat, minimizer);
}
