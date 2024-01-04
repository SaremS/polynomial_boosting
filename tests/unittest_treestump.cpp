#include <vector>

#include "gtest/gtest.h"
#include "linalg.h"
#include "regression_models/linear_regression_model.h"
#include "tree/treestump.h"
#include "loss_functions/quadratic_loss.h"

TEST(testtree, simple_treestump) {

	std::vector<double> inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7,0,
		8,0,
		9,0,
		10.0,
		11.0
	};
	Matrix X = Matrix(inputs);
	Matrix y = Matrix(inputs);

	LossFunction* loss_function = new QuadraticLoss();

	TreeStump model = TreeStump(loss_function, 1);
	model.fit(X,y);

	Matrix predictions = model.predict(X);
	
	double mse = (predictions - y).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}
