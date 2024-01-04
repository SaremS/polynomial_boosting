#include <vector>

#include "gtest/gtest.h"
#include "linalg.h"
#include "regression_models/linear_regression_model.h"

TEST(testregression, simple_regression) {
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
	X.prepend_ones();

	Matrix y = Matrix(inputs);

	LinearRegressionModel model = LinearRegressionModel();
	model.fit(X,y);

	Matrix predictions = model.predict(X);
	
	double mse = (predictions - y).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}

