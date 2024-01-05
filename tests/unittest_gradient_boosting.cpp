#include <vector>

#include "gtest/gtest.h"
#include "linalg.h"
#include "loss_functions/quadratic_loss.h"
#include "gradient_boosting/gradient_boosting.h"

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

	TreeStump model = GradientBoosting(loss_function);
	model.fit(X,y);

	Matrix predictions = model.predict(X);
	
	double mse = (predictions - y).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}


TEST(testtree, simple_importances) {

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

	TreeStump model = GradientBoosting(loss_function);
	model.fit(X,y);

	std::vector<double> importances = model.get_feature_importances();

	EXPECT_TRUE(importances.size() == 1);
	EXPECT_TRUE(importances[0] >= 0.0);
}
