#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "linalg.h"
#include "regression_models/linear_regression_model.h"
#include "regression_models/fast_linear_regression.h"

TEST(testregression, simple_regression) {
	std::vector<double> inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
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

TEST(testregression, fast_regression) {
	std::vector<double> inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};
	Matrix X = Matrix(inputs);

	Matrix y = Matrix(inputs);

	FastLinearRegression model = FastLinearRegression();
	model.fit(X,y);

	Matrix predictions = model.predict(X);
	
	double mse = (predictions - y).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}


TEST(testregression, fast_regression_constant_ones) {
	std::vector<double> inputs = {
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
		1.0
	};

	std::vector<double> targets = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	Matrix X = Matrix(inputs);
	Matrix y = Matrix(targets);

	FastLinearRegression model = FastLinearRegression();
	model.fit(X,y);

	Matrix predictions = model.predict(X);
	
 	double y_mean = y.sum() / 10.0;	
	Matrix y_mean_mat = Matrix(std::vector<double>(10, y_mean));


	double mse = (predictions - y_mean_mat).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}


TEST(testregression, comp_normal_fast_regression) {
	std::vector<double> inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};
	Matrix X = Matrix(inputs);
	Matrix Xnormal = Matrix(inputs);
	Xnormal.prepend_ones();

	Matrix y = Matrix(inputs);

	LinearRegressionModel normal_model = LinearRegressionModel();
	FastLinearRegression fast_model = FastLinearRegression();
	
	normal_model.fit(Xnormal,y);
	
	fast_model.fit(X,y);


	Matrix predictions_normal = normal_model.predict(Xnormal);


	Matrix predictions_fast = fast_model.predict(X);

	double mse = (predictions_normal - predictions_fast).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}


TEST(testregression, comp_regularized_normal_fast_regression) {
	std::vector<double> inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};
	Matrix X = Matrix(inputs);
	Matrix Xnormal = Matrix(inputs);
	Xnormal.prepend_ones();

	Matrix y = Matrix(inputs);

	LinearRegressionModel normal_model = LinearRegressionModel(5.0);
	FastLinearRegression fast_model = FastLinearRegression(5.0);
	
	normal_model.fit(Xnormal,y);
	
	fast_model.fit(Xnormal,y);

	std::cout << normal_model.get_coefficients().get_as_eigen() << std::endl;
	std::cout << std::endl;
	std::cout << fast_model.get_coefficients().get_as_eigen() << std::endl;

	Matrix predictions_normal = normal_model.predict(Xnormal);
	Matrix predictions_fast = fast_model.predict(Xnormal);

	std::cout << predictions_normal.get_as_eigen() << std::endl;
	std::cout << predictions_fast.get_as_eigen() << std::endl;

	double mse = (predictions_normal - predictions_fast).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}

TEST(testregression, verify_fast_ols_sse) {
	std::vector<double> inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};
	Matrix X = Matrix(inputs);

	std::vector<double> targets = {
		2.0,
		2.0,
		6.4,
		5.0,
		6.0,
		7.0,
		8.2,
		9.0,
		11.0,
		11.5
	};
	Matrix y = Matrix(targets);

	FastLinearRegression fast_model = FastLinearRegression(5.0);
	fast_model.fit(X,y);

	Matrix predictions_fast = fast_model.predict(X);

	double mse_manual = 0.5*(y - predictions_fast).pow_elementwise(2.0).sum() / 10.0;
	double mse_fast = 0.5*fast_model.get_ols_sse() / 10.0;

	EXPECT_NEAR(mse_manual, mse_fast, 1e-9);
}

TEST(testregression, fast_regression_add) {
	std::vector<double> full_inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	std::vector<double> full_targets = {
		2.0,
		2.0,
		6.4,
		5.0,
		6.0,
		7.0,
		8.2,
		9.0,
		11.0,
		11.0
	};

	std::vector<double> first_inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0
	};
	
	std::vector<double> first_targets = {
		2.0,
		2.0,
		6.4,
		5.0,
		6.0
	};

	std::vector<double> second_inputs = {
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	std::vector<double> second_targets = {
		7.0,
		8.2,
		9.0,
		11.0,
		11.0
	};

	Matrix X = Matrix(full_inputs);
	Matrix y = Matrix(full_targets);
	Matrix X1 = Matrix(first_inputs);
	Matrix y1 = Matrix(first_targets);
	Matrix X2 = Matrix(second_inputs);
	Matrix y2 = Matrix(second_targets);

	FastLinearRegression model = FastLinearRegression();
	model.fit(X1,y1);
	model.update_coefficients_add(X2,y2);

	Matrix predictions = model.predict(X);

	FastLinearRegression model2 = FastLinearRegression();
	model2.fit(X,y);

	Matrix predictions2 = model2.predict(X);

	double mse = (predictions - predictions2).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}


TEST(testregression, fast_regression_add_regularized) {
	std::vector<double> full_inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	std::vector<double> full_targets = {
		2.0,
		2.0,
		6.4,
		5.0,
		6.0,
		7.0,
		8.2,
		9.0,
		11.0,
		11.0
	};

	std::vector<double> first_inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0
	};
	
	std::vector<double> first_targets = {
		2.0,
		2.0,
		6.4,
		5.0,
		6.0
	};

	std::vector<double> second_inputs = {
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	std::vector<double> second_targets = {
		7.0,
		8.2,
		9.0,
		11.0,
		11.0
	};

	Matrix X = Matrix(full_inputs);
	Matrix y = Matrix(full_targets);
	Matrix X1 = Matrix(first_inputs);
	Matrix y1 = Matrix(first_targets);
	Matrix X2 = Matrix(second_inputs);
	Matrix y2 = Matrix(second_targets);

	FastLinearRegression model = FastLinearRegression(5.0);
	model.fit(X1,y1);
	model.update_coefficients_add(X2,y2);

	Matrix predictions = model.predict(X);

	FastLinearRegression model2 = FastLinearRegression(5.0);
	model2.fit(X,y);

	Matrix predictions2 = model2.predict(X);

	double mse = (predictions - predictions2).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}


TEST(testregression, fast_regression_drop) {
	std::vector<double> full_inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	std::vector<double> full_targets = {
		2.0,
		2.0,
		6.4,
		5.0,
		6.0,
		7.0,
		8.2,
		9.0,
		11.0,
		11.0
	};

	std::vector<double> first_inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0
	};
	
	std::vector<double> first_targets = {
		2.0,
		2.0,
		6.4,
		5.0,
		6.0
	};

	std::vector<double> second_inputs = {
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	std::vector<double> second_targets = {
		7.0,
		8.2,
		9.0,
		11.0,
		11.0
	};

	Matrix X = Matrix(full_inputs);
	Matrix y = Matrix(full_targets);
	Matrix X1 = Matrix(first_inputs);
	Matrix y1 = Matrix(first_targets);
	Matrix X2 = Matrix(second_inputs);
	Matrix y2 = Matrix(second_targets);

	FastLinearRegression model = FastLinearRegression();
	model.fit(X,y);
	model.update_coefficients_drop(X2,y2);

	Matrix predictions = model.predict(X);

	FastLinearRegression model2 = FastLinearRegression();
	model2.fit(X1,y1);

	Matrix predictions2 = model2.predict(X);

	double mse = (predictions - predictions2).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}


TEST(testregression, fast_regression_drop_regularized) {
	std::vector<double> full_inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	std::vector<double> full_targets = {
		2.0,
		2.0,
		6.4,
		5.0,
		6.0,
		7.0,
		8.2,
		9.0,
		11.0,
		11.0
	};

	std::vector<double> first_inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0
	};
	
	std::vector<double> first_targets = {
		2.0,
		2.0,
		6.4,
		5.0,
		6.0
	};

	std::vector<double> second_inputs = {
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	std::vector<double> second_targets = {
		7.0,
		8.2,
		9.0,
		11.0,
		11.0
	};

	Matrix X = Matrix(full_inputs);
	Matrix y = Matrix(full_targets);
	Matrix X1 = Matrix(first_inputs);
	Matrix y1 = Matrix(first_targets);
	Matrix X2 = Matrix(second_inputs);
	Matrix y2 = Matrix(second_targets);

	FastLinearRegression model = FastLinearRegression(5.0);
	model.fit(X,y);
	model.update_coefficients_drop(X2,y2);

	Matrix predictions = model.predict(X);

	FastLinearRegression model2 = FastLinearRegression(5.0);
	model2.fit(X1,y1);

	Matrix predictions2 = model2.predict(X);

	double mse = (predictions - predictions2).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}
