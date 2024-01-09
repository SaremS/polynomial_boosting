#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "linalg.h"
#include "regression_models/linear_regression_model.h"
#include "regression_models/fast_linear_regression.h"
#include "tree/treestump.h"
#include "loss_functions/quadratic_loss.h"

TEST(testtree, simple_treestump) {

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

	LossFunction* loss_function = new QuadraticLoss();

	TreeStump model = TreeStump(loss_function, 1);
	model.fit(X,y);

	Matrix predictions = model.predict(X);
	
	double mse = (predictions - y).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}

TEST(testtree, simple_treestump_with_weights) {

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

	Matrix weights = Matrix(1.0, 10, 1);

	LossFunction* loss_function = new QuadraticLoss();

	TreeStump model = TreeStump(loss_function, 1);
	TreeStump weighted_model = TreeStump(loss_function, 1);


	model.fit(X,y);
	weighted_model.fit_with_weights(X,y,weights);

	Matrix predictions = model.predict(X);
	Matrix predictions_weighted = weighted_model.predict(X);
	
	double mse = (predictions - predictions_weighted).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);
}

TEST(testtree, weighted_simple_versus_fast) {

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

	Matrix weights = Matrix(1.0, 10, 1);

	LossFunction* loss_function = new QuadraticLoss();

	TreeStump weighted_model = TreeStump(loss_function, 2);
	TreeStump fast_model = TreeStump(loss_function, 2);

	weighted_model.fit_with_weights(X,y,weights);
	fast_model.fit_fast_with_weights(X,y,weights);

	Matrix predictions_weighted = weighted_model.predict(X);
	Matrix predictions_fast = fast_model.predict_fast(X);

	double mse = (predictions_fast - predictions_weighted).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);

	int split_feature_weighted = weighted_model.get_split_feature();
	int split_feature_fast = fast_model.get_split_feature();

	EXPECT_EQ(split_feature_weighted, split_feature_fast);

	double split_value_weighted = weighted_model.get_split_value();
	double split_value_fast = fast_model.get_split_value();

	EXPECT_EQ(split_value_weighted, split_value_fast);

	double head_loss_weighted = weighted_model.get_loss_at_head();
	double head_loss_fast = fast_model.get_loss_at_head();

	EXPECT_EQ(head_loss_weighted, head_loss_fast);

	double node_loss_weighted = weighted_model.get_weighted_node_loss();
	double node_loss_fast = fast_model.get_weighted_node_loss();

	EXPECT_EQ(node_loss_weighted, node_loss_fast);
	

}


TEST(testtree, simple_versus_fast_complex) {
	
	//random inputs
	std::vector<std::vector<double>> inputs = {
		{1.2, 2.4, 4.0, 6.1},
		{-1.1, 3.5, 5.34, -1.0},
		{0.345, 4.6, -0.234, 1.235},
		{1.123, 5.7, 1.0, 2.0},
		{-1.356, 6.8, -2.45, -4.12},
		{0.867, 7.9, 4.12, -3.1},
		{1.893, 8.0, 3.1, 0.0},
		{-1.34, 9.1, -3.12, 1.454},
		{0.467, 10.2, 5.345, 2.345},
		{1.66, 11.3, 1.0, 3.0}
	};
	Matrix X = Matrix(inputs);

	//random targets
	std::vector<std::vector<double>> targets = {
		{6.0},
		{1.5},
		{-3.5},
		{1.1},
		{-1.7},
		{0.4},
		{1.0},
		{4.0},
		{4.2},
		{1.0}
	};
	Matrix y = Matrix(targets);
	
	std::vector<double> random_weights = {
		1.0,
		1.0,
		1.0,
		1.0,
		1.0,
		3.234,
		3.234,
		3.234,
		3.234,
		3.234
	};
	Matrix weights = Matrix(random_weights);

	LossFunction* loss_function = new QuadraticLoss();

	TreeStump weighted_model = TreeStump(loss_function, 4);
	TreeStump fast_model = TreeStump(loss_function, 4);

	weighted_model.fit_with_weights(X,y,weights);
	fast_model.fit_fast_with_weights(X,y,weights);

	Matrix predictions_weighted = weighted_model.predict(X);
	Matrix predictions_fast = fast_model.predict_fast(X);

	double mse = (predictions_fast - predictions_weighted).pow_elementwise(2.0).sum() / 10.0;

	EXPECT_TRUE(mse < 1e-6);

	int split_feature_weighted = weighted_model.get_split_feature();
	int split_feature_fast = fast_model.get_split_feature();

	EXPECT_EQ(split_feature_weighted, split_feature_fast);

	double split_value_weighted = weighted_model.get_split_value();
	double split_value_fast = fast_model.get_split_value();

	EXPECT_EQ(split_value_weighted, split_value_fast);

	double head_loss_weighted = weighted_model.get_loss_at_head();
	double head_loss_fast = fast_model.get_loss_at_head();

	EXPECT_EQ(head_loss_weighted, head_loss_fast);

	double node_loss_weighted = weighted_model.get_weighted_node_loss();
	double node_loss_fast = fast_model.get_weighted_node_loss();

	double node_mse = (node_loss_fast - node_loss_weighted) * (node_loss_fast - node_loss_weighted);

	EXPECT_TRUE(node_mse < 1e-6);
	

}
