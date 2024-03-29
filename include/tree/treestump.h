#ifndef TREESTUMP_H
#define TREESTUMP_H

#include "loss_functions/loss_function.h"
#include "loss_functions/quadratic_loss.h"
#include "regression_models/regression_model.h"
#include "linalg.h"

class TreeStump {
private:
	LossFunction* loss_function;
	
	int min_obs_per_leaf;

	int polynomial_level;

	int split_feature;
	double split_value;

	double lambda_regularization;

	double loss_at_head;
	double weighted_node_loss;

	RegressionModel* left_model;
	RegressionModel* right_model;

	std::function<double(double, double)> get_loss_lambda() const;
	std::function<double(double, double, double)> get_weighted_loss_lambda() const;	

	Matrix get_polynomial_features(const Matrix &X) const;

public:
	TreeStump(
			int min_obs_per_leaf,
			double lambda_regularization=0.0,
			int polynomial_level=1): 
		loss_function(new QuadraticLoss()),
		min_obs_per_leaf(min_obs_per_leaf),
		polynomial_level(polynomial_level),
		lambda_regularization(lambda_regularization),
		left_model(nullptr),
		right_model(nullptr){};
	TreeStump(
			LossFunction* loss_function,
			int min_obs_per_leaf,
			double lambda_regularization=0.0,
			int polynomial_level=1): 
		loss_function(loss_function),
		min_obs_per_leaf(min_obs_per_leaf),
		polynomial_level(polynomial_level),
		lambda_regularization(lambda_regularization),
		left_model(nullptr),
		right_model(nullptr){};
	void fit(const Matrix &X, const Matrix &y);
	void fit_with_weights(const Matrix &X, const Matrix &y, const Matrix &weights);
	void fit_fast_with_weights(const Matrix &X, const Matrix &y, const Matrix &weights);
	Matrix predict(const Matrix &X) const;
	Matrix predict_fast(const Matrix &X) const;
	int get_split_feature() const;
	double get_split_value() const;
	double get_feature_importance() const;
	double get_loss_at_head() const;
	double get_weighted_node_loss() const;

	Matrix get_left_model_coefficients() const;
	Matrix get_right_model_coefficients() const;
	
	void fit_eigen(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y) {
		Matrix X_matrix(X);
		Matrix y_matrix(y);
		this->fit(X_matrix, y_matrix);
	}

	Eigen::MatrixXd predict_eigen(const Eigen::MatrixXd &X) const {
		Matrix X_matrix(X);
		Matrix y_matrix = this->predict(X_matrix);
		return y_matrix.get_as_eigen();
	}
	// Class destructor, deletes all pointers
	~TreeStump() {

		//Loss function point already deletd in gradient_boosting.cpp
		//as long as we don't use TreeStump directly, but only through
		//GradientBoosting, we don't need to delete it here (it even
		//seems to cause a segfault if we do)
		/*if (loss_function != nullptr) {
			delete loss_function;
			loss_function = nullptr;
		}*/
		delete left_model;
		left_model = nullptr;
		delete right_model;
		right_model = nullptr;
	}
};

#endif
