#include <functional>
#include <iostream>
	
#include "loss_functions/loss_function.h"
#include "regression_models/regression_model.h"
#include "regression_models/linear_regression_model.h"
#include "regression_models/fast_linear_regression.h"
#include "linalg.h"
#include "tree/treestump.h"
#include "tree/data_iterator.h"



std::function<double(double, double)> TreeStump::get_loss_lambda() const {
	return [this](double prediction, double actual) {
		return this->loss_function->loss(prediction, actual);
	};
}

std::function<double(double, double, double)> TreeStump::get_weighted_loss_lambda() const {
	return [this](double prediction, double actual, double weight) {
		return this->loss_function->weighted_loss(prediction, actual, weight);
	};
}

void TreeStump::fit(const Matrix &X, const Matrix &y) {

	int n_features = X.get_n_cols();
	int n_samples = X.get_n_rows();	
	int min_obs_per_leaf = this->min_obs_per_leaf;

	double best_loss = INFINITY;
	LinearRegressionModel* best_left_model = new LinearRegressionModel(this->lambda_regularization);
	LinearRegressionModel* best_right_model = new LinearRegressionModel(this->lambda_regularization);

	auto loss_lambda = [this](double prediction, double actual) {
		return this->loss_function->loss(prediction, actual);
	};

	Matrix loss_minimizer = this->loss_function->minimizer_matrix(y);
	this->loss_at_head = apply_binary(loss_minimizer, 0, y, 0, loss_lambda).sum() / n_samples;

	for (int feature = 0; feature < n_features; feature++) {
		for (int sample = 0; sample < n_samples; sample++) {
			double split_value = X.get_element_at(sample, feature);
			
			auto [X_left, X_right] = X.split_col_at_value(feature, split_value);

			int n_left = X_left.get_n_rows();
			int n_right = X_right.get_n_rows();

			// If either of the splits is too small, skip this split
			if (n_left < min_obs_per_leaf || n_right < min_obs_per_leaf) {
				continue;
			}

			auto [y_left, y_right] = y.split_col_by_other_colval(0, X, feature, split_value);

			X_left.prepend_ones();
			X_right.prepend_ones();

			LinearRegressionModel the_left_model = LinearRegressionModel(this->lambda_regularization);
			the_left_model.fit(X_left, y_left);
			LinearRegressionModel the_right_model = LinearRegressionModel(this->lambda_regularization);
			the_right_model.fit(X_right, y_right);
			
			Matrix pred_left = the_left_model.predict(X_left);
			Matrix pred_right = the_right_model.predict(X_right);

			double loss_left = apply_binary(pred_left, 0, y_left, 0, loss_lambda).sum();
			double loss_right = apply_binary(pred_right, 0, y_right, 0, loss_lambda).sum();

			double loss = (loss_left + loss_right)/n_samples;

			if (loss < best_loss) {
				best_loss = loss;
				this->split_feature = feature;
				this->split_value = split_value;
				
				delete best_left_model;
				delete best_right_model;

				best_left_model = new LinearRegressionModel(the_left_model);
				best_right_model = new LinearRegressionModel(the_right_model);
			}
		}
	}

	this->left_model = best_left_model;
	this->right_model = best_right_model;
	this->weighted_node_loss = best_loss;

	best_left_model = nullptr;
	best_right_model = nullptr;
}

void TreeStump::fit_with_weights(const Matrix &X, const Matrix &y, const Matrix &weights) {

	int n_features = X.get_n_cols();
	int n_samples = X.get_n_rows();	
	int min_obs_per_leaf = this->min_obs_per_leaf;

	double best_loss = INFINITY;
	LinearRegressionModel* best_left_model = new LinearRegressionModel(this->lambda_regularization);
	LinearRegressionModel* best_right_model = new LinearRegressionModel(this->lambda_regularization);

	auto loss_lambda = [this](double prediction, double actual, double weight) {
		return this->loss_function->weighted_loss(prediction, actual, weight);
	};

	Matrix loss_minimizer = this->loss_function->minimizer_matrix(y);
	this->loss_at_head = apply_triary(loss_minimizer, 0, y, 0, weights, 0, loss_lambda).sum() / n_samples;

	for (int feature = 0; feature < n_features; feature++) {
		for (int sample = 0; sample < n_samples; sample++) {

			double split_value = X.get_element_at(sample, feature);
			
			auto [X_left, X_right] = X.split_col_at_value(feature, split_value);

			int n_left = X_left.get_n_rows();
			int n_right = X_right.get_n_rows();

			// If either of the splits is too small, skip this split
			if (n_left < min_obs_per_leaf || n_right < min_obs_per_leaf) {
				continue;
			}

			auto [y_left, y_right] = y.split_col_by_other_colval(0, X, feature, split_value);
			auto [w_left, w_right] = weights.split_col_by_other_colval(0, X, feature, split_value);

			X_left.prepend_ones();
			X_right.prepend_ones();

			LinearRegressionModel the_left_model = LinearRegressionModel(this->lambda_regularization);
			the_left_model.fit(X_left, y_left);
			LinearRegressionModel the_right_model = LinearRegressionModel(this->lambda_regularization);
			the_right_model.fit(X_right, y_right);
			
			Matrix pred_left = the_left_model.predict(X_left);
			Matrix pred_right = the_right_model.predict(X_right);

			double loss_left = apply_triary(pred_left, 0, y_left, 0, w_left, 0, loss_lambda).sum();
			double loss_right = apply_triary(pred_right, 0, y_right, 0, w_right, 0, loss_lambda).sum();

			double loss = (loss_left + loss_right)/n_samples;

			if (loss*loss < 1e-9) {
				loss = 0.0;
			}

			if (loss < best_loss) {
				best_loss = loss;
				this->split_feature = feature;
				this->split_value = split_value;
				
				delete best_left_model;
				delete best_right_model;

				best_left_model = new LinearRegressionModel(the_left_model);
				best_right_model = new LinearRegressionModel(the_right_model);
			}
		}
	}

	this->left_model = best_left_model;
	this->right_model = best_right_model;
	this->weighted_node_loss = best_loss;

	best_left_model = nullptr;
	best_right_model = nullptr;
}

Matrix TreeStump::predict(const Matrix &X) const {
	int n_samples = X.get_n_rows();

	std::vector<double> y_pred(n_samples);

	for (int sample = 0; sample < n_samples; sample++) {
		double x = X.get_element_at(sample, this->split_feature);
		std::vector<double> Xvec = {x};
		Matrix Xrow = Matrix(Xvec);
		Xrow.prepend_ones();
		if (x < this->split_value) {
			y_pred[sample] = this->left_model->predict(Xrow).get_element_at(0, 0);
		} else {
			y_pred[sample] = this->right_model->predict(Xrow).get_element_at(0, 0);
		}
	}

	Matrix y_pred_mat = Matrix(y_pred);

	return y_pred_mat;
}

Matrix TreeStump::predict_fast(const Matrix &X) const {
	int n_samples = X.get_n_rows();

	Matrix X_col = X.get_column(this->split_feature);
	Matrix X_poly = this->get_polynomial_features(X_col);

	std::vector<double> y_pred(n_samples);

	for (int sample = 0; sample < n_samples; sample++) {
		Matrix Xrow = X_poly.get_row(sample);
		double x = X_col.get_element_at(sample, 0);
		if (x < this->split_value) {
			y_pred[sample] = this->left_model->predict(Xrow).get_element_at(0, 0);
		} else {
			y_pred[sample] = this->right_model->predict(Xrow).get_element_at(0, 0);
		}
	}

	Matrix y_pred_mat = Matrix(y_pred);

	return y_pred_mat;
}

void TreeStump::fit_fast_with_weights(const Matrix &X, const Matrix &y, const Matrix &weights) {

	int n_features = X.get_n_cols();
	int n_samples = X.get_n_rows();	
	int min_obs_per_leaf = this->min_obs_per_leaf;

	FastLinearRegression* best_left_model = new FastLinearRegression(this->lambda_regularization);
	FastLinearRegression* best_right_model = new FastLinearRegression(this->lambda_regularization);

	auto loss_lambda = this->get_weighted_loss_lambda(); 
	double best_loss = INFINITY;

	Matrix loss_minimizer = this->loss_function->minimizer_matrix(y);
	this->loss_at_head = apply_triary(loss_minimizer, 0, y, 0, weights, 0, loss_lambda).sum() / n_samples;

	for (int feature = 0; feature < n_features; feature++) {
		Matrix X_col = X.get_column(feature);
		Matrix X_poly = this->get_polynomial_features(X_col);

		SortingDataIterator data_iterator = SortingDataIterator(
				X_poly,
				y,
				weights,
				X_col,
				this->min_obs_per_leaf
		);

		DataSplit* first_split = data_iterator.next();

		if (first_split == nullptr) {
			continue;
		}
		auto [X_left, y_left, w_left, X_right, y_right, w_right, X_split, y_split, split_value] = *first_split;
		delete first_split;

		FastLinearRegression the_left_model = FastLinearRegression(this->lambda_regularization);
		FastLinearRegression the_right_model = FastLinearRegression(this->lambda_regularization);

		the_left_model.fit(X_left, y_left);
		the_right_model.fit(X_right, y_right);


		double loss_left = 0.5*the_left_model.get_ols_sse();
		double loss_right = 0.5*the_right_model.get_ols_sse();

		double loss = (loss_left + loss_right)/n_samples;

		if (loss < best_loss) {
			best_loss = loss;
			this->split_feature = feature;
			this->split_value = split_value;
				
			delete best_left_model;
			delete best_right_model;

			best_left_model = new FastLinearRegression(the_left_model);
			best_right_model = new FastLinearRegression(the_right_model);
		}

		while (true) {
			DataSplit* split = data_iterator.next();
			if (split == nullptr) {
				break;
			}
			auto [X_left, y_left, w_left, X_right, y_right, w_right, X_split, y_split, split_value] = *split;
			delete split;
			
			the_left_model.update_coefficients_add(X_split, y_split);
			the_right_model.update_coefficients_drop(X_split, y_split);

			double loss_left = 0.5*the_left_model.get_ols_sse();
			double loss_right = 0.5*the_right_model.get_ols_sse();

			double loss = (loss_left + loss_right)/n_samples;

			if (loss < best_loss) {
				best_loss = loss;
				this->split_feature = feature;
				this->split_value = split_value;
				
				delete best_left_model;
				delete best_right_model;

				best_left_model = new FastLinearRegression(the_left_model);
				best_right_model = new FastLinearRegression(the_right_model);
			}
		}
	}

	this->left_model = best_left_model;
	this->right_model = best_right_model;
	this->weighted_node_loss = best_loss;

	best_left_model = nullptr;
	best_right_model = nullptr;
}

int TreeStump::get_split_feature() const {
	return this->split_feature;
}

double TreeStump::get_split_value() const {
	return this->split_value;
}

double TreeStump::get_feature_importance() const {
	return this->loss_at_head - this->weighted_node_loss;
}

double TreeStump::get_loss_at_head() const {
	return this->loss_at_head;
}

double TreeStump::get_weighted_node_loss() const {
	return this->weighted_node_loss;
}

Matrix TreeStump::get_left_model_coefficients() const {
	return this->left_model->get_coefficients();
}

Matrix TreeStump::get_right_model_coefficients() const {
	return this->right_model->get_coefficients();
}

Matrix TreeStump::get_polynomial_features(const Matrix &X) const {
	int n_samples = X.get_n_rows();

	std::vector<Matrix> X_poly_vec;

	for (int p=0; p<=this->polynomial_level; p++) {
		X_poly_vec.push_back(X.pow_elementwise(p));
	}

	Matrix X_poly = concat_matrices_colwise(X_poly_vec);

	return X_poly;
}
