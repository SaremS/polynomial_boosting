#include <iostream>

#include "loss_functions/loss_function.h"
#include "regression_models/regression_model.h"
#include "regression_models/linear_regression_model.h"
#include "regression_models/fast_linear_regression.h"
#include "linalg.h"
#include "tree/treestump.h"


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

			X_left.prepend_ones();
			X_right.prepend_ones();

			LinearRegressionModel the_left_model = LinearRegressionModel(this->lambda_regularization);
			the_left_model.fit(X_left, y_left);
			LinearRegressionModel the_right_model = LinearRegressionModel(this->lambda_regularization);
			the_right_model.fit(X_right, y_right);
			
			Matrix pred_left = the_left_model.predict(X_left);
			Matrix pred_right = the_right_model.predict(X_right);

			double loss_left = apply_triary(pred_left, 0, y_left, 0, weights, 0, loss_lambda).sum();
			double loss_right = apply_triary(pred_right, 0, y_right, 0, weights, 0, loss_lambda).sum();

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

	std::vector<double> y_pred(n_samples);

	for (int sample = 0; sample < n_samples; sample++) {
		double x = X.get_element_at(sample, this->split_feature);
		std::vector<double> Xvec = {x};
		Matrix Xrow = Matrix(Xvec);
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

	double best_loss = INFINITY;
	FastLinearRegression* best_left_model = new FastLinearRegression(this->lambda_regularization);
	FastLinearRegression* best_right_model = new FastLinearRegression(this->lambda_regularization);

	auto loss_lambda = [this](double prediction, double actual, double weight) {
		return this->loss_function->weighted_loss(prediction, actual, weight);
	};

	Matrix loss_minimizer = this->loss_function->minimizer_matrix(y);
	this->loss_at_head = apply_triary(loss_minimizer, 0, y, 0, weights, 0, loss_lambda).sum() / n_samples;

	for (int feature = 0; feature < n_features; feature++) {

		Matrix Xcol = X.get_column(feature);

		Matrix ysort = y.get_rows_by_other_col_rank(-Xcol, 0, n_samples);
		Matrix Xsort = Xcol.get_rows_by_other_col_rank(-Xcol, 0, n_samples);
		Matrix wsort = weights.get_rows_by_other_col_rank(-Xcol, 0, n_samples);

		int first_split_index = min_obs_per_leaf - 1;
		double split_value = Xsort.get_element_at(first_split_index, 0);

		Matrix X_left = Xsort.get_row_range(0, first_split_index + 1);
		Matrix X_right = Xsort.get_row_range(first_split_index + 1, n_samples);
		
		Matrix y_left = ysort.get_row_range(0, first_split_index + 1);
		Matrix y_right = ysort.get_row_range(first_split_index + 1, n_samples);

		FastLinearRegression the_left_model = FastLinearRegression(this->lambda_regularization);
		FastLinearRegression the_right_model = FastLinearRegression(this->lambda_regularization);

		the_left_model.fit(X_left, y_left);
		the_right_model.fit(X_right, y_right);

		Matrix pred_left = the_left_model.predict(X_left);
		Matrix pred_right = the_right_model.predict(X_right);

		double loss_left = apply_triary(pred_left, 0, y_left, 0, wsort, 0, loss_lambda).sum();
		double loss_right = apply_triary(pred_right, 0, y_right, 0, wsort, 0, loss_lambda).sum();

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

		for (int sample = min_obs_per_leaf; sample < n_samples - min_obs_per_leaf; sample++) {
			// skip if split value did not change
			if (Xsort.get_element_at(sample, 0) == Xsort.get_element_at(sample-1, 0)) {
				continue;
			}
			split_value = Xsort.get_element_at(sample, 0);

			Matrix X_split = Xsort.get_row(sample);
			Matrix y_split = ysort.get_row(sample);

			Matrix X_left = Xsort.get_row_range(0, sample+1);
			Matrix X_right = Xsort.get_row_range(sample+1, n_samples);

			Matrix y_left = ysort.get_row_range(0, sample+1);
			Matrix y_right = ysort.get_row_range(sample+1, n_samples);
			
			the_left_model.update_coefficients_add(X_split, y_split);
			the_right_model.update_coefficients_drop(X_split, y_split);

			Matrix pred_left = the_left_model.predict(X_left);
			Matrix pred_right = the_right_model.predict(X_right);

			double loss_left = apply_triary(pred_left, 0, y_left, 0, wsort, 0, loss_lambda).sum();
			double loss_right = apply_triary(pred_right, 0, y_right, 0, wsort, 0, loss_lambda).sum();

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
