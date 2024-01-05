#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "gradient_boosting/gradient_boosting.h"
#include "loss_functions/loss_function.h"
#include "loss_functions/quadratic_loss.h"
#include "tree/treestump.h"

namespace py = pybind11;

PYBIND11_MODULE(polynomial_boosting,m) {
	py::class_<GradientBoosting>(m, "PolynomialBoostingModel")
		.def(py::init<double, double,int, int>())
		.def("fit", &GradientBoosting::fit_eigen)
		.def("predict", &GradientBoosting::predict_eigen)
		.def("get_feature_importances", &GradientBoosting::get_feature_importances)
		.def("get_losses_at_head", &GradientBoosting::get_losses_at_head)
		.def("get_weighted_node_losses", &GradientBoosting::get_weighted_node_losses)
		.def("get_n_trees", &GradientBoosting::get_n_trees);
}

void GradientBoosting::fit(const Matrix &X, const Matrix &y) {
	// Initialize the first tree
	this->initial_prediction = this->loss_function->minimizer_matrix(y).get_row(0);

	auto loss_lambda = [this](double prediction, double actual) {
		return this->loss_function->first_derivative(prediction, actual);
	};

	this->n_features = X.get_n_cols();
	
	// Fit the remaining trees
	for (int i=0; i<this->n_trees; i++) {
		// Calculate the pseudo-residuals
		Matrix y_pred = this->predict(X);
		Matrix pseudo_residuals = apply_binary(y_pred, 0, y, 0, loss_lambda);
		// Fit a tree to the pseudo-residuals
		TreeStump* new_tree = new TreeStump(this->loss_function, this->min_obs_per_leaf, this->lambda_regularization);
		new_tree->fit(X, pseudo_residuals);
		this->trees.push_back(new_tree);
	}
}

Matrix GradientBoosting::predict(const Matrix &X) const {
	Matrix y_pred = this->initial_prediction.replicate(X.get_n_rows(), 1);
	for (int i=0; i<this->trees.size(); i++) {
		y_pred = y_pred - this->learning_rate * this->trees[i]->predict(X);
	}
	return y_pred;
}

std::vector<double> GradientBoosting::get_feature_importances() const {
	std::vector<double> feature_importances(this->n_features, 0.0);
	double total_importance = 0.0;

	for (int i=0; i<this->trees.size(); i++) {
		int tree_feature = this->trees[i]->get_split_feature();
		double tree_importance = this->trees[i]->get_feature_importance();

		feature_importances[tree_feature] += tree_importance;
		total_importance += tree_importance;
	}
	
	for (int i=0; i<this->trees.size(); i++) {
		feature_importances[i] /= total_importance;
	}

	return feature_importances;
}

std::vector<double> GradientBoosting::get_losses_at_head() const {
	std::vector<double> losses_at_head(this->n_features, 0.0);

	for (int i=0; i<this->trees.size(); i++) {
		losses_at_head[i] = this->trees[i]->get_loss_at_head();
	}
	return losses_at_head;
}

std::vector<double> GradientBoosting::get_weighted_node_losses() const {
	std::vector<double> weighted_node_losses(this->trees.size(), 0.0);

	for (int i=0; i<this->trees.size(); i++) {
		weighted_node_losses[i] = this->trees[i]->get_weighted_node_loss();
	}
	return weighted_node_losses;
}

int GradientBoosting::get_n_trees() const {
	return this->trees.size();
}
