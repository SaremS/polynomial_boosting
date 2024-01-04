#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "gradient_boosting/gradient_boosting.h"
#include "loss_functions/loss_function.h"
#include "loss_functions/quadratic_loss.h"
#include "tree/treestump.h"

namespace py = pybind11;

PYBIND11_MODULE(polynomial_boosting,m) {
	py::class_<GradientBoosting>(m, "PolynomialBoostingModel")
		.def(py::init<double, double,int, int>())
		.def("fit", &GradientBoosting::fit_eigen)
		.def("predict", &GradientBoosting::predict_eigen);
}

void GradientBoosting::fit(const Matrix &X, const Matrix &y) {
	// Initialize the first tree
	TreeStump* first_tree = new TreeStump(this->loss_function, this->min_obs_per_leaf, this->lambda_regularization);
	first_tree->fit(X, y);
	this->trees.push_back(first_tree);

	auto loss_lambda = [this](double prediction, double actual) {
		return this->loss_function->first_derivative(prediction, actual);
	};
	
	// Fit the remaining trees
	for (int i=0; i<this->n_trees-1; i++) {
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
	Matrix y_pred = this->trees[0]->predict(X);
	for (int i=1; i<this->trees.size(); i++) {
		y_pred = y_pred - this->learning_rate * this->trees[i]->predict(X);
	}
	return y_pred;
}
