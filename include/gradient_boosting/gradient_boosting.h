#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

#include <vector>
#include <Eigen/Dense>

#include "loss_functions/loss_function.h"
#include "loss_functions/quadratic_loss.h"
#include "tree/treestump.h"
#include "seedable_rng.h"
#include "goss_sampler.h"

class GradientBoosting {
private:
	LossFunction* loss_function;
	Matrix initial_prediction;
	std::vector<TreeStump*> trees;
	double learning_rate;
	double lambda_regularization;

	int min_obs_per_leaf;
	int n_trees;
	int n_features;

	GossSampler goss_sampler;
	SeedableRNG rng;
public:
	GradientBoosting(
			double learning_rate = 0.1,
			double lambda_regularization = 0.0,
			int n_trees = 100,
			int min_obs_per_leaf = 5,
			double goss_alpha = 0.5,
			double goss_beta = 0.5,
			int seed = 0):
		loss_function(new QuadraticLoss()),
		learning_rate(learning_rate),
		lambda_regularization(lambda_regularization),
		n_trees(n_trees),
		min_obs_per_leaf(min_obs_per_leaf),
		goss_sampler(goss_alpha, goss_beta),
		rng(seed) {};
	GradientBoosting(
			LossFunction* loss_function,
			double learning_rate = 0.1,
			double lambda_regularization = 0.0,
			int n_trees = 100,
			int min_obs_per_leaf = 1,
			double goss_alpha = 0.5,
			double goss_beta = 0.5,
			int seed = 0): 
		loss_function(loss_function),
		learning_rate(learning_rate),
		lambda_regularization(lambda_regularization),
		n_trees(n_trees),
		min_obs_per_leaf(min_obs_per_leaf),
		goss_sampler(goss_alpha, goss_beta),
		rng(seed) {};
	void fit(const Matrix &X, const Matrix &y);
	Matrix predict(const Matrix &X) const;
	std::vector<double> get_feature_importances() const;
	std::vector<double> get_losses_at_head() const;
	std::vector<double> get_weighted_node_losses() const;

	int get_n_trees() const;
	
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
	~GradientBoosting() {
		delete loss_function;
		loss_function = nullptr;
		for (auto tree : trees) {
			delete tree;
			tree = nullptr;
		}
	}
};

#endif
