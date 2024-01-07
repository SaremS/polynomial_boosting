#include "linalg.h"
#include "gradient_boosting/goss_sampler.h"

GossResult GossSampler::sample(const Matrix &X, const Matrix &gradients, const int &seed) {
	int n_obs = X.get_n_rows();
	int gradient_cols = gradients.get_n_cols();

	//N highest squared gradients
	int alpha_n_obs = (int) (n_obs * this->alpha);
	
	//N sampling from remaining observations
	int sample_n_obs = n_obs - alpha_n_obs;
	int beta_n_obs = (int) (sample_n_obs * this->beta);

	//Get rows with highest squared gradients
	Matrix squared_gradients = gradients.pow_elementwise(2.0);

	Matrix X_alpha = X.get_rows_by_other_col_rank(squared_gradients, 0, alpha_n_obs);
	Matrix y_alpha = gradients.get_rows_by_other_col_rank(squared_gradients, 0, alpha_n_obs);

	//Return early, if no samples from remaining observations should be drawn
	if (beta_n_obs == 0) {
		return GossResult{X_alpha, y_alpha, Matrix(1.0, n_obs, gradient_cols)};
	}

	//Get samples from remaining observations
	Matrix neg_sq_grads = Matrix(0.0, n_obs, gradient_cols) - gradients;

	Matrix X_sample = X.get_rows_by_other_col_rank(neg_sq_grads, 0, sample_n_obs);
	Matrix y_sample = gradients.get_rows_by_other_col_rank(neg_sq_grads, 0, sample_n_obs);

	Matrix X_beta = X_sample.sample_rows(beta_n_obs, seed);
	Matrix y_beta = y_sample.sample_rows(beta_n_obs, seed);

	//Combine
	Matrix X_final = X_alpha.append_rows(X_beta);
	Matrix y_final = y_alpha.append_rows(y_beta);

	//Make loss weights
	Matrix loss_weights = this->make_loss_weights(alpha_n_obs, beta_n_obs);

	return GossResult{X_final, y_final, loss_weights};
}

Matrix GossSampler::make_loss_weights(const int &alpha_n_obs, const int &beta_n_obs) {
	std::vector<double> weights;

	for (int i=0; i<alpha_n_obs; i++) {
		weights.push_back(1.0);
	}

	double beta_weight = (1.0 - this->alpha) / this->beta; 

	for (int i=0; i<beta_n_obs; i++) {
		weights.push_back(beta_weight);
	}

	return Matrix(weights);
}
