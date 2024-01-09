#include <vector>

#include "regression_models/fast_linear_regression.h"
#include "linalg.h"

void FastLinearRegression::fit(const Matrix &X, const Matrix &y) {
	//initialize sums
	sum_xy = 0.0;
	sum_x = 0.0;
	sum_y = 0.0;
	sum_x_sq = 0.0;
	sum_sq_x = 0.0;
	sum_y_sq = 0.0;

	//calculate sums
	for (size_t i=0; i<X.get_n_rows(); i++) {
		sum_xy += X.get_element_at(i,0) * y.get_element_at(i,0);
		sum_x += X.get_element_at(i,0);
		sum_y += y.get_element_at(i,0);
		sum_x_sq += X.get_element_at(i,0) * X.get_element_at(i,0);
		sum_y_sq += y.get_element_at(i,0) * y.get_element_at(i,0);
	}
	
	sum_sq_x = sum_x * sum_x;

	double N = X.get_n_rows();
	this->n_obs = N;

	//calculate coefficients
	double denom = (sum_x_sq) - sum_sq_x / N + this->lambda_regularization;

	if (denom*denom > 1e-9) {
		beta1 = (sum_xy - sum_x * sum_y / N) / denom;
	} else {
		beta1 = 0.0;
	}

	beta0 = (sum_y - beta1 * sum_x) / N;

	is_trained = true;
}

void FastLinearRegression::update_coefficients_add(const Matrix &X, const Matrix &y) {
	//update sums
	for (size_t i=0; i<X.get_n_rows(); i++) {
		sum_xy += X.get_element_at(i,0) * y.get_element_at(i,0);
		sum_x += X.get_element_at(i,0);
		sum_y += y.get_element_at(i,0);
		sum_x_sq += X.get_element_at(i,0) * X.get_element_at(i,0);
		sum_y_sq += y.get_element_at(i,0) * y.get_element_at(i,0);
	}

	sum_sq_x = sum_x * sum_x;
	
	this->n_obs += X.get_n_rows();
	double N = this->n_obs;

	//calculate coefficients
	double denom = (sum_x_sq) - sum_sq_x / N + this->lambda_regularization;

	if (denom*denom > 1e-6) {
		beta1 = (sum_xy - sum_x * sum_y / N) / denom;
	} else {
		beta1 = 0.0;
	}
	beta0 = (sum_y - beta1 * sum_x) / N;
}

void FastLinearRegression::update_coefficients_drop(const Matrix &X, const Matrix &y) {
	//update sums
	for (size_t i=0; i<X.get_n_rows(); i++) {
		sum_xy -= X.get_element_at(i,0) * y.get_element_at(i,0);
		sum_x -= X.get_element_at(i,0);
		sum_y -= y.get_element_at(i,0);
		sum_x_sq -= X.get_element_at(i,0) * X.get_element_at(i,0); 
		sum_y_sq -= y.get_element_at(i,0) * y.get_element_at(i,0);
	}

	sum_sq_x = sum_x * sum_x;
	
	this->n_obs -= X.get_n_rows();
	double N = this->n_obs;

	//calculate coefficients
	double denom = (sum_x_sq) - sum_sq_x / N + this->lambda_regularization;
	
	if (denom*denom > 1e-6) {
		beta1 = (sum_xy - sum_x * sum_y / N) / denom;
	} else {
		beta1 = 0.0;
	}
	beta0 = (sum_y - beta1 * sum_x) / N;
}

Matrix FastLinearRegression::predict(const Matrix &X) const {
	if (!is_trained) {
		throw std::runtime_error("Model not trained");
	}
	
	return this->beta0 + this->beta1 * X;
}

Matrix FastLinearRegression::get_coefficients() const {
	if (!is_trained) {
		throw std::runtime_error("Model not trained");
	}

	std::vector<double> coefficients = {beta0, beta1};
	Matrix result = Matrix(coefficients);

	return result;
}

double FastLinearRegression::get_ols_sse() const {
	if (!is_trained) {
		throw std::runtime_error("Model not trained");
	}

	double sum_xy = this->sum_xy;	
	double sum_x = this->sum_x;
	double sum_y = this->sum_y;
	double sum_x_sq = this->sum_x_sq;
	double sum_y_sq = this->sum_y_sq;

	double beta0 = this->beta0;
	double beta1 = this->beta1;
	double n = this->n_obs;

	double sse = sum_y_sq - 2*beta1*sum_xy - 2*beta0*sum_y + n*beta0*beta0 + 2*beta0*beta1*sum_x + beta1*beta1*sum_x_sq;

	return sse;
}
