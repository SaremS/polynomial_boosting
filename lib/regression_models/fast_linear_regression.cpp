#include <vector>
#include <iostream>

#include "regression_models/fast_linear_regression.h"
#include "linalg.h"

void FastLinearRegression::fit(const Matrix &X, const Matrix &y) {
	//initialize sums
	Matrix X_transpose = X.transpose();
	this->XX = X_transpose.matrix_multiply(X); 
	this->Xy = X_transpose.matrix_multiply(y);
	this->yy = y.dot(y);

	double N = X.get_n_rows();
	this->n_obs = N;

	//calculate coefficients
	Matrix regularizer = make_regularization_matrix(X.get_n_cols(), this->lambda_regularization);

	this->coefficients = (this->XX + regularizer).inverse().matrix_multiply(this->Xy);

	Matrix sse = this->yy - 2 * this->Xy.dot(this->coefficients) + this->coefficients.dot(this->XX).matrix_multiply(this->coefficients);

	this->ols_sse = sse.get_element_at(0,0);
	
	is_trained = true;
}

void FastLinearRegression::update_coefficients_add(const Matrix &X, const Matrix &y) {
	//update sums
	Matrix X_transpose = X.transpose();

	this->XX = this->XX + (X_transpose.matrix_multiply(X)); 
	this->Xy = this->Xy + (X_transpose.matrix_multiply(y));
	this->yy = this->yy + (y.dot(y));

	this->n_obs += X.get_n_rows();

	//calculate coefficients
	Matrix regularizer = make_regularization_matrix(X.get_n_cols(), this->lambda_regularization);

	this->coefficients = (this->XX + regularizer).inverse().matrix_multiply(this->Xy);

	Matrix sse = this->yy - 2 * this->Xy.dot(this->coefficients) + this->coefficients.dot(this->XX).matrix_multiply(this->coefficients);

	this->ols_sse = sse.get_element_at(0,0);
}

void FastLinearRegression::update_coefficients_drop(const Matrix &X, const Matrix &y) {
	//update sums
	Matrix X_transpose = X.transpose();
	this->XX = this->XX - (X_transpose.matrix_multiply(X)); 
	this->Xy = this->Xy - (X_transpose.matrix_multiply(y));
	this->yy = this->yy - y.dot(y);
	
	this->n_obs -= X.get_n_rows();

	//calculate coefficients
	Matrix regularizer = make_regularization_matrix(X.get_n_cols(), this->lambda_regularization);

	this->coefficients = (this->XX + regularizer).inverse().matrix_multiply(this->Xy);

	Matrix sse = this->yy - 2 * this->Xy.dot(this->coefficients) + this->coefficients.dot(this->XX).matrix_multiply(this->coefficients);

	this->ols_sse = sse.get_element_at(0,0);
}

Matrix FastLinearRegression::predict(const Matrix &X) const {
	if (!is_trained) {
		throw std::runtime_error("Model not trained");
	}

	return X.matrix_multiply(this->coefficients);
}

Matrix FastLinearRegression::get_coefficients() const {
	if (!is_trained) {
		throw std::runtime_error("Model not trained");
	}

	return this->coefficients;
}

double FastLinearRegression::get_ols_sse() const {
	if (!is_trained) {
		throw std::runtime_error("Model not trained");
	}

	return this->ols_sse;
}
