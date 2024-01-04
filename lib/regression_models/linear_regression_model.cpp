#include <vector>

#include "regression_models/linear_regression_model.h"
#include "linalg.h"

void LinearRegressionModel::fit(const Matrix &X, const Matrix &y) {

	std::vector<std::vector<double>> diag_content = {
		{0.0,0.0},
		{0.0,this->lambda_regularization}
	};
	
	Matrix regularization_matrix = Matrix(diag_content);

	this->coefficients = (X.dot(X) + regularization_matrix).inverse().matrix_multiply(X.dot(y));
	this->is_trained = true;
}

Matrix LinearRegressionModel::predict(const Matrix &X) const {
	return X.matrix_multiply(this->coefficients);
}

Matrix LinearRegressionModel::get_coefficients() const {
	return this->coefficients;
}
