#include <math.h>

#include "loss_functions/quadratic_loss.h"
#include "linalg.h"

double QuadraticLoss::loss(double prediction, double actual) {
	return 0.5 * pow(prediction - actual, 2.0);
}

double QuadraticLoss::first_derivative(double prediction, double actual) {
	return (prediction - actual);
}

double QuadraticLoss::second_derivative(double prediction, double actual) {
	return 1.0;
}

Matrix QuadraticLoss::minimizer_matrix(Matrix const &y) {
	double mean = y.sum() / y.get_n_rows();
	return Matrix(mean, y.get_n_rows(), y.get_n_cols());
}
