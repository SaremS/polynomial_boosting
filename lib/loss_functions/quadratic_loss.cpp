#include <math.h>

#include "loss_functions/quadratic_loss.h"

double QuadraticLoss::loss(double prediction, double actual) {
	return 0.5 * pow(prediction - actual, 2.0);
}

double QuadraticLoss::first_derivative(double prediction, double actual) {
	return (prediction - actual);
}

double QuadraticLoss::second_derivative(double prediction, double actual) {
	return 1.0;
}
