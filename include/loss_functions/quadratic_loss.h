#ifndef QUADRATIC_LOSS_H
#define QUADRATIC_LOSS_H

#include "loss_function.h"

class QuadraticLoss: public LossFunction {
public:
	double loss(double prediction, double actual);
	double first_derivative(double prediction, double actual);
	double second_derivative(double prediction, double actual);
};

#endif
