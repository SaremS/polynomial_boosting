#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

class LossFunction {

public:
	virtual ~LossFunction() {};
	virtual double loss(double prediction, double actual) = 0;
	virtual double first_derivative(double prediction, double actual) = 0;
	virtual double second_derivative(double prediction, double actual) = 0;
};

#endif
