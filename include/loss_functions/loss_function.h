#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <linalg.h>

class LossFunction {

public:
	virtual ~LossFunction() {};
	virtual double loss(double prediction, double actual) = 0;
	virtual double first_derivative(double prediction, double actual) = 0;
	virtual double second_derivative(double prediction, double actual) = 0;

	// Returns the minimizing value(s) of the loss function
	// for a given target matrix as a matrix of the same size
	// (i.e. the minimizer(s) are repeated N times)
	//
	// Example:
	// In case of a quadratic loss function and a single target column, 
	// the minimizer is the mean of the target column. 
	// The mean is then repeated N times to form a matrix
	virtual Matrix minimizer_matrix(Matrix const &y) = 0;
};

#endif
