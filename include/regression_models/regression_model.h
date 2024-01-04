#ifndef REGRESSION_MODEL_H
#define REGRESSION_MODEL_H

#include "linalg.h"

class RegressionModel {
public:
	virtual ~RegressionModel() {};
	virtual void fit(const Matrix &X, const Matrix &y) = 0;
	virtual Matrix predict(const Matrix &X) const = 0;
	virtual Matrix get_coefficients() const = 0;
};

#endif
