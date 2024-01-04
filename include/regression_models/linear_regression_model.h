#ifndef LINEAR_REGRESSION_MODEL_H
#define LINEAR_REGRESSION_MODEL_H

#include "linalg.h"
#include "regression_model.h"

class LinearRegressionModel: public RegressionModel {
private:
	double lambda_regularization;
	bool is_trained;
	Matrix coefficients;

public:
	LinearRegressionModel(double lambda_regularization = 0.0):
		lambda_regularization(lambda_regularization),
		is_trained(false),
		coefficients(Matrix()){};
	void fit(const Matrix &X, const Matrix &y);
	Matrix predict(const Matrix &X) const;
	Matrix get_coefficients() const;
};

#endif
