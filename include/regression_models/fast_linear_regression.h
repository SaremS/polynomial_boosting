#ifndef FAST_LINEAR_REGRESSION_H
#define FAST_LINEAR_REGRESSION_H

#include "linalg.h"
#include "regression_model.h"

//fast updates for linear regression (currently only supports 1 feature)
class FastLinearRegression: public RegressionModel {
private:
	Matrix coefficients;

	Matrix XX;
	Matrix Xy;
	Matrix yy;
	
	double ols_sse;

	double n_obs;

	double lambda_regularization;
	bool is_trained;
public:
	FastLinearRegression(double lambda_regularization = 0.0):
		n_obs(0.0),
		lambda_regularization(lambda_regularization),
		is_trained(false) {};

	void fit(const Matrix &X, const Matrix &y);

	//update coefficients by adding more data
	void update_coefficients_add(const Matrix &X, const Matrix &y);
	//update coefficients by dropping data
	void update_coefficients_drop(const Matrix &X, const Matrix &y);
	Matrix predict(const Matrix &X) const;
	Matrix get_coefficients() const;

	double get_ols_sse() const; //sum of squared errors (without weights)
};

#endif
