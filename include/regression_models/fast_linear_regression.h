#ifndef FAST_LINEAR_REGRESSION_H
#define FAST_LINEAR_REGRESSION_H

#include "linalg.h"
#include "regression_model.h"

//fast updates for linear regression (currently only supports 1 feature)
class FastLinearRegression: public RegressionModel {
private:
	double beta0;
	double beta1;

	double sum_xy;
	double sum_x;
	double sum_y;
	double sum_x_sq; //sum of x^2
	double sum_sq_x; //sum of x, squared
	double sum_y_sq; //sum of y^2
	
	double n_obs;

	double lambda_regularization;
	bool is_trained;
public:
	FastLinearRegression(double lambda_regularization = 0.0):
		lambda_regularization(lambda_regularization),
		sum_xy(0.0),
		sum_x(0.0),
		sum_y(0.0),
		sum_x_sq(0.0),
		sum_sq_x(0.0),
		sum_y_sq(0.0),
		n_obs(0.0),
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
