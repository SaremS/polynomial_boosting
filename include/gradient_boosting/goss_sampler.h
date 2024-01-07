#ifndef GOSS_SAMPLER_H
#define GOSS_SAMPLER_H

#include "linalg.h"

struct GossResult {
	Matrix X;
	Matrix y;

	Matrix loss_weights;
};

//helper class for Gradient Boosting GOSS sampling
class GossSampler {
private:
	double alpha;
	double beta;

	Matrix make_loss_weights(const int &alpha_n_obs, const int &beta_n_obs);
public:
	GossSampler(double alpha, double beta): alpha(alpha), beta(beta) {};

 	//https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
	//Algorithm 2
	GossResult sample(const Matrix &X, const Matrix &gradients, const int &seed);
};


#endif
