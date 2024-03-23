#include "Linear.h"
#include "Eigen/Dense"



Eigen::VectorXd LinearLayer::forward(const Eigen::VectorXd & input) {
	Eigen::VectorXd output = weights * input;
	for (int i = 0; i < _input_size; ++i) {
		grad = input.transpose();
	}
	return output;
}

void LinearLayer::backward(Eigen::VectorXd v,double lr) {
	weights -= lr * v * grad;
}