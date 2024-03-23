#ifndef Linear
#define Linear

#include <Eigen/Dense>
#include "Base.h"

class LinearLayer : public BaseModel {
public:
	LinearLayer(int input_size, int output_size) : weights{ Eigen::MatrixXd::Random(output_size, input_size) }, _input_size{ input_size }
		, output_size{ output_size }, grad{ Eigen::MatrixXd{weights}} {}
	Eigen::VectorXd forward(const Eigen::VectorXd & );
	// v is the before element in chain rule.
	void backward(Eigen::VectorXd v,double lr = 0.0001);

private:
	Eigen::MatrixXd weights;
	Eigen::MatrixXd grad;
	int _input_size, output_size;
};
#endif