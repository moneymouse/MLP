#include <iostream>
//#include <iomanip>
//#include <vector>
//#include <array>
#include <string>
#include "Linear.h"
#include <Eigen/Dense>
#include <fstream>

class A {
public:
	virtual void p() { std::cout << "As p"; }
};

class B : public A {
public:
	void p() { std::cout << "Bs p"; }
};

double loss_grad(double y, double pred_y, int batch_size=1) {
	return 2 * (pred_y - y) / batch_size;
}

int main(){ 
	LinearLayer l1{2,1};
	Eigen::Vector2d train_input[3], test_input[1];
	double train_output[3];
	train_input[0] << 1, 2;
	train_input[1] << 1, 3;
	train_input[2] << 1, 5;
	train_output[0] = 5;
	train_output[1] = 6;
	train_output[2] = 8;
	test_input[0] << 1, 4;

	int epoch = 10000;
	for (int i = 0; i < epoch; ++i) {
		for (int x_i = 0; x_i < 3;++x_i) {
			const Eigen::VectorXd & x = train_input[x_i];
			double y_pred = l1.forward(x)(0);
			double chain = loss_grad(train_output[x_i], y_pred);
			Eigen::VectorXd temp{1};
			temp << chain;
			l1.backward(temp); 
		}
	}
	std::cout << l1.forward(test_input[0]);
	/*double chain{ 0.3 };
	Eigen::VectorXd temp{chain};
	*///std::cout << temp;
	LinearLayer* a = new LinearLayer[2]{ {1,3},{3,4} };
}
