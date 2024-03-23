#ifndef user_base
#define user_base
#include<vector>
#include<Eigen/Dense>

class BaseModel {
public:
	virtual Eigen::VectorXd forward(const Eigen::VectorXd &) =0 {};
	virtual void backward(Eigen::VectorXd,double) = 0 {};
};

#endif // !user_matrix
