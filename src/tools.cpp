#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Calculate the RMSE here.

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // Check the validity of the following inputs:
  // The estimation vector size should not be zero
  if(estimations.size() == 0){
    cout << "Input is empty" << endl;
    return rmse;
  }

  // The estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size()){
    cout << "Invalid estimation or ground_truth. Data should have the same size" << endl;
    return rmse;
  }

  for(int i=0; i < estimations.size(); i++){
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x) {

  // Calculate a Jacobian here.

  MatrixXd Hj(3, 4);
  Hj << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

  double px = x(0);
  double py = x(1);
  double vx = x(2);
  double vy = x(3);

  double c1 = px * px + py * py;
  double c2 = sqrt(c1);
  double c3 = c1 * c2;

  if (fabs(c1) < 0.0001) {
    std::cout << "CalculateJacobian: Divide by zero" << std::endl;
    return Hj;
  }

  Hj << (px/c2),               (py/c2),               0,     0,
        -(py/c1),              (px/c1),               0,     0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
