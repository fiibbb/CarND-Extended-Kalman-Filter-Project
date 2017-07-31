#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // predict the state
  x_ = F_ * x_;
  P_ = F_ * P_ * (F_.transpose()) + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // update the state by using Kalman Filter equations
  VectorXd y = z - H_ * x_; // error calculation
  Calculate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // update the state by using Extended Kalman Filter equations
  double rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  double theta = atan(x_(1) / x_(0));
  double rho_dot = (x_(0)*x_(2) + x_(1)*x_(3)) / rho;
  VectorXd h = VectorXd(3);
  h << rho, theta, rho_dot;
  VectorXd y = z - h;
  Calculate(y);
}

void KalmanFilter::Calculate(const Eigen::VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd K =  P_ * Ht * ((H_ * P_ * Ht + R_).inverse());
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}
