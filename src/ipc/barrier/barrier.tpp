#pragma once
#include "barrier.hpp"

#include <limits>
#include <cmath>

namespace ipc {

template <typename T> T barrier(const T& d, const double dhat)
{
    if (d <= 0.0) {
        return T(std::numeric_limits<double>::infinity());
    }
    if (d >= dhat) {
        return T(0);
    }
    // b(d) = -(d-d̂)²ln(d / d̂)
    const T d_minus_dhat = (d - dhat);
    //return -d_minus_dhat * d_minus_dhat * log(d / dhat);
    return d_minus_dhat * d_minus_dhat;
}

double barrier_gradient(const double d, const double dhat)
{
    if (d <= 0.0 || d >= dhat) {
        return 0.0;
    }
    // b(d) = -(d - d̂)²ln(d / d̂)
    // b'(d) = -2(d - d̂)ln(d / d̂) - (d-d̂)²(1 / d)
    //       = (d - d̂) * (-2ln(d/d̂) - (d - d̂) / d)
    //       = (d̂ - d) * (2ln(d/d̂) - d̂/d + 1)
    // return (dhat - d) * (2 * log(d / dhat) - dhat / d + 1);
    return 2 * (dhat - d);
}

double barrier_hessian(const double d, const double dhat)
{
    if (d <= 0.0 || d >= dhat) {
        return 0.0;
    }
    const double dhat_d = dhat / d;
    //return (dhat_d + 2) * dhat_d - 2 * log(d / dhat) - 3;
    return -2;
}


} // namespace ipc
