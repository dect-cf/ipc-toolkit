#pragma once

#include <ipc/utils/eigen_ext.hpp>

#include <Eigen/Geometry>

namespace ipc {

/// @brief Compute the distance between a two infinite lines in 3D.
/// @note The distance is actually squared distance.
/// @warning If the lines are parallel this function returns a distance of zero.
/// @param ea0 The first vertex of the edge defining the first line.
/// @param ea1 The second vertex of the edge defining the first line.
/// @param ea0 The first vertex of the edge defining the second line.
/// @param ea1 The second vertex of the edge defining the second line.
/// @return The distance between the two lines.
template <
    typename DerivedEA0,
    typename DerivedEA1,
    typename DerivedEB0,
    typename DerivedEB1>
auto line_line_distance(
    const Eigen::MatrixBase<DerivedEA0>& ea0,
    const Eigen::MatrixBase<DerivedEA1>& ea1,
    const Eigen::MatrixBase<DerivedEB0>& eb0,
    const Eigen::MatrixBase<DerivedEB1>& eb1)
{
    assert(ea0.size() == 3);
    assert(ea1.size() == 3);
    assert(eb0.size() == 3);
    assert(eb1.size() == 3);

    const auto normal = cross(ea1 - ea0, eb1 - eb0);
    const auto line_to_line = (eb0 - ea0).dot(normal);
    return line_to_line * line_to_line / normal.squaredNorm();
}

/// @brief Compute the distance between a two infinite lines in 3D.
/// @note The distance is actually squared distance.
/// @warning If the lines are parallel this function returns a distance of zero.
/// @param ea0 The first vertex of the edge defining the first line.
/// @param ea1 The second vertex of the edge defining the first line.
/// @param na The outward normal associated with the edge defining the first line.
/// @param ea0 The first vertex of the edge defining the second line.
/// @param ea1 The second vertex of the edge defining the second line.
/// @return The signed squared distance between the two lines.
template <
    typename DerivedEA0,
    typename DerivedNA,
    typename DerivedEA1,
    typename DerivedEB0,
   typename DerivedEB1>
auto line_line_distance(
    const Eigen::MatrixBase<DerivedEA0>& ea0,
    const Eigen::MatrixBase<DerivedEA1>& ea1,
    const Eigen::MatrixBase<DerivedNA>& na,
    const Eigen::MatrixBase<DerivedEB0>& eb0,
    const Eigen::MatrixBase<DerivedEB1>& eb1)
{
    assert(ea0.size() == 3);
    assert(ea1.size() == 3);
    assert(na.size() == 3);
    assert(eb0.size() == 3);
    assert(eb1.size() == 3);

    /// These expressions can be derived by defining the lines
    /// la(ta) = (1 - ta) * ea0 + ta * ea1
    /// lb(tb) = (1 - tb) * eb0 + tb * eb1
    /// and requiring that
    /// (la(ta) - lb(tb)).dot(ea) = 0
    /// (la(ta) - lb(tb)).dot(eb) = 0
    /// where ea = ea1 - ea0
    /// and eb = eb1 - eb0.
    /// We also define d=eb0 - ea0.
    /// This leads to the linear system
    /// [ea*ea  ea*eb][ta] = [ea*d]
    /// [ea*eb  eb*eb][tb] = [eb*d]
    /// where * indicates the dot product between vectors.

    const auto ea = ea1 - ea0;
    const double ea2 = ea.squaredNorm();
    const auto eb = eb1 - eb0;
    const double eb2 = eb.squaredNorm();
    const auto d = eb0 - ea0;

    const double ea_d = ea.dot( d );
    const double eb_d = eb.dot( d );
    const double ea_eb = ea.dot( eb );
    const double det = ( ea2 * eb2 + ea_eb * ea_eb );
    const double inv_det = 1 / det;

    const double ta = inv_det * ( ea_d * eb2 + eb_d * ea_eb );
    const double tb = inv_det * ( ea_eb * ea_d - eb2 * eb_d );

    const auto pa = ( 1 - ta ) * ea0 + ta * ea1;
    const auto pb = ( 1 - tb ) * eb0 + tb * eb1;
    const auto vab = pb - pa;

    /// na is the outward normal associated with the first edge.
    /// vab represents a vector pointing from the nearest point on la to the nearest point on lb.
    /// If these vectors point in the same direction then eb is outside
    const int s = vab.dot( na ) < 0 ? -1 : 1;

    return s * vab.squaredNorm();
}

// Symbolically generated derivatives;
namespace autogen {
    void line_line_distance_gradient(
        double v01,
        double v02,
        double v03,
        double v11,
        double v12,
        double v13,
        double v21,
        double v22,
        double v23,
        double v31,
        double v32,
        double v33,
        double g[12]);

    void line_line_distance_hessian(
        double v01,
        double v02,
        double v03,
        double v11,
        double v12,
        double v13,
        double v21,
        double v22,
        double v23,
        double v31,
        double v32,
        double v33,
        double H[144]);
} // namespace autogen

/// @brief Compute the gradient of the distance between a two lines in 3D.
/// @note The distance is actually squared distance.
/// @warning If the lines are parallel this function returns a distance of zero.
/// @param[in] ea0 The first vertex of the edge defining the first line.
/// @param[in] ea1 The second vertex of the edge defining the first line.
/// @param[in] ea0 The first vertex of the edge defining the second line.
/// @param[in] ea1 The second vertex of the edge defining the second line.
/// @param[out] grad The gradient of the distance wrt ea0, ea1, eb0, and eb1.
template <
    typename DerivedEA0,
    typename DerivedEA1,
    typename DerivedEB0,
    typename DerivedEB1,
    typename DerivedGrad>
void line_line_distance_gradient(
    const Eigen::MatrixBase<DerivedEA0>& ea0,
    const Eigen::MatrixBase<DerivedEA1>& ea1,
    const Eigen::MatrixBase<DerivedEB0>& eb0,
    const Eigen::MatrixBase<DerivedEB1>& eb1,
    Eigen::PlainObjectBase<DerivedGrad>& grad)
{
    assert(ea0.size() == 3);
    assert(ea1.size() == 3);
    assert(eb0.size() == 3);
    assert(eb1.size() == 3);

    grad.resize(ea0.size() + ea1.size() + eb0.size() + eb1.size());
    autogen::line_line_distance_gradient(
        ea0[0], ea0[1], ea0[2], ea1[0], ea1[1], ea1[2], eb0[0], eb0[1], eb0[2],
        eb1[0], eb1[1], eb1[2], grad.data());
}

/// @brief Compute the hessian of the distance between a two lines in 3D.
/// @note The distance is actually squared distance.
/// @warning If the lines are parallel this function returns a distance of zero.
/// @param[in] ea0 The first vertex of the edge defining the first line.
/// @param[in] ea1 The second vertex of the edge defining the first line.
/// @param[in] ea0 The first vertex of the edge defining the second line.
/// @param[in] ea1 The second vertex of the edge defining the second line.
/// @param[out] hess The hessian of the distance wrt ea0, ea1, eb0, and eb1.
template <
    typename DerivedEA0,
    typename DerivedEA1,
    typename DerivedEB0,
    typename DerivedEB1,
    typename DerivedHess>
void line_line_distance_hessian(
    const Eigen::MatrixBase<DerivedEA0>& ea0,
    const Eigen::MatrixBase<DerivedEA1>& ea1,
    const Eigen::MatrixBase<DerivedEB0>& eb0,
    const Eigen::MatrixBase<DerivedEB1>& eb1,
    Eigen::PlainObjectBase<DerivedHess>& hess)
{
    assert(ea0.size() == 3);
    assert(ea1.size() == 3);
    assert(eb0.size() == 3);
    assert(eb1.size() == 3);

    hess.resize(
        ea0.size() + ea1.size() + eb0.size() + eb1.size(),
        ea0.size() + ea1.size() + eb0.size() + eb1.size());
    autogen::line_line_distance_hessian(
        ea0[0], ea0[1], ea0[2], ea1[0], ea1[1], ea1[2], eb0[0], eb0[1], eb0[2],
        eb1[0], eb1[1], eb1[2], hess.data());
}

} // namespace ipc
