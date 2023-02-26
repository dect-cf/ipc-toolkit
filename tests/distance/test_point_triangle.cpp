#include <catch2/catch.hpp>

#include <finitediff.hpp>
#include <igl/PI.h>

#include <ipc/distance/point_triangle.hpp>
#include <ipc/utils/eigen_ext.hpp>

using namespace ipc;

inline Eigen::Vector2d
edge_normal(const Eigen::Vector2d& e0, const Eigen::Vector2d& e1)
{
    Eigen::Vector2d e = e1 - e0;
    Eigen::Vector2d normal(-e.y(), e.x());
    return normal.normalized();
}

int tri_normal_normal_sign(
    const Eigen::Vector3d& n,
    const Eigen::Vector3d& t0,
    const Eigen::Vector3d& t1,
    const Eigen::Vector3d& t2 )
{
    const auto normal = (t1 - t0).cross(t2 - t0);
    return n.dot( normal ) > 0 ? -1 : 1;
}

TEST_CASE("Point-triangle distance", "[distance][point-triangle]")
{
    double py = GENERATE(-10, -1, -1e-12, 0, 1e-12, 1, 10);
    Eigen::Vector3d p(0, py, 0);
    Eigen::Vector3d t0(-1, 0, 1);
    Eigen::Vector3d t1(1, 0, 1);
    Eigen::Vector3d t2(0, 0, -1);

    Eigen::Vector3d closest_point;
    SECTION("closest to triangle")
    {
        double pz = GENERATE(0, -1 + 1e-12, -1, 1, 1 - 1e-12);
        p.z() = pz;
        closest_point = p;
        closest_point.y() = 0;
    }
    SECTION("closest to t0")
    {
        double px = GENERATE(-1, -1 - 1e-12, -11);
        p.x() = px;
        p.z() = t0.z();
        closest_point = t0;
    }
    SECTION("closest to t1")
    {
        double px = GENERATE(1, 1 + 1e-12, 11);
        p.x() = px;
        p.z() = t1.z();
        closest_point = t1;
    }
    SECTION("closest to t2")
    {
        double pz = GENERATE(-1, -1 - 1e-12, -11);
        p.z() = pz;
        closest_point = t2;
    }
    SECTION("closest to t0t1")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        closest_point = (t1 - t0) * alpha + t0;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t1.x(), t1.z()));
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }
    SECTION("closest to t1t2")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        closest_point = (t2 - t1) * alpha + t1;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t1.x(), t1.z()), Eigen::Vector2d(t2.x(), t2.z()));
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }
    SECTION("closest to t2t0")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        closest_point = (t0 - t2) * alpha + t2;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t2.x(), t2.z()), Eigen::Vector2d(t0.x(), t0.z()));
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }

    double distance = point_triangle_distance(p, t0, t1, t2);
    CAPTURE(py, closest_point.x(), closest_point.y(), closest_point.z());
    CHECK(
        distance
        == Approx(point_point_distance(p, closest_point)).margin(1e-12));
}

template <typename DerivedP, typename DerivedN, typename DerivedOrigin, typename DerivedNormal>
auto print_point_plane_signed_distance(
    const Eigen::MatrixBase<DerivedP>& p,
    const Eigen::MatrixBase<DerivedN>& n,
    const Eigen::MatrixBase<DerivedOrigin>& origin,
    const Eigen::MatrixBase<DerivedNormal>& normal)
{
    auto point_to_plane = (p - origin).dot(normal);
    const double s = n.dot( normal ) > 0 ? -1 : 1;
    std::cout << "p: " << p.transpose() << std::endl;
    std::cout << "n: " << n.transpose() << std::endl;
    std::cout << "origin: " << origin.transpose() << std::endl;
    std::cout << "normal: " << normal.transpose() << std::endl;
    std::cout << "p - origin: " << (p - origin).transpose() << std::endl;
    std::cout << "point_to_plane: " << point_to_plane << std::endl;
    std::cout << "n.normal: " << n.dot( normal ) << std::endl;
    std::cout << "sign: " << s << std::endl;
}

template <typename DerivedP, typename DerivedN, typename DerivedT0, typename DerivedT1, typename DerivedT2>
auto print_point_plane_signed_distance(
    const Eigen::MatrixBase<DerivedP>& p,
    const Eigen::MatrixBase<DerivedN>& n,
    const Eigen::MatrixBase<DerivedT0>& t0,
    const Eigen::MatrixBase<DerivedT1>& t1,
    const Eigen::MatrixBase<DerivedT2>& t2 )
{
   std::cout << "t0: " << t0.transpose() << std::endl;
   std::cout << "t1: " << t1.transpose() << std::endl;
   std::cout << "t2: " << t2.transpose() << std::endl;
    auto normal = cross(t1 - t0, t2 - t0);
    print_point_plane_signed_distance(p, n, t0, normal);
}

TEST_CASE("Point-triangle signed distance", "[signed-distance][point-triangle]")
{
    double py = GENERATE(-10, -1, -1e-12, 0, 1e-12, 1, 10);
    Eigen::Vector3d p(0, py, 0);
    Eigen::Vector3d t0(-1, 0, 1);
    Eigen::Vector3d t1(1, 0, 1);
    Eigen::Vector3d t2(0, 0, -1);

    Eigen::Vector3d closest_point;
    SECTION("closest to triangle")
    {
        double pz = GENERATE(0, -1 + 1e-12, -1, 1, 1 - 1e-12);
        p.z() = pz;
        closest_point = p;
        closest_point.y() = 0;
    }
    SECTION("closest to t0")
    {
        double px = GENERATE(-1, -1 - 1e-12, -11);
        p.x() = px;
        p.z() = t0.z();
        closest_point = t0;
    }
    SECTION("closest to t1")
    {
        double px = GENERATE(1, 1 + 1e-12, 11);
        p.x() = px;
        p.z() = t1.z();
        closest_point = t1;
    }
    SECTION("closest to t2")
    {
        double pz = GENERATE(-1, -1 - 1e-12, -11);
        p.z() = pz;
        closest_point = t2;
    }
    SECTION("closest to t0t1")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        closest_point = (t1 - t0) * alpha + t0;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t1.x(), t1.z()));
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }
    SECTION("closest to t1t2")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        closest_point = (t2 - t1) * alpha + t1;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t1.x(), t1.z()), Eigen::Vector2d(t2.x(), t2.z()));
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }
    SECTION("closest to t2t0")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        closest_point = (t0 - t2) * alpha + t2;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t2.x(), t2.z()), Eigen::Vector2d(t0.x(), t0.z()));
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }

    CAPTURE(py, closest_point.x(), closest_point.y(), closest_point.z());
    Eigen::Vector3d n( 0, -1, 0 );
    double distance = point_triangle_signed_distance(p, n, t0, t1, t2);
    int dist_sign = distance < 0 ? -1 : 1;
    int compare_sign = tri_normal_normal_sign( n, t0, t1, t2);
    double compare = compare_sign * point_point_distance(p, closest_point);
    if(not (distance == Approx(compare).margin(1e-12)))
    {
       switch (point_triangle_distance_type(p, t0, t1, t2)) {
	  case PointTriangleDistanceType::P_T0:
	     std::cout << "type: P_T0" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_T1:
	     std::cout << "type: P_T1" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_T2:
	     std::cout << "type: P_T2" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_E0:
	     std::cout << "type: P_E0" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_E1:
	     std::cout << "type: P_E1" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_E2:
	     std::cout << "type: P_E2" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_T:
	     std::cout << "type: P_T" << std::endl;
	     break;
	  case PointTriangleDistanceType::AUTO:
	     break;
	  default:
	     throw std::invalid_argument(
		"Invalid distance type for point-triangle distance!");
       }
       point_triangle_distance_type(p, t0, t1, t2);
       print_point_plane_signed_distance(p, n, t0, t1, t2);
       std::cout << "distance: " << distance << std::endl;
       std::cout << "compare: " << compare << std::endl;
    }
    CHECK(distance == Approx(compare).margin(1e-12));
    CHECK(dist_sign == compare_sign);
    n[ 1 ] = 1;
    distance = point_triangle_signed_distance(p, n, t0, t1, t2);
    dist_sign = distance < 0 ? -1 : 1;
    compare_sign = tri_normal_normal_sign( n, t0, t1, t2);
    compare = compare_sign * point_point_distance(p, closest_point);
    if(not (distance == Approx(compare).margin(1e-12)))
    {
       switch (point_triangle_distance_type(p, t0, t1, t2)) {
	  case PointTriangleDistanceType::P_T0:
	     std::cout << "type: P_T0" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_T1:
	     std::cout << "type: P_T1" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_T2:
	     std::cout << "type: P_T2" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_E0:
	     std::cout << "type: P_E0" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_E1:
	     std::cout << "type: P_E1" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_E2:
	     std::cout << "type: P_E2" << std::endl;
	     break;
	  case PointTriangleDistanceType::P_T:
	     std::cout << "type: P_T" << std::endl;
	     break;
	  case PointTriangleDistanceType::AUTO:
	     break;
	  default:
	     throw std::invalid_argument(
		"Invalid distance type for point-triangle distance!");
       }
       print_point_plane_signed_distance(p, n, t0, t1, t2);
       std::cout << "distance: " << distance << std::endl;
       std::cout << "compare: " << compare << std::endl;
    }
    CHECK(distance == Approx(compare).margin(1e-12));
}

TEST_CASE(
    "Point-triangle distance gradient", "[distance][point-triangle][grad]")
{
    double py = GENERATE(-10, -1, -1e-12, 0, 1e-12, 1, 10);
    Eigen::Vector3d p(0, py, 0);
    Eigen::Vector3d t0(-1, 0, 1);
    Eigen::Vector3d t1(1, 0, 1);
    Eigen::Vector3d t2(0, 0, -1);

    SECTION("closest to triangle")
    {
        double pz = GENERATE(0, -1 + 1e-12, -1, 1, 1 - 1e-12);
        p.z() = pz;
    }
    SECTION("closest to t0")
    {
        double px = GENERATE(-1, -1 - 1e-12, -11);
        p.x() = px;
        p.z() = t0.z();
    }
    SECTION("closest to t1")
    {
        double px = GENERATE(1, 1 + 1e-12, 11);
        p.x() = px;
        p.z() = t1.z();
    }
    SECTION("closest to t2")
    {
        double pz = GENERATE(-1, -1 - 1e-12, -11);
        p.z() = pz;
    }
    SECTION("closest to t0t1")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        Eigen::Vector3d closest_point = (t1 - t0) * alpha + t0;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t1.x(), t1.z()));
        // double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }
    SECTION("closest to t1t2")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        Eigen::Vector3d closest_point = (t2 - t1) * alpha + t1;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t1.x(), t1.z()), Eigen::Vector2d(t2.x(), t2.z()));
        // double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }
    SECTION("closest to t2t0")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        Eigen::Vector3d closest_point = (t0 - t2) * alpha + t2;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t2.x(), t2.z()), Eigen::Vector2d(t0.x(), t0.z()));
        // double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }

    Eigen::VectorXd x(12);
    x.segment<3>(0) = p;
    x.segment<3>(3) = t0;
    x.segment<3>(6) = t1;
    x.segment<3>(9) = t2;
    // Compute the gradient using finite differences
    auto f = [&](const Eigen::VectorXd& x) {
        return point_triangle_distance(
            x.segment<3>(0), x.segment<3>(3), x.segment<3>(6), x.segment<3>(9));
    };

    Eigen::VectorXd grad;
    point_triangle_distance_gradient(p, t0, t1, t2, grad);

    Eigen::VectorXd fgrad;
    fd::finite_gradient(x, f, fgrad);

    CHECK(fd::compare_gradient(grad, fgrad));
}

TEST_CASE("Point-triangle distance hessian", "[distance][point-triangle][hess]")
{
    double py = GENERATE(-10, -1, -1e-12, 0, 1e-12, 1, 10);
    Eigen::Vector3d p(0, py, 0);
    Eigen::Vector3d t0(-1, 0, 1);
    Eigen::Vector3d t1(1, 0, 1);
    Eigen::Vector3d t2(0, 0, -1);

    SECTION("closest to triangle")
    {
        double pz = GENERATE(0, -1 + 1e-12, -1, 1, 1 - 1e-12);
        p.z() = pz;
    }
    SECTION("closest to t0")
    {
        double px = GENERATE(-1, -1 - 1e-12, -11);
        p.x() = px;
        p.z() = t0.z();
    }
    SECTION("closest to t1")
    {
        double px = GENERATE(1, 1 + 1e-12, 11);
        p.x() = px;
        p.z() = t1.z();
    }
    SECTION("closest to t2")
    {
        double pz = GENERATE(-1, -1 - 1e-12, -11);
        p.z() = pz;
    }
    SECTION("closest to t0t1")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        Eigen::Vector3d closest_point = (t1 - t0) * alpha + t0;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t0.x(), t0.z()), Eigen::Vector2d(t1.x(), t1.z()));
        // double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }
    SECTION("closest to t1t2")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        Eigen::Vector3d closest_point = (t2 - t1) * alpha + t1;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t1.x(), t1.z()), Eigen::Vector2d(t2.x(), t2.z()));
        // double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }
    SECTION("closest to t2t0")
    {
        double alpha = GENERATE(0.0, 1e-4, 0.5, 1.0 - 1e-4, 1.0);
        Eigen::Vector3d closest_point = (t0 - t2) * alpha + t2;
        Eigen::Vector2d perp = edge_normal(
            Eigen::Vector2d(t2.x(), t2.z()), Eigen::Vector2d(t0.x(), t0.z()));
        // double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11, 1000);
        double scale = GENERATE(0, 1e-12, 1e-4, 1, 2, 11);
        p.x() = closest_point.x() + scale * perp.x();
        p.z() = closest_point.z() + scale * perp.y();
    }

    PointTriangleDistanceType dtype =
        point_triangle_distance_type(p, t0, t1, t2);

    Eigen::VectorXd x(12);
    x.segment<3>(0) = p;
    x.segment<3>(3) = t0;
    x.segment<3>(6) = t1;
    x.segment<3>(9) = t2;
    // Compute the gradient using finite differences
    auto f = [&](const Eigen::VectorXd& x) {
        return point_triangle_distance(
            x.segment<3>(0), x.segment<3>(3), x.segment<3>(6), x.segment<3>(9),
            dtype);
    };

    Eigen::MatrixXd hess;
    point_triangle_distance_hessian(p, t0, t1, t2, hess);

    Eigen::MatrixXd fhess;
    fd::finite_hessian(x, f, fhess);

    CAPTURE(dtype);
    CHECK(fd::compare_hessian(hess, fhess, 1e-2));
}
