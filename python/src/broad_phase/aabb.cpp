#include "../common.hpp"

#include <ipc/broad_phase/aabb.hpp>

namespace py = pybind11;
using namespace ipc;

void define_aabb(py::module_& m)
{
    py::class_<AABB>(m, "AABB")
        .def(py::init(), "")
        .def(
            py::init<const ArrayMax3d&, const ArrayMax3d&>(), "",
            py::arg("min"), py::arg("max"))
        .def(
            py::init<const AABB&, const AABB&>(), "", py::arg("aabb1"),
            py::arg("aabb2"))
        .def(
            py::init<const AABB&, const AABB&, const AABB&>(), "",
            py::arg("aabb1"), py::arg("aabb2"), py::arg("aabb3"))
        .def_static(
            "from_point",
            py::overload_cast<const VectorMax3d&, double>(&AABB::from_point),
            R"ipc_Qu8mg5v7(
            Compute a AABB for a static point.

            Parameters:
                p: The point's position.
                inflation_radius: Radius of a sphere around the point which the AABB encloses.

            Returns:
                The constructed AABB.
            )ipc_Qu8mg5v7",
            py::arg("p"), py::arg("inflation_radius") = 0)
        .def_static(
            "from_point",
            py::overload_cast<const VectorMax3d&, const VectorMax3d&, double>(
                &AABB::from_point),
            R"ipc_Qu8mg5v7(
            Compute a AABB for a moving point (i.e. temporal edge).

            Parameters:
                p_t0: The point's position at time t=0.
                p_t1: The point's position at time t=1.
                inflation_radius: Radius of a capsule around the temporal edge which the AABB encloses.

            Returns:
                The constructed AABB.
            )ipc_Qu8mg5v7",
            py::arg("p_t0"), py::arg("p_t1"), py::arg("inflation_radius") = 0)
        .def(
            "intersects", &AABB::intersects,
            R"ipc_Qu8mg5v7(
            Check if another AABB intersects with this one.

            Parameters:
                other: The other AABB.

            Returns:
                If the two AABBs intersect.
            )ipc_Qu8mg5v7",
            py::arg("other"))
        .def_readwrite("min", &AABB::min, "Minimum corner of the AABB.")
        .def_readwrite("max", &AABB::max, "Maximum corner of the AABB.")
        .def_readwrite(
            "vertex_ids", &AABB::vertex_ids,
            "Vertex IDs attached to the AABB.");

    m.def(
        "build_vertex_boxes",
        [](const Eigen::MatrixXd& V, double inflation_radius = 0) {
            std::vector<AABB> vertex_boxes;
            build_vertex_boxes(V, vertex_boxes, inflation_radius);
            return vertex_boxes;
        },
        "Build one AABB per vertex position (row of V).", py::arg("V"),
        py::arg("inflation_radius") = 0);

    m.def(
        "build_vertex_boxes",
        [](const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1,
           double inflation_radius = 0) {
            std::vector<AABB> vertex_boxes;
            build_vertex_boxes(V0, V1, vertex_boxes, inflation_radius);
            return vertex_boxes;
        },
        "", py::arg("V0"), py::arg("V1"), py::arg("inflation_radius") = 0);

    m.def(
        "build_edge_boxes",
        [](const std::vector<AABB>& vertex_boxes, const Eigen::MatrixXi& E) {
            std::vector<AABB> edge_boxes;
            build_edge_boxes(vertex_boxes, E, edge_boxes);
            return edge_boxes;
        },
        "", py::arg("vertex_boxes"), py::arg("E"));

    m.def(
        "build_face_boxes",
        [](const std::vector<AABB>& vertex_boxes, const Eigen::MatrixXi& F) {
            std::vector<AABB> face_boxes;
            build_face_boxes(vertex_boxes, F, face_boxes);
            return face_boxes;
        },
        "", py::arg("vertex_boxes"), py::arg("F"));
}
