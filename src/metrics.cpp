#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "FEVV/Filters/CGAL/Surface_mesh/msdm2.h"
#include "FEVV/Filters/Generic/minmax_map.h"
#include "FEVV/Filters/Generic/color_mesh.h"
#include "FEVV/Filters/Generic/calculate_face_normals.hpp"

namespace py = pybind11;

typedef Eigen::Matrix<long long, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixX3i;
typedef Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixX3d;

void populate_mesh(
        FEVV::MeshSurface &mesh,
        FEVV::PMapsContainer &pmaps_bag,
        Eigen::Ref<MatrixX3d> verts,
        Eigen::Ref<MatrixX3i> faces) {
    bool obj_file = true;
    unsigned int duplicated_vertices_nbr;
    FEVV::Filters::MeshFromVectorReprParameters< FEVV::MeshSurface > mfvr_params;

    FEVV::Types::MVR< double,
                      double,
                      float,
                      float,
                      long long > mvr;
    FEVV::Geometry_traits< FEVV::MeshSurface > gt(mesh);

    mvr.points_coords.resize(verts.rows(), std::vector<double>(verts.cols(), 0));
    for(size_t i = 0; i < verts.rows(); ++i) {
        for(size_t j = 0; j < verts.cols(); ++j) {
            mvr.points_coords[i][j] = verts(i, j);
        }
    }
    mvr.faces_indices.resize(faces.rows(), std::vector<long long>(faces.cols(), 0));
    for(size_t i = 0; i < faces.rows(); ++i) {
        for(size_t j = 0; j < faces.cols(); ++j) {
            mvr.faces_indices[i][j] = faces(i, j);
        }
    }

    mesh_from_vector_representation(
        mesh,
        pmaps_bag,
        duplicated_vertices_nbr,
        mvr,
        mfvr_params.use_corner_texcoord(obj_file),
        gt);
}

double MSDM2_wrapper(
        Eigen::Ref<MatrixX3d> verts_a,
        Eigen::Ref<MatrixX3i> faces_a,
        Eigen::Ref<MatrixX3d> verts_b,
        Eigen::Ref<MatrixX3i> faces_b) {
    double msdm2;
    int nb_levels = 3;

    FEVV::MeshSurface m_original;
    FEVV::PMapsContainer pmaps_bag_original;

    FEVV::MeshSurface m_degraded;
    FEVV::PMapsContainer pmaps_bag_degraded;

    populate_mesh(m_original, pmaps_bag_original, verts_a, faces_a);
    populate_mesh(m_degraded, pmaps_bag_degraded, verts_b, faces_b);

    auto pm_degrad = get(boost::vertex_point, m_degraded);
    auto pm_original = get(boost::vertex_point, m_original);

    using FaceNormalMap =
        typename FEVV::PMap_traits< FEVV::face_normal_t,
                                    FEVV::MeshSurface >::pmap_type;
    FaceNormalMap fnm_degrad;
    FaceNormalMap fnm_original;

    fnm_degrad = make_property_map(FEVV::face_normal, m_degraded);
    // store property map in property maps bag
    put_property_map(
        FEVV::face_normal, m_degraded, pmaps_bag_degraded, fnm_degrad);
    FEVV::Filters::calculate_face_normals(m_degraded, pm_degrad, fnm_degrad);

    fnm_original = make_property_map(FEVV::face_normal, m_original);
    // store property map in property maps bag
    put_property_map(
        FEVV::face_normal, m_original, pmaps_bag_original, fnm_original);
    FEVV::Filters::calculate_face_normals(
        m_original, pm_original, fnm_original);

    typedef typename FEVV::Vertex_pmap< FEVV::MeshSurface, double >
        VertexMSDM2Map;
    VertexMSDM2Map msdm2_pmap;
    msdm2_pmap =
        FEVV::make_vertex_property_map< FEVV::MeshSurface, double >(m_degraded);
    pmaps_bag_degraded["v:msdm2"] = msdm2_pmap;

    FEVV::Filters::process_msdm2_multires(
        m_degraded,
        pm_degrad,
        fnm_degrad,
        pmaps_bag_degraded,
        m_original,
        pm_original,
        fnm_original,
        pmaps_bag_original,
        nb_levels,
        msdm2_pmap,
        msdm2);
    return msdm2;
}

PYBIND11_MODULE(py_mepp2, m) {
    m.doc() = R"pbdoc(
        MEPP2 bindings for python for MSDM2, SDCD
        -----------------------
        .. currentmodule:: py_mepp2
        .. autosummary::
           :toctree: _generate
           MSDM2
    )pbdoc";

    m.def("MSDM2", &MSDM2_wrapper, R"pbdoc(
        compute MSDM2 metric
        between two meshes.
    )pbdoc");
}
