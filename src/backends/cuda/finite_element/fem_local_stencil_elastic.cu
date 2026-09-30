#include <finite_element/fem_local_stencil_solver.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/local_stencil_elastic.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/codim_thickness.h>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

namespace uipc::backend::cuda
{
void FEMLocalStencilSolver::init_elastic()
{
    using Term = fem::StencilElasticTerm;
    vector<Vector4i> stencils;
    vector<Term>     terms;
    vector<Vector3>  rest(m_fem->x_bars().size());
    vector<Float>    thickness(rest.size());
    m_fem->x_bars().copy_to(rest.data());
    m_fem->thicknesses().copy_to(thickness.data());
    vector<Matrix3x3> inverses(m_fem->tets().size());
    vector<Float>     volumes(inverses.size());
    if(!inverses.empty())
    {
        m_fem->Dm3x3_invs().copy_to(inverses.data());
        m_fem->rest_volumes().copy_to(volumes.data());
    }
    m_fem->for_each(
        world().scene().geometries(),
        [&](const FiniteElementMethod::ForEachInfo& info, geometry::SimplicialComplex& sc)
        {
            const auto uid = info.geo_info().dim_uid.uid;
            if(!((uid == 10 && sc.dim() == 3)
                 || ((uid == 819 || uid == 11) && sc.dim() == 2)))
                throw SimEngineException("local_stencil supports SNK1 tetrahedra, BW/NH membranes and optional discrete shell bending.");
            const IndexT offset = static_cast<IndexT>(info.geo_info().vertex_offset);
            const auto mus = (sc.dim() == 3 ? sc.tetrahedra().find<Float>("mu") :
                                              sc.triangles().find<Float>("mu"))
                                 ->view();
            const auto lambdas =
                (sc.dim() == 3 ? sc.tetrahedra().find<Float>("lambda") :
                                 sc.triangles().find<Float>("lambda"))
                    ->view();
            auto rates = sc.triangles().find<Float>("strain_rate");
            for(SizeT i = 0; i < mus.size(); ++i)
            {
                Term t;
                t.type   = static_cast<IndexT>(uid);
                t.mu     = mus[i];
                t.lambda = lambdas[i];
                if(!std::isfinite(t.mu) || !std::isfinite(t.lambda) || t.mu <= 0
                   || t.lambda <= 0)
                    throw SimEngineException("local_stencil requires positive finite material parameters.");
                Vector4i ids = Vector4i::Constant(-1);
                if(sc.dim() == 3)
                {
                    ids = sc.tetrahedra().topo().view()[i].array() + offset;
                    const SizeT e = info.geo_info().primitive_offset + i;
                    t.inverse     = inverses[e];
                    t.weight      = volumes[e];
                }
                else
                {
                    ids.head<3>() = sc.triangles().topo().view()[i].array() + offset;
                    const Vector3 &a = rest[ids(0)], &b = rest[ids(1)],
                                  &c = rest[ids(2)];
                    t.rest_area2     = fem::stencil_triangle_measure(a, b, c);
                    if(!std::isfinite(t.rest_area2) || t.rest_area2 <= 0)
                        throw SimEngineException("local_stencil requires nondegenerate rest triangles.");
                    t.weight = 0.5 * t.rest_area2;
                    if(uid == 819)
                    {
                        t.shell_inverse =
                            sym::strainlimiting_baraff_witkin_shell_2d::Dm2x2(a, b, c)
                                .inverse();
                        t.strain_rate = rates ? rates->view()[i] : 100;
                        if(!std::isfinite(t.strain_rate) || t.strain_rate < 0)
                            throw SimEngineException("local_stencil requires finite nonnegative strain_rate.");
                    }
                    else
                    {
                        Vector9 X;
                        X << a, b, c;
                        Matrix2x2 B;
                        sym::neo_hookean_shell_2d::A(B, X);
                        t.shell_inverse = B.inverse();
                        t.weight *= 2
                                    * triangle_thickness(thickness[ids(0)],
                                                         thickness[ids(1)],
                                                         thickness[ids(2)]);
                    }
                }
                stencils.push_back(ids);
                terms.push_back(t);
            }
            bool bending = false;
            if(auto extras = sc.meta().find<VectorXu64>(builtin::extra_constitution_uids))
                for(auto extra : extras->view().front())
                {
                    if(extra != 17 || sc.dim() != 2)
                        throw SimEngineException("local_stencil only supports DiscreteShellBending as a shell extra constitution.");
                    bending = true;
                }
            if(!bending)
                return;
            // Match DiscreteShellBending's edge/opposite-vertex convention.
            std::map<std::pair<IndexT, IndexT>, std::set<IndexT>> opposite;
            for(auto tri : sc.triangles().topo().view())
                for(int a = 0; a < 3; ++a)
                {
                    IndexT b = tri((a + 1) % 3), c = tri((a + 2) % 3);
                    if(b > c)
                        std::swap(b, c);
                    opposite[{b, c}].insert(tri(a));
                }
            auto stiffness = sc.edges().find<Float>("bending_stiffness")->view();
            const auto edges = sc.edges().topo().view();
            for(SizeT i = 0; i < edges.size(); ++i)
            {
                IndexT b = edges[i](0), c = edges[i](1);
                if(b > c)
                    std::swap(b, c);
                const auto& verts = opposite[{b, c}];
                if(verts.size() > 2)
                    throw SimEngineException("local_stencil bending requires manifold triangle edges.");
                if(verts.size() != 2)
                    continue;
                Vector4i ids{*verts.begin() + offset, b + offset, c + offset, *verts.rbegin() + offset};
                Term t;
                t.type = 17;
                t.mu   = stiffness[i];
                if(!std::isfinite(t.mu) || t.mu < 0)
                    throw SimEngineException("local_stencil requires finite nonnegative bending stiffness.");
                detail::compute_discrete_shell_bending_reference(t.length,
                                                                 t.height,
                                                                 t.angle,
                                                                 t.weight,
                                                                 rest[ids(0)],
                                                                 rest[ids(1)],
                                                                 rest[ids(2)],
                                                                 rest[ids(3)]);
                stencils.push_back(ids);
                terms.push_back(t);
            }
        });
    m_elastic_stencils.copy_from(stencils);
    m_elastic_terms.copy_from(terms);
}
}  // namespace uipc::backend::cuda
