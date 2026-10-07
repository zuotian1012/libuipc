#include <finite_element/fem_local_stencil_solver.h>
#include <algorithm>
#include <numeric>

namespace uipc::backend::cuda
{
void FEMLocalStencilSolver::init_coloring(span<const Vector4i> elements,
                                          const vector<vector<IndexT>>& incident)
{
    if(!m_hybrid && !m_gs)
        return;
    const SizeT    n = elements.size();
    vector<IndexT> order(n), degree(n), colors(n), packed;
    vector<U64>    rank(n, 0), occupied(incident.size(), 0);
    vector<bool>   covered(n, false);
    std::iota(order.begin(), order.end(), 0);
    for(SizeT t = 0; t < n; ++t)
    {
        vector<IndexT> adjacent;
        for(int a = 0; a < 4; ++a)
            if(elements[t](a) >= 0)
                adjacent.insert(adjacent.end(),
                                incident[elements[t](a)].begin(),
                                incident[elements[t](a)].end());
        std::sort(adjacent.begin(), adjacent.end());
        adjacent.erase(std::unique(adjacent.begin(), adjacent.end()), adjacent.end());
        degree[t] = static_cast<IndexT>(adjacent.size()) - 1;
    }
    if(m_gs)
    {
        // Full coloring for a pure GS baseline: no grey elasticity stencils.
        // Dynamic contact stencils are colored against these groups at each
        // outer iterate. This host greedy implementation is not GPU Vivace.
        std::stable_sort(order.begin(),
                         order.end(),
                         [&](IndexT a, IndexT b)
                         { return degree[a] > degree[b]; });
        m_gs_vertex_colors.resize(incident.size());
        m_gs_elastic_colors.resize(n);
        m_gs_color_count = 0;
        for(IndexT t : order)
        {
            vector<bool> used(m_gs_color_count, false);
            for(int a = 0; a < 4; ++a)
                if(elements[t](a) >= 0)
                    for(IndexT c : m_gs_vertex_colors[elements[t](a)])
                        used[c] = true;
            IndexT color = 0;
            while(color < used.size() && used[color])
                ++color;
            m_gs_color_count       = std::max(m_gs_color_count, color + 1);
            m_gs_elastic_colors[t] = color;
            for(int a = 0; a < 4; ++a)
                if(elements[t](a) >= 0)
                    m_gs_vertex_colors[elements[t](a)].push_back(color);
        }
        for(IndexT color = 0; color < m_gs_color_count; ++color)
            m_color_passes.emplace_back(0, color);
        logger::info("Local stencil full GS elasticity coloring: {} colors, {} stencils",
                     m_gs_color_count,
                     n);
        return;
    }
    SizeT remaining = n;
    // Encode grey as a rank penalty larger than every finite color accumulated
    // within the bounded construction. This avoids floating infinity ties.
    const U64 grey_rank         = static_cast<U64>(m_color_limit) * (2 * n + 2);
    bool      complete_coverage = false;
    for(SizeT scheme = 0; remaining && scheme <= 2 * n; ++scheme)
    {
        const SizeT before = remaining;
        std::stable_sort(order.begin(),
                         order.end(),
                         [&](IndexT a, IndexT b)
                         {
                             // Grey stencils must enter an independent group in some
                             // scheme. Prefer uncovered stencils after stalled progress.
                             if(complete_coverage && covered[a] != covered[b])
                                 return !covered[a];
                             if(scheme == 0)
                                 return degree[a] > degree[b];
                             if(rank[a] != rank[b])
                                 return rank[a] > rank[b];
                             return degree[a] > degree[b];
                         });
        std::fill(occupied.begin(), occupied.end(), U64{0});
        std::fill(colors.begin(), colors.end(), IndexT{-1});
        vector<bool> used(m_color_limit, false);
        for(IndexT t : order)
        {
            U64 unavailable = 0;
            for(int a = 0; a < 4; ++a)
                if(elements[t](a) >= 0)
                    unavailable |= occupied[elements[t](a)];
            for(IndexT c = 0; c < m_color_limit; ++c)
                if(!(unavailable & (U64{1} << c)))
                {
                    colors[t] = c;
                    used[c]   = true;
                    for(int a = 0; a < 4; ++a)
                        if(elements[t](a) >= 0)
                            occupied[elements[t](a)] |= U64{1} << c;
                    break;
                }
            if((m_primary_coverage ? colors[t] == 0 : colors[t] >= 0) && !covered[t])
            {
                covered[t] = true;
                --remaining;
            }
            rank[t] += colors[t] < 0 ? grey_rank : static_cast<U64>(colors[t]);
        }
        packed.insert(packed.end(), colors.begin(), colors.end());
        for(IndexT c = 0; c < m_color_limit; ++c)
            if(used[c])
                m_color_passes.emplace_back(static_cast<IndexT>(scheme), c);
        complete_coverage = remaining == before;
    }
    if(remaining)
        throw SimEngineException("Complementary coloring failed to cover every elastic stencil.");
    m_colors.copy_from(packed);
    logger::info("Local stencil complementary coloring: {} schemes, {} color passes, all {} elastic stencils covered ({})",
                 packed.size() / n,
                 m_color_passes.size(),
                 n,
                 m_primary_coverage ? "primary" : "independent");
}

void FEMLocalStencilSolver::prepare_gs_coloring()
{
    vector<Vector4i> contacts(m_contact_stencils.size());
    if(!contacts.empty())
        m_contact_stencils.cview().copy_to(contacts.data());
    auto occupied    = m_gs_vertex_colors;
    auto colors      = m_gs_elastic_colors;
    m_gs_color_count = static_cast<IndexT>(m_color_passes.size());
    for(const auto& stencil : contacts)
    {
        vector<bool> used(m_gs_color_count, false);
        for(int a = 0; a < 4; ++a)
            if(stencil(a) >= 0)
                for(IndexT color : occupied[stencil(a)])
                    used[color] = true;
        IndexT color = 0;
        while(color < used.size() && used[color])
            ++color;
        m_gs_color_count = std::max(m_gs_color_count, color + 1);
        colors.push_back(color);
        for(int a = 0; a < 4; ++a)
            if(stencil(a) >= 0)
                occupied[stencil(a)].push_back(color);
    }
    m_gs_colors.copy_from(colors);
}
}  // namespace uipc::backend::cuda
