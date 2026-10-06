#include <finite_element/fem_local_stencil_solver.h>
#include <algorithm>
#include <numeric>

namespace uipc::backend::cuda
{
void FEMLocalStencilSolver::init_coloring(span<const Vector4i> elements,
                                          const vector<vector<IndexT>>& incident)
{
    if(!m_hybrid)
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
}  // namespace uipc::backend::cuda
