# Preliminary contact bottleneck evidence

**Status:** reproducible small-scene observation, not a causal bottleneck or a novelty claim. Prepared for the 2026-09-17 10:30 group meeting.

## Controlled throughput result

Each value below is the median of three fresh-process run medians over frames 6–40. All 36 runs completed 40 frames and passed strict convergence and the scene-specific separation checks. The `contact=0` control keeps the same cloth mesh and solver settings but disables all contact and omits the ground primitive.

A grid with 8 quads per side has 81 vertices and 128 triangles per sheet. A grid with 16 quads per side has 289 vertices and 512 triangles per sheet; the four-layer 16-grid case therefore has 1,156 cloth vertices and 2,048 triangles.

| Grid quads per side | Layers | Contact off, ms/frame | Contact on, ms/frame | Ratio |
|---:|---:|---:|---:|---:|
| 8 | 1 | 2.647 | 3.158 | 1.19 |
| 8 | 2 | 3.797 | 10.607 | 2.79 |
| 8 | 4 | 4.529 | 13.621 | 3.01 |
| 16 | 1 | 4.497 | 5.243 | 1.17 |
| 16 | 2 | 5.311 | 13.991 | 2.63 |
| 16 | 4 | 5.369 | 20.160 | 3.76 |

The four-layer 16-grid condition costs 14.79 ms/frame more than its no-contact control. The contact run's median Newton count is 1 and median linear iteration count is 10, versus 2 and 15 in the control. Thus the additional wall time is not explained by *more* global solver iterations in this scene. This is an observation about this configuration, not yet a partition of the wall-time difference.

## Separate diagnostic runs

For the four-layer 16-grid contact condition, synchronized stage timers accumulated over 40 frames show 971.1 ms in `Pipeline`, including 205.5 ms in the two `Detect DCD Candidates` scopes (40 initial detections and 28 additional Newton detections) and 121.6 ms in `Detect Trajectory Candidates`. Together these scopes are 33.7% of the synchronized pipeline time. Their throughput contribution cannot be read directly from this percentage because timer synchronization changes wall time. The corresponding no-contact diagnostic pipeline took 243.0 ms, but its algorithmic path differs.

At frame 40, the same diagnostic reports 34,074 DCD candidate pairs, 55,844 swept-trajectory candidate pairs, and 34,074 filtered active primitive pairs. This does not show a large candidate-to-active inflation in the settled state. At frame 1, however, 8,708 trajectory candidates were produced while there were zero active pairs; the cause of those pairs remains to be isolated.

Canonical active-pair sets were compared after DCD filtering. During the contact-formation phase, 13 nonempty within-frame comparisons in the four-layer 16-grid scene had median `R_new` around 21.5%; the low-churn-per-Newton hypothesis is **not supported there**. During settled frames 33–40, eight across-frame comparisons had median `R_new` around 0.078%, with roughly 19,400 active pairs per frame. Similar settled-phase medians occurred in the 8-grid 2-layer, 8-grid 4-layer, and 16-grid 2-layer variants (0%, 0.033%, and 0%, respectively). These are variations of one scene family, not the required three independent scenes.

The crossing and ground checks are scene-specific. The final 36-run throughput batch also required the lowest vertex of each upper sheet to remain above the highest vertex of the sheet below it, a sufficient nonintersection check for these horizontally separated sheets. For the four-layer 16-grid contact runs, the minimum observed inter-layer matching-vertex gap was 0.01034, the minimum vertical-slab gap was 0.00992, and the lowest vertex height was 0.00834 above the ground. A general mesh-intersection check for other geometries and a conservative refresh certificate are still required for any production change.

## Interpretation and next gate

The most promising *question* is whether persistent collision information can receive a conservative, regional validity certificate across frames, so the global DCD search is performed only when new pairs may emerge. The current data establish temporal coherence in a settled parallel-cloth case and a material collision-detection cost. They do **not** establish that full DCD can safely be skipped, that the same behavior generalizes to twisting or sliding, or that this differs from existing local CCD approaches.

The next diagnostic was completed after this report: six complete-baseline runs classified every DCD call by new active pairs and measured its core time. One of 246 active-set calls found no new pair; the optimistic whole-call time removal remained under 4% of synchronized pipeline time in every run, below the 1.2x gate. See `next_step/2026-09-16-followup.md`. Regional certification and per-query CCD cost remain open questions; neither is supported as a performance improvement yet.

Raw evidence: `probe/summary.json`, `probe_grid16/summary.json`, `probe_grid16/frames.csv`, `probe_grid16/timers_l4_c1.json`, and `churn_canonical_g*_l*.txt`. See `env.md` for commands and runtime caveats.
