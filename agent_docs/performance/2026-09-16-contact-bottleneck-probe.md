# Contact bottleneck probe (2026-09-16)

This is preliminary research evidence, not an accepted optimization. The full protocol, environment, raw logs, and limitations are in [`experiments/2026-09-16-preliminary.md`](../../experiments/2026-09-16-preliminary.md) and [`experiments/env.md`](../../experiments/env.md).

The in-repository `94_contact_density_probe` generated parallel cloth stacks on an RTX 3060 Ti. All 36 fresh-process, 40-frame throughput runs completed and converged. For a four-layer, 16x16-grid scene, the median frame time over frames 6–40 was 20.454 ms with contact and 5.792 ms for the same cloth mesh with contact disabled. The ratio was 3.53x; the contact run had fewer Newton iterations, so increased Newton count does not explain this case's difference.

A separate synchronized timing run attributed 205.5 ms of its 971.1 ms pipeline total to DCD detection and 121.6 ms to trajectory-candidate detection over 40 frames. Timer synchronization changes wall time, so those figures describe stage attribution only. In canonical active-pair diagnostics, settled frames 33–40 had median across-frame `R_new = 0.078%` in the four-layer 16-grid variant. During contact formation, median within-frame `R_new` was 21.5% across 13 nonempty comparisons. Three other grid/layer variants showed similarly low settled across-frame churn, but they remain one scene family.

Next gate: an offline refresh oracle that preserves the baseline trajectory for correctness comparison, followed by a conservative regional validity certificate and replication in at least three independent scene families. Neither a safe skip nor a novel algorithm has been established.
