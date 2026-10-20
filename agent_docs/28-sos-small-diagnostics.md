# 28 - Small SoS diagnostics (2026-10-20)

The owner requested targeted small tests before another large run. The new
`python/examples/diagnose_local_stencil.py` runs serial, isolated CUDA processes
with bounded rounds and a per-arm timeout. It preserves each failure, budget
acceptance and configuration, rather than treating process completion as
convergence. This change publishes the previously local opt-in GS/Jacobi/batch
energy instrumentation; it does not change solver directions or acceptance.

## Controlled cases

All meshes occupy a 0.2 m cube, with density 1000 kg/m^3, Young's modulus
1 MPa, Poisson ratio 0.3, dt=0.01 s and local velocity residual tolerance
0.01 m/s. Each cell is split into six tetrahedra. The beam case pins x=0,
disables contact and starts at rest. The impact case is free, starts 0.7 mm
above the floor and has downward speed 0.8 m/s for resolutions 2, 4 and 8.
The initial resolution-1 smoke case uses zero initial speed.

All local arms use 15 inner passes, sequential hybrid and independent
coverage. Controls are assembled/packed elasticity, native IPC/custom SoS
contact, monotone/local merging and warm off/on. Custom contact uses expanded
activation with IPC stiffness calibration. Warm initialization uses after_local
projection and at most 10 auxiliary passes. Native uses its existing
displacement stopping norm, so these are not matched-accuracy speedups.

## Results

Contact-free mesh refinement:

| Cells per edge | Vertices | Tetrahedra | Outer rounds | Total inner passes |
|---|---:|---:|---:|---:|
| 1 | 8 | 6 | 2 | 7 |
| 2 | 27 | 48 | 3 | 30 |
| 4 | 125 | 384 | 11 | 147 |
| 8 | 729 | 3072 | 46 | 671 |

Both assembled and packed arms give the same counts and agree to roundoff in
their endpoints. Slow mesh-scale propagation is present even without contact,
warm initialization or customized barriers. This supports investigating the
global error modes left by local sweeps; it does not by itself prove a coding
defect or establish the cause of every dense bunny oscillation.

Ground impact, outer rounds:

| Vertices | Native IPC | Local IPC | Custom SoS, warm off | Custom SoS, warm on |
|---|---:|---:|---:|---:|
| 27 | 3 | 5 | 5 | 14 |
| 125 | 3 | 14 | 15 | 32 |
| 729 | 3 | 52 | 52 | Not converged at 64; 108 with budget 128 |

Monotone and local merging agree in these impact controls. No accepted
GS/Jacobi/batch energy increase above 1e-16 was recorded in the tested arms.
This is a statement about these trajectories, not a proof of simultaneous
Jacobi descent for arbitrary meshes.

The first full-elastic residual after warm handoff is about 200.09 m/s at 125
vertices and 797.42 m/s at 729 vertices, versus 0.898 m/s at the initial cold
state. In the 729-vertex case, the auxiliary warm solve itself converges in
seven passes (residual 0.00304 m/s); it is not being cut off at its ten-pass
limit. Its different inertia/contact objective therefore produces an
initialization that takes more work for the full elastic objective here.
At 64 rounds the full endpoint residual is 0.03891 m/s; the extended monotone
arm eventually converges at 108 rounds. This motivates testing warm candidate
selection/projection against the full objective, not simply increasing the
auxiliary warm iteration count.

## Independent checks

For the 27-vertex beam, the runner independently reconstructs lumped masses,
SNK1 energy and analytic gradients on CPU, checks the gradient by finite
differences, and solves the free-DOF problem with a dense finite-difference
Newton reference. It includes the frontend's converted tetrahedral coefficients
(`mu'=4*mu/3`, `lambda'=lambda+5*mu/6`), not the raw engineering moduli.

The independent local endpoint velocity residual is 0.009637870923 m/s,
matching the GPU log and passing 0.01. The dense reference reaches
1.75e-9 m/s; the local endpoint is at most 3.16e-5 m away. Native's independent
residual is 0.000208985 m/s and its max reference distance is 2.06e-7 m.
Identical tolerance numbers thus do not imply equal accuracy across solvers.

The existing CUDA `[stencil]` suite passes 207 assertions in 13 cases, including
local CG versus direct solve, Hessian products, contact derivatives and the
collapsed-endpoint regression. These checks do not rule out every local CG
failure at highly deformed contact states. No relaxation of physical stopping
thresholds or hidden native fallback was introduced.

## Reproduction and provenance

With a current built package:

```shell
python python/examples/diagnose_local_stencil.py --output output/small-smoke
python python/examples/diagnose_local_stencil.py --resolution 2 --impact-speed .8 --output output/small-impact
python python/examples/diagnose_local_stencil.py --resolution 8 --impact-speed .8 --rounds 128 --cases ground-sos-monotone ground-warm-monotone --output output/small-warm-control
```

`UIPC_LOCAL_PACKAGE` selects the package. On this workstation the verified
runtime is `output/sos-budget-failure-20261018/runtime`; backend SHA256 is
`088c9b84e37fcd4800c62ef8799cb0d6bd2bae79493032d6648f097ff01ccf24`.
It includes the preserved working-tree changes from its build; a clean
published-checkout rebuild has not been independently validated. Timings
include diagnostic logging and are not performance claims.

Results live under `output/sos-small-diagnostics-20261020/`: `final-smoke`,
`beam-certificate`, `grid4`, `grid8`, and `warm-extended`. Each has a report,
per-arm log, scene config, geometry and engine stats. Earlier exploratory
`refined-impact` and `verified-impact` CPU certificates used incorrect legacy
energy/coefficient assumptions and are superseded by `beam-certificate`;
their failed oracle comparisons are not backend defects.

Recommended next experiments: isolate long-wavelength error correction in
the contact-free refinement case, and compare warm candidates at the same
physical frame using the full objective. A modified method must be labeled
separately from paper reproduction and pass these controls before a dense
120-frame rerun. No full-scene stability or paper-equivalent speedup is claimed.
