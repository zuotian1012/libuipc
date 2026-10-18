# Scene Configuration Reference

`Scene::default_config()` / `Scene.default_config()` provides the values for a
simulation scene. `Scene::config_schema()` / `Scene.config_schema()` is the
machine-readable contract for every registered key: defaults, storage and JSON
types, units, hard constraints, lifecycle, implementation status, descriptions,
and source-level consumers. The tables below present the same contract for
humans. A single typed declaration creates both the runtime defaults and schema
metadata, preventing the two views from drifting as keys evolve. It lives in
[`scene_default_config.cpp`](https://github.com/spiriMirror/libuipc/blob/main/src/core/core/scene_default_config.cpp),
while effective-value rules and selector values were checked against the CUDA
backend that consumes them.

!!! important

    Scene configuration is strict: unknown keys, wrong storage types, non-finite
    values, unsupported selectors, and values outside the documented hard
    constraints are rejected. Validation runs when constructing a `Scene` and
    again in `World::init(scene)`, after any mutable `scene.config()` edits.

## Querying the contract

The schema is available without reading backend source and can be consumed by
tools, agents, editors, and configuration generators.

=== "C++"

    ```cpp
    auto schema = uipc::core::Scene::config_schema();
    auto dt = schema["entries"]["dt"];
    std::cout << dt.dump(2) << '\n';
    ```

=== "Python"

    ```python
    from uipc import Scene

    schema = Scene.config_schema()
    print(schema["entries"]["dt"])
    ```

The installed command-line interface can dump either the complete schema or a
single slash-separated key:

```bash
python -m uipc config-schema
python -m uipc config-schema contact/d_hat
```

## Creating and editing a configuration

The usual path is to copy the defaults, change only what the scene needs, and
then construct the scene.

=== "C++"

    ```cpp
    auto config = uipc::core::Scene::default_config();
    config["dt"] = 0.01;
    config["gravity"] = uipc::Vector3{0.0, -9.8, 0.0};
    config["contact"]["friction"]["enable"] = true;

    uipc::core::Scene scene{config};
    ```

=== "Python"

    ```python
    from uipc.core import Scene

    config = Scene.default_config()
    config["dt"] = 0.01
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    config["contact"]["friction"]["enable"] = True

    scene = Scene(config)
    ```

A scene also exposes its registered configuration as attributes. This is
useful when a scene has already been created or loaded. Make these changes
**before** `world.init(scene)`: several backend systems cache configuration
values during initialization, so `scene.config()` is not a general hot-reload
interface. Call `scene.validate_config()` for an earlier explicit check;
`world.init(scene)` calls it automatically.

In Python, `Scene.default_config()` returns a dictionary, but `scene.config()`
returns a **ConfigAttributes facade**, not a dictionary. Use `find("dt")` and
`uipc.view(slot)` for attribute access; retain the owning `scene` while using the
facade. Post-initialization attribute writes are not automatically revalidated by
every `world.advance()` call and are not a general backend reconfiguration API.

=== "C++"

    ```cpp
    auto config = scene.config();
    auto dt = config.find<uipc::Float>("dt");
    uipc::geometry::view(*dt)[0] = 0.02;
    ```

=== "Python"

    ```python
    from uipc import view

    dt = scene.config().find("dt")
    view(dt)[:] = 0.02
    ```

Do not use `scene.config().create(...)` as an application metadata store.
Unregistered values have no backend consumer; attach custom attributes to the
appropriate geometry instead.

## Types and units

- `flag` is stored as an integer (`0` or `1`); Python `False`/`True` is
  accepted.
- Length, time, velocity, acceleration, pressure, and density use SI units:
  m, s, m/s, m/s², Pa, and kg/m³.
- A vector is a three-component column vector. Python JSON therefore displays
  gravity as `[[x], [y], [z]]`.
- A selector must match one of the documented strings exactly. An unsupported
  selector is rejected by scene configuration validation. Optional collision
  selectors additionally depend on the build options described below.
- Numeric hard constraints apply even when a subsystem is disabled or a relative
  override is active. For example, the stored absolute `contact/d_hat` must still
  be positive when `contact/d_hat_relative > 0`. All numeric values must be finite.

## Time integration

| Key | Type | Default | Valid domain / choices | Meaning |
| --- | --- | --- | --- | --- |
| `dt` | float, s | `0.01` | finite, `> 0` | Time represented by one call to `world.advance()`. |
| `gravity` | Vector3, m/s² | `[0, -9.8, 0]` | finite components | Global acceleration applied to dynamic bodies and FEM vertices. |
| `integrator/type` | string | `"bdf1"` | `"bdf1"`, `"bdf2"` | Backward differentiation formula used for time integration. Start with BDF1; BDF2 changes numerical damping and transient response. |
| `cfl/enable` | flag | `0` | `0`, `1` | Enables the contact-system CFL step filter. This is an advanced/experimental high-speed-contact control; leave it off unless a scene has been diagnosed to need it. |

!!! warning "BDF2 startup and variable-step limitation"

    The 2026-09-11 audit of `4757859c` found that BDF2 history initialization does
    not preserve a prescribed nonzero constant velocity at startup. A contact-free
    particle with velocity 1 m/s and dt=0.01 s advanced 0.0066667 m on its first
    BDF2 step rather than 0.01 m. The implemented BDF2 coefficients also assume a
    constant step size; live dt reads do not establish variable-step BDF2 support.
    Use default BDF1 when relying on these cases until the integration contract is
    corrected and tested. This is an identified algorithm/history limitation,
    not a requirement for bitwise repeatability or a change to the solver here.

## Newton solve

| Key | Type | Default | Valid domain / choices | Meaning |
| --- | --- | --- | --- | --- |
| `newton/solver` | string | `"global"` | `"global"`, `"local_stencil"` | Unitless selector, set before initialization. `global` uses the standard global linear solve. Experimental `local_stencil` uses elastic/contact Newton-CG with unprojected Hessians: StableNeoHookean tetrahedra, StrainLimitingBaraffWitkinShell or NeoHookeanShell triangles, optional DiscreteShellBending and fixed half planes; mixed cloth/solid scenes are supported. Requires BDF1, IPC, static topology and no differentiable simulation. Contact uses standard IPC barriers and CCD; friction must be disabled when contact is enabled. Uses the selected `local_stencil/stopping` norm; the default velocity mode uses absolute `velocity_tol` on the mass-scaled total gradient divided by `dt`. Relative tolerance must be zero and semi-implicit termination is bypassed. Other constitutions (including ABD, rods, plastic bending and stitching) remain unsupported. |
| `newton/local_stencil/sweep` | string | `"hybrid"` | `"hybrid"`, `"jacobi"`, `"gs"`, `"additive"` | Unitless, set before initialization; effective only with `newton/solver="local_stencil"`. Hybrid protects the current complementary elastic color and averages secondary updates. Jacobi averages all incident stencil displacements per vertex. GS computes/submits only the current full color, including dynamically colored contacts; host greedy coloring is not GPU Vivace. Additive retains legacy uniform scaling without local filtering. All use the same local Newton-CG objective and selected stopping norm. |
| `newton/local_stencil/globalization` | string | `"local"` | `"local"`, `"regular"` | Unitless, set before initialization. `local` applies per-stencil CCD/inversion/Armijo before merging. `regular` skips per-stencil filters. Both retain the combined elastic inversion bound and engine global CCD/energy acceptance. Additive ignores this option. |
| `newton/local_stencil/hybrid_submission` | string | `"frozen"` | `"frozen"`, `"sequential"` | Unitless selector, set before initialization. `frozen` preserves simultaneous submission of independently filtered directions. `sequential` requires hybrid/local mode: commit the current independent color, then filter the unowned components of the original Jacobi directions at the updated state with recomputed slopes. The Jacobi search uses the actual per-vertex averaged contribution, including its recomputed slope, CCD and energy increment. Both phases and the composed outer path retain geometry checks. This interpretation of separate searches is under integration verification. |
| `newton/local_stencil/color_limit` | integer | `8` | `[1, 64]` | Unitless, set before initialization; effective only for hybrid local stencils. Maximum independent colors per complementary scheme; excess elastic stencils remain grey. Coverage follows `color_coverage`. GS uses an uncapped full coloring instead. |
| `newton/local_stencil/merge_acceptance` | string | `"monotone"` | `"monotone"`, `"local"` | Unitless, set before initialization; local stencils only. `monotone` additionally requires globally decreasing merged energy and rejects non-descent merged directions. `local` requires local globalization and accepts independently Armijo-filtered proposals after combined CCD/inversion and finite-energy certification, including inner batches; it does not impose global monotonicity or weaken final residual stopping. |
| `newton/local_stencil/barrier_activation` | string | `"reference"` | `"reference"`, `"expanded"` | Unitless selector, set before initialization; effective for `barrier_model=sos`. `reference` uses frame-origin clearance literally. `expanded` uses the larger of that clearance and the global activation distance, compensating kappa with the chosen distance. This is an explicit persistent-contact stabilization extension, not a lower bound stated by Eq. 11. |
| `newton/local_stencil/barrier_stiffness` | string | `"ipc"` | `"ipc"`, `"distance"` | Unitless selector, set before initialization; effective for `barrier_model=sos`. `ipc` converts the native squared-distance log-squared barrier coefficient to a distance-log coefficient by matching normal force at half the global activation clearance, then applies paper per-pair compensation. This fixes coefficient units but does not equate the entire potentials. `distance` accepts a coefficient already in Eq. 11 units without conversion. |
| `newton/local_stencil/barrier_model` | string | `"ipc"` | `"ipc"`, `"sos"` | Unitless, set before initialization; local stencils only. `sos` requires local globalization, retains full PT/EE/plane pairs within each physical step and uses Eq. 11 with reference clearance and compensated stiffness in every energy/derivative consumer. Thickness is subtracted from geometric distance. The standard IPC objective remains the default control. |
| `newton/local_stencil/warm_start_inversion` | string | `"iterative"` | `"iterative"`, `"after"`, `"after_local"` | Unitless, set before initialization; effective with `warm_start=1`. `iterative` retains local and combined inversion caps during every warm iteration. `after` keeps CCD during the inertia/barrier solve and measures inversion bounds without capping the scratch solve. If a proposal threatens inversion, it restores the physical frame origin and bounds the converged guess before elasticity. `after_local` first scales each vertex by the minimum path bound of its incident elastic/contact terms, rechecking the actual mixed path for up to 32 rounds; a final combined inversion and regular CCD certificate is always required. Already certified warm paths need no second projection. New contacts that invalidate a residual-converged projected guess restart the inertia/barrier solve while its optional budget remains. A budgeted initializer is checked against the expanded objective and geometry before full elasticity, without treating its warm residual as a certificate. Intermediate warm iterates may have inverted volume elements; no physical frame is accepted without the full residual and positive-element checks. These opt-in controls interpret the post-solve searches in section 6; the specific projection procedure is not specified by the paper. |
| `newton/local_stencil/warm_start_max_iter` | integer | `0` | `[0, +inf)` | Warm passes per physical frame, set before initialization; effective with `warm_start=1`. `0` retains residual-based warm termination. Positive values cap auxiliary initialization work and submit a geometrically checked approximate guess when exhausted; the log explicitly distinguishes this from warm convergence. Full elasticity and the selected final residual threshold remain mandatory. The paper does not prescribe a fixed warm-pass budget. |
| `newton/local_stencil/warm_start_relaxation` | number | `1.0` | `(0, 1]` | Unitless, set before initialization; effective during inertia-barrier warm start only. Sets the maximum local trial fraction before Armijo filtering. `1` is the unrelaxed Jacobi control; smaller values explicitly under-relax it. This numerical stabilization is not specified by the paper and does not change the final residual tolerance. |
| `newton/local_stencil/warm_start` | integer | `0` | `0`, `1` | Unitless, set before initialization; local stencils only. `1` requires `barrier_model=sos`, overrides `initial_guess`, and enables acceleration prediction followed by inertia-barrier Jacobi Newton-CG before enabling elasticity. CCD remains active in both phases; inversion ordering follows `warm_start_inversion`. Warm initialization uses the selected residual tolerance unless its explicit positive `warm_start_max_iter` budget expires. Full elasticity always uses the selected residual threshold before physical frame completion. Velocity history is checkpointed; first-step acceleration uses external acceleration. |
| `newton/local_stencil/hessian_product` | string | `"assembled"` | `"assembled"`, `"csfd"`, `"packed"` | Unitless selector, set before initialization. `assembled` uses explicit local matrices; `csfd` evaluates restricted SNK1 stress products directly. `packed` retains separate vertex/stencil direction slots and distributes energy/stencil products across CUDA threads with parallel accumulation; it requires local globalization. Shell/contact operators are analytic in both CSFD modes. All modes retain the same 12-step projection-free CG and residual criterion. |
| `newton/local_stencil/initial_guess` | string | `"previous"` | `"previous"`, `"inertial"` | Unitless, set before initialization; effective only for local stencils. `previous` retains the previous position. `inertial` proposes `x_tilde - x` at the first pass of a frame when it is a descent direction; the combined inversion bound, global CCD and energy search still apply. This option alone does not run the inertia-barrier warm start or change the convergence tolerance; `warm_start=1` overrides it. |
| `newton/local_stencil/stopping` | string | `"velocity"` | `"velocity"`, `"gradient_l2"` | Unitless selector, set before initialization. `velocity` uses max free-vertex `norm(g_i)/(m_i*dt)` and `newton/velocity_tol`; `gradient_l2` uses the absolute free-DOF incremental-potential gradient norm and `gradient_tol`. Both norms are logged. |
| `newton/local_stencil/gradient_tol` | Float | `1e-4` | Finite and `>0` | `kg*m`, effective only for `gradient_l2`; gradient of `J = 0.5*(x-x_tilde)^T*M*(x-x_tilde) + dt^2*V`, without mass normalization. This is not a universal accuracy match to a paper or another solver. |
| `newton/local_stencil/color_coverage` | string | `"independent"` | `"independent"`, `"primary"` | Unitless, set before initialization. `independent` requires every elastic stencil to enter a non-grey independent group in some scheme (Fig. 5 interpretation). `primary` requires every stencil to be color zero in some scheme, as stated in section 5.2 of the SoS paper. |
| `newton/local_stencil/inner_sweeps` | IndexT | `1` | `[1,256]` | Count of local color passes per global acceptance round. Values above one require `sweep="hybrid"` and `globalization="local"`; a frozen contact set is re-differentiated for every inner pass. Inner merged steps require local contact CCD, elastic bounds and frozen-objective Armijo acceptance. The accumulated displacement still undergoes global CCD/energy acceptance from the restored outer start. |
| `newton/local_stencil/restart_schedule_on_recover` | IndexT | `0` | `0`, `1` | Unitless flag. `1` explicitly resets the color cursor when recovering physical state for a checkpoint comparison, allowing a changed schedule; `0` restores the stored cursor and requires a matching pass count. |
| `newton/max_iter` | integer | `1024` | `>= 1` | Maximum nonlinear iterations in one frame (outer acceptance rounds for local stencils). Reaching it warns, or throws in strict mode unless the local-stencil budget policy below certifies an approximate endpoint. |
| `newton/local_stencil/iteration_limit_policy` | string | `"error"` | `error`, `accept_last_valid` | Unitless selector, consumed by the IPC local-stencil path. `error` preserves existing strict-mode behavior. `accept_last_valid` permits an unconverged full-elastic endpoint at `newton/max_iter`, after ordinary outer motion checks and fresh endpoint contact/gradient/energy evaluation. It does not apply another displacement or waive line-search, warm-handoff, animation-target or geometry failures. Frame stats retain `converged=false`, set `hit_newton_limit=true`, and report `iteration_limit.accepted`, endpoint velocity residual (m/s), gradient L2 (kg*m), energy and endpoint residual status. Native/global and AL paths do not use this selector. |
| `newton/min_iter` | integer | `0` | `0 <= value <= max_iter` | Hard floor before ordinary Newton convergence may terminate. `0` disables the floor. |
| `newton/use_adaptive_tol` | reserved integer | `0` | exactly `0` | Reserved for compatibility. Because no adaptive-tolerance consumer exists, setting it to `1` is rejected rather than silently doing nothing. |
| `newton/velocity_tol` | float, m/s | `0.05` | `> 0` | Absolute velocity tolerance. The displacement test is `max_axis_displacement <= velocity_tol * dt`. |
| `newton/velocity_tol_relative` | float | `0.0` | `> 0` enables; `<= 0` disables | Scene-relative override. Effective velocity tolerance becomes `value * rest_scene_bbox_diagonal`. |
| `newton/ccd_tol` | float | `1.0` | `(0, 1]` | Newton convergence additionally requires the latest CCD step fraction to be at least this value. |
| `newton/transrate_tol` | float, 1/s | `0.1` | `>= 0` | ABD transform-rate tolerance. The per-step threshold is `transrate_tol * dt`; irrelevant when no affine bodies exist. |
| `newton/semi_implicit/enable` | flag | `1` | `0`, `1` | Enables cumulative-step termination in IPC and the configured `K_min` delay in AL-IPC. |
| `newton/semi_implicit/beta_tol` | float | `1e-3` | `[0, 1]` | Standard IPC early-exit threshold for accumulated beta. AL-IPC uses `contact/al-ipc/toi_threshold` instead. |
| `newton/semi_implicit/K_min` | integer | `6` | `>= 0` | Delays cumulative-progress attenuation until the configured step count. It is not a hard ordinary-Newton floor in IPC; in AL-IPC it prevents safe-path termination before that many completed outer steps. Values below `1` are treated as `1` by AL-IPC. |

See [Newton and Linear Solvers](scene_configs/newton.md) for the exact
termination logic and tuning guidance.

## Linear system

| Key | Type | Default | Valid domain / choices | Meaning |
| --- | --- | --- | --- | --- |
| `linear_system/tol_rate` | float | `1e-3` | `(0, 1)` | Relative PCG residual tolerance. Smaller is more accurate and usually more expensive. |
| `linear_system/solver` | string | `"fused_pcg"` | `"fused_pcg"`, `"linear_pcg"` | Global iterative solver. `fused_pcg` is the optimized default; `linear_pcg` supports detailed PCG vector dumps. |
| `linear_system/fem_preconditioner` | string | `"diag"` | `"diag"`, `"mas"` | FEM local preconditioner. MAS auto-partitions every non-empty FEM geometry into fixed-size clusters and is intended for stiff/ill-conditioned FEM scenes. |
| `linear_system/use_cuda_graph` | integer mode | `1` | `0`, `1`, `2` | Fused-PCG launch mode: `0` plain launches; `1` host-checked block replay; `2` full-GPU while-loop graph. Mode 2 requires CUDA 12.4+ and falls back when unsupported. Non-IPC pipelines currently force graphs off. |
| `linear_system/check_interval` | integer | `5` | `>= 1` | Number of fused-PCG iterations between host convergence checks in modes 0/1. Larger values reduce checks but make exit granularity coarser. |

## Line search

| Key | Type | Default | Valid domain / choices | Meaning |
| --- | --- | --- | --- | --- |
| `line_search/max_iter` | integer | `8` | `>= 1` | Maximum backtracking iterations per Newton step. |
| `line_search/report_energy` | flag | `0` | `0`, `1` | Logs the energy contribution of every line-search reporter. Useful for diagnosis, noisy for normal runs. |

## Contact and friction

| Key | Type | Default | Valid domain / choices | Meaning |
| --- | --- | --- | --- | --- |
| `contact/enable` | flag | `1` | `0`, `1` | Builds contact detection and response systems. Disable for a deliberately contact-free scene. |
| `contact/d_hat` | float, m | `0.01` | `> 0` | Absolute IPC activation distance. |
| `contact/d_hat_relative` | float | `0.0` | `> 0` enables; `<= 0` disables | Overrides `d_hat` with `value * rest_scene_bbox_diagonal`. |
| `contact/friction/enable` | flag | `1` | `0`, `1` | Enables frictional contact terms. Normal non-penetration remains active when disabled. |
| `contact/eps_velocity` | float, m/s | `0.01` | `> 0` | Absolute friction transition velocity. |
| `contact/eps_velocity_relative` | float | `0.0` | `> 0` enables; `<= 0` disables | Overrides `eps_velocity` with `value * rest_scene_bbox_diagonal`. |
| `contact/constitution` | string | `"ipc"` | `"ipc"`, `"al-ipc"` | Selects the standard IPC or augmented-Lagrangian IPC pipeline. |

These global keys do not define pairwise material behavior. Friction and
contact resistance for geometry pairs are set through `ContactTabular`; see
[Contact and Collision](scene_configs/contact.md).

## AL-IPC parameters

The following keys are read only by the `"al-ipc"` pipeline. They are
algorithm parameters rather than material contact resistance.

| Key | Type | Default | Valid domain | Meaning |
| --- | --- | --- | --- | --- |
| `contact/al-ipc/mu_scale_mode` | string | `"per_vertex"` | `"per_vertex"`, `"diag_norm"` | Selects the stable mass-based per-vertex estimate or experimental uniform Hessian-diagonal scaling. |
| `contact/al-ipc/mu_scale_diag_norm` | float | `0.1` | `> 0` | In `diag_norm` mode, sets `mu = value * max_i(abs(H_E(i,i)))`, with AL contact excluded from `H_E`. |
| `contact/al-ipc/mu_scale_fem` | float | `5e7` | `> 0` | FEM scale used only in `per_vertex` mode: `mu_i = mass_i * value * dt²`. |
| `contact/al-ipc/mu_scale_abd` | float | `1e5` | `> 0` | ABD scale used only in `per_vertex` mode: `mu_i = body_mass * value * dt²`. |
| `contact/al-ipc/toi_threshold` | float | `0.1` | `(0, 1]` | Remaining cumulative safe-path weight tolerated by AL termination. Smaller values require more outer progress. |
| `contact/al-ipc/alpha_lower_bound` | float | `1e-6` | `(0, 1]` | CCD steps at or below this value do not advance the collision-free state or termination progress. |
| `contact/al-ipc/decay_factor` | float | `0.3` | `(0, 1)` | Multiplies an inactive constraint's weight after each outer AL update. The pair is removed once the accumulated weight is below `0.01`. |

`per_vertex` is the default because it remains stable when one scene mixes
cloth, volumetric FEM, affine bodies, and very different vertex masses.
`diag_norm` follows the conditioning-aware initialization in the AL-IPC paper,
but applies one uniform value to every vertex and is retained as an explicit
experimental comparison mode. Validate its trajectory and line-search status
before using it for production scenes. If its assembled diagonal norm is empty
or non-finite, the backend logs a warning and falls back to `per_vertex` for
that frame.

## Adaptive contact resistance

| Key | Type | Default | Valid domain | Meaning |
| --- | --- | --- | --- | --- |
| `contact/adaptive/min_kappa` | float, Pa | `1e8` (100 MPa) | `> 0` | Lower fallback bound and the effective default resistance when the user never calls `default_model(...)`. |
| `contact/adaptive/init_kappa` | float, Pa | `1e9` (1 GPa) | `> 0` | Initial resistance used by the adaptive-kappa strategy. |
| `contact/adaptive/max_kappa` | float, Pa | `1e11` (100 GPa) | `> 0` | Upper fallback bound. Keep `min <= init <= max`. |
| `contact/adaptive/kappa_eval_scale` | float | `1e-16` | `> 0` | Evaluation scale for the scene-adaptive kappa corridor. This is an expert parameter; keep the default unless reproducing a calibrated method. |

If a user explicitly sets a non-negative default contact resistance, the
backend clamps it into `[min_kappa, max_kappa]` and reports the range. A
negative resistance is the explicit opt-in marker for adaptive kappa and is
not clamped. When a scene-derived corridor is computable, it takes precedence
over the configured fallback bounds.

## Collision detection, validation, differentiation, and diagnostics

| Key | Type | Default | Valid domain / choices | Meaning |
| --- | --- | --- | --- | --- |
| `collision_detection/method` | string | `"info_stackless_bvh"` | `"info_stackless_bvh"`; optionally `"info_stackless_bvh_v0"`, `"stackless_bvh"`, `"linear_bvh"` | Broad-phase trajectory filter. Keep the default unless benchmarking or diagnosing the broad phase. |
| `sanity_check/enable` | flag | `1` | `0`, `1` | Runs pre-initialization intersection and distance checks. A failed check makes the world invalid. |
| `sanity_check/mode` | string | `"normal"` | `"normal"`, `"quiet"` | `normal` also writes diagnostic geometry when a check fails; `quiet` reports the failure without exporting that geometry. |
| `diff_sim/enable` | flag | `0` | `0`, `1` | Initializes differentiable-simulation state. Calling non-const `scene.diff_sim()` sets this flag automatically; do so before world initialization. |
| `extras/debug/dump_surface` | flag | `0` | `0`, `1` | Dumps intermediate surface state during the nonlinear solve. Produces substantial output. |
| `extras/debug/dump_linear_system` | flag | `0` | `0`, `1` | Dumps assembled global linear systems for diagnosis. |
| `extras/debug/dump_linear_pcg` | flag | `0` | `0`, `1` | Dumps PCG vectors for `linear_pcg`. `fused_pcg` warns and ignores this option. |
| `extras/debug/dump_mas_matrices` | flag | `0` | `0`, `1` | Dumps MAS matrices when the MAS FEM preconditioner is active. |
| `extras/strict_mode/enable` | flag | `0` | `0`, `1` | Converts nonlinear/line-search limit warnings into engine errors. Recommended for automated validation, not exploratory tuning. |

The three alternate collision selectors are compiled only when
`UIPC_WITH_CUDA_LEGACY_COLLISION=ON` (CMake, the default) or
`cuda_legacy_collision=true` (XMake). Builds with the option disabled omit the
filter implementations and remove their names from the schema enum, so scene
construction rejects a serialized or manually edited unavailable selector.
The machine-readable entry exposes this state in `conditionalValues`.

## Effective-value precedence

Three pairs of absolute/relative controls follow the same pattern. Let `L` be
the diagonal of the rest-scene bounding box:

| Effective quantity | Rule |
| --- | --- |
| Newton velocity tolerance | `velocity_tol_relative > 0 ? velocity_tol_relative * L : velocity_tol` |
| Contact activation distance | `d_hat_relative > 0 ? d_hat_relative * L : d_hat` |
| Friction transition velocity | `eps_velocity_relative > 0 ? eps_velocity_relative * L : eps_velocity` |

Relative controls are useful when the same scene recipe is run at different
scales. Absolute controls are easier to reason about when mesh units are known
and stable. Do not enable both with the expectation that they are added: a
positive relative value overrides the absolute value.

## Source map

For audits and future documentation updates, the main implementation points
are:

- schema, defaults, and unknown-key rejection:
  [`src/core/core/scene_default_config.cpp`](https://github.com/spiriMirror/libuipc/blob/main/src/core/core/scene_default_config.cpp)
- scene construction and mutable config attributes:
  [`src/core/core/scene.cpp`](https://github.com/spiriMirror/libuipc/blob/main/src/core/core/scene.cpp)
- backend initialization and cached values:
  [`src/backends/cuda/engine/sim_engine_do_init.cu`](https://github.com/spiriMirror/libuipc/blob/main/src/backends/cuda/engine/sim_engine_do_init.cu)
- relative tolerances:
  [`max_translation_checker.cu`](https://github.com/spiriMirror/libuipc/blob/main/src/backends/cuda/newton_tolerance/max_translation_checker.cu)
  and
  [`global_contact_manager.cu`](https://github.com/spiriMirror/libuipc/blob/main/src/backends/cuda/contact_system/global_contact_manager.cu)
- solver modes:
  [`linear_fused_pcg.cu`](https://github.com/spiriMirror/libuipc/blob/main/src/backends/cuda/linear_system/linear_fused_pcg.cu)
