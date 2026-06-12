# Pendulum branch-restriction bug — diagnosis & fix design

Tracking: GitHub issue #18. Status: **FIXED & verified** (see §6).

This document is the careful write-up for later review. It records (1) the bug and
evidence, (2) the reference algorithm decoded from Han & Yang's code, (3) why the
current port is wrong, and (4) the fix design.

---

## 1. The bug

`restrict_trajectory_to_curve` (`src/OpenLoop/pendulum/nonsmooth.py`) truncates every
backward-PMP trajectory ~25× too early. Retained Pendulum value samples occupy only a
tiny neighborhood of the upright equilibrium and **do not contain the swing-up
nonsmoothness** the benchmark exists to provide.

### Evidence (dataset `Pendulum_20260609_19f7876b...`, identical to `20260607`)

| | θ | ω | value |
|---|---|---|---|
| retained samples (all 34,887) | [−0.15, 0.24] | [−0.34, 0.20] | [0, **1.13**] |
| nonsmooth curve Γ (switching set) | [−4.8, 10.5] | [±11.2] | [**26**, 100] |

- Nearest sample to Γ: **0.56**; zero samples near the switching set.
- 108,367 / 143,254 trajectory points (76%) discarded by the cut.
- Solver `value_max=100`, yet retained values cap at 1.13.
- Value-sorted polyline has **1,687 / 2,635 consecutive gaps > 2 units** (max 22.8) —
  the phantom cross-arm chords.

---

## 2. Reference algorithm (Han & Yang, arXiv:2312.17467)

Code: `github.com/ComputationalRobotics/InvertedPendulumOptimalValue`, file
`main code/github_contour_line_nosat.m` (no control saturation = our default).

### 2a. Nonsmooth curve Γ — `compute the nonsmooth curve`

Γ is a set of **spiral lines** near the bottom positions θ = π + 2kπ, extending in θ̇ as
value grows (paper Fig. 2, middle column). Per value level `Vc_v(k)` (0 : 0.1 : 80):

1. Build the **equal-value contour** `cont`: for each of N trajectories, the state at the
   point whose value is closest to `Vc_v(k)`. Ordered by trajectory index (= boundary
   angle), so `cont` traces a closed loop around the origin.
2. Detect wraparound jumps: `cont_dist > 4` between consecutive points → insert bridge
   points (`insert_points`) so the polygon is well-formed across the 2π seam.
3. `poly1 = polyshape(cont_final)`, `poly2 = poly1` shifted by **+2π** in θ.
   `intersect(poly1, poly2)` → the vertices with `SID == 0` are the **new** intersection
   vertices = the nonsmooth points (expected **4** per level).
4. **Track the 4 arms across levels** by nearest-neighbor to the previous level's points
   (`onpoints`), so each arm is an ordered monotone polyline (not a bag of points).

Result: `boundary_spiral`, shape `(4 · num_levels, 2)`, four interleaved ordered arms.

### 2b. Closed basin boundary `bsp` — symmetry assembly

```matlab
boundary_spiral = boundary_spiral(1:1252,:);            % manual, PARAMETER-SPECIFIC trim
a_bsp = boundary_spiral(4*(1:K),  :);                   % arm 4 across levels
c_bsp = boundary_spiral(4*(1:K)+2,:);                   % arm 2 across levels
bsp = [flip([2*pi-c_bsp(:,1), -c_bsp(:,2)]); a_bsp];    % join reflected arm2 + arm4
bsp = [bsp; flip([2*pi-bsp(:,1), -bsp(:,2)])];          % reflect through (π,0) -> close loop
bsp = [bsp; flip([bsp(:,1)-2*pi, bsp(:,2)])];           % add the -2π period copy
```

- Reflection used: `(θ, ω) → (2π − θ, −ω)` (mirror through the bottom point (π, 0)).
- The result is a **closed polygon** bounding the smooth basin.
- ⚠️ Hand-tuned: the `1:1252` trim and the specific arm choices (2 and 4) are parameter-
  specific; the code comments warn "refine the nonsmooth line yourself" if (m, l, g)
  change. Our port must not hard-code these.

### 2c. Restriction — `inpolygon` containment (`github_main_nosat.m`)

```matlab
[in,on] = inpolygon(y(:,1), y(:,2), bsp(:,1), bsp(:,2));
nin = find(in ~= 1);
if size(nin,1) ~= 0, y(nin(1):end,:) = []; end          % truncate at FIRST exit
```

Keep trajectory points **inside** the basin; cut at the first exit.

---

## 3. Why the current port is wrong

| step | reference | this repo (`nonsmooth.py`) |
|---|---|---|
| Γ points | 4 SID==0 vertices/level, arm-tracked, ordered | all intersection points, **sorted by value** (no arm tracking) |
| boundary | **closed** basin polygon via symmetry | **open** value-ordered `LineString` (`as_linestring`) |
| cut test | **point-in-polygon** containment | `LineString.intersection` |

The open value-ordered polyline connects points across different spiral arms; those
chords sweep through the low-θ̇ origin region, so every trajectory's `LineString` hits a
phantom chord almost immediately (value ≈ 1). Prototype confirms the dumped Γ points
**cannot be re-clustered post-hoc** into a clean arm (assembled polygon is self-
intersecting, zero area) — arm tracking must happen at curve-computation time.

---

## 4. Fix design (3 pieces + verify)

1. **`compute_nonsmooth_curve`** — port arm tracking: per level extract the intersection
   vertices and match to the previous level's arms by nearest neighbor, yielding ordered
   arms. Avoid the hard-coded trim/arm-index choices; identify the basin-bounding arms
   geometrically (the arms nearest θ = ±π that enclose the origin).
2. **`basin_polygon()`** (new) — assemble the closed smooth-basin boundary from the
   relevant arm(s) via the symmetry `(θ,ω)→(2π−θ,−ω)` + the −2π copy.
3. **`restrict_trajectory_to_curve`** — replace `LineString.intersection` with point-in-
   polygon containment (truncate at first exit), matching the reference.

**Verification (diagnose loop):** `scripts/investigation/diagnose_restriction_loop.py`
caches the 256 raw trajectories once (~200 s) and re-runs only the geometry per
iteration. Pass signal: retained value max ≫ 20 and min-distance-to-Γ ≈ 0 (vs. bug:
1.13 and 0.56). Visually verify `bsp` + truncated trajectories before regenerating and
promoting the dataset.

---

## 4b. Diagnose-loop findings (in progress)

Loop: `scripts/investigation/diagnose_restriction_loop.py` caches the 256 raw
trajectories (integration is only ~32 s; the cache mainly saves repeated runs).
Profiling the geometry split:

- `compute_nonsmooth_curve`: **0.1 s**
- `build_value_samples` (restriction): **176 s**  ← the `LineString.intersection`
  ×256 against a 2636-vertex line. The fix (inpolygon) also removes this cost.

Bug reproduced exactly via the loop: retained value max 1.126, min-dist-to-Γ 0.562.

**Clean Γ extraction (replaces the noisy `LineString` crossings).** Using **polygon**
boundary crossings — `Polygon(contour).boundary ∩ translate(+2π).boundary` — yields
**exactly 4 vertices per value level** (452 total over levels 26–99), matching the
reference's `SID==0` count. The old `LineString.intersection` produced ~2592 points
including a spurious central blob (plot: `rawdata/plots/debug_arms.png` vs
`debug_arms_poly.png`).

**Arm structure (plot: `rawdata/plots/debug_arms_tracked.png`).** NN-tracking the 4
crossings from value 26 upward gives 4 clean arms (59 pts each, step median ~0.12):
- arms **0 & 2**: outer straight diagonals, reflection pair about (π,0)
  [`(−2.5,9.4) ↔ (8.78,−9.4)`].
- arms **1 & 3**: inner spiral curling into (π,0), the other reflection pair.

**Value extent insight.** Per-level crossings start at **value 26** (first branch
collision). The reference's manual `boundary_spiral(1:1252)` trim keeps only ~levels
≤ 31 — i.e. the origin's smooth basin extends only to value ≈ 26–31, **not** 100.
So corrected retained samples should reach value ~26–31 and ω ≈ ±5 (not value 100).
Grade threshold updated accordingly.

**Open domain question (checkpoint):** which arms bound the *upright (origin)* basin
(vs. the (π,0) bottom basin)? This determines the `inpolygon` boundary and is the
remaining crux before assembling the closed polygon.

## 4c. Resume point — remaining work

Solved so far (all in the fast loop, validated by plots):
- clean 4-per-level Γ extraction via polygon-boundary crossings;
- NN arm-tracking → 4 ordered arms (2 reflection pairs about (π,0));
- raw-trajectory plot (`rawdata/plots/debug_raw_traj.png`) confirms trajectories
  emanate from the origin and must be cut at the first diagonal-arm crossing.

**Decision (KW):** use option 3 — faithfully port the reference's `bsp` assembly. Ad-hoc
basin constructions fail: e.g. "near diagonal arm + its origin-reflection" produces a
degenerate sliver (the arm passes near the origin, so its reflection nearly coincides),
giving value max only ~19 along a thin line (`rawdata/plots/debug_basin_cut.png`). The
correct basin needs the reference's exact arm picks + `(θ,ω)→(2π−θ,−ω)` reflections +
−2π copy, transcribed from `github_contour_line_nosat.m` (saved at `/tmp/contour_nosat.m`).
NOTE: "restrict to inside Γ" is the paper-level method; the explicit `bsp` reflection
formula is reference *code* only, and is hand-tuned (port the structure, data-driven trim).

Remaining (the intricate crux):
1. **Close the origin basin into a compact polygon.** The two diagonal arms (0 and its
   origin-reflection) form an *open wedge* through the origin — not compact. Need the
   reference's full `bsp` assembly: pick the right arms (ref uses `a_bsp`/`c_bsp`),
   apply `(θ,ω)→(2π−θ,−ω)` + the −2π copy to close the loop. **Domain question for KW:
   which arms bound the upright (origin) basin vs. the bottom (π,0) basin?**
2. Port clean extraction + arm-tracking + `basin_polygon()` into `nonsmooth.py`.
3. Rewrite `restrict_trajectory_to_curve` → `inpolygon` containment (first exit).
4. Regenerate dataset; grade (retained value → ~26–31, min-dist-to-Γ → ~0); rebuild
   distance cache; confirm region-split now has near-region samples.

Resume assets: trajectory cache `rawdata/data/_debug_raw_trajectories_256.pkl`;
harness `scripts/investigation/diagnose_restriction_loop.py`; debug plots under
`rawdata/plots/debug_*.png`. (All throwaway — delete after the fix lands.)

## 4d. VALIDATED fix (prototype `scripts/investigation/proto_bsp.py`)

The faithful transcription now passes the loop grade. Validated algorithm:

1. **Per value level** (26 → 35, step 0.25; the value-35 cap matches the reference's
   value-31 trim intent, replacing the hard-coded `1:1252`):
   build the equal-value contour (one point/trajectory, trajectories ordered by
   `boundary_angle`); take the **polygon-boundary crossings** with the +2π shift
   (`Polygon(contour).boundary ∩ translate(+2π).boundary`) → the 4 `SID==0` vertices.
2. Keep the 4 crossings nearest (π,0) (drops higher-order spiral wraps); **track 4 arms
   across levels by optimal (Hungarian) assignment** — robust to the arm-swaps that
   greedy NN produced (greedy collapsed the basin into a pinched channel).
3. **`bsp` assembly** (reference): `part1 = [refl(c)[::-1]; a]`,
   `part2 = [part1; refl(part1)[::-1]]`, `bsp = [part2; (part2−2π)[::-1]]`, where
   `refl(θ,ω)=(2π−θ,−ω)`. Arm pair selected empirically: the one giving a valid
   origin-containing basin with value_max in [26,35] and max area (here arms `a=2,c=1`).
4. Take the **origin-containing component** (drop the ±2π neighbor copies the assembly
   also produces).
5. **Restriction** = `inpolygon` containment: cut each trajectory at the first exit.

Grade (vs. the bug):

| metric | bug | fixed |
|---|---|---|
| retained value max | 1.13 | **31.19** |
| retained samples | 34,887 | **88,292** |
| min dist to Γ | 0.56 | **< 0.3** |
| extent | \|x\|<0.41 | θ∈[−2.2,2.2], ω∈[−3.5,3.5] |

Plot: `rawdata/plots/debug_bsp_faithful.png` — a clean diagonal swing-up corridor around
the origin, samples filling it out to the switching arms.

### Port plan (Phase 5)
- `nonsmooth.py`: rewrite `compute_nonsmooth_curve` to track arms (Hungarian); add
  `basin_polygon()` (steps 3–4); keep `NonsmoothCurve` carrying both arms and basin.
- `nonsmooth.py`: rewrite `restrict_trajectory_to_curve` → `inpolygon` (step 5).
- Regenerate dataset (~32 s integration + fast geometry); rebuild distance cache; confirm
  region-split now has near-region samples. Then delete the debug harness/cache/plots.
- Open question to resolve in the port: arm-pair selection — empirical (max-area in band)
  vs. transcribe the reference's exact `a_bsp=arm4 / c_bsp=arm2` index picks.

## 5. Downstream impact

- The Pendulum value-sample benchmark does not currently exercise the switching-set
  nonsmoothness; `activationsearch` pendulum rows should be revisited after the fix.
- Blocks the region-split (near/far) evaluation (`conf/eval/region_split.yaml`,
  `region_split_errors`, the `train.py` hook), which has no near-region samples until the
  data extends to Γ.

## 6. FIX LANDED (Phase 5/6)

Ported into `src/OpenLoop/pendulum/nonsmooth.py`:
- `compute_nonsmooth_curve` now tracks 4 spiral arms (polygon crossings + Hungarian
  assignment, fine value step ≤ 0.25, capped at `basin_value_max=35`) and builds the
  upright smooth basin via the reference reflection assembly; arm pair chosen empirically
  by **first-exit grading** (retained value_max ≤ cap; trajectories spiral and re-enter,
  so containment grading is wrong — this was the last porting bug).
- `restrict_trajectory_to_curve` rewritten to `inpolygon` containment (first exit).
- `NonsmoothCurve` carries the basin ring; persisted in the curve `.npz`.

Regression (production code on cached trajectories, `diagnose_restriction_loop.py`):
geometry 1.1 s (was 176 s), retained 88,266 / discarded 54,988, value max **31.19**,
θ∈[−2.2,2.2], ω∈[−3.5,3.5], min-dist-to-Γ 0.005. Regenerated dataset:
`Pendulum_20260609b_8f70cffb...` (32 s); configs repointed (`conf/data/pendulum.yaml`,
`conf/eval/region_split.yaml`).

### Still open — region-split `near` threshold (separate from this bug)
With the fixed data, samples DO reach Γ (dist 0.005–1.8), but `near = d < 2h` still
captures ~0 because `h` = median NN spacing = 0.0011 is dominated by *along-trajectory*
density, not the transverse scale. The band must use a transverse scale or a
distance-distribution percentile, not the isotropic NN spacing. Revisit
`precompute_region_distances.py` / `EvalConfig.band_k` semantics.

### Throwaway debug assets (delete when done reviewing)
`scripts/investigation/{diagnose_restriction_loop,proto_bsp}.py`,
`rawdata/data/_debug_raw_trajectories_256.pkl`, `rawdata/plots/debug_*.png`.
