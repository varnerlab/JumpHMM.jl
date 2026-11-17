# JumpHMM.jl 0.5.0: refactor to the varnerlab course style

Date: 2026-11-17. Status: draft for Jeff's approval (builder-check answers included).

## Intent

JumpHMM.jl is the synthetic-market generator behind CHEME 5660 week 13 (L13a, L13b) and the
HMM-with-jumps paper. Its calling style, structure, and code do not match the varnerlab course
packages, and Jeff, who maintains it, cannot maintain code he does not understand. Working
through the L13a SPY example made this plain: changing the jump settings takes a 4-argument
`JumpParameters(...)` call and an 8-argument positional `JumpHiddenMarkovModel(...)` call
(example cell 36).

Version 0.5.0 rewrites the whole package in the course style, with plain, documented, commented
code and complete, current documentation. Every caller moves to the new syntax: the L13a and
L13b notebooks, the course MPC code, and every experiment in the paper. The paper code is
rewritten in the same plain style and rerun, so every number in the paper comes from one code
version that a reviewer can reproduce.

## Decisions Jeff made (2026-11-17)

1. Refactor everything, including the parts only the paper uses (copulas, vines, single-index
   and hybrid single-index models, validation, decoding).
2. No compatibility layer and no old code kept in the package. Every line of 0.5.0 is in the
   course style. Old calls stop working.
3. The jump settings (`ϵ`, `λ`, and the episode settings) live in the model, as in the course
   package's `MyHiddenMarkovModelWithJumps`.
4. L13a is updated in place: the same notebook, figures, and numbers. In each code cell, the old
   calls stay as comments above the new calls where that works. Sentences that name old
   functions change to the new names.
5. L13b, the course MPC code, and the paper move to 0.5.0.
6. Callers always write the package name in front: `JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel,
   (...))`. This keeps L13a's existing rule and avoids the course package's own `build`.
7. Paper numbers may change. The paper is not published, and today its seven code folders run
   five different JumpHMM versions, so a reviewer cannot reproduce it with the latest code. After
   the rerun, every number comes from 0.5.0. Updating the paper text to the new numbers is a later
   step, when Jeff asks.
8. All paper code is rewritten in the plain style, not only the lines the new syntax touches,
   because readers of the paper need to understand it.
9. The two other repos that follow JumpHMM's `main` branch (eCornell-AI-finance-lectures and
   modified_cobb_douglas_example) are pinned to the 0.4.1 commit before 0.5.0 reaches `main`.
10. L13b's market model is the SPY model that L13a fits and saves, not the eCornell surrogate.
    L13b's numbers may change; it has not been taught yet.

Released weekly bundles (week-13.0, week-13.1) are not touched; each carries its own
environment pinned to 0.4.1 and keeps working.

## Approaches considered

- One new code base plus a thin layer that maps the old calls onto it, so L13a runs unchanged.
  Rejected by Jeff: the layer is code in a style he does not maintain.
- Keep L13a on 0.4.1 in its own environment. Rejected: the course would depend on two
  JumpHMM versions, the MPC code would have to leave the course package (one environment cannot
  load two versions), and L13b could not read the model file L13a saves.
- Chosen: rewrite the package in place as 0.5.0 and migrate every caller, L13a included.

## Goals

- Every type and function follows the course style (below), so Jeff can read every line.
- Same numbers as 0.4.1 for the package: with the same inputs and seeds, 0.5.0 returns the same
  values as 0.4.1, bit for bit, so L13a's printed results stay valid. A difference means a bug,
  unless Jeff approves it after seeing why.
- Complete documentation: every exported type and function has a course-format docstring, and
  the Documenter site covers the whole package, starting from the L13a workflow.
- All callers run on 0.5.0: L13a, L13b, `code/src/mpc`, the course tests, and every paper
  script.
- The paper is reproducible: every paper environment pins the same JumpHMM 0.5.0 commit, is
  resolved under the same Julia version, and every random step in the paper code takes an
  explicit seed (some runs, such as untuned seeds in `tune`, do not today).

## Non-goals and anti-goals

- No new methods or model changes. This is a rewrite of how the code reads, not of what it
  computes. Known data questions (for example the unadjusted splits in the paper's universe
  data) are out of scope.
- No compatibility shims, deprecation wrappers, or dual names.
- No changes to the released week-13.0 and week-13.1 bundles, and no new L13a example.
- No edits to the paper's `audits/` folder; it keeps old copies on purpose.
- No updates to the paper's text numbers in this project; that comes later. The JFDS
  manuscript is synced later, when Jeff asks.
- Failure even if everything runs: code Jeff or a paper reader cannot follow (inner functions,
  closures, ternary chains, `x || throw` one-liners, `do` blocks, dense broadcasting chains), or
  package numbers that silently change.

## The course style (the standard to follow)

Reference: `VLQuantitativeFinancePackage.jl` (Types.jl, Factory.jl, Compute.jl) and Jeff's
global rules for student code. It applies to the package, the course MPC files that call
JumpHMM, and all paper code.

- **Files:** `Include.jl` (all `using` lines), `Types.jl` (all types), `Factory.jl` (all
  `build` methods), then one file per topic. The module file only includes `Include.jl`,
  the topic files, and lists exports, grouped and commented. Paper experiment folders follow the
  same split for their helper code.
- **Types:** `mutable struct My…Model <: Abstract…` with documented fields and an empty
  constructor `My…Model() = new()`. No inner constructors, no validation inside the struct.
  No derived fields that can go stale (compute `Poisson(λ)` where it is used).
- **Construction:** `build(MyType, (key = value, ...))::MyType` with a NamedTuple; checks of the
  inputs happen in `build`, written as plain `if` blocks that throw `ArgumentError`.
- **Functions:** documented top-level functions; explicit `for` loops and `if` blocks; no inner
  or nested functions, closures, `do` blocks, ternary chains, or `x || throw` / `&& return`
  one-liners. Comments say what each step does, in the course voice (`# initialize -`,
  `# main simulation loop -`, `# return -`).
- **Docstrings:** the course format: the signature, a plain description, `### Arguments`, the
  required NamedTuple keys, `### Returns`, `### Optional Arguments`, and a short example.
- **Names:** course names for settings (`number_of_states`, `number_of_paths`,
  `number_of_steps`, `Δt`), Greek letters where the paper and lecture use them (`ϵ`, `λ`, `ν`,
  `α`, `β`, `δ`, `ψ`).

## The new single-asset syntax

```julia
# estimate the chain and the emissions from growth rates (1/yr); no jump episodes yet
model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = g,
    number_of_states = 100, ν = 5.0, ϵ = 0.0, λ = 1.0, p_neg = 0.52, N_tail = 5, Δt = 1/252));

# tune ϵ and λ on a grid: returns a new model that holds the best pair
tuned_model = JumpHMM.tune(model, g; ϵ_range = ..., λ_range = ..., number_of_paths = 100,
    seed = 5660);

# study other jump settings: build again from the same growth rates (the fit is deterministic)
study_model = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = g,
    number_of_states = 100, ν = 5.0, ϵ = 1e-3, λ = 90.0, p_neg = 0.52, N_tail = 5, Δt = 1/252));

# simulate: a NamedTuple of days × paths matrices
paths = JumpHMM.simulate(study_model, number_of_days; number_of_paths = 100, seed = 5660);
paths.growth_rates   # Array{Float64,2}, 1/yr
paths.states         # Array{Int64,2}
paths.episodes       # Array{Bool,2}, true on jump-episode days
```

My calls, with reasons:
- `build` takes growth rates, not prices; students already compute `g` in L13a, and
  `excess_growth_rates(prices; rf, Δt)` stays as a helper.
- `build` has one NamedTuple shape. Changing the jump settings means building again from the
  data; the fit takes about a second. Because the model is mutable, code that needs many jump
  variants of one fitted model (the paper's factorial runs) may copy the model and set its
  jump fields; the docs show the `build` form.
- The function stays `simulate` (Jeff's word for it). The course package's `sample` would
  collide with `StatsBase.sample`.
- `simulate` returns matrices instead of a vector of path objects, as the course package's
  `sample` does for the geometric Brownian motion model.
- `n_paths` becomes `number_of_paths` everywhere.

## Mapping for the rest of the package (builder fills in the details)

| 0.4.1 | 0.5.0 |
|---|---|
| `LaplacePartition`, `StudentTEmission`, `JumpParameters` | Fields of `MyJumpHiddenMarkovModel` (bin boundaries, emission `μ`, `σ`, `ν`, jump settings), or small `My…Model` types built with `build`; no positional constructors |
| `fit(JumpHiddenMarkovModel, prices; ...)` | `build(MyJumpHiddenMarkovModel, (growth_rates = ..., ...))` |
| `tune`, `simulate`, `validate`, `decode`, `forward_filter`, `log_likelihood` | Same names, course-style signatures, `number_of_paths`; `validate` returns a `MyValidationReportModel` |
| `GaussianCopula`, `StudentTCopula`, `VineCopula`, bivariate families | `MyGaussianCopulaModel`, `MyStudentTCopulaModel`, `MyVineCopulaModel`, `My…BivariateCopulaModel`, each made with `build` |
| `SingleIndexModel`, `HybridSingleIndexModel` | `My…` names that do not collide with the course package's `MySingleIndexModel` (for example `MySingleIndexDependenceModel`) |
| `SectorFactorModel`, `sample_draws`, `compose`, `compose_name`, `with_parameters`, `sample_dependence` | `MySectorFactorModel` made with `build`; the draw-and-compose functions keep their roles with course-style signatures |
| `PortfolioModel`, `PortfolioSimulationResult` | `MyPortfolioModel` made with `build`; `simulate` returns NamedTuples of matrices |
| `excess_growth_rates`, `prices_from_growth_rates`, `ema_rate`, `drift_path`, ... | Same roles, course-style code and docstrings |

The builder settles the field names, the exact NamedTuple keys, and which helpers stay
exported, using the style rules above, and records each choice in the API page of the docs.
The sector and own models are fit from compounded level series today; 0.5.0 keeps that round
trip so their numbers match.

## Verification

1. **Reference outputs from 0.4.1.** Before changing the package, a script in a separate
   environment pinned to 0.4.1 runs every public function on fixed inputs (SPY and a small
   universe from the paper's cached data) with fixed seeds and saves the results as plain arrays
   (JLD2 of `Array`s, no package types).
2. **Package equivalence tests.** A matching script on 0.5.0 compares every result: exact
   equality, or a tolerance of 1e-12 where floating-point order changes and only then. Random
   draws must be consumed in the same order as 0.4.1.
3. **Unit tests.** `test/` is rewritten for the new API and passes.
4. **Docs.** `docs/make.jl` builds with no warnings; every exported symbol appears.
5. **Course.** L13a, L13b, and `code/test/mpc` run on 0.5.0. L13a's re-executed outputs match
   its stored outputs, and every number in its text is unchanged. L13b is re-executed and its
   text numbers are checked against the new outputs.
6. **Paper code rewrite.** Each paper script is first moved to the 0.5.0 API with the smallest
   edit and run on reduced settings (same seeds, fewer paths or replications); the plain-style
   rewrite must reproduce those reduced-run outputs exactly. Then every script runs once at full
   size on 0.5.0.
7. **Paper rerun record.** A table lists, for every number in the arXiv draft's tables, figures,
   and text, the draft value and the 0.5.0 value, so Jeff can update the text later. The
   0-warning build, `verify.py`, `make source`, and the empty-folder compile pass on the rerun
   outputs.

## Migration list

- JumpHMM.jl: `src/`, `test/`, `docs/` (rewrite every page), `README.md`, `CLAUDE.md` (it says
  "all structs are immutable"), `Project.toml` version 0.5.0. The stale notes
  (`SPECIFICATION.md`, `code_review.md`, `jump_hmm_sim_plan.md`) move to an archive folder
  (my call).
- Course: L13a example (in place, old calls commented out, function names in the text updated),
  its `figs/Fig-L13a-Episode-Shape-Data.jl` and saved models in `L13a/data/`; L13b SectorFactor
  and MPC-SyntheticData examples and `scripts/fit_generator.jl`, switched to L13a's SPY model;
  the eCornell surrogate file in `L13b/data/` is no longer used and is removed; `code/src/mpc`
  files that call JumpHMM (`Synthetic.jl`, `MPCPortfolio.jl`) and `code/test/mpc`, in the plain
  style; both JumpHMM pins (the root `Project.toml` and `code/Project.toml`, including its
  `[compat]` entry); every link to the JumpHMM docs in the L13a and L13b notebooks and READMEs,
  because the page anchors change.
- Paper: every script in the experiment folders that use JumpHMM (arxiv-revision,
  baseline-comparison, copula-experiment, downstream-evaluation, gbm-experiment,
  other-ticker-experiment, sim-experiment, spy-experiment), including exploratory scripts whose
  results are not in the draft; all their environments pinned to the same 0.5.0 commit;
  `Revision.jl`'s 0.4.1 check; cached fits that hold old types (for example
  `arxiv-revision/results/fit.jld2`), which are regenerated; the code README's version notes.
  The text's "JumpHMM 0.4.1" mentions (`method_sector.tex`) change with the number update, later.
- Other repos: eCornell-AI-finance-lectures and modified_cobb_douglas_example pinned to the 0.4.1
  commit (`e5821a8`), committed and pushed only with Jeff's go-ahead, before 0.5.0 reaches
  `main`.

## Order of work and checkpoints

1. Reference outputs from 0.4.1.
2. Single-asset core: `Types.jl`, `Factory.jl`, fitting, `simulate`, `tune`, with equivalence
   tests. **Checkpoint: Jeff reads this code before the rest is converted**, so the style is
   settled on a small piece first.
3. The rest of the package, module by module, each with equivalence tests.
4. Docs.
5. Course migration (L13a, then L13b and the MPC code).
6. Paper migration, rewrite, and rerun. **This phase starts after the paper's current
   uncommitted work (the arXiv sections and two downstream-evaluation scripts) is committed;
   ask Jeff for that go-ahead first.** The rerun record (verification 7) is compared with that
   committed draft.
7. Release steps, each with Jeff's go-ahead: pin the two other repos, tag v0.4.1 so its docs stay
   online, push 0.5.0 and tag it, then switch the course and paper pins to the pushed commit.
   Students of the released bundles who follow the old `/dev/` docs links will see the 0.5.0
   docs.

## Constraints

- No commits, pushes, or tags without Jeff's go-ahead (course repo: other sessions edit it
  concurrently, so commit with a temporary index when asked). The course and paper pin JumpHMM by
  GitHub URL and commit, so until the 0.5.0 commit is pushed, development uses a local path
  source.
- Course text follows the course writing rules (no lecture-number references, names before
  symbols, short plain sentences).
- Long paper runs go in the background with logs, and the plan records how long each took.

## Left to the builder

Field names, NamedTuple keys, internal helper layout, file split inside the style rules, the
reference-input choice, the reduced settings for the paper equivalence runs, the order of paper
reruns, and how long runs are scheduled. Each choice is recorded in the docs' API page or the
plan.
