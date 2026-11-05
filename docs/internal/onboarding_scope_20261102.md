# Installation and agent UX: bounded initiative

## Product decision

Installation ends with the installed identity, environment readiness and WSL
workspace protection. It does not create a model or run a simulation. Offer one
useful next step, such as a small example or the user's own model, and wait for
their choice. A request that already includes an example is sufficient consent;
do not add another confirmation round.

Everyday use is task-specific: run a project, inspect a result, change a model,
or diagnose a concrete failure. Do not turn all four into a universal checklist.
Reuse known environment/capability information until it changes. Consult the
installed provider's options when the website describes a newer version.

## Research informing the design

- [Microsoft Research, Guidelines for Human-AI Interaction (2019)](https://www.microsoft.com/en-us/research/articles/guidelines-for-human-ai-interaction-eighteen-best-practices-for-human-centered-ai-design/):
  make capabilities clear, time assistance to context, support dismissal and
  limit scope when uncertain. Here, readiness is a visible completion point;
  the user controls whether a suggestion becomes a computation.
- [Google PAIR, Feedback + Control](https://pair.withgoogle.com/guidebook-v2/chapter/feedback-controls/):
  balance automation and user control. Automate the requested installation;
  retain brief handoffs for permission, restart and private account entry.
- [Anthropic, Building effective agents (2024)](https://www.anthropic.com/research/building-effective-agents):
  prefer simple, composable workflows and justify the extra cost of autonomy.
  No new autonomous task queue or agent orchestration framework is needed here.
- [Anthropic, Writing effective tools for agents (2025)](https://www.anthropic.com/research/writing-tools-for-agents):
  clear tool boundaries and relevant, compact responses improve usability;
  evaluate outcomes rather than exact wording. Keep skill discovery short and
  scientific details in references loaded when needed.

These sources motivate design choices, not a measured claim of token savings or
proof that every third-party agent will follow the guidance.

## Changes and acceptance

1. Homepage, installation manual, Windows route, agent guide and generated
   manifest agree on readiness as completion. The optional example remains easy
   to find; developer/release acceptance still executes real solves.
2. The Linux setup helper defaults to no project/solve. `--first-solve` explicitly
   requests the example. Its JSON records the request, readiness, example outcome
   and next action separately. Failure/cancellation remains nonzero and recorded.
3. The source WSL bundle installer calls the verifier in readiness-only mode.
   Older downloaded bundles are unchanged. A readiness report never claims
   a fresh finite-element JIT/solve was exercised.
4. Skill entry routes install, existing-project and result-only tasks without
   loading all material/migration rules. Relevant scientific rules remain in
   the distributed reference directory.
5. Example command sequences use the same public CLI. Result-only work does not
   start a solve. Health/upgrade investigation is conditional, not every-run work.

## UX review

1. **Discover installation — needs correction, addressed.** The public homepage
   still highlighted release 0.3.8 while its repository badge showed 0.4.4.
   Use the stable releases link. Do not advertise an offline artifact that the
   current release does not contain.
2. **Install — scope mismatch, addressed.** The published copyable prompt
   requested a solve even when the user only wanted installation. The new prompt
   ends at readiness; the implementation now matches that boundary.
3. **Continue — usable with a clear choice.** Keep the existing page styling,
   show an optional first example immediately below installation, and link to
   manual/Windows routes. No modal, new account or additional service is added.
4. **Repeat use — reduced overhead.** Use a short task route and optional detail,
   rather than repeatedly requiring doctor, upgrade and a full capability dump.

Public-page and local-preview screenshots were inspected at desktop width.
This is not a complete accessibility audit or a new clean-host Windows install.
Behavior tests cover default/opt-in execution, readiness failure, structured
completion and bundle request scope. Remote PowerShell 5.1/7 CI is required for
the changed installer contract; release CI retains the opt-in installed solve.

Local validation on 2026-11-02: 136 targeted tests passed, generated entrypoints
were current, and the documentation passed a strict build. An isolated wheel
installation completed the static-solid template through check, run, show and
verify: maximum displacement 0.000427424 m and strain energy 42.7303 J. This was
developer acceptance, not a default installation side effect or a new benchmark.
