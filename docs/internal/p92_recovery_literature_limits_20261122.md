# Recovery literature: evidence admitted and inference boundaries

This review supplements the already recorded P92, P91 and 10%Cr references.
It does not treat another material's fitted coefficients as a transferable
P92 card. The current BAM task remains one continuous three-protocol card,
not independent fitting of each hold.

## Microstructural evidence

[Li et al., Acta Materialia (2021), 117130](https://doi.org/10.1016/j.actamat.2021.117130)
reports 316H stress-varying experiments and diffraction/microscopy. The
publisher-indexed text distinguishes relaxation/reversal of internal stress
from dislocation-density changes. Its density measurements are near the
technique's reliability limit; the discussion emphasizes trends. This supports
separating signed backstress memory, flow-driven regeneration and isotropic
resistance, not interpreting their fitted scalar states as measured densities.
Neither its 650 C coefficients nor its microscopic interpretation is proof
of the present P92 mechanism.
The full published text was subsequently retrieved from Empa's repository;
the resistance discriminator records the applicable equations and rejects
its own scalar hypothesis rather than claiming to reproduce that model.

[Comparison of primary creep regeneration and anelastic recovery (2020),
103474](https://doi.org/10.1016/j.mechmat.2020.103474) compares 316H at 650 C
with 10%Cr martensitic steel at 600 C. The accessible publisher abstract
reports loading-history sensitivity and a stronger recoverable response in
the martensitic steel. It reinforces retaining recoverability but supplies
no verified BAM 620 C parameter set. The publisher PDF was not retrieved.

## Existing constitutive routes, not newly discovered solutions

[Hosseini et al. (2020), 105546](https://doi.org/10.1016/j.ijmecsci.2020.105546)
describes a two-term overstress power law, common drag/backstress concepts,
and strain hardening, dynamic recovery and static recovery for 10%Cr steel.
Its accessible introduction makes explicit why a single stress-range law
can fail to transfer. This is a related formulation family, not a validated
replacement for the existing two-flow P92 reference. An unverified secondary
page retrieved under this title contains a different 2018 paper; its equations
and tables are therefore not admitted under the 2020 identity.

[Kazemi et al. (2023), 104821](https://doi.org/10.1016/j.euromechsol.2022.104821)
was already used in the two-time recoverability design. Re-finding it does
not create a new validated mechanism. The accessible formulation excerpt
motivates two Kelvin components and viscoplasticity; our state-dependent
mobility function is expressly our hypothesis, not their published equation.

## Decisions for this batch

- No material-label switch, prescribed measured creep strain, hidden hold
  reset or protocol-specific coefficient.
- Preserve the original directional law, recoverable energy and signed
  component accounting. Independently verify any new kinetic replacement.
- Treat numerical agreement, positive dissipation, lower training cost,
  frozen response accuracy and life qualification as separate statements.
- The new mobility fit cannot be called a reproduction of any cited paper.
  Inspect hold entry, accumulated creep, resolved relaxation secants and
  later cycles separately, even if its joint objective improves.

## Hybrid-control model identity recheck

The publisher's accessible sections of
[Zhang et al. (2022), 107369](https://doi.org/10.1016/j.ijmecsci.2022.107369)
describe a modified, yield-surface-free Walker formulation and 19 identified
parameters for mixed-control experiments. The full constitutive equations
and numerical parameter table remain unavailable in this retrieval. Do not
reconstruct them from the abstract or confuse cited-by articles with its text.

The openly documented [NEML Alloy 617 Walker implementation](https://neml.readthedocs.io/en/main/walker.html)
is **not** that P92 formulation: its flow rule explicitly contains a threshold
Y and Macaulay brackets, and its documentation states that the original
overshoot contribution is omitted. The shared model name is not equation
identity or evidence that switching packages would solve the BAM discrepancy.

The related authors' public
[CN115931532A description](https://patents.google.com/patent/CN115931532A/zh)
is a modified time-fraction life-assessment route using fatigue, creep and
creep-fatigue observations. It is not the missing local material-update law;
no implementation or lifetime claim is inferred from it. No coefficients or
new validation observations were admitted by these searches.

## Public hybrid-control disclosure: useful, not implementation-ready

The separate [CN113742914A disclosure](https://patents.google.com/patent/CN113742914A/zh)
provides equations and a staged four-test identification strategy at 650 C.
It uses two backstresses, state-dependent recovery and accumulated-strain
hardening. Its numerical material card is not supplied in the inspected
15-page document. Its identity with the 2022 paper has not been established.

Visual inspection of PDF pages 7--8 also reveals unresolved implementation
questions: Eq. 3 has no absolute-value bars before the real power; Eq. 4
prints recovery without a multiplier D. The corresponding page 7 of the
[B publication](https://patents.google.com/patent/CN113742914B/zh) retains both.
Do not silently repair the expressions and call that a published reproduction.
Our mathematical checks are that the literal first expression is not generally
real under reversal, and the literal second identifies only the difference
of its first two coefficients. Neither observation proves which equations
were used in the authors' calculations.

The useful design lesson is to constrain reversal, short/long creep and
cycle evolution together. Public availability is not an implementation
license. No formula, parameter, claimed validation or new runtime material
has been imported from this disclosure.

Retrieved A PDF SHA256:
`7f70b94510518beed51e5d08e34360f8443fc314ddca1ceb672c3b8391523efe`.

## Source reproduction still separate from BAM mechanism selection

The indexed full text of [Li et al., 106044](https://doi.org/10.1016/j.ijmecsci.2020.106044)
again reports n2=278 in Table 2 and explicitly points to supplementary MATLAB
code. Crossref identifies the publisher article as `S0020740320319779`.
Direct repository retrieval failed with a TLS connection error, PDF screenshot
retrieval failed, and the publisher returned 403. The source-code supplement
was not obtained. The earlier visual-verification/loading-profile gate stays
open; the provisional card and its retained failure are not changed. Seeking
the original implementation is preferable to choosing a convenient exponent.
