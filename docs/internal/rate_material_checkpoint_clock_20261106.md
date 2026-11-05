# Physical time in rate-dependent restart identity

Inspection and an executable discriminator found identical checkpoint identities
for otherwise identical second- and hour-labelled Models in pure creep, joint
creep/plasticity and Maxwell Steps when no point recorder was selected and the
temperature history had no clock label. A successful restore could therefore
reinterpret physical time without an identity mismatch.

Bind an explicitly declared `physical_time_unit` into both same-layout and
portable Step identities. Do not infer units when they were never declared.
Keep the state format, solver, material laws and numerical time grid unchanged.
Check live thermal labels before saving or loading, before any output is
published or accepted state is replaced.

Acceptance: solve, save and restore with the same clock; reject a different
clock without changing displacement, material state or accepted time. Exercise
both checkpoint formats and all three consumers. Verify that legacy unlabelled
identities still omit the new key and incompatible thermal labels cannot be
saved.

Compatibility is intentionally explicit: a newly declared clock requires a
matching saved identity. Older archives which never recorded that clock are
not proof of its unit and are not silently upgraded. Finish such existing jobs
with their original version or use a separately reviewed migration; do not
remove unit declarations merely to force restoration. Unlabelled legacy paths
remain unlabelled, not implicitly seconds.

The restart tests also exposed a separate Maxwell v1 history defect: its
identity included the active temperature sample, so a fresh Step with the same
complete thermal path could reject a valid interrupted run. Match the existing
creep policy: compare the full recorded history identity, not the currently
applied sample. Normalize that wrapper in older v1 archives without discarding
the path hash or inventing a clock unit. Portable Maxwell recovery reconstructs
the tangent for the next increment; compare committed state and continued
response, not a completed increment's transient tangent buffer.

Local verification: 153 checkpoint, Maxwell, joint-creep/plasticity and CI-scope
tests passed in 119.93 seconds. The real FEniCSx portable Maxwell driver passed
2-to-1 and 1-to-2 rank recovery sequentially, including collective rejection of
a different clock and unchanged displacement/time. Runs used the TCP provider
and one BLAS/OpenMP thread. The scoped CI route now repeats this restart driver
when its consumer or driver changes. These checks do not extend the serial
small-strain user-material consumer to MPI.
