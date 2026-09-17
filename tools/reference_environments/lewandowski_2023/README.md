# Lewandowski 2023 external J2 reference

This directory provides an auditable rerun recipe for the public legacy
FEniCS/MGIS/MFront beam without installing that historical stack into the
AgentFEM runtime. It is an external-oracle builder, not an AgentFEM solver
image. The base image is digest-pinned; the generated execution manifest
records every resolved package version and the curve content digest.

Download the two files pinned in
`tests/lewandowski_2023_self_weight_beam_fixture.py` into an empty working
directory as `reference.py` and `LogarithmicStrainPlasticity.mfront`. Verify
their SHA-256 digests, copy `run_reference.py` beside them, and create an empty
`results/` directory. Then build and run:

```bash
docker build --platform linux/amd64 \
  -t agentfem/lewandowski-j2-reference:2023 \
  tools/reference_environments/lewandowski_2023

docker run --rm --platform linux/amd64 \
  -e OMPI_MCA_plm=isolated \
  -e OMPI_MCA_btl=self,vader \
  -v "$PWD:/reference" \
  agentfem/lewandowski-j2-reference:2023 \
  python run_reference.py
```

The runner rejects mutable or mismatched source bytes before compiling the
MFront behaviour. Its outputs are `reference_curve.csv` and
`reference_execution.json`.
