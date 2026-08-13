# Documentation versions

AgentFEM documentation is prepared for two distinct channels:

- **Development (`dev`)** follows the current `main` branch and may describe
  interfaces that are still changing.
- **Released versions** are immutable snapshots built from Git tags. A stable
  release can receive the `latest` alias; a prerelease can receive `preview`.

The header version selector is provided by Mike through the standard Material
for MkDocs version contract. The selector becomes multi-version as soon as the
first documentation snapshots are published to the documentation branch.

## Publishing policy

```text
main branch        → dev
vX.Y.ZaN/bN/rcN    → version + preview alias
vX.Y.Z             → version + latest alias
```

Version `0.2.0` is the first non-prerelease package and receives the `latest`
documentation alias. Capability maturity remains independent of the package
version: experimental formulations remain labelled experimental in every
released documentation snapshot. Earlier preview documentation remains
available from the corresponding Git tags.

Version deployment is an explicit release operation. Ordinary documentation
checks remain read-only and cannot silently rewrite the published-documentation
branch.
