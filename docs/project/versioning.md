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

The current package is a public alpha, so the site must not label it as a
stable release. The source documentation for the existing preview remains
available at the
[`v0.2.0a1` tag](https://github.com/haoming-luo/agentfem/tree/v0.2.0a1/docs).

Version deployment is an explicit release operation. Ordinary documentation
checks remain read-only and cannot silently rewrite the published-documentation
branch.
