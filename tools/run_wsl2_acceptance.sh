#!/usr/bin/env bash
set -euo pipefail

# Run this script inside WSL2, from an AgentFEM checkout and an activated
# FEniCSx environment.  It deliberately refuses native Linux and WSL1.
kernel_release="$(uname -r | tr '[:upper:]' '[:lower:]')"
case "${kernel_release}" in
  *microsoft-standard-wsl2*|*wsl2*) ;;
  *)
    echo "WSL2 acceptance requires a microsoft-standard-WSL2 kernel; found: ${kernel_release}" >&2
    exit 2
    ;;
esac

python_bin="${PYTHON:-python}"
evidence_dir="${1:-wsl2-acceptance}"
mkdir -p "${evidence_dir}"
dist_dir="$(mktemp -d)"
trap 'rm -rf "${dist_dir}"' EXIT

"${python_bin}" -m build --outdir "${dist_dir}"
"${python_bin}" release_gate.py \
  --dist "${dist_dir}" \
  --smoke \
  --mpi-ranks 2 \
  --require-platform wsl2 \
  --report "${evidence_dir}/agent-acceptance.json" \
  --platform-report "${evidence_dir}/platform-acceptance-wsl2.json"

"${python_bin}" promotion_gate.py \
  --evidence "${evidence_dir}/agent-acceptance.json" \
  --evidence "${evidence_dir}/platform-acceptance-wsl2.json" \
  --report "${evidence_dir}/promotion-audit-wsl2.json"

echo "WSL2 acceptance evidence: ${evidence_dir}"
