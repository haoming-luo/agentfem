#!/bin/bash
set -euo pipefail

. "$PREFIX/etc/profile.d/conda.sh"
conda activate "$PREFIX"

wheel=""
wheel_count=0
for candidate in "$PREFIX"/share/agentfem/agentfem-*.whl; do
  [ -f "$candidate" ] || continue
  wheel="$candidate"
  wheel_count=$((wheel_count + 1))
done
if [ "$wheel_count" -ne 1 ]; then
  echo "Expected exactly one canonically named AgentFEM wheel" >&2
  exit 1
fi
"$PREFIX/bin/python" -m pip install --no-deps --force-reinstall "$wheel"

# Do not mutate .zshrc or replace the user's Python. A visible launcher opens
# an already activated AgentFEM shell instead.
runtime_parent=$(dirname "$PREFIX")
user_home=$(dirname "$runtime_parent")
applications="$user_home/Applications"
launcher="$applications/AgentFEM Terminal.command"
mkdir -p "$applications" "$user_home/AgentFEMProjects" "$user_home/Library/Logs/AgentFEM"

cat > "$launcher" <<LAUNCHER
#!/bin/zsh
set -u
RUNTIME_PREFIX="$PREFIX"
source "\$RUNTIME_PREFIX/etc/profile.d/conda.sh"
conda activate "\$RUNTIME_PREFIX"
clear
echo "AgentFEM Runtime $INSTALLER_VER"
echo "Projects: \$HOME/AgentFEMProjects"
echo
if [ ! -f "\$HOME/Library/Logs/AgentFEM/doctor-$INSTALLER_VER.ok" ]; then
  agentfem doctor || {
    echo
    echo "The health check needs attention. Copy the report when asking for help."
  }
  touch "\$HOME/Library/Logs/AgentFEM/doctor-$INSTALLER_VER.ok"
fi
cd "\$HOME/AgentFEMProjects"
exec /bin/zsh -i
LAUNCHER
chmod 755 "$launcher"

"$PREFIX/bin/agentfem" doctor > "$user_home/Library/Logs/AgentFEM/install-$INSTALLER_VER.log" 2>&1 || true
rm -f "$wheel"
