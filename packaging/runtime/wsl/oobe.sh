#!/bin/bash
set -euo pipefail

default_uid=1000
version=$(cat /usr/lib/agentfem/runtime-version 2>/dev/null || echo unknown)

if getent passwd "$default_uid" >/dev/null; then
  exit 0
fi

echo "AgentFEM Runtime $version"
echo "Create the Linux user used for your simulation projects."
preset_username=${AGENTFEM_UPGRADE_USER:-}
if [[ "$preset_username" =~ ^[a-z_][a-z0-9_-]*$ ]] && [[ "$preset_username" != root ]]; then
  username=$preset_username
  echo "Reusing Linux username '$username' from the previous AgentFEM runtime."
else
  while true; do
    read -r -p "Username: " username
    if [[ "$username" =~ ^[a-z_][a-z0-9_-]*$ ]] && [[ "$username" != root ]]; then
      break
    fi
    echo "Use lowercase letters, numbers, '_' or '-', starting with a letter."
  done
fi

adduser --uid "$default_uid" --disabled-password --gecos "" "$username"
echo "Choose the Linux password used when this account runs sudo."
passwd "$username"
usermod -aG sudo "$username"
mkdir -p "/home/$username/.local/state/agentfem"
chown -R "$username:$username" "/home/$username/.local"

if [[ -n "$preset_username" ]]; then
  # A replacement upgrade restores the previous home first; the installer
  # invokes the same workspace transaction afterwards.
  mkdir -p "/home/$username/AgentFEMProjects"
  chown -R "$username:$username" "/home/$username/AgentFEMProjects"
else
  workspace_path=${AGENTFEM_PROJECTS_HOME:-}
  workspace_args=(workspace --protect)
  if [[ -n "$workspace_path" ]]; then
    workspace_args+=(--path "$workspace_path")
  fi
  if ! runuser -u "$username" -- /opt/conda/bin/agentfem "${workspace_args[@]}" \
      > "/home/$username/.local/state/agentfem/first-workspace.log" 2>&1; then
    echo "Persistent Windows workspace setup did not complete."
    echo "No project data was removed. Run: agentfem workspace --protect"
    mkdir -p "/home/$username/AgentFEMProjects"
    chown -R "$username:$username" "/home/$username/AgentFEMProjects"
  fi
fi

su - "$username" -c "/opt/conda/bin/agentfem doctor > ~/.local/state/agentfem/first-doctor.log 2>&1" || true
echo
echo "AgentFEM is ready. Projects belong in ~/AgentFEMProjects."
echo "This path is protected on the Windows drive when installed with the AgentFEM installer."
echo "Check: agentfem workspace"
echo "Run: agentfem doctor"
