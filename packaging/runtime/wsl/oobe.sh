#!/bin/bash
set -euo pipefail

default_uid=1000
version=$(cat /usr/lib/agentfem/runtime-version 2>/dev/null || echo unknown)

if getent passwd "$default_uid" >/dev/null; then
  exit 0
fi

echo "AgentFEM Runtime $version"
echo "Create the Linux user used for your simulation projects."
while true; do
  read -r -p "Username: " username
  if [[ "$username" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    break
  fi
  echo "Use lowercase letters, numbers, '_' or '-', starting with a letter."
done

adduser --uid "$default_uid" --disabled-password --gecos "" "$username"
echo "Choose the Linux password used when this account runs sudo."
passwd "$username"
usermod -aG sudo "$username"
mkdir -p "/home/$username/AgentFEMProjects" "/home/$username/.local/state/agentfem"
chown -R "$username:$username" "/home/$username/AgentFEMProjects" "/home/$username/.local"

su - "$username" -c "/opt/conda/bin/agentfem doctor > ~/.local/state/agentfem/first-doctor.log 2>&1" || true
echo
echo "AgentFEM is ready. Projects belong in ~/AgentFEMProjects."
echo "Run: agentfem doctor"
