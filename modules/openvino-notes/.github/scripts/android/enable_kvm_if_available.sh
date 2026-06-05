#!/usr/bin/env bash
set -euo pipefail

if [[ -e /dev/kvm ]]; then
  echo 'KERNEL=="kvm", GROUP="kvm", MODE="0666", OPTIONS+="static_node=kvm"' \
    | sudo tee /etc/udev/rules.d/99-kvm4all.rules
  sudo udevadm control --reload-rules
  sudo udevadm trigger --name-match=kvm
else
  echo "/dev/kvm is not available on this runner"
fi
