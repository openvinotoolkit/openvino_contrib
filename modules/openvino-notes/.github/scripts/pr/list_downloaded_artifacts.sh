#!/usr/bin/env bash
set -euo pipefail

find "${1:-pr-artifacts}" -maxdepth 4 -type f | sort
