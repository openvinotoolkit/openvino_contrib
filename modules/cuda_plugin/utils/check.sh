#!/bin/bash

set -exuo pipefail

if [[ $(type -P clang-format-12) ]]; then
	ver=-12
elif [[ $(type -P clang-format-11) ]]; then
	ver=-11
elif [[ $(type -P clang-format-10) ]]; then
	ver=-10
elif [[ $(type -P clang-format-9) ]]; then
	ver=-9
elif [[ $(type -P clang-format) ]]; then
	ver=
else
	echo "no clang-format-10/clang-format-9/clang-format found in PATH"
	exit 1
fi

cd "$(git rev-parse --show-toplevel)"

git diff --diff-filter=ACMR -U0 origin/develop | perl -ne '
  if (m|^\+\+\+ b/(.*)|) {
    $newname = $1;
    if ($name =~ /^.+(\.cpp)|(\.hpp)|(\.h)|(\.cu)|(\.cuh)$/) {
      print "clang-format'$ver'$lines -Werror -dry-run -style=file $name\n"
    }
    $name = $newname;
    $lines = "";
  }
  if (m|^@@.*\+(\d+),(\d+)|) {
    $to = $1 + $2;
    $lines = "$lines -lines=$1:$to"
  }
  END {
    if ($name =~ /^.+(\.cpp)|(\.hpp)|(\.h)|(\.cu)|(\.cuh)$/) {
      print "clang-format'$ver'$lines -Werror -dry-run -style=file $name\n"
    }
  }' | parallel
