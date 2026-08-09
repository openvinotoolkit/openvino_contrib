# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Safely extract a bounded module build artifact."""

import tarfile
from pathlib import Path

MAX_MEMBERS = 100_000
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024


def extract_archive(
    archive: Path,
    destination: Path,
    max_members: int = MAX_MEMBERS,
    max_extracted_bytes: int = MAX_ARTIFACT_BYTES,
) -> None:
    if not archive.is_file():
        raise ValueError(f"artifact archive not found: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"destination must be empty: {destination}")

    with tarfile.open(archive, "r:gz") as source:
        members = []
        extracted_bytes = 0
        for member in source:
            members.append(member)
            if len(members) > max_members:
                raise ValueError(f"archive contains more than {max_members} members")
            if member.isreg():
                extracted_bytes += member.size
                if extracted_bytes > max_extracted_bytes:
                    raise ValueError(f"archive expands beyond {max_extracted_bytes} bytes")
        source.extractall(destination, members=members, filter="data")
