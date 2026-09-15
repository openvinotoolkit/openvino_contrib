# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from ov_contrib_license.discovery.cmake import parse_cmake
from ov_contrib_license.discovery.repository import (
    DiscoveryOptions,
    RepositoryDiscovery,
)


class CMakeParserTests(unittest.TestCase):
    def test_multiline_command_and_comments(self) -> None:
        commands = parse_cmake(
            """
            # FetchContent_Declare(ignored GIT_REPOSITORY https://example.invalid/ignored)
            FetchContent_Declare(
                fmt
                GIT_REPOSITORY "https://github.com/fmtlib/fmt.git"
                GIT_TAG 10.2.1 # exact upstream tag
            )
            """
        )

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].name, "fetchcontent_declare")
        self.assertIn("https://github.com/fmtlib/fmt.git", commands[0].arguments)
        self.assertIn("10.2.1", commands[0].arguments)

    def test_balanced_nested_syntax(self) -> None:
        commands = parse_cmake(
            "FetchContent_Declare(foo URL $<IF:$<BOOL:${FLAG}>,https://example/a,https://example/b>)"
        )

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].arguments[0], "foo")

    def test_bracket_comment_is_ignored(self) -> None:
        commands = parse_cmake(
            """
            #[=[
            ExternalProject_Add(ignored URL https://example.invalid/archive)
            ]=]
            find_package(OpenVINO REQUIRED)
            """
        )

        self.assertEqual([command.name for command in commands], ["find_package"])

    def test_unresolved_revision_is_preserved_as_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cmake = root / "modules" / "sample" / "CMakeLists.txt"
            cmake.parent.mkdir(parents=True)
            cmake.write_text(
                """
                FetchContent_Declare(
                    example
                    GIT_REPOSITORY https://github.com/example/project.git
                    GIT_TAG ${EXAMPLE_REVISION}
                )
                """,
                encoding="utf-8",
            )

            inventory = RepositoryDiscovery(root, DiscoveryOptions()).run()

        self.assertEqual(len(inventory.components), 1)
        component = inventory.components[0]
        self.assertIsNone(component.version)
        self.assertEqual(component.relationships[0].value, "FETCHED_AT_BUILD")
        self.assertEqual(
            inventory.findings[0].code, "DISCOVERY_CMAKE_UNRESOLVED_REVISION"
        )
        evidence_details = dict(component.evidence[0].details)
        self.assertEqual(
            evidence_details["unresolved_revision_expression"], "${EXAMPLE_REVISION}"
        )


if __name__ == "__main__":
    unittest.main()
