# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass


class ExpressionError(ValueError):
    pass


TOKEN = re.compile(
    r"\s*(\(|\)|AND\b|OR\b|WITH\b|[A-Za-z0-9][A-Za-z0-9.+:-]*)", re.IGNORECASE
)


@dataclass(frozen=True)
class Expression:
    operator: str
    value: str | None = None
    left: Expression | None = None
    right: Expression | None = None

    def render(self, parent_precedence: int = 0) -> str:
        if self.operator == "LICENSE":
            assert self.value is not None
            return self.value
        precedence = {"OR": 1, "AND": 2, "WITH": 3}[self.operator]
        assert self.left is not None and self.right is not None
        rendered = f"{self.left.render(precedence)} {self.operator} {self.right.render(precedence + 1)}"
        return f"({rendered})" if precedence < parent_precedence else rendered

    def identifiers(self) -> tuple[str, ...]:
        if self.operator == "LICENSE":
            assert self.value is not None
            return (self.value,)
        assert self.left is not None and self.right is not None
        return self.left.identifiers() + self.right.identifiers()


def _tokens(value: str) -> tuple[str, ...]:
    result: list[str] = []
    position = 0
    while position < len(value):
        match = TOKEN.match(value, position)
        if not match:
            raise ExpressionError(f"Invalid SPDX expression near {value[position:]!r}")
        result.append(match.group(1))
        position = match.end()
    return tuple(result)


class _Parser:
    def __init__(self, tokens: tuple[str, ...]) -> None:
        self.tokens = tokens
        self.position = 0

    def parse(self) -> Expression:
        if not self.tokens:
            raise ExpressionError("SPDX expression must not be empty")
        result = self._or()
        if self.position != len(self.tokens):
            raise ExpressionError(f"Unexpected token {self.tokens[self.position]!r}")
        return result

    def _or(self) -> Expression:
        result = self._and()
        while self._peek("OR"):
            self.position += 1
            result = Expression("OR", left=result, right=self._and())
        return result

    def _and(self) -> Expression:
        result = self._with()
        while self._peek("AND"):
            self.position += 1
            result = Expression("AND", left=result, right=self._with())
        return result

    def _with(self) -> Expression:
        result = self._primary()
        if self._peek("WITH"):
            self.position += 1
            exception = self._primary()
            if result.operator != "LICENSE" or exception.operator != "LICENSE":
                raise ExpressionError(
                    "WITH requires a license and an exception identifier"
                )
            result = Expression("WITH", left=result, right=exception)
        return result

    def _primary(self) -> Expression:
        if self.position >= len(self.tokens):
            raise ExpressionError("Unexpected end of SPDX expression")
        token = self.tokens[self.position]
        if token == "(":
            self.position += 1
            result = self._or()
            if self.position >= len(self.tokens) or self.tokens[self.position] != ")":
                raise ExpressionError("Missing closing parenthesis in SPDX expression")
            self.position += 1
            return result
        if token.upper() in {"AND", "OR", "WITH"} or token == ")":
            raise ExpressionError(f"Unexpected token {token!r}")
        self.position += 1
        value = "NOASSERTION" if token.upper() == "NOASSERTION" else token
        return Expression("LICENSE", value=value)

    def _peek(self, value: str) -> bool:
        return (
            self.position < len(self.tokens)
            and self.tokens[self.position].upper() == value
        )


def parse_expression(value: str | None) -> Expression:
    return _Parser(_tokens(value or "NOASSERTION")).parse()


def normalize_expression(value: str | None) -> str:
    return parse_expression(value).render()
