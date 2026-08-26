"""A tiny, safe expression language for claims an agent makes about artifacts.

This exists because of one live finding. `p6_observer` reported:

    hierarchy_meta records n_leaves=29 but leaves_per_family sums to 32

That is not an opinion. It is arithmetic over an artifact, and it was true — but
the pipeline had no way to know, so the gate warned, the run finished, and a human
verified it by hand the next day. The observer did not need *permission* to act;
it needed a way to be **proven right**.

So an observation may now carry a `check`: an expression over the same artifacts
it cites, which the pipeline evaluates.

    sum(hierarchy_meta.leaves_per_family.values()) == hierarchy_meta.n_leaves

- evaluates **False** -> the defect is real, measured, and may block. It is no
  longer an LLM claim; it is a failed assertion.
- evaluates **True**  -> the observation is refuted by its own test and dropped.
  This is the agent's own false-positive filter, and it costs nothing.
- does not parse, or names something absent -> nothing is claimed either way and
  the observation stays advisory, exactly as before.

**Why an evaluator and not `eval`.** The string comes from a model. Every node
type is whitelisted, there are no imports, no lambdas, no assignment, and no
Python attribute access at all: `a.b` is a **dict lookup**, never `getattr`, so
`__class__` and friends resolve to nothing rather than to an object. Calls are
restricted to a fixed table of pure functions. The failure mode of an expression
this module cannot handle is `Unverifiable`, never an exception escaping into the
run and never code executing.

**What a check may not do.** It reads artifacts and returns a boolean. It cannot
write, cannot re-run a phase, and cannot change a parameter. The authority a
confirmed check earns is exactly the authority a failing assertion has: it stops
the line and names what is wrong. Deciding what to do about it stays with the
operator — see `ops/findings.py` for how a confirmed finding survives until
somebody does.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any

#: An expression longer than this is not a check, it is a program.
MAX_EXPR_CHARS = 400
#: Bound on parsed nodes, so a pathological expression cannot burn the run.
MAX_NODES = 250
#: Bound on comprehension iterations, for the same reason.
MAX_ITER = 20000


class CheckError(RuntimeError):
    """The expression could not be evaluated. Never a verdict about the claim."""


@dataclass
class CheckResult:
    """The outcome of evaluating one check.

    `verdict` is deliberately three-valued. Collapsing `unverifiable` into
    `refuted` would silently discard every claim whose test we could not run —
    turning a gap in the checker into apparent good news, which is the failure
    mode a guardrail can least afford.
    """

    verdict: str                      # "confirmed" | "refuted" | "unverifiable"
    expression: str = ""
    value: Any = None
    detail: str = ""

    @property
    def confirmed(self) -> bool:
        return self.verdict == "confirmed"

    def as_record(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "expression": self.expression,
                "value": _plain(self.value), "detail": self.detail}


def _plain(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in list(v.items())[:20]}
    if isinstance(v, (list, tuple, set)):
        return [_plain(x) for x in list(v)[:20]]
    return str(v)[:200]


# ------------------------------------------------------------------ functions
def _values(x: Any) -> list[Any]:
    return list(x.values()) if isinstance(x, dict) else list(x)


def _keys(x: Any) -> list[Any]:
    return list(x.keys()) if isinstance(x, dict) else list(range(len(x)))


def _items(x: Any) -> list[Any]:
    return [list(kv) for kv in x.items()] if isinstance(x, dict) else list(enumerate(x))


def _count(seq: Any, val: Any = True) -> int:
    return sum(1 for v in seq if v == val)


#: Pure, total, and bounded. Anything that could mutate, import, format, or reach
#: an object's internals is absent by construction rather than by filtering.
FUNCS: dict[str, Any] = {
    "len": lambda x: len(x), "sum": lambda x: sum(x), "abs": abs,
    "min": lambda *a: min(a[0]) if len(a) == 1 else min(a),
    "max": lambda *a: max(a[0]) if len(a) == 1 else max(a),
    "round": round, "int": int, "float": float, "str": str, "bool": bool,
    "set": lambda x: set(x), "sorted": lambda x: sorted(x), "list": lambda x: list(x),
    "any": lambda x: any(x), "all": lambda x: all(x),
    "values": _values, "keys": _keys, "items": _items, "count": _count,
    "isfinite": lambda x: isinstance(x, (int, float)) and math.isfinite(x),
}

#: Method-style spellings an agent naturally writes: `d.values()`. Resolved to the
#: same functions. Being tolerant here costs nothing — a spelling we reject is a
#: real finding we fail to prove — while the safety comes from the whitelist, not
#: from the syntax.
METHODS = {"values": _values, "keys": _keys, "items": _items, "count": _count}

_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b if b else float("nan"),
    ast.FloorDiv: lambda a, b: a // b if b else float("nan"),
    ast.Mod: lambda a, b: a % b if b else float("nan"),
    ast.Pow: lambda a, b: a ** b,
}
_CMPOPS = {
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}

_ALLOWED = (
    ast.Expression, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Constant,
    ast.Name, ast.Attribute, ast.Subscript, ast.Call, ast.List, ast.Tuple, ast.Set,
    ast.Dict, ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.comprehension,
    ast.Load, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd, ast.Slice,
    *_BINOPS, *_CMPOPS,
)

_MISSING = object()


class _Eval:
    def __init__(self, artifacts: dict[str, Any]) -> None:
        self.a = artifacts
        self.scope: dict[str, Any] = {}
        self.iters = 0

    # -- resolution -------------------------------------------------------
    def _lookup(self, obj: Any, key: Any) -> Any:
        """Dict/sequence lookup ONLY.

        Never `getattr`. `x.__class__` is a key that does not exist in a JSON
        artifact, so it resolves to `_MISSING` and the check becomes
        unverifiable — the same as any other typo — instead of reaching a Python
        object. That property is what makes this safe to run on model output.
        """
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            sk = str(key)
            return obj[sk] if sk in obj else _MISSING
        if isinstance(obj, (list, tuple)):
            try:
                i = int(key)
            except (TypeError, ValueError):
                return _MISSING
            return obj[i] if -len(obj) <= i < len(obj) else _MISSING
        return _MISSING

    def visit(self, n: ast.AST) -> Any:
        if not isinstance(n, _ALLOWED):
            raise CheckError(f"{type(n).__name__} is not allowed in a check")
        m = getattr(self, "_" + type(n).__name__, None)
        if m is None:
            raise CheckError(f"{type(n).__name__} is not allowed in a check")
        return m(n)

    # -- nodes ------------------------------------------------------------
    def _Expression(self, n: ast.Expression) -> Any:
        return self.visit(n.body)

    def _Constant(self, n: ast.Constant) -> Any:
        return n.value

    def _Name(self, n: ast.Name) -> Any:
        if n.id in self.scope:
            return self.scope[n.id]
        if n.id in self.a:
            return self.a[n.id]
        if n.id in FUNCS:
            return FUNCS[n.id]
        raise CheckError(f"{n.id!r} is not one of this phase's artifacts")

    def _Attribute(self, n: ast.Attribute) -> Any:
        base = self.visit(n.value)
        got = self._lookup(base, n.attr)
        if got is _MISSING:
            if n.attr in METHODS:
                return ("__method__", METHODS[n.attr], base)
            raise CheckError(f"{n.attr!r} is not a key of the cited artifact")
        return got

    def _Subscript(self, n: ast.Subscript) -> Any:
        base = self.visit(n.value)
        sl = n.slice
        if isinstance(sl, ast.Slice):
            lo = self.visit(sl.lower) if sl.lower else None
            hi = self.visit(sl.upper) if sl.upper else None
            try:
                return base[lo:hi]
            except TypeError as exc:
                raise CheckError(str(exc)) from exc
        got = self._lookup(base, self.visit(sl))
        if got is _MISSING:
            raise CheckError("subscript does not resolve in the cited artifact")
        return got

    def _Call(self, n: ast.Call) -> Any:
        if n.keywords:
            raise CheckError("keyword arguments are not allowed in a check")
        fn = self.visit(n.func)
        args = [self.visit(a) for a in n.args]
        if isinstance(fn, tuple) and fn and fn[0] == "__method__":
            _, f, base = fn
            return f(base, *args)
        if not callable(fn) or fn not in FUNCS.values():
            raise CheckError("only the built-in check functions may be called")
        try:
            return fn(*args)
        except (TypeError, ValueError, ZeroDivisionError, AttributeError) as exc:
            raise CheckError(f"{type(exc).__name__}: {exc}") from exc

    def _BinOp(self, n: ast.BinOp) -> Any:
        return _BINOPS[type(n.op)](self.visit(n.left), self.visit(n.right))

    def _UnaryOp(self, n: ast.UnaryOp) -> Any:
        v = self.visit(n.operand)
        return (not v) if isinstance(n.op, ast.Not) else (-v if isinstance(n.op, ast.USub) else +v)

    def _BoolOp(self, n: ast.BoolOp) -> Any:
        vals = [self.visit(v) for v in n.values]
        return all(vals) if isinstance(n.op, ast.And) else any(vals)

    def _Compare(self, n: ast.Compare) -> Any:
        left = self.visit(n.left)
        for op, cmp in zip(n.ops, n.comparators):
            right = self.visit(cmp)
            fn = _CMPOPS.get(type(op))
            if fn is None:
                raise CheckError(f"{type(op).__name__} is not allowed in a check")
            try:
                if not fn(left, right):
                    return False
            except TypeError as exc:
                raise CheckError(f"cannot compare those values: {exc}") from exc
            left = right
        return True

    def _List(self, n: ast.List) -> Any:
        return [self.visit(e) for e in n.elts]

    def _Tuple(self, n: ast.Tuple) -> Any:
        return tuple(self.visit(e) for e in n.elts)

    def _Set(self, n: ast.Set) -> Any:
        return {self.visit(e) for e in n.elts}

    def _Dict(self, n: ast.Dict) -> Any:
        return {self.visit(k): self.visit(v) for k, v in zip(n.keys, n.values) if k}

    def _comp(self, n: Any) -> list[Any]:
        if len(n.generators) != 1:
            raise CheckError("a check may use at most one `for`")
        gen = n.generators[0]
        if gen.is_async or not isinstance(gen.target, ast.Name):
            raise CheckError("only a simple `for x in ...` is allowed")
        out = []
        for item in self.visit(gen.iter):
            self.iters += 1
            if self.iters > MAX_ITER:
                raise CheckError("check iterates over too much data")
            self.scope[gen.target.id] = item
            if all(self.visit(c) for c in gen.ifs):
                out.append(self.visit(n.elt))
        self.scope.pop(gen.target.id, None)
        return out

    _ListComp = _GeneratorExp = _comp

    def _SetComp(self, n: ast.SetComp) -> Any:
        return set(self._comp(n))


def evaluate(expression: str, artifacts: dict[str, Any]) -> CheckResult:
    """Evaluate one check against a phase's artifacts.

    The claim is CONFIRMED when the expression is **False**: an observation is a
    report of something wrong, and its check is the assertion that should have
    held. Reading it the other way round would make every silent artifact look
    like a defect.
    """
    expr = (expression or "").strip().strip("`").strip()
    if not expr:
        return CheckResult("unverifiable", detail="no check supplied")
    if len(expr) > MAX_EXPR_CHARS:
        return CheckResult("unverifiable", expr[:80], detail="check expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return CheckResult("unverifiable", expr, detail=f"not a valid expression: {exc.msg}")
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        return CheckResult("unverifiable", expr, detail="check expression too complex")
    try:
        val = _Eval(artifacts).visit(tree)
    except CheckError as exc:
        return CheckResult("unverifiable", expr, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 — a check must never break the run
        return CheckResult("unverifiable", expr, detail=f"{type(exc).__name__}: {exc}")
    if not isinstance(val, bool):
        # A check that returns 32 has not settled anything, and treating a
        # truthy number as "the assertion held" would silently confirm nothing.
        return CheckResult("unverifiable", expr, value=val,
                           detail=f"check returned {type(val).__name__}, not a true/false claim")
    return CheckResult("refuted" if val else "confirmed", expr, value=val,
                       detail="the assertion held — the observation is refuted by its own test"
                       if val else "the assertion FAILED — the observation is measured, not claimed")
