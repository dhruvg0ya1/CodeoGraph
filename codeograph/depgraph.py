"""The repository's dependency graph, at file AND symbol level.

Nodes are files (`pkg/auth.py`) and symbols (`pkg/auth.py::Session.send`). Edges point from the user to the used:
  file   --imports-->  file      (Python absolute + relative imports and JS/TS relative imports, resolved to files)
  file   --defines-->  symbol
  symbol --calls---->  symbol    (Python: names resolved through local defs, `self.`, imports and module aliases;
                                  JS/TS: calls of names imported from a local module)
So "what breaks if I change X" is a reverse walk from X, and the hop count says how direct the dependency is.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
from collections import deque
from pathlib import PurePosixPath

import networkx as nx

from .chunking import Chunk

JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


# --------------------------------------------------------------------------- python module map
def _module_names(path: str) -> list[str]:
    """Dotted names a file can be imported as: `src/requests/api.py` -> requests.api (and src.requests.api)."""
    p = PurePosixPath(path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    names = []
    for i in range(len(parts)):                      # every suffix, so src/, lib/ or a monorepo prefix resolves
        if parts[i:]:
            names.append(".".join(parts[i:]))
    return names


def build_module_map(py_files: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    # shorter prefixes first so the longest (most specific) path wins only when it is the only candidate
    for f in sorted(py_files, key=lambda x: (x.count("/"), x)):
        for name in _module_names(f):
            mapping.setdefault(name, f)
    return mapping


def _resolve_relative(cur: str, module: str | None, level: int) -> str:
    pkg = PurePosixPath(cur).parent.parts
    base = list(pkg[: len(pkg) - (level - 1)]) if level > 1 else list(pkg)
    return ".".join(base + ([module] if module else []))


class _PyFile(ast.NodeVisitor):
    """Collects defs, imports and calls for one Python file."""

    def __init__(self, path: str, modmap: dict[str, str]):
        self.path, self.modmap = path, modmap
        self.imports: set[str] = set()                    # files imported
        self.aliases: dict[str, str] = {}                 # local name -> "file" (module) or "file::name"
        self.defs: dict[str, str] = {}                    # local qualified name -> node id
        self.calls: list[tuple[str, str]] = []            # (caller node id, raw callee expression)
        self._stack: list[str] = []
        self._class: list[str] = []
        self._types: list[dict[str, str]] = []            # per function: local variable -> class expression

    def _mod_file(self, dotted: str) -> str | None:
        return self.modmap.get(dotted)

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            f = self._mod_file(a.name)
            if f:
                self.imports.add(f)
                self.aliases[a.asname or a.name.split(".")[0]] = f if a.asname else (self._mod_file(a.name.split(".")[0]) or f)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        dotted = _resolve_relative(self.path, node.module, node.level) if node.level else (node.module or "")
        base = self._mod_file(dotted)
        for a in node.names:
            sub = self._mod_file(f"{dotted}.{a.name}") if dotted else None
            if sub:                                        # `from pkg import module`
                self.imports.add(sub)
                self.aliases[a.asname or a.name] = sub
            elif base:                                     # `from module import name`
                self.imports.add(base)
                self.aliases[a.asname or a.name] = f"{base}::{a.name}"

    def _def(self, node: ast.AST, name: str) -> None:
        qual = ".".join(self._class + [name]) if self._class and not self._stack else name
        nid = f"{self.path}::{qual}"
        if not self._stack:                                # top-level function, class or method
            self.defs[qual] = nid
            self._types.append(_local_types(node))
        self._stack.append(nid if not self._stack else self._stack[-1])
        self.generic_visit(node)
        self._stack.pop()
        if not self._stack:
            self._types.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._def(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._stack:                                    # class inside a function: part of that function
            self.generic_visit(node)
            return
        self.defs[".".join(self._class + [node.name])] = f"{self.path}::{'.'.join(self._class + [node.name])}"
        self._class.append(node.name)
        for base in node.bases:                            # inheritance counts as a use of the base class
            self.calls.append((f"{self.path}::{'.'.join(self._class)}", _expr(base)))
        self.generic_visit(node)
        self._class.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self._stack:
            raw = _expr(node.func)
            head, _, rest = raw.partition(".")
            types = self._types[-1] if self._types else {}
            if rest and head in types:                     # cart.total() with `cart: Cart` -> Cart.total
                raw = f"{types[head]}.{rest}"
            self.calls.append((self._stack[-1], raw))
        self.generic_visit(node)


def _local_types(fn: ast.AST) -> dict[str, str]:
    """Variables whose class is known: annotated parameters and `x = SomeClass(...)` assignments."""
    types: dict[str, str] = {}
    args = getattr(fn, "args", None)
    if args is not None:
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            ann = _expr(a.annotation) if a.annotation is not None else ""
            if ann and ann[:1].isupper():
                types[a.arg] = ann
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call):
            cls = _expr(sub.value.func)
            if cls and cls.rsplit(".", 1)[-1][:1].isupper():
                for t in sub.targets:
                    if isinstance(t, ast.Name):
                        types[t.id] = cls
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name) and sub.annotation is not None:
            ann = _expr(sub.annotation)
            if ann and ann[:1].isupper():
                types[sub.target.id] = ann
    return types


def _expr(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _expr(node.value)
        return f"{inner}.{node.attr}" if inner else node.attr
    if isinstance(node, ast.Call):
        return _expr(node.func)
    return ""


def _resolve_py_call(raw: str, caller: str, pf: _PyFile, all_defs: dict[str, dict[str, str]]) -> str | None:
    if not raw:
        return None
    parts = raw.split(".")
    cls = caller.split("::", 1)[1].split(".")[0] if "::" in caller and "." in caller.split("::", 1)[1] else None
    if parts[0] in ("self", "cls") and len(parts) == 2 and cls:
        return pf.defs.get(f"{cls}.{parts[1]}")
    if len(parts) == 1:
        if parts[0] in pf.defs:
            return pf.defs[parts[0]]
        target = pf.aliases.get(parts[0])
        if target and "::" in target:
            f, name = target.split("::", 1)
            return all_defs.get(f, {}).get(name)
        return None
    head = pf.aliases.get(parts[0])
    if head and "::" not in head:                          # module.func
        return all_defs.get(head, {}).get(".".join(parts[1:]))
    if head and "::" in head:                              # ImportedClass.method
        f, name = head.split("::", 1)
        return all_defs.get(f, {}).get(f"{name}.{parts[1]}") or all_defs.get(f, {}).get(name)
    return None


# --------------------------------------------------------------------------- javascript / typescript
_JS_IMPORT = re.compile(r"""(?:import\s+(?P<what>[^'";]+?)\s+from\s+|require\s*\(\s*|import\s*\(\s*)['"](?P<src>\.[^'"]+)['"]""")


def _resolve_js(cur: str, spec: str, files: set[str]) -> str | None:
    base = posixpath.normpath(posixpath.join(posixpath.dirname(cur), spec))
    for cand in [base, *(base + e for e in JS_EXTS), *(f"{base}/index{e}" for e in JS_EXTS)]:
        if cand in files:
            return cand
    stem, ext = posixpath.splitext(base)                    # "./x.js" written for a .ts source
    if ext in (".js", ".jsx"):
        for e in (".ts", ".tsx"):
            if stem + e in files:
                return stem + e
    return None


def _js_imported_names(what: str) -> list[tuple[str, str]]:
    """`{ a, b as c }` / `Def, { x }` / `* as ns` -> [(local, imported)]"""
    out = []
    what = what.strip()
    m = re.search(r"\{([^}]*)\}", what)
    if m:
        for part in m.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if " as " in part:
                imp, loc = (s.strip() for s in part.split(" as ", 1))
            else:
                imp = loc = part.replace("type ", "").strip()
            out.append((loc, imp))
        what = what[: m.start()] + what[m.end():]
    for part in what.split(","):
        part = part.strip()
        if part.startswith("* as "):
            out.append((part[5:].strip(), "*"))
        elif part and re.fullmatch(r"[A-Za-z_$][\w$]*", part):
            out.append((part, "default"))
    return out


# --------------------------------------------------------------------------- build
def build_graph(files: dict[str, str], chunks: list[Chunk]) -> nx.DiGraph:
    """`files` maps repo-relative path -> source text."""
    g = nx.DiGraph()
    for path in files:
        g.add_node(path, kind="file")
    for c in chunks:
        if c.symbol_type in ("module", "block"):
            continue
        nid = f"{c.file_path}::{c.qualified_name}"
        g.add_node(nid, kind=c.symbol_type, file=c.file_path, name=c.symbol_name, line=c.start_line)
        g.add_edge(c.file_path, nid, type="defines")

    # python
    py = [p for p in files if p.endswith(".py")]
    modmap = build_module_map(py)
    parsed: dict[str, _PyFile] = {}
    for p in py:
        try:
            tree = ast.parse(files[p])
        except (SyntaxError, ValueError):
            continue
        v = _PyFile(p, modmap)
        v.visit(tree)
        parsed[p] = v
    all_defs = {p: v.defs for p, v in parsed.items()}
    for p, v in parsed.items():
        for f in v.imports:
            if f != p:
                g.add_edge(p, f, type="imports")
        for caller, raw in v.calls:
            callee = _resolve_py_call(raw, caller, v, all_defs)
            if callee and callee != caller and g.has_node(callee) and g.has_node(caller):
                g.add_edge(caller, callee, type="calls")

    # javascript / typescript
    js_files = {p for p in files if p.endswith(JS_EXTS)}
    by_file: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_file.setdefault(c.file_path, []).append(c)
    for p in js_files:
        src = files[p]
        imported: dict[str, str] = {}
        for m in _JS_IMPORT.finditer(src):
            target = _resolve_js(p, m.group("src"), js_files)
            if not target:
                continue
            g.add_edge(p, target, type="imports")
            for local, name in _js_imported_names(m.group("what") or ""):
                if name not in ("*", "default"):
                    imported[local] = f"{target}::{name}"
        for c in by_file.get(p, []):
            if c.symbol_type in ("module", "block", "class"):
                continue
            caller = f"{p}::{c.qualified_name}"
            for local, callee in imported.items():
                if g.has_node(callee) and re.search(rf"(?<![\w$.]){re.escape(local)}\s*[(<]", c.code):
                    g.add_edge(caller, callee, type="calls")
    return g


# --------------------------------------------------------------------------- queries
def find_nodes(g: nx.DiGraph, target: str) -> list[str]:
    """A file path, a qualified symbol (`Session.send`), a bare name (`send`) or `file::symbol`."""
    target = target.strip().strip("`")
    if g.has_node(target):
        return [target]
    exact = [n for n in g.nodes if "::" in n and n.split("::", 1)[1] == target]
    if exact:
        return exact
    named = [n for n in g.nodes if "::" in n and n.rsplit(".", 1)[-1].split("::")[-1] == target]
    if named:
        return named
    return [n for n in g.nodes if "::" not in n and (n.endswith("/" + target) or n == target)]


def dependents(g: nx.DiGraph, nodes: list[str], max_hops: int = 3) -> list[tuple[str, int, str]]:
    """Reverse BFS over imports + calls. Returns (node, hops, edge type that reached it), nearest first."""
    seen: dict[str, tuple[int, str]] = {}
    start = set(nodes)
    for n in nodes:                         # a file's dependents include those of every symbol it defines
        if g.nodes[n].get("kind") == "file":
            start |= {s for _, s, d in g.out_edges(n, data=True) if d.get("type") == "defines"}
    q = deque((n, 0) for n in start)
    while q:
        node, hops = q.popleft()
        if hops >= max_hops:
            continue
        for pred, _, data in g.in_edges(node, data=True):
            et = data.get("type")
            if et not in ("calls", "imports") or pred in start or pred in seen:
                continue
            seen[pred] = (hops + 1, et)
            q.append((pred, hops + 1))
    return sorted(((n, h, t) for n, (h, t) in seen.items()), key=lambda x: (x[1], x[0]))


def to_json(g: nx.DiGraph) -> str:
    return json.dumps(nx.node_link_data(g, edges="edges"))


def from_json(text: str) -> nx.DiGraph:
    return nx.node_link_graph(json.loads(text), directed=True, edges="edges")
