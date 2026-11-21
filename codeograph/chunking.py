"""Source files -> semantic chunks.

Python, JavaScript, TypeScript and TSX are parsed with Tree-sitter: every function, class and method becomes its
own chunk with its qualified name (`Session.send`), signature, docstring and exact line range. A class also gets a
compact "outline" chunk (header, docstring, attributes and method signatures) so "what does class X do" questions
find it. Code outside any definition becomes one module chunk. Every other language is cut into overlapping line
windows, so nothing in the repository is invisible to search.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tree_sitter import Language, Node, Parser

LANGUAGE_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp", ".cc": "cpp",
    ".c": "c", ".h": "c", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php", ".kt": "kotlin",
    ".swift": "swift", ".scala": "scala",
}
WINDOW, OVERLAP = 60, 10          # line windows for languages without a grammar here
MAX_EMBED_CHARS = 2000            # what the embedder sees of a long chunk (BM25 always sees all of it)


@dataclass
class Chunk:
    file_path: str
    symbol_name: str               # short name: "send"
    qualified_name: str            # "Session.send" for a method, "send" for a function
    symbol_type: str               # function | method | class | module | type | block
    language: str
    start_line: int                # 1-based, inclusive
    end_line: int
    code: str
    signature: str = ""
    docstring: str = ""
    parent: str = ""               # enclosing class for a method
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"{self.file_path}:{self.start_line}-{self.end_line}:{self.qualified_name}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(f"{self.file_path}\0{self.qualified_name}\0{self.code}".encode()).hexdigest()

    def embed_text(self) -> str:
        """What the dense embedder sees: where it lives and what it is, then the code."""
        head = f"{self.file_path} :: {self.qualified_name} ({self.symbol_type})"
        parts = [head, self.signature, self.docstring, self.code]
        return "\n".join(p for p in parts if p)[:MAX_EMBED_CHARS]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# --------------------------------------------------------------------------- grammars
def _load_parsers() -> dict[str, Parser]:
    parsers: dict[str, Parser] = {}
    try:
        import tree_sitter_python as tspy
        parsers["python"] = Parser(Language(tspy.language()))
    except ImportError:  # pragma: no cover
        pass
    try:
        import tree_sitter_javascript as tsjs
        parsers["javascript"] = Parser(Language(tsjs.language()))
    except ImportError:  # pragma: no cover
        pass
    try:
        import tree_sitter_typescript as tsts
        parsers["typescript"] = Parser(Language(tsts.language_typescript()))
        parsers["tsx"] = Parser(Language(tsts.language_tsx()))
    except ImportError:  # pragma: no cover
        pass
    return parsers


PARSERS = _load_parsers()


def _text(node: Node | None) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None and node.text is not None else ""


def _lines(src_lines: list[str], node: Node) -> str:
    return "\n".join(src_lines[node.start_point[0]: node.end_point[0] + 1])


# --------------------------------------------------------------------------- python
def _py_docstring(body: Node | None) -> str:
    if body is None:
        return ""
    for stmt in body.children:
        if stmt.type == "expression_statement" and stmt.children and stmt.children[0].type == "string":
            raw = _text(stmt.children[0])
            for q in ('"""', "'''", '"', "'"):
                if raw.startswith(q) and raw.endswith(q) and len(raw) >= 2 * len(q):
                    return raw[len(q): -len(q)].strip()
            return raw.strip()
        if stmt.type not in ("comment",):
            break
    return ""


def _py_signature(node: Node, src: bytes) -> str:
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    return src[node.start_byte:end].decode("utf-8", errors="replace").strip().rstrip(":").strip()


def _chunk_python(src: str, path: str) -> list[Chunk]:
    data = src.encode("utf-8")
    tree = PARSERS["python"].parse(data)
    lines = src.splitlines()
    chunks: list[Chunk] = []
    covered: set[int] = set()

    def unwrap(node: Node) -> tuple[Node, Node]:
        """(outer node incl. decorators, the def/class itself)"""
        if node.type == "decorated_definition":
            inner = node.child_by_field_name("definition")
            return node, inner if inner is not None else node
        return node, node

    def add_def(outer: Node, node: Node, parent: str = "") -> None:
        name = _text(node.child_by_field_name("name")) or "<anonymous>"
        body = node.child_by_field_name("body")
        start, end = outer.start_point[0] + 1, outer.end_point[0] + 1
        if node.type == "class_definition":
            methods, header_end = [], end
            for child in body.children if body is not None else []:
                o, d = unwrap(child)
                if d.type == "function_definition":
                    methods.append((o, d))
                    header_end = min(header_end, o.start_point[0])
            header = "\n".join(lines[start - 1: max(start, header_end)])
            sigs = [f"    {_py_signature(d, data)}" for _, d in methods]
            outline = header + ("\n    # methods:\n" + "\n".join(sigs) if sigs else "")
            chunks.append(Chunk(path, name, name, "class", "python", start, end, outline,
                                signature=_py_signature(node, data), docstring=_py_docstring(body)))
            for o, d in methods:
                add_def(o, d, parent=name)
        else:
            qual = f"{parent}.{name}" if parent else name
            chunks.append(Chunk(path, name, qual, "method" if parent else "function", "python", start, end,
                                _lines(lines, outer), signature=_py_signature(node, data),
                                docstring=_py_docstring(body), parent=parent))
        if not parent:
            covered.update(range(start - 1, end))

    for child in tree.root_node.children:
        outer, node = unwrap(child)
        if node.type in ("function_definition", "class_definition"):
            add_def(outer, node)
    _add_module_chunk(chunks, lines, covered, path, "python")
    return chunks


# --------------------------------------------------------------------------- javascript / typescript
_JS_FUNCS = {"function_declaration", "generator_function_declaration"}
_JS_TYPES = {"interface_declaration", "type_alias_declaration", "enum_declaration"}
_JS_FN_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}


def _js_doc(node: Node) -> str:
    prev = node.prev_sibling
    if prev is not None and prev.type == "comment" and prev.end_point[0] >= node.start_point[0] - 1:
        doc = _text(prev).strip()
        if doc.startswith("/**"):
            return "\n".join(ln.strip(" */") for ln in doc.splitlines()).strip()
        return doc.lstrip("/ ").strip()
    return ""


def _js_signature(node: Node, src: bytes) -> str:
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else min(node.end_byte, node.start_byte + 200)
    return " ".join(src[node.start_byte:end].decode("utf-8", errors="replace").split())


def _chunk_js(src: str, path: str, language: str) -> list[Chunk]:
    data = src.encode("utf-8")
    tree = PARSERS[language].parse(data)
    lines = src.splitlines()
    lang_label = "typescript" if language == "tsx" else language
    chunks: list[Chunk] = []
    covered: set[int] = set()

    def emit(outer: Node, node: Node, name: str, kind: str, parent: str = "") -> None:
        start, end = outer.start_point[0] + 1, outer.end_point[0] + 1
        qual = f"{parent}.{name}" if parent else name
        chunks.append(Chunk(path, name, qual, kind, lang_label, start, end, _lines(lines, outer),
                            signature=_js_signature(node, data), docstring=_js_doc(outer), parent=parent))
        if not parent:
            covered.update(range(start - 1, end))

    def handle(outer: Node, node: Node) -> None:
        t = node.type
        if t in _JS_FUNCS:
            emit(outer, node, _text(node.child_by_field_name("name")) or "<anonymous>", "function")
        elif t in ("class_declaration", "abstract_class_declaration"):
            cname = _text(node.child_by_field_name("name")) or "<anonymous>"
            body = node.child_by_field_name("body")
            start, end = outer.start_point[0] + 1, outer.end_point[0] + 1
            members = [m for m in (body.children if body is not None else []) if m.type == "method_definition"]
            sigs = [f"  {_js_signature(m, data)}" for m in members]
            head = lines[start - 1] if start - 1 < len(lines) else ""
            outline = head + ("\n  // methods:\n" + "\n".join(sigs) if sigs else "")
            chunks.append(Chunk(path, cname, cname, "class", lang_label, start, end, outline,
                                signature=_js_signature(node, data), docstring=_js_doc(outer)))
            covered.update(range(start - 1, end))
            for m in members:
                emit(m, m, _text(m.child_by_field_name("name")) or "<anonymous>", "method", parent=cname)
        elif t in _JS_TYPES:
            emit(outer, node, _text(node.child_by_field_name("name")) or "<anonymous>", "type")
        elif t in ("lexical_declaration", "variable_declaration"):
            for decl in node.children:
                if decl.type != "variable_declarator":
                    continue
                value = decl.child_by_field_name("value")
                if value is not None and value.type in _JS_FN_VALUES:
                    emit(outer, value, _text(decl.child_by_field_name("name")) or "<anonymous>", "function")
        elif t == "export_statement":
            decl = node.child_by_field_name("declaration")
            if decl is not None:
                handle(outer, decl)

    for child in tree.root_node.children:
        handle(child, child)
    _add_module_chunk(chunks, lines, covered, path, lang_label)
    return chunks


# --------------------------------------------------------------------------- shared
def _add_module_chunk(chunks: list[Chunk], lines: list[str], covered: set[int], path: str, lang: str) -> None:
    rest = [ln for i, ln in enumerate(lines) if i not in covered]
    code = "\n".join(rest).strip()
    if code and sum(1 for ln in rest if ln.strip()) >= 3:
        chunks.append(Chunk(path, Path(path).stem, Path(path).stem, "module", lang, 1, max(1, len(lines)), code))


def _chunk_windows(src: str, path: str, language: str) -> list[Chunk]:
    lines = src.splitlines()
    out: list[Chunk] = []
    if not any(ln.strip() for ln in lines):
        return out
    step = WINDOW - OVERLAP
    for start in range(0, max(1, len(lines)), step):
        seg = lines[start: start + WINDOW]
        if not any(ln.strip() for ln in seg):
            continue
        s, e = start + 1, start + len(seg)
        name = f"{Path(path).stem}#L{s}"
        out.append(Chunk(path, name, name, "block", language, s, e, "\n".join(seg)))
        if start + WINDOW >= len(lines):
            break
    return out


def chunk_source(src: str, path: str) -> list[Chunk]:
    """Chunks for one file's text. `path` is repo-relative with forward slashes."""
    language = LANGUAGE_BY_EXT.get(Path(path).suffix.lower(), "text")
    try:
        if language == "python" and "python" in PARSERS:
            return _chunk_python(src, path)
        if language in ("javascript", "typescript", "tsx") and language in PARSERS:
            return _chunk_js(src, path, language)
    except Exception:  # a grammar crash on one odd file must not stop an index
        pass
    return _chunk_windows(src, path, "typescript" if language == "tsx" else language)
