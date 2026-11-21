from codeograph.chunking import chunk_source

PY = '''"""module doc"""
import os

X = 1


@decorator
def top(a, b=2):
    """Adds."""
    return a + b


class Box:
    """A box."""
    size = 3

    def open(self):
        return self.size

    @property
    def area(self):
        """Area."""
        return self.size ** 2


if __name__ == "__main__":
    print(top(1))
'''


def test_python_functions_classes_methods():
    chunks = {c.qualified_name: c for c in chunk_source(PY, "pkg/m.py")}
    assert chunks["top"].symbol_type == "function"
    assert chunks["top"].docstring == "Adds."
    assert chunks["top"].start_line == 7                     # the decorator line is part of the chunk
    assert "@decorator" in chunks["top"].code
    assert chunks["top"].signature.startswith("def top(a, b=2)")
    assert chunks["Box"].symbol_type == "class" and chunks["Box"].docstring == "A box."
    assert "def open(self)" in chunks["Box"].code and "# methods:" in chunks["Box"].code
    assert chunks["Box.open"].symbol_type == "method" and chunks["Box.open"].parent == "Box"
    assert chunks["Box.area"].docstring == "Area."
    module = [c for c in chunks.values() if c.symbol_type == "module"][0]
    assert "import os" in module.code and "def top" not in module.code


TS = '''import { x } from "./x";

/** Loads things. */
export async function load(id: string): Promise<void> {
  return;
}

export const double = (n: number) => n * 2;

class Store {
  get(key: string) { return key; }
  set(key: string, v: number) { }
}

export interface Item { id: string }
'''


def test_typescript_symbols():
    chunks = {c.qualified_name: c for c in chunk_source(TS, "src/a.ts")}
    assert chunks["load"].symbol_type == "function" and chunks["load"].docstring == "Loads things."
    assert chunks["double"].symbol_type == "function"
    assert chunks["Store"].symbol_type == "class"
    assert chunks["Store.get"].symbol_type == "method"
    assert chunks["Item"].symbol_type == "type"
    assert all(c.language == "typescript" for c in chunks.values())


def test_other_languages_get_line_windows():
    src = "\n".join(f"line {i}" for i in range(150))
    chunks = chunk_source(src, "a/b.go")
    assert [c.symbol_type for c in chunks] == ["block"] * len(chunks)
    assert chunks[0].start_line == 1 and chunks[0].end_line == 60
    assert chunks[1].start_line == 51                         # 10-line overlap
    assert chunks[-1].end_line == 150


def test_broken_python_still_indexed():
    chunks = chunk_source("def broken(:\n    pass\n" * 3, "x.py")
    assert chunks                                             # tree-sitter is error-tolerant


def test_embed_text_has_location_and_signature():
    c = [c for c in chunk_source(PY, "pkg/m.py") if c.qualified_name == "top"][0]
    t = c.embed_text()
    assert t.startswith("pkg/m.py :: top (function)") and "Adds." in t
