from __future__ import annotations

from google.genai import types


_LIST_TREE = types.FunctionDeclaration(
    name="list_tree",
    description=(
        "List directories and files under `path`, up to `max_depth` levels deep. "
        "Use this to get oriented in an unfamiliar repo before reading any file."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "path": types.Schema(
                type="STRING",
                description="Directory relative to repo root. Use '.' for the root.",
            ),
            "max_depth": types.Schema(
                type="INTEGER",
                description="How many levels deep to descend (1-4). Default 2.",
            ),
        },
        required=["path"],
    ),
)

_READ_FILE = types.FunctionDeclaration(
    name="read_file",
    description=(
        "Read a contiguous range of lines from a text file. Lines are returned "
        "with line numbers prefixed (e.g. '42│def foo():'). USE THIS to confirm "
        "the exact lines and contents you intend to cite. Always read enough "
        "context around the lines you'll cite."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "path": types.Schema(
                type="STRING",
                description="File path relative to repo root.",
            ),
            "start_line": types.Schema(
                type="INTEGER",
                description="1-indexed first line to read. Default 1.",
            ),
            "end_line": types.Schema(
                type="INTEGER",
                description=(
                    "1-indexed last line to read (inclusive). Omit to read up to "
                    "800 lines from start_line."
                ),
            ),
        },
        required=["path"],
    ),
)

_GREP = types.FunctionDeclaration(
    name="grep",
    description=(
        "Search for a Python regex `pattern` across text files. Returns up to 60 "
        "matches with file path, line number, and the matching line. Use this to "
        "locate symbols, imports, route handlers, error patterns, etc."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "pattern": types.Schema(
                type="STRING",
                description="Python-flavored regex.",
            ),
            "path": types.Schema(
                type="STRING",
                description="Subdirectory to search (default '.').",
            ),
            "glob": types.Schema(
                type="STRING",
                description="Optional filename glob, e.g. '*.py' or 'src/**/*.ts'.",
            ),
            "case_insensitive": types.Schema(
                type="BOOLEAN",
                description="If true, match case-insensitively.",
            ),
        },
        required=["pattern"],
    ),
)

_FIND_FILES = types.FunctionDeclaration(
    name="find_files",
    description=(
        "Find files whose name matches a glob (e.g. '*.py', 'auth*.ts'). "
        "Use this to locate likely-relevant files by naming convention."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "name_glob": types.Schema(
                type="STRING",
                description="Filename glob pattern.",
            ),
            "path": types.Schema(
                type="STRING",
                description="Subdirectory to search under (default '.').",
            ),
        },
        required=["name_glob"],
    ),
)


def build_tool() -> types.Tool:
    return types.Tool(
        function_declarations=[_LIST_TREE, _READ_FILE, _GREP, _FIND_FILES]
    )
