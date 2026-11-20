"""Safe, zero-hard-template Manim Community renderer (engine='manim', $0/shot).

Addresses F21:
- Allows execution of bespoke CreativeBible-driven Manim Python scene code.
- Enforces strict AST sandbox BEFORE spawning any subprocess:
  * Blocks unsafe imports (including dotted submodules & relative imports).
  * Blocks unsafe builtins/calls and dunder introspection escape hatches.
  * Blocks LaTeX classes (Tex, MathTex, SingleStringMathTex) when dvisvgm/LaTeX is unavailable.
  * Validates that `scene_class` actually exists as a Scene subclass in the AST (preventing
    Manim v0.22.1's silent single-scene fallback).
- Enforces strict process-group timeout and CancellationToken termination in an isolated temp dir.
- Supports preview (`-ql`) and final (`-qh`) quality modes with explicit `--resolution` and `--fps`.
- Provides honest `is_manim_available()` and `manim_diagnostics()`.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from gtm_videogen.core.config import get_settings
from gtm_videogen.core.job_registry import CancellationToken
from gtm_videogen.media._subprocess import (
    aspect_to_dimensions,
    probe_media,
    run_subprocess,
)
from gtm_videogen.media.ffmpeg_stitcher import _move_output_atomically
from gtm_videogen.models.schemas import RenderedClip

ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "manim",
        "math",
        "random",
        "colorsys",
        "itertools",
        "numpy",
    }
)

BLOCKED_CALL_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "input",
        "memoryview",
        "type",
    }
)

BLOCKED_INTROSPECTION_ATTRS: frozenset[str] = frozenset(
    {
        "f_globals",
        "f_builtins",
        "f_locals",
        "f_back",
        "f_code",
        "gi_frame",
        "gi_code",
        "cr_frame",
        "cr_code",
        "ag_frame",
        "ag_code",
        "tb_frame",
        "tb_next",
    }
)

BLOCKED_NUMPY_IO_ATTRS: frozenset[str] = frozenset(
    {
        "loadtxt",
        "fromfile",
        "tofile",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "load",
        "genfromtxt",
        "memmap",
        "fromregex",
        "DataSource",
        "ctypeslib",
        "f2py",
        "ctypes",
        "CDLL",
        "PyDLL",
        "testing",
        "distutils",
        "dump",
        "dumps",
    }
)

BLOCKED_MANIM_SYMBOLS: frozenset[str] = frozenset(
    {
        "config",
        "tempconfig",
        "os",
        "sys",
        "subprocess",
        "shutil",
        "Path",
        "pathlib",
        "PosixPath",
        "WindowsPath",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
        "open_file",
        "capture",
        "file_ops",
        "commands",
        "cli",
        "logger",
        "console",
        "error_console",
        "save_text",
        "save_html",
        "save_svg",
        "save_image",
        "from_file",
        "CairoRenderer",
        "Manager",
        "MediaFile",
        "ensure_executable",
        "utils",
        "plugins",
        "module_ops",
        "tex_file_writing",
        "typst_file_writing",
        "Typst",
        "MathTypst",
        "SVGMobject",
        "ImageMobject",
        "ImageMobjectFromCamera",
        "add_sound",
        "register_font",
        "get_full_raster_image_path",
        "get_full_sound_file_path",
        "seek_full_path_from_defaults",
        "guarantee_empty_existence",
        "guarantee_existence",
        "get_dir_layout",
        "add_extension_if_not_present",
        "modify_atime",
        "get_video_metadata",
        "open_media_file",
        "add_version_before_extension",
        "config_ops",
        "caching",
        "debug",
        "scene_file_writer",
        "SceneFileWriter",
        "renderer",
        "file_writer",
        "cairo",
        "ImageSurface",
        "PDFSurface",
        "SVGSurface",
        "av",
        "moderngl",
        "SHADER_FOLDER",
        "shader",
        "shader_wrapper",
        "write_to_png",
    }
)

BLOCKED_MANIM_SUBPACKAGES: frozenset[str] = frozenset(
    {
        "animation",
        "camera",
        "cli",
        "core",
        "data_structures",
        "manager",
        "mobject",
        "opengl",
        "plugins",
        "renderer",
        "scene",
        "typing",
        "unit",
        "utils",
    }
)

BLOCKED_RUNTIME_SYMBOLS: frozenset[str] = frozenset(
    {
        "system",
        "popen",
        "modules",
        "spawnv",
        "execv",
        "builtins",
        "posix_spawn",
        "posix_spawnp",
        "fork",
        "execl",
        "execle",
        "execlp",
        "execvp",
        "spawnl",
        "spawnlp",
        "spawnvp",
        "unlink",
        "rmdir",
        "chmod",
        "chown",
        "operator",
        "op",
        "attrgetter",
        "itemgetter",
        "methodcaller",
        "inspect",
        "getmembers",
        "currentframe",
        "getmodule",
        "getsource",
        "bltns",
        "types",
        "warnings",
        "contextlib",
        "copyreg",
        "io",
        "FileIO",
        "BytesIO",
        "StringIO",
        "tempfile",
        "NamedTemporaryFile",
        "TemporaryFile",
        "TemporaryDirectory",
        "mkstemp",
        "mkdtemp",
        "platform",
        "threading",
        "write_text",
        "read_text",
        "write_bytes",
        "read_bytes",
        "environ",
        "getenv",
        "putenv",
        "unsetenv",
        "logging",
        "FileHandler",
        "RotatingFileHandler",
        "TimedRotatingFileHandler",
        "touch",
        "symlink_to",
        "hardlink_to",
        "iterdir",
        "glob",
        "rglob",
        "mkdir",
        "rename",
        "nx",
        "networkx",
        "read_adjlist",
        "write_adjlist",
        "read_edgelist",
        "write_edgelist",
        "read_gml",
        "write_gml",
        "read_graphml",
        "write_graphml",
    }
)

BLOCKED_PRIVATE_NAMES: frozenset[str] = frozenset(
    {
        "_os",
        "_sys",
        "_config",
        "_builtins",
        "_io",
        "_posix",
        "_subprocess",
    }
)

BLOCKED_FORMAT_ATTRS: frozenset[str] = frozenset({"format", "format_map"})

_ALL_BLOCKED_SYMBOLS: frozenset[str] = (
    BLOCKED_CALL_NAMES
    | BLOCKED_INTROSPECTION_ATTRS
    | BLOCKED_MANIM_SYMBOLS
    | BLOCKED_NUMPY_IO_ATTRS
    | BLOCKED_RUNTIME_SYMBOLS
    | BLOCKED_PRIVATE_NAMES
)

LATEX_CLASSES: frozenset[str] = frozenset({"Tex", "MathTex", "SingleStringMathTex"})

KNOWN_SCENE_BASES: frozenset[str] = frozenset(
    {
        "Scene",
        "MovingCameraScene",
        "ThreeDScene",
        "ZoomedScene",
        "VectorScene",
        "LinearTransformationScene",
        "VoiceoverScene",
    }
)


class ManimSandboxError(ValueError):
    """Raised when Manim scene code violates AST sandbox security or structural rules."""


class ManimUnavailableError(RuntimeError):
    """Raised honestly when Manim or required native libraries/FFmpeg are unavailable."""


class ManimTimeoutError(TimeoutError):
    """Raised when a Manim subprocess exceeds its configured timeout."""


class ManimRenderError(RuntimeError):
    """Raised when Manim exits non-zero or fails to produce a valid MP4 output."""


def is_latex_available() -> bool:
    """Return True only when both dvisvgm and a LaTeX engine are available on PATH."""
    has_dvisvgm = shutil.which("dvisvgm") is not None
    has_latex = shutil.which("latex") is not None or shutil.which("xelatex") is not None
    return has_dvisvgm and has_latex


def is_manim_available() -> bool:
    """Return True when manim, pycairo, manimpango, and ffmpeg are all available."""
    has_manim = importlib.util.find_spec("manim") is not None
    has_cairo = importlib.util.find_spec("cairo") is not None
    has_pango = importlib.util.find_spec("manimpango") is not None
    has_ffmpeg = shutil.which(get_settings().ffmpeg_bin) is not None
    return has_manim and has_cairo and has_pango and has_ffmpeg


def _pkg_version(pkg_name: str) -> str | None:
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def manim_diagnostics() -> dict[str, Any]:
    """Return honest runtime diagnostics for Manim, Cairo/Pango bindings, FFmpeg, and LaTeX."""
    return {
        "available": is_manim_available(),
        "manim_version": _pkg_version("manim"),
        "pycairo_version": _pkg_version("pycairo"),
        "manimpango_version": _pkg_version("manimpango"),
        "ffmpeg_available": shutil.which(get_settings().ffmpeg_bin) is not None,
        "latex_available": is_latex_available(),
        "dvisvgm_available": shutil.which("dvisvgm") is not None,
        "python_executable": sys.executable,
    }


def _extract_base_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _is_super_init_attr(node: ast.Attribute) -> bool:
    return (
        node.attr == "__init__"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "super"
    )


def _has_non_instance_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in {"staticmethod", "classmethod"}:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr in {"staticmethod", "classmethod"}:
            return True
    return False


def _validate_function_self_params(
    args: ast.arguments,
    *,
    allow_first_positional_self: bool,
) -> None:
    pos_args = list(args.posonlyargs) + list(args.args)
    for idx, arg in enumerate(pos_args):
        if arg.arg == "self" and not (allow_first_positional_self and idx == 0):
            raise ManimSandboxError(
                "Parameter 'self' is only allowed as the first positional parameter "
                "of a direct class method."
            )
    other_args: list[ast.arg] = list(args.kwonlyargs)
    if args.vararg is not None:
        other_args.append(args.vararg)
    if args.kwarg is not None:
        other_args.append(args.kwarg)
    for arg in other_args:
        if arg.arg == "self":
            raise ManimSandboxError(
                "Parameter 'self' is only allowed as the first positional parameter "
                "of a direct class method."
            )


def _collect_direct_class_methods(tree: ast.Module) -> set[int]:
    """Return node IDs of functions defined directly inside a ClassDef body."""
    direct_class_methods: set[int] = set()
    for cls_node in ast.walk(tree):
        if not isinstance(cls_node, ast.ClassDef):
            continue
        for stmt in cls_node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                direct_class_methods.add(id(stmt))
    return direct_class_methods


def _is_code_ref(expr: ast.expr) -> bool:
    return (isinstance(expr, ast.Name) and expr.id == "Code") or (
        isinstance(expr, ast.Attribute) and expr.attr == "Code"
    )


def validate_manim_code(
    code: str,
    scene_class: str | None = None,
    *,
    allow_latex: bool | None = None,
) -> None:
    """Validate Manim Python scene code via AST BEFORE executing any subprocess."""
    if not code or not code.strip():
        raise ManimSandboxError("Manim scene code must not be empty.")
    if scene_class is not None and (not scene_class or not scene_class.isidentifier()):
        raise ManimSandboxError(f"Invalid scene_class identifier: {scene_class!r}")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ManimSandboxError(f"Syntax error in Manim scene code: {exc}") from exc

    latex_allowed = is_latex_available() if allow_latex is None else allow_latex

    # Collect ONLY module-level class definitions to verify scene_class is a real Scene subclass
    class_bases: dict[str, list[str]] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            bases = [b for b in (_extract_base_name(base) for base in stmt.bases) if b]
            class_bases[stmt.name] = bases

    direct_class_methods = _collect_direct_class_methods(tree)
    direct_code_call_funcs: set[int] = {
        id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_code_ref(n.func)
    }

    for node in ast.walk(tree):
        # 1. Strict import allowlist, no dotted submodules, no non-manim '*', no private symbols
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == "self":
                    raise ManimSandboxError(
                        "Aliasing an import as 'self' is forbidden in Manim sandbox."
                    )
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    raise ManimSandboxError(
                        f"Blocked unsafe import {alias.name!r} "
                        f"(root module {root!r} is not in allowlist)."
                    )
                if "." in alias.name:
                    raise ManimSandboxError(
                        f"Blocked dotted submodule import {alias.name!r}. "
                        f"Only top-level imports (e.g. 'import {root}') are allowed."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                raise ManimSandboxError("Relative imports are forbidden in Manim sandbox.")
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ManimSandboxError(
                    f"Blocked unsafe import from {mod!r} "
                    f"(root module {root!r} is not in allowlist)."
                )
            if "." in mod:
                raise ManimSandboxError(
                    f"Blocked dotted submodule import from {mod!r}. "
                    f"Only top-level 'from {root} import ...' is allowed."
                )
            for alias in node.names:
                if alias.asname == "self":
                    raise ManimSandboxError(
                        "Aliasing an import as 'self' is forbidden in Manim sandbox."
                    )
                if alias.name == "*" and root != "manim":
                    raise ManimSandboxError(
                        f"Wildcard import 'from {mod} import *' is only allowed for 'manim'."
                    )
                if root == "manim" and alias.name in BLOCKED_MANIM_SUBPACKAGES:
                    raise ManimSandboxError(
                        f"Blocked internal manim subpackage import {alias.name!r} from {mod!r}."
                    )
                if root == "manim" and alias.name == "Code" and alias.asname is not None:
                    raise ManimSandboxError(
                        "Aliasing 'Code' on import is forbidden in Manim sandbox."
                    )
                if alias.name.startswith("_") or alias.name in _ALL_BLOCKED_SYMBOLS:
                    raise ManimSandboxError(
                        f"Blocked unsafe imported symbol {alias.name!r} from {mod!r}."
                    )
                if not latex_allowed and alias.name in LATEX_CLASSES:
                    raise ManimSandboxError(
                        f"LaTeX class {alias.name!r} is blocked because dvisvgm/LaTeX is "
                        f"unavailable on this host. Use Text or MarkupText instead."
                    )

        # 2. Forbid rebinding/shadowing 'self' via assignments, loops, with, except, match, defs
        elif isinstance(node, ast.Name):
            if node.id == "self" and isinstance(node.ctx, (ast.Store, ast.Del)):
                raise ManimSandboxError(
                    "Assigning, rebinding, or deleting 'self' is forbidden in Manim sandbox."
                )
            if node.id in _ALL_BLOCKED_SYMBOLS or _is_dunder(node.id):
                raise ManimSandboxError(f"Blocked unsafe builtin/name reference: {node.id!r}.")
            if node.id == "Code" and id(node) not in direct_code_call_funcs:
                raise ManimSandboxError(
                    "Referencing 'Code' outside a direct Code(code_string=...) call is forbidden."
                )
            if not latex_allowed and node.id in LATEX_CLASSES:
                raise ManimSandboxError(
                    f"LaTeX class {node.id!r} is blocked because dvisvgm/LaTeX is "
                    f"unavailable on this host. Use Text or MarkupText instead."
                )

        elif isinstance(node, ast.ExceptHandler):
            if node.name == "self":
                raise ManimSandboxError(
                    "Binding 'self' in an except handler is forbidden in Manim sandbox."
                )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "self":
                raise ManimSandboxError(
                    "Defining a function named 'self' is forbidden in Manim sandbox."
                )
            is_direct_instance_method = (
                id(node) in direct_class_methods and not _has_non_instance_decorator(node)
            )
            _validate_function_self_params(
                node.args,
                allow_first_positional_self=is_direct_instance_method,
            )

        elif isinstance(node, ast.Lambda):
            _validate_function_self_params(node.args, allow_first_positional_self=False)

        elif isinstance(node, ast.ClassDef):
            if node.name == "self":
                raise ManimSandboxError(
                    "Defining a class named 'self' is forbidden in Manim sandbox."
                )

        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name == "self":
                raise ManimSandboxError(
                    "Binding 'self' in a match pattern is forbidden in Manim sandbox."
                )

        elif isinstance(node, ast.MatchMapping):
            if node.rest == "self":
                raise ManimSandboxError(
                    "Binding 'self' in a match pattern is forbidden in Manim sandbox."
                )

        # 3. Block private/dunder attrs unconditionally (only super().__init__ allowed)
        elif isinstance(node, ast.Attribute):
            if node.attr in _ALL_BLOCKED_SYMBOLS or node.attr in BLOCKED_FORMAT_ATTRS:
                raise ManimSandboxError(f"Blocked unsafe attribute access: {node.attr!r}.")
            if node.attr == "Code" and id(node) not in direct_code_call_funcs:
                raise ManimSandboxError(
                    "Referencing 'Code' outside a direct Code(code_string=...) call is forbidden."
                )
            if node.attr in BLOCKED_MANIM_SUBPACKAGES and not (
                node.attr == "camera"
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                raise ManimSandboxError(
                    f"Blocked internal manim subpackage attribute access: {node.attr!r}."
                )
            if node.attr.startswith("_") and not _is_super_init_attr(node):
                raise ManimSandboxError(
                    f"Blocked private/dunder attribute access: {node.attr!r}."
                )
            if not latex_allowed and node.attr in LATEX_CLASSES:
                raise ManimSandboxError(
                    f"LaTeX class {node.attr!r} is blocked because dvisvgm/LaTeX is "
                    f"unavailable on this host. Use Text or MarkupText instead."
                )

        # 4. Block file-reading kwargs & unsafe Code(...) invocations (BUG-M3-S18)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "code_file":
                    raise ManimSandboxError(
                        "Blocked unsafe keyword argument 'code_file' (disk reads forbidden)."
                    )
            if _is_code_ref(node.func):
                if node.args:
                    raise ManimSandboxError(
                        "Positional arguments to Code(...) are forbidden; "
                        "pass explicit keyword arguments such as "
                        "Code(code_string=..., language=...)."
                    )
                for kw in node.keywords:
                    if kw.arg is None:
                        raise ManimSandboxError(
                            "Unpacked **kwargs in Code(...) are forbidden; "
                            "pass explicit keyword arguments such as "
                            "Code(code_string=..., language=...)."
                        )

        # 5. Block string-subscript access to dunders / blocked builtins / escape symbols
        elif isinstance(node, ast.Subscript):
            slc = node.slice
            if (
                isinstance(slc, ast.Constant)
                and isinstance(slc.value, str)
                and (
                    _is_dunder(slc.value)
                    or slc.value in _ALL_BLOCKED_SYMBOLS
                    or slc.value == "code_file"
                )
            ):
                raise ManimSandboxError(f"Blocked unsafe subscript key access: {slc.value!r}.")

    # Verify requested scene_class exists at module level and inherits from a known Scene base
    if scene_class is not None:
        if scene_class not in class_bases:
            raise ManimSandboxError(
                f"Requested scene_class {scene_class!r} not found at module level in Manim code. "
                f"Defined classes: {sorted(class_bases)}"
            )

        def _is_scene_subclass(cls_name: str, visited: set[str]) -> bool:
            if cls_name in visited:
                return False
            visited.add(cls_name)
            for base in class_bases.get(cls_name, []):
                if base in KNOWN_SCENE_BASES:
                    return True
                if base in class_bases and _is_scene_subclass(base, visited):
                    return True
            return False

        if not _is_scene_subclass(scene_class, set()):
            raise ManimSandboxError(
                f"Class {scene_class!r} must subclass Scene (bases: {class_bases[scene_class]})."
            )


validate_manim_scene_code = validate_manim_code


def _locate_rendered_mp4(media_dir: Path, preferred_name: str) -> Path | None:
    videos_dir = media_dir / "videos"
    if not videos_dir.exists():
        return None
    # First search for exact preferred filename (ignoring partial_movie_files)
    candidates = [p for p in videos_dir.rglob("*.mp4") if "partial_movie_files" not in p.parts]
    for p in candidates:
        if p.name == preferred_name:
            return p
    return candidates[0] if candidates else None


def _locate_rendered_png(media_dir: Path) -> Path | None:
    images_dir = media_dir / "images"
    if not images_dir.exists():
        return None
    candidates = sorted(
        images_dir.rglob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _build_sanitized_manim_env(tmp_dir: Path) -> dict[str, str]:
    """Build a minimal sanitized environment for the Manim subprocess (no host secrets)."""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(tmp_dir),
        "TMPDIR": str(tmp_dir),
        "XDG_CACHE_HOME": str(tmp_dir),
        "MPLCONFIGDIR": str(tmp_dir),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
    }
    for key in ("VIRTUAL_ENV", "FONTCONFIG_PATH", "FONTCONFIG_FILE", "LD_LIBRARY_PATH"):
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


class ManimRenderer:
    """Sandboxed Manim Community renderer supporting preview (-ql) and final (-qh) modes."""

    def __init__(
        self,
        *,
        python_bin: str | None = None,
        ffmpeg_bin: str | None = None,
        ffprobe_bin: str | None = None,
    ) -> None:
        settings = get_settings()
        self.python_bin = python_bin or sys.executable
        self.ffmpeg_bin = ffmpeg_bin or settings.ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin or settings.ffprobe_bin

    async def render_scene(
        self,
        code: str,
        scene_class: str,
        output_path: Path | str,
        *,
        aspect: str = "16:9",
        fps: int = 24,
        quality: Literal["preview", "final"] = "preview",
        timeout_sec: float = 60.0,
        shot_id: str = "manim_shot",
        static_duration_sec: float = 1.0,
        allow_latex: bool | None = None,
        cancel: CancellationToken | None = None,
    ) -> RenderedClip:
        """Validate Manim code via AST sandbox and render to MP4 in an isolated temp dir."""
        if not is_manim_available():
            diag = manim_diagnostics()
            raise ManimUnavailableError(
                f"Manim Community renderer is unavailable on this host: {diag}"
            )

        # AST sandbox validation MUST run before any subprocess is spawned
        validate_manim_code(code, scene_class, allow_latex=allow_latex)

        if cancel is not None:
            cancel.raise_if_cancelled()

        is_preview = quality == "preview"
        w, h = aspect_to_dimensions(aspect, preview=is_preview)
        quality_flag = "-ql" if is_preview else "-qh"

        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        tmp_dir = Path(tempfile.mkdtemp(prefix="gtm_manim_"))
        try:
            scene_file = tmp_dir / "scene.py"
            media_dir = tmp_dir / "media"
            scene_file.write_text(code, encoding="utf-8")

            out_name = "rendered_scene.mp4"
            cmd = [
                self.python_bin,
                "-m",
                "manim",
                quality_flag,
                "--media_dir",
                str(media_dir),
                "--resolution",
                f"{w},{h}",
                "--fps",
                str(fps),
                "-o",
                out_name,
                str(scene_file),
                scene_class,
            ]

            sanitized_env = _build_sanitized_manim_env(tmp_dir)
            try:
                rc, _, stderr_b = await run_subprocess(
                    cmd,
                    cancel=cancel,
                    timeout_sec=timeout_sec,
                    cwd=tmp_dir,
                    env=sanitized_env,
                    check=False,
                )
            except TimeoutError as exc:
                raise ManimTimeoutError(
                    f"Manim render of {scene_class!r} exceeded timeout of {timeout_sec}s"
                ) from exc

            if rc != 0:
                stderr = stderr_b.decode("utf-8", errors="replace")
                raise ManimRenderError(
                    f"Manim render of {scene_class!r} failed (exit {rc}): {stderr[-1000:]}"
                )

            mp4_path = _locate_rendered_mp4(media_dir, out_name)
            if mp4_path is None or not mp4_path.exists():
                png_path = _locate_rendered_png(media_dir)
                if png_path is not None and png_path.exists():
                    dur = max(0.2, float(static_duration_sec))
                    encoded_mp4 = tmp_dir / "static_scene.mp4"
                    vf = (
                        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
                    )
                    encode_cmd = [
                        self.ffmpeg_bin,
                        "-y",
                        "-loop",
                        "1",
                        "-framerate",
                        str(fps),
                        "-i",
                        str(png_path),
                        "-t",
                        f"{dur:.3f}",
                        "-vf",
                        vf,
                        "-c:v",
                        "libx264",
                        "-profile:v",
                        "high",
                        "-level:v",
                        "4.1",
                        "-pix_fmt",
                        "yuv420p",
                        "-color_range",
                        "tv",
                        "-colorspace",
                        "bt709",
                        "-color_primaries",
                        "bt709",
                        "-color_trc",
                        "bt709",
                        "-video_track_timescale",
                        "90000",
                        "-r",
                        str(fps),
                        "-movflags",
                        "+faststart",
                        str(encoded_mp4),
                    ]
                    await run_subprocess(
                        encode_cmd,
                        cancel=cancel,
                        timeout_sec=min(timeout_sec, 60.0),
                        check=True,
                    )
                    mp4_path = encoded_mp4
                else:
                    raise ManimRenderError(
                        f"Manim exited 0 for {scene_class!r} but no MP4 or PNG was found "
                        f"under {media_dir}."
                    )

            _move_output_atomically(mp4_path, out)
            probe = await probe_media(out, ffprobe_bin=self.ffprobe_bin, cancel=cancel)
            if not probe.has_video or probe.duration_sec <= 0.0:
                raise ManimRenderError(
                    f"Manim output {out} failed ffprobe validation (duration={probe.duration_sec})."
                )

            return RenderedClip(
                shot_id=shot_id,
                path=str(out),
                duration_sec=probe.duration_sec,
                fps=round(probe.fps) if probe.fps > 0 else fps,
                width=probe.width or w,
                height=probe.height or h,
                cache_hit=False,
                usd=0.0,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


async def render_manim_scene(
    code: str,
    scene_class: str,
    output_path: Path | str,
    *,
    aspect: str = "16:9",
    fps: int = 24,
    quality: Literal["preview", "final"] = "preview",
    timeout_sec: float = 60.0,
    shot_id: str = "manim_shot",
    static_duration_sec: float = 1.0,
    allow_latex: bool | None = None,
    cancel: CancellationToken | None = None,
) -> RenderedClip:
    """Module-level convenience wrapper around ManimRenderer.render_scene."""
    return await ManimRenderer().render_scene(
        code=code,
        scene_class=scene_class,
        output_path=output_path,
        aspect=aspect,
        fps=fps,
        quality=quality,
        timeout_sec=timeout_sec,
        shot_id=shot_id,
        static_duration_sec=static_duration_sec,
        allow_latex=allow_latex,
        cancel=cancel,
    )
