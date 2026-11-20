"""AST sandbox, timeout, diagnostics, and real Manim Community render tests (F21)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gtm_videogen.media._subprocess import probe_media
from gtm_videogen.media.manim_renderer import (
    ManimRenderer,
    ManimSandboxError,
    ManimTimeoutError,
    _build_sanitized_manim_env,
    is_manim_available,
    manim_diagnostics,
    validate_manim_code,
)
from gtm_videogen.models.schemas import Shot

VALID_SCENE_CODE = """
from manim import Scene, Circle, Text, Create, Write, DOWN

class BrandReveal(Scene):
    def construct(self):
        dot = Circle(radius=1.0, color="#F4A261")
        label = Text("Solara Botanics", font_size=36).next_to(dot, DOWN)
        self.play(Create(dot), Write(label), run_time=0.5)
        self.wait(0.3)
"""


def test_shot_schema_accepts_manim_engine() -> None:
    shot = Shot(
        id="s_manim",
        index=0,
        beat="metric_callout",
        duration_sec=2.5,
        engine="manim",
        visual_prompt="Animated metric bar chart in brand palette",
    )
    assert shot.engine == "manim"


def test_manim_diagnostics_and_availability() -> None:
    diag = manim_diagnostics()
    assert isinstance(diag["available"], bool)
    assert diag["available"] == is_manim_available()
    assert "manim_version" in diag
    assert "pycairo_version" in diag
    assert "manimpango_version" in diag
    assert "latex_available" in diag
    assert "dvisvgm_available" in diag


@pytest.mark.parametrize(
    ("bad_code", "scene_name", "reason"),
    [
        (
            "import os\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): os.system('id')\n",
            "S",
            "os import",
        ),
        (
            "import os.path\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "dotted os.path import",
        ),
        (
            "import subprocess\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "subprocess import",
        ),
        (
            "import socket\nfrom manim import Scene\nclass S(Scene):\n def construct(self): pass\n",
            "S",
            "socket import",
        ),
        (
            "from urllib.request import urlopen\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "urllib.request import",
        ),
        (
            "from pathlib import Path\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "pathlib import",
        ),
        (
            "from . import helper\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "relative import",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n def construct(self): open('/etc/passwd')\n",
            "S",
            "open call",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n def construct(self): __import__('os')\n",
            "S",
            "__import__ call",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n def construct(self): eval('1+1')\n",
            "S",
            "eval call",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n def construct(self): exec('x=1')\n",
            "S",
            "exec call",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n def construct(self): getattr(self, 'x')\n",
            "S",
            "getattr call",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self): ().__class__.__bases__[0].__subclasses__()\n",
            "S",
            "dunder introspection",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self): x = d['__builtins__']\n",
            "S",
            "__builtins__ subscript",
        ),
        (
            VALID_SCENE_CODE,
            "NonExistentScene",
            "missing scene_class",
        ),
        (
            "class NotAScene:\n pass\n",
            "NotAScene",
            "not a Scene subclass",
        ),
        # BUG-M3-S1: dunder & introspection escapes
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self):\n"
            "  ga = object.__getattribute__\n"
            "  ga((), '__cl' + 'ass__')\n",
            "S",
            "object.__getattribute__ dunder escape",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self): x = self.f_globals\n",
            "S",
            "frame introspection f_globals",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self): x = g.gi_frame\n",
            "S",
            "generator introspection gi_frame",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self): x = __name__\n",
            "S",
            "dunder Name reference",
        ),
        # BUG-M3-S2: strict import allowlist & numpy file I/O
        (
            "import posix\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): posix.system('id')\n",
            "S",
            "posix import blocked by allowlist",
        ),
        (
            "import io\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): io.FileIO('/tmp/x', 'w')\n",
            "S",
            "io import blocked by allowlist",
        ),
        (
            "import inspect\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "inspect import blocked by allowlist",
        ),
        (
            "import asyncio\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "asyncio import blocked by allowlist",
        ),
        (
            "import numpy as np\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): np.loadtxt('/etc/passwd')\n",
            "S",
            "numpy.loadtxt file I/O blocked",
        ),
        (
            "from numpy import save\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): save('/tmp/x', [1])\n",
            "S",
            "from numpy import save blocked",
        ),
        # BUG-M3-S3: Manim internal submodules & config/OS escapes
        (
            "from manim.utils.commands import capture\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): capture(['id'])\n",
            "S",
            "manim.utils.commands submodule blocked",
        ),
        (
            "import manim.utils.file_ops as fo\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "manim.utils.file_ops submodule blocked",
        ),
        (
            "import manim\nclass S(manim.Scene):\n"
            " def construct(self): manim.config.output_file = '/tmp/hijack.mp4'\n",
            "S",
            "manim.config attribute escape blocked",
        ),
        (
            "from manim import Scene, config\nclass S(Scene):\n"
            " def construct(self): config.output_file = '/tmp/hijack.mp4'\n",
            "S",
            "from manim import config blocked",
        ),
        (
            "from manim import *\nclass S(Scene):\n"
            " def construct(self): tempconfig({'output_file': '/tmp/x.mp4'})\n",
            "S",
            "tempconfig reference blocked",
        ),
        # BUG-M3-S4: memoryview and type builtins
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self): memoryview(b'abc')\n",
            "S",
            "memoryview builtin blocked",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self): type('X', (), {})\n",
            "S",
            "type builtin blocked",
        ),
        # BUG-M3-S5: Scene subclass spoofing & nested ClassDef
        (
            "class FakeScene:\n pass\nclass MyScene(FakeScene):\n def construct(self): pass\n",
            "MyScene",
            "FakeScene suffix spoofing blocked",
        ),
        (
            "from manim import Scene\ndef make():\n class NestedScene(Scene):\n"
            "  def construct(self): pass\n",
            "NestedScene",
            "nested ClassDef not at module level",
        ),
        # BUG-M3-S6 & BUG-M3-S7: private single-underscore imports, attributes & non-manim '*'
        (
            "import random\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): random._os.system('id')\n",
            "S",
            "random._os.system RCE blocked",
        ),
        (
            "from random import _os\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "from random import _os blocked",
        ),
        (
            "from random import *\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "non-manim wildcard import blocked",
        ),
        (
            "import collections\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): collections._sys.modules['os']\n",
            "S",
            "collections._sys.modules['os'] RCE blocked",
        ),
        (
            "import manim\nclass S(manim.Scene):\n"
            " def construct(self): x = manim._config\n",
            "S",
            "manim._config private attr blocked",
        ),
        # BUG-M3-S8: dotted submodule imports on non-manim allowlisted roots
        (
            "import numpy.ctypeslib as nc\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "import numpy.ctypeslib dotted submodule blocked",
        ),
        (
            "import numpy.f2py\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "import numpy.f2py dotted submodule blocked",
        ),
        (
            "from numpy.lib._datasource import DataSource\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "from numpy.lib._datasource import DataSource blocked",
        ),
        # BUG-M3-S9: numpy ndarray.tofile & defense-in-depth runtime/subscript blocks
        (
            "import numpy as np\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  np.frombuffer(b'x', dtype=np.uint8).tofile('/tmp/x')\n",
            "S",
            "numpy ndarray.tofile file write blocked",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self): x = d['os']\n",
            "S",
            "subscript key 'os' blocked",
        ),
        # BUG-M3-S10: self rebinding, parameter shadowing, alias, and external self._attr
        (
            "import random\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  self = random\n"
            "  self._os.posix_spawn('/bin/sh', ['sh', '-c', 'id'], {})\n",
            "S",
            "self = random rebinding and posix_spawn RCE blocked",
        ),
        (
            "import random\nfrom manim import Scene\n"
            "def leak(self): return self._os\n"
            "class S(Scene):\n"
            " def construct(self): leak(random)\n",
            "S",
            "non-method function parameter self shadowing blocked",
        ),
        (
            "import random\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  fn = lambda self: self._os\n"
            "  fn(random)\n",
            "S",
            "lambda parameter self shadowing blocked",
        ),
        (
            "import random as self\nfrom manim import Scene\n"
            "class S(Scene):\n def construct(self): pass\n",
            "S",
            "import alias as self blocked",
        ),
        (
            "import random\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  for self in [random]:\n"
            "   pass\n",
            "S",
            "for-loop rebinding self blocked",
        ),
        (
            "from manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self): x = self._external_private\n",
            "S",
            "self._attr not defined on class or assigned in class method blocked",
        ),
        (
            "from manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  self._os = 1\n",
            "S",
            "BLOCKED_PRIVATE_NAMES checked unconditionally on self._os",
        ),
        # BUG-M3-S11 & BUG-M3-S12: stdlib re-exports & manim internal submodules
        (
            "import fractions\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self): fractions.operator.attrgetter('__import__')\n",
            "S",
            "fractions.operator.attrgetter blocked",
        ),
        (
            "import typing\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self): typing.operator.attrgetter('__globals__')\n",
            "S",
            "typing.operator.attrgetter blocked",
        ),
        (
            "import enum\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self): x = enum.bltns\n",
            "S",
            "enum.bltns blocked",
        ),
        (
            "import dataclasses\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self): dataclasses.inspect.getmembers(dataclasses)\n",
            "S",
            "dataclasses.inspect.getmembers blocked",
        ),
        (
            "import manim\nclass S(manim.Scene):\n"
            " def construct(self): x = manim.utils\n",
            "S",
            "manim.utils blocked",
        ),
        (
            "import manim\nclass S(manim.Scene):\n"
            " def construct(self): x = manim.plugins\n",
            "S",
            "manim.plugins blocked",
        ),
        (
            "from manim import Scene\nclass S(Scene):\n"
            " def construct(self): x = attrgetter('foo')\n",
            "S",
            "defense-in-depth attrgetter symbol blocked",
        ),
        # BUG-M3-S13: deep public manim.* re-exports
        (
            "import manim\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  p = manim.camera.camera.pathlib.PosixPath('/tmp/pwn')\n"
            "  p.write_text('x')\n",
            "S",
            "manim.camera.camera.pathlib.PosixPath blocked",
        ),
        (
            "import manim\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  f = manim.scene.scene_file_writer.NamedTemporaryFile()\n",
            "S",
            "manim.scene.scene_file_writer.NamedTemporaryFile blocked",
        ),
        (
            "import manim\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  p = manim.scene.scene.platform\n",
            "S",
            "manim.scene.scene.platform blocked",
        ),
        # BUG-M3-S14: str.format / str.format_map dunder traversal
        (
            "import random\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            '  v = "{0.seed.__globals__[_os].environ[HOME]}".format(random.Random)\n',
            "S",
            "str.format dunder traversal blocked",
        ),
        (
            "from manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            '  v = "{x}".format_map({"x": 1})\n',
            "S",
            "str.format_map blocked",
        ),
        # BUG-M3-S15: unbound class method self._private & decorator aliasing
        (
            "import numpy\nfrom manim import Scene\n"
            "class Stealer:\n"
            " def steal(self):\n"
            "  if False:\n"
            "   self._private = None\n"
            "  return self._private\n"
            "class S(Scene):\n"
            " def construct(self): Stealer.steal(numpy)\n",
            "S",
            "unbound Stealer.steal(numpy) accessing self._private blocked",
        ),
        (
            "import numpy\nfrom manim import Scene\n"
            "sm = staticmethod\n"
            "class Stealer:\n"
            " @sm\n"
            " def steal(self): return self._utils\n"
            "class S(Scene):\n"
            " def construct(self): Stealer.steal(numpy)\n",
            "S",
            "decorator alias sm = staticmethod + self._utils blocked",
        ),
        # BUG-M3-S16: manim internal subpackage traversal, alias propagation & file_writer
        (
            "import manim\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  manim.renderer.shader_wrapper.logging.FileHandler('/etc/hostname')\n",
            "S",
            "manim.renderer.shader_wrapper.logging.FileHandler blocked",
        ),
        (
            "import manim as mn\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  mn.renderer.shader.SHADER_FOLDER.touch()\n",
            "S",
            "mn.renderer.shader.SHADER_FOLDER.touch blocked",
        ),
        (
            "import manim\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  m = manim\n"
            "  x = m.camera.camera.cairo.ImageSurface\n",
            "S",
            "rebound manim alias m.camera.camera.cairo.ImageSurface blocked",
        ),
        (
            "import manim\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  m = manim\n"
            "  x = m.camera\n",
            "S",
            "rebound manim alias m.camera subpackage blocked structurally",
        ),
        (
            "from manim import Scene, renderer\n"
            "class S(Scene):\n"
            " def construct(self): pass\n",
            "S",
            "from manim import renderer blocked",
        ),
        (
            "from manim import Scene, camera\n"
            "class S(Scene):\n"
            " def construct(self): pass\n",
            "S",
            "from manim import camera blocked",
        ),
        (
            "from manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  fw = self.renderer.file_writer\n",
            "S",
            "self.renderer.file_writer blocked",
        ),
        # BUG-M3-S17: unconditional BLOCKED_MANIM_SUBPACKAGES & networkx I/O blocks
        (
            "import manim\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  g = (lambda x: x.mobject)(manim).graph.nx\n",
            "S",
            "(lambda x: x.mobject)(manim).graph.nx blocked",
        ),
        (
            "import manim\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            "  cam = (lambda x: x.camera)(manim)\n",
            "S",
            "(lambda x: x.camera)(manim) blocked",
        ),
        (
            "import manim\nfrom manim import Scene\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            '  d = {"k": manim}\n'
            '  x = d["k"].mobject\n',
            "S",
            "dict lookup d['k'].mobject blocked unconditionally",
        ),
        # BUG-M3-S18: Code(code_file=...) / positional / **kwargs / alias disk reads blocked
        (
            "from manim import Scene, Code\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            '  c = Code(code_file="/etc/hostname", language="text")\n',
            "S",
            "Code(code_file=...) disk read blocked",
        ),
        (
            "from manim import Scene, Code\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            '  c = Code("/etc/hostname")\n',
            "S",
            "Code(positional) disk read blocked",
        ),
        (
            "from manim import Scene, Code\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            '  c = Code(**{"code_file": "/etc/hostname"})\n',
            "S",
            "Code(**kwargs) disk read blocked",
        ),
        (
            "from manim import Scene, Code as C\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            '  c = C("/etc/hostname")\n',
            "S",
            "aliased Code import blocked",
        ),
        (
            "from manim import Scene, Code\n"
            "class S(Scene):\n"
            " def construct(self):\n"
            '  c = (lambda f: f("/etc/hostname"))(Code)\n',
            "S",
            "lambda-wrapped Code reference blocked",
        ),
    ],
)
def test_ast_sandbox_blocks_unsafe_or_invalid_code(
    bad_code: str, scene_name: str, reason: str
) -> None:
    with pytest.raises(ManimSandboxError):
        validate_manim_code(bad_code, scene_name, allow_latex=False)


def test_ast_sandbox_blocks_latex_when_unavailable() -> None:
    latex_code = """
from manim import Scene, MathTex, Write

class FormulaScene(Scene):
    def construct(self):
        eq = MathTex(r"E = mc^2")
        self.play(Write(eq))
"""
    with pytest.raises(ManimSandboxError, match="LaTeX"):
        validate_manim_code(latex_code, "FormulaScene", allow_latex=False)


def test_ast_sandbox_accepts_valid_scene() -> None:
    validate_manim_code(VALID_SCENE_CODE, "BrandReveal", allow_latex=False)

    # Verify all 6 allowlisted roots, super().__init__(), public helpers, self.remove(),
    # np.stack(), self.camera.background_color, f-strings, and `for _`.
    advanced_scene = """
import colorsys
import itertools
import math
import random
import numpy as np
from manim import Scene, VGroup, Circle, Text, Create

class CustomBadge(VGroup):
    def __init__(self):
        super().__init__()
        self.accent = colorsys.hsv_to_rgb(0.1, 0.7, 0.9)
        jitter = random.uniform(0.0, 0.05)
        pts = np.stack([np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])])
        self.add(Circle(radius=math.sin(np.pi / 4) + jitter + float(pts[0][0])))

class BadgeScene(Scene):
    def build_badge(self):
        group = VGroup()
        for _ in itertools.islice(range(10), 2):
            group.add(CustomBadge())
        return group

    def construct(self):
        self.camera.background_color = "#0B0F1A"
        self.badge = self.build_badge()
        c = Circle(radius=0.5)
        title = "Solara_Botanics".replace("_", " ")
        label = Text(f"{title} {1.23:.1f}", font_size=28)
        self.add(c, label)
        self.play(Create(self.badge), run_time=0.3)
        self.remove(c)
"""
    validate_manim_code(advanced_scene, "BadgeScene", allow_latex=False)

    code_snippet_scene = """
from manim import Scene, Code

class CodeSnippetScene(Scene):
    def construct(self):
        self.camera.background_color = "#111827"
        snippet = Code(code_string="x = 1", language="python")
        self.add(snippet)
"""
    validate_manim_code(code_snippet_scene, "CodeSnippetScene", allow_latex=False)


def test_build_sanitized_manim_env_strips_host_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret-elevenlabs")
    monkeypatch.setenv("FAL_KEY", "secret-fal")
    monkeypatch.setenv("HOME", "/home/real-host-home")

    env = _build_sanitized_manim_env(tmp_path)

    assert "OPENAI_API_KEY" not in env
    assert "GEMINI_API_KEY" not in env
    assert "ELEVENLABS_API_KEY" not in env
    assert "FAL_KEY" not in env
    assert env["HOME"] == str(tmp_path)
    assert env["TMPDIR"] == str(tmp_path)
    assert env["XDG_CACHE_HOME"] == str(tmp_path)
    assert env["MPLCONFIGDIR"] == str(tmp_path)
    assert "PATH" in env


@pytest.mark.asyncio
async def test_manim_real_render_preview_and_portrait(tmp_path: Path) -> None:
    if not is_manim_available():
        pytest.skip("Manim Community is not installed in this environment")

    renderer = ManimRenderer()

    # 1. Real -ql preview 16:9 @ 24fps MP4 render
    out_16_9 = tmp_path / "brand_reveal_16_9.mp4"
    clip = await renderer.render_scene(
        code=VALID_SCENE_CODE,
        scene_class="BrandReveal",
        output_path=out_16_9,
        aspect="16:9",
        fps=24,
        quality="preview",
        timeout_sec=45.0,
        shot_id="manim_1",
    )
    assert out_16_9.exists()
    assert clip.usd == 0.0
    assert clip.width == 854 and clip.height == 480
    probe = await probe_media(out_16_9)
    assert probe.has_video
    assert probe.width == 854 and probe.height == 480
    assert abs(probe.fps - 24.0) < 0.2
    assert probe.duration_sec > 0.5

    # 2. Real -ql portrait 9:16 @ 30fps MP4 render
    out_9_16 = tmp_path / "brand_reveal_9_16.mp4"
    clip_v = await renderer.render_scene(
        code=VALID_SCENE_CODE,
        scene_class="BrandReveal",
        output_path=out_9_16,
        aspect="9:16",
        fps=30,
        quality="preview",
        timeout_sec=45.0,
        shot_id="manim_2",
    )
    assert out_9_16.exists()
    assert clip_v.width == 480 and clip_v.height == 854
    probe_v = await probe_media(out_9_16)
    assert probe_v.has_video
    assert probe_v.width == 480 and probe_v.height == 854
    assert abs(probe_v.fps - 30.0) < 0.2


@pytest.mark.asyncio
async def test_manim_timeout_enforcement(tmp_path: Path) -> None:
    if not is_manim_available():
        pytest.skip("Manim Community is not installed in this environment")

    infinite_scene = """
from manim import Scene

class InfiniteLoop(Scene):
    def construct(self):
        while True:
            x = 1 + 1
"""
    renderer = ManimRenderer()
    with pytest.raises(ManimTimeoutError):
        await renderer.render_scene(
            code=infinite_scene,
            scene_class="InfiniteLoop",
            output_path=tmp_path / "timeout.mp4",
            aspect="16:9",
            fps=24,
            quality="preview",
            timeout_sec=2.0,
        )


@pytest.mark.asyncio
async def test_manim_static_scene_png_encoded_to_mp4(tmp_path: Path) -> None:
    """DEFECT-01: Static Manim scenes using self.add(...) without play/wait emit PNG -> MP4."""
    if not is_manim_available():
        pytest.skip("Manim Community is not installed in this environment")

    static_scene = """
from manim import Scene, Text

class StaticTitleCard(Scene):
    def construct(self):
        self.add(Text("STATIC MANIM CARD", font_size=40))
"""
    out_mp4 = tmp_path / "static_card.mp4"
    renderer = ManimRenderer()
    clip = await renderer.render_scene(
        code=static_scene,
        scene_class="StaticTitleCard",
        output_path=out_mp4,
        aspect="16:9",
        fps=24,
        quality="preview",
        static_duration_sec=1.2,
        timeout_sec=45.0,
        shot_id="manim_static",
    )
    assert out_mp4.exists()
    assert clip.usd == 0.0
    assert clip.width == 854 and clip.height == 480
    probe = await probe_media(out_mp4)
    assert probe.has_video
    assert probe.width == 854 and probe.height == 480
    assert abs(probe.fps - 24.0) < 0.2
    assert abs(probe.duration_sec - 1.2) < 0.15


@pytest.mark.parametrize(
    ("bad_code", "reason"),
    [
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        self.add(SVGMobject('/etc/passwd'))\n",
            "SVGMobject host file read",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        self.add(ImageMobject('/tmp/secret.png'))\n",
            "ImageMobject host file read",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        self.add(ImageMobjectFromCamera(self.camera))\n",
            "ImageMobjectFromCamera",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        self.add_sound('/tmp/secret.wav')\n",
            "self.add_sound host file read",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        register_font('/tmp/font.ttf')\n",
            "register_font host file read",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        self.add(Typst('#read(\"/etc/passwd\")'))\n",
            "Typst subprocess/file read",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        self.add(MathTypst('x^2'))\n",
            "MathTypst subprocess/file read",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        get_full_raster_image_path('/tmp/secret.png')\n",
            "get_full_raster_image_path",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        get_full_sound_file_path('/tmp/secret.wav')\n",
            "get_full_sound_file_path",
        ),
    ],
)
def test_manim_sandbox_blocks_host_media_readers_and_typst(bad_code: str, reason: str) -> None:
    """M6 Tier 5 Challenge 1: Block host media loaders and Typst unconditionally."""
    with pytest.raises(ManimSandboxError):
        validate_manim_code(bad_code, "S", allow_latex=True)


@pytest.mark.parametrize(
    ("bad_code", "reason"),
    [
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        seek_full_path_from_defaults('/tmp/victim', '', [])\n",
            "seek_full_path_from_defaults host path construction",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        guarantee_empty_existence('/tmp/victim')\n",
            "guarantee_empty_existence host rmtree",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        guarantee_existence('/tmp/pwned')\n",
            "guarantee_existence host mkdir",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        get_dir_layout('/etc')\n",
            "get_dir_layout host directory listing",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        add_extension_if_not_present('/tmp/x', '.mp4')\n",
            "add_extension_if_not_present path helper",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        modify_atime('/tmp/x')\n",
            "modify_atime host utime mutation",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        get_video_metadata('/tmp/secret.mp4')\n",
            "get_video_metadata host video probe",
        ),
    ],
)
def test_manim_sandbox_blocks_file_ops_and_commands_exports(bad_code: str, reason: str) -> None:
    """M6 Iter 2 Challenge 1: Block all top-level file_ops & commands host-filesystem helpers."""
    with pytest.raises(ManimSandboxError):
        validate_manim_code(bad_code, "S", allow_latex=False)
    with pytest.raises(ManimSandboxError):
        validate_manim_code(bad_code, "S", allow_latex=True)


@pytest.mark.parametrize(
    ("bad_code", "reason"),
    [
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        error_console.record = True\n"
            "        error_console.print('pwn')\n"
            "        error_console.save_text('/tmp/console_pwn.txt')\n",
            "error_console.save_text host file write",
        ),
        (
            "from manim import Scene, error_console\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        error_console.print('hi')\n",
            "direct error_console import and reference",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        x.save_text('/tmp/pwn.txt')\n",
            "save_text attribute call",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        x.save_html('/tmp/pwn.html')\n",
            "save_html attribute call",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        x.save_svg('/tmp/pwn.svg')\n",
            "save_svg attribute call",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        Circle().save_image('/tmp/pwn')\n",
            "Mobject.save_image arbitrary host PNG write",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        TexTemplate.from_file('/etc/passwd')\n",
            "TexTemplate.from_file arbitrary host file read",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        CairoRenderer()\n",
            "CairoRenderer direct reference",
        ),
        (
            "from manim import *\n"
            "class S(Scene):\n"
            "    def construct(self):\n"
            "        Manager()\n",
            "Manager direct reference",
        ),
    ],
)
def test_manim_sandbox_blocks_error_console_and_file_writers(bad_code: str, reason: str) -> None:
    """M6 Iter 3: Block error_console, file-save helpers, and renderer internals."""
    with pytest.raises(ManimSandboxError):
        validate_manim_code(bad_code, "S", allow_latex=False)
    with pytest.raises(ManimSandboxError):
        validate_manim_code(bad_code, "S", allow_latex=True)



