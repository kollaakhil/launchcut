# Bundled Open-License Fonts

These 5 open-license TrueType font families are bundled directly inside `assets/fonts/` so FFmpeg ASS subtitle rendering (`:fontsdir=...`) and Pillow motion graphics cards work deterministically across Linux, macOS, and Docker without hardcoded OS font paths (fixing BUG-24).

`CreativeBible.typography_and_layout.font_family` selects among these families at runtime based on the brand and creative direction—no single house font is forced.

| Role | Family Name | Bundled Files | System Source Path | License |
|---|---|---|---|---|
| Geometric / Modern Sans | `Roboto` | `Roboto-Regular.ttf`, `Roboto-Bold.ttf` | `/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/` | Apache License 2.0 |
| Condensed Display / Impact | `Roboto Condensed` | `RobotoCondensed-Regular.ttf`, `RobotoCondensed-Bold.ttf` | `/usr/share/fonts/truetype/roboto/unhinted/` | Apache License 2.0 |
| Editorial Serif | `Liberation Serif` | `LiberationSerif-Regular.ttf`, `LiberationSerif-Bold.ttf` | `/usr/share/fonts/truetype/liberation/` | SIL Open Font License 1.1 |
| Monospace / Technical | `Liberation Mono` | `LiberationMono-Regular.ttf`, `LiberationMono-Bold.ttf` | `/usr/share/fonts/truetype/liberation/` | SIL Open Font License 1.1 |
| Humanist / Neutral Sans | `Liberation Sans` | `LiberationSans-Regular.ttf`, `LiberationSans-Bold.ttf` | `/usr/share/fonts/truetype/liberation/` | SIL Open Font License 1.1 |
