"""Apply patches to salsa-clrs that enable additional algorithms.

Run once after installing dependencies:
    python scripts/patch_salsaclrs.py

Patches applied:
1. Uncomment bellman_ford, articulation_points, bridges in SAMPLERS
2. Add fast_mis_2 spec alias (algorithm references 'fast_mis_2' but spec is 'fast_mis')
"""

import importlib
import site
import sys
from pathlib import Path


def _find_salsaclrs() -> Path:
    """Locate the salsaclrs package directory."""
    try:
        import salsaclrs
        return Path(salsaclrs.__file__).parent
    except ImportError:
        # Search site-packages
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            candidate = Path(sp) / "salsaclrs"
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Cannot find salsaclrs package")


def patch_sampler(pkg_dir: Path) -> None:
    """Enable bellman_ford, articulation_points, bridges in SAMPLERS."""
    sampler_path = pkg_dir / "sampler.py"
    text = sampler_path.read_text()

    replacements = [
        ("# 'articulation_points': ArticulationSampler,", "'articulation_points': ArticulationSampler,"),
        ("# 'bridges': ArticulationSampler,", "'bridges': ArticulationSampler,"),
        ("# 'bellman_ford': BellmanFordSampler,", "'bellman_ford': BellmanFordSampler,"),
    ]

    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True

    if changed:
        sampler_path.write_text(text)
        print(f"Patched {sampler_path}: enabled bellman_ford, articulation_points, bridges")
    else:
        print(f"Sampler already patched or has different format")


def patch_specs(pkg_dir: Path) -> None:
    """Add fast_mis_2 alias so the algorithm implementation can find its spec."""
    specs_path = pkg_dir / "specs.py"
    text = specs_path.read_text()

    alias_line = "SPECS['fast_mis_2'] = SPECS['fast_mis']"
    if alias_line not in text:
        text = text.rstrip() + f"\n{alias_line}\n"
        specs_path.write_text(text)
        print(f"Patched {specs_path}: added fast_mis_2 spec alias")
    else:
        print(f"Specs already patched")


def main():
    pkg_dir = _find_salsaclrs()
    print(f"Found salsaclrs at: {pkg_dir}")
    patch_sampler(pkg_dir)
    patch_specs(pkg_dir)
    print("Done.")


if __name__ == "__main__":
    main()
