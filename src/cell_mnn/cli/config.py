"""
Shared `--config` loading for the training CLIs.
"""

import argparse
import tomllib
from pathlib import Path
from typing import Optional, Sequence

from cell_mnn.checks import require


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--config', type=str, default=None,
                         help='Path to a TOML file of flag-name -> value, applied as '
                              'defaults (CLI flags still override it)')


def apply_config_defaults(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]]) -> None:
    """
    Pre-parse `argv` for `--config` and, if given, `set_defaults` the parser
    from its (flat) TOML table. Call after every `add_argument`, before the
    real `parser.parse_args(argv)`.
    """
    known, _ = parser.parse_known_args(argv)
    config_path = known.config
    if config_path is None:
        return

    path = Path(config_path)
    require(path.is_file(), f"no run config at {path}")

    try:
        with path.open("rb") as f:
            values = tomllib.load(f)
    except tomllib.TOMLDecodeError as err:
        raise ValueError(f"{path}: {err}") from None

    # The parser's own flags are the schema -- a typo'd key is a clear error, not silently
    # ignored by set_defaults.
    actions_by_dest = {a.dest: a for a in parser._actions if a.dest not in ("help", "config")}
    unknown = set(values) - set(actions_by_dest)
    require(not unknown,
            f"{path}: unknown key(s) {sorted(unknown)}; must be a subset of "
            f"{sorted(actions_by_dest)}")

    # set_defaults bypasses argparse's own `choices` check, so an invalid enum value in
    # the config would otherwise pass straight through unnoticed.
    for key, value in values.items():
        choices = actions_by_dest[key].choices
        if choices is not None:
            require(value in choices,
                    f"{path}: {key}={value!r} not in {sorted(choices)}")

    parser.set_defaults(**values)
