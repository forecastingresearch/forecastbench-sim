"""Config-file loading for the scripts in worlds/micropolis/scripts/.

Every script takes an optional config file path as its first positional argument
and reads all of its non-flag parameters from there; behavior toggles
(--dry-run, --quiet, --plot) stay on the command line, as do --seed, --cities,
--disasters, --models and --label, which override the parameters that most
often vary run to run. Omitting the path falls back to configs/default.json5,
or to whatever a script passes as add_config_args(default=...) when its own
needs differ.

A script asks for the keys it needs via Config.get* and errors out if one is
missing, so a single config file can carry the union of every script's
parameters and still be usable by each of them individually.

Configs are parsed as JSON5, so they may contain // and /* */ comments and
trailing commas. Plain JSON is a subset of JSON5 and loads unchanged.
"""

import argparse
import functools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import json5

from . import messages as msg
from . import module_globals as g

# The installed synthetic default travels with the wheel. User configs may
# be supplied explicitly; no production config is selected implicitly.
CONFIG_DIR = Path(__file__).resolve().parent / "examples"

# The hand-edited data beside the code: the prompt preamble/epilogue texts a
# config names, and the model tables. Separate from configs/ so a directory
# listing of the configs is a listing of the runnable configs.
DATAFILES_DIR = Path(__file__).resolve().parent / "datafiles"

# The orders build_corpus can emit its questions in, and so the order they are
# numbered in each batch prompt. "turn" groups every metric for one horizon
# together; "metric" groups every horizon for one metric together. The order is
# part of the prompt text, so switching simply hashes to a new cache entry.
QUESTIONS_SORT_TURN = "turn"
QUESTIONS_SORT_METRIC = "metric"
QUESTIONS_SORTS = (QUESTIONS_SORT_TURN, QUESTIONS_SORT_METRIC)

# How a batch's questions are tagged in the prompt, and so how the answer lines
# are matched back to them. "numeric" numbers the questions and reads "Q<n>:"
# lines; "semantic" tags each with "<metric label>@<turn>" and reads lines
# carrying that tag, which lets a model answer out of order without its
# answers sliding onto the wrong questions.
QUESTION_TAGGING_NUMERIC = "numeric"
QUESTION_TAGGING_SEMANTIC = "semantic"
QUESTION_TAGGINGS = (QUESTION_TAGGING_NUMERIC, QUESTION_TAGGING_SEMANTIC)

# The floor analyze_continuous.py's per-question normalization modes put under
# a denominator, as a share of the metric's own global scale. 1% is low enough
# that it binds only where a metric barely moves over the continuations — a
# city whose traffic is pinned at 0 — and high enough to keep such a question's
# normalized CRPS inside two orders of magnitude of the rest. Overridable per
# config ('norm_global_frac') and per run (--norm-global-frac); each report
# says how many questions it bound.
DEFAULT_NORM_GLOBAL_FRAC = 0.01

# .json5 rather than .json so editors don't flag the comments as syntax errors.
# Either extension loads; the parser is the same.
DEFAULT_CONFIG_PATH = CONFIG_DIR / "example.json5"


class ConfigError(Exception):
    """A config file is missing, malformed, or missing a key a script needs."""


class Config:
    """Parameters loaded from a JSON5 config file.

    Access is by explicit getter so that a missing key is a hard error naming
    both the key and the file it should have been in, rather than a KeyError or
    a silent default.
    """

    def __init__(self, data: dict[str, Any], path: Path):
        self.data = data
        self.path = path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        try:
            # JSON5, so configs can carry // and /* */ comments and trailing
            # commas. Plain JSON is a subset, so existing configs still load.
            data = json5.loads(path.read_text())
        except ValueError as e:
            raise ConfigError(f"config file {path} is not valid JSON5: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError(f"config file {path} must contain a JSON object")
        cfg = cls(data, path)
        # Loading a config selects the data directory the whole process works
        # in, so it happens here rather than in each script: every reader of a
        # cached path goes through module_globals, and none of them takes a
        # config. Absent, the key means the default world, which is also a
        # choice — a later config naming another directory is an error.
        try:
            g.set_data_dir(cfg.get_str_or("data_dir", g.DEFAULT_DATA_SUBDIR), path)
        except ValueError as e:
            raise ConfigError(f"parameter 'data_dir' in {path}: {e}") from e
        return cfg

    def _require(self, key: str) -> Any:
        if key not in self.data:
            raise ConfigError(f"missing required parameter '{key}' in {self.path}")
        return self.data[key]

    def get_int(self, key: str) -> int:
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be an integer, got {value!r}"
            )
        return value

    def get_int_or(self, key: str, default: int) -> int:
        """The integer at `key`, or `default` if the config doesn't set it.

        Same reasoning as get_bool_or: a parameter added after configs were
        already in use should not turn every existing config into an error.
        """
        if key not in self.data:
            return default
        return self.get_int(key)

    def get_float(self, key: str) -> float:
        value = self._require(key)
        # An int is a fine float here — a config writing 1 rather than 1.0 for
        # a ratio means the same thing — but a bool is an int in Python and
        # never means a number in a config.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a number, got {value!r}"
            )
        return float(value)

    def get_float_or(self, key: str, default: float) -> float:
        """The number at `key`, or `default` if the config doesn't set it.

        Same reasoning as get_bool_or: a parameter added after configs were
        already in use should not turn every existing config into an error.
        """
        if key not in self.data:
            return default
        return self.get_float(key)

    def get_bool(self, key: str) -> bool:
        value = self._require(key)
        if not isinstance(value, bool):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a boolean, got {value!r}"
            )
        return value

    def get_bool_or(self, key: str, default: bool) -> bool:
        """The boolean at `key`, or `default` if the config doesn't set it.

        For flags added after configs were already in use: an absent key means
        the old behavior rather than an error, so existing config files keep
        working without having to name every new toggle.
        """
        if key not in self.data:
            return default
        return self.get_bool(key)

    def get_str(self, key: str) -> str:
        value = self._require(key)
        if not isinstance(value, str):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a string, got {value!r}"
            )
        return value

    def get_str_or(self, key: str, default: str) -> str:
        """The string at `key`, or `default` if the config doesn't set it.

        Same reasoning as get_bool_or: a parameter added after configs were
        already in use should not turn every existing config into an error.
        """
        if key not in self.data:
            return default
        return self.get_str(key)

    def get_int_list(self, key: str) -> list[int]:
        value = self._require(key)
        if (
            not isinstance(value, list)
            or not value
            or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
        ):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a non-empty list of "
                f"integers, got {value!r}"
            )
        return value

    def get_str_list(self, key: str) -> list[str]:
        value = self._require(key)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(v, str) for v in value)
        ):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a non-empty list of "
                f"strings, got {value!r}"
            )
        return value

    def get_bool_list(self, key: str) -> list[bool]:
        value = self._require(key)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(v, bool) for v in value)
        ):
            raise ConfigError(
                f"parameter '{key}' in {self.path} must be a non-empty list of "
                f"booleans, got {value!r}"
            )
        return value

    def get_concurrency(self) -> int | None:
        """The optional 'concurrency' key, or None when unset.

        The maximum number of API calls in flight at once, across every model
        — one global cap, since every call leaves through the same gateway.
        None (the usual case) means prompting.DEFAULT_CONCURRENCY applies.
        """
        if "concurrency" not in self.data:
            return None
        value = self.data["concurrency"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(
                f"parameter 'concurrency' in {self.path} must be a positive "
                f"integer, got {value!r}"
            )
        return value

    def get_preamble_path(self) -> Path | None:
        """The optional 'preamble_path', resolved against DATAFILES_DIR.

        Names the file holding the forecasting prompt's preamble, which must
        contain a "{sources}" placeholder (see scenarios.prompt_preamble). None
        means scenarios.DEFAULT_PREAMBLE_PATH applies, so a config written
        before the key existed keeps the preamble it was run with.
        """
        if "preamble_path" not in self.data:
            return None
        path = DATAFILES_DIR / self.get_str("preamble_path")
        if not path.exists():
            raise ConfigError(
                f"preamble file not found: {path} "
                f"(named by parameter 'preamble_path' in {self.path})"
            )
        return path

    def get_epilogue_path(self) -> Path | None:
        """The optional 'epilogue_path', resolved against DATAFILES_DIR.

        Names the file holding the forecasting prompt's epilogue — the answer
        format instructions after the questions — which may contain an "{n}"
        placeholder (see scenarios.read_epilogue). None means
        scenarios.DEFAULT_EPILOGUE_PATH applies, so a config written before the
        key existed keeps the epilogue it was run with.
        """
        if "epilogue_path" not in self.data:
            return None
        path = DATAFILES_DIR / self.get_str("epilogue_path")
        if not path.exists():
            raise ConfigError(
                f"epilogue file not found: {path} "
                f"(named by parameter 'epilogue_path' in {self.path})"
            )
        return path

    def get_questions_sort(self) -> str:
        """The optional 'questions_sort', saying how a batch's questions are ordered.

        "turn" (the default) numbers every metric for one horizon before moving
        to the next horizon; "metric" numbers every horizon for one metric
        first. The order is part of the prompt text, so switching misses the
        response cache rather than mixing variants, and an absent key means the
        order configs were already run with.
        """
        value = self.get_str_or("questions_sort", QUESTIONS_SORT_TURN)
        if value not in QUESTIONS_SORTS:
            raise ConfigError(
                f"parameter 'questions_sort' in {self.path} must be one of "
                f"{', '.join(QUESTIONS_SORTS)}, got {value!r}"
            )
        return value

    def get_question_tagging(self) -> str:
        """The optional 'question_tagging', saying how questions are tagged.

        "numeric" (the default) numbers the questions and expects "Q<n>:"
        answer lines; "semantic" tags them "<metric label>@<turn>" and expects
        answer lines carrying the same tag. Part of the prompt text, so
        switching misses the response cache rather than mixing variants, and an
        absent key means the numbering configs were already run with.
        """
        value = self.get_str_or("question_tagging", QUESTION_TAGGING_NUMERIC)
        if value not in QUESTION_TAGGINGS:
            raise ConfigError(
                f"parameter 'question_tagging' in {self.path} must be one of "
                f"{', '.join(QUESTION_TAGGINGS)}, got {value!r}"
            )
        return value

    def get_questions_per_prompt(self) -> int:
        """The optional 'questions_per_prompt', capping questions per prompt.

        -1 (the default) means no cap: every question a scenario asks at one
        snapshot goes in a single prompt, which is the cheapest way to ask them
        since they share a game report. A positive value splits a batch that
        would exceed it, repeating the report in each prompt (see
        continuous_eval.group_into_batches). Part of what the prompt asks, so
        changing it misses the response cache rather than mixing variants.
        """
        value = self.get_int_or("questions_per_prompt", -1)
        if value == 0 or value < -1:
            raise ConfigError(
                f"parameter 'questions_per_prompt' in {self.path} must be -1 "
                f"(no cap) or a positive integer, got {value!r}"
            )
        return value

    def get_cities(self, override: list[str] | None = None) -> list[str]:
        """The 'cities' list, or override when one was passed on the command line.

        Either way the names are validated against the known Micropolis cities,
        so a typo on the command line fails the same way one in a config does.
        """
        if override is not None:
            cities, source = override, "--cities"
        else:
            cities = self.get_str_list("cities")
            source = f"parameter 'cities' in {self.path}"
        unknown = [c for c in cities if c not in g.CITY_CHOICES]
        if unknown:
            raise ConfigError(
                f"{source} names unknown "
                f"{'city' if len(unknown) == 1 else 'cities'}: {', '.join(unknown)}"
            )
        return cities

    def get_disasters(self, override: list[bool] | None = None) -> list[bool]:
        """The 'disasters' list, or override when one was passed on the command line."""
        if override is not None:
            return override
        return self.get_bool_list("disasters")

    def get_models(self, override: list[str] | None = None) -> list[str]:
        """The 'models' list, or override when one was passed on the command line.

        'models' may also be a string: the path, relative to the config file's
        directory, of a JSON5 file holding the list itself, so configs that
        prompt the same model set can share one definition.

        Model ids are not validated against a known set the way cities are:
        the providers add and retire names constantly, so the only real check
        is whether the call succeeds.
        """
        if override is not None:
            return override
        value = self._require("models")
        if not isinstance(value, str):
            return self.get_str_list("models")
        path = self.path.parent / value
        if not path.exists():
            raise ConfigError(
                f"models file not found: {path} "
                f"(named by parameter 'models' in {self.path})"
            )
        try:
            models = json5.loads(path.read_text())
        except ValueError as e:
            raise ConfigError(f"models file {path} is not valid JSON5: {e}") from e
        if (
            not isinstance(models, list)
            or not models
            or any(not isinstance(m, str) for m in models)
        ):
            raise ConfigError(
                f"models file {path} must contain a non-empty list of "
                f"strings, got {models!r}"
            )
        return models

    def get_norm_global_frac(self, override: float | None = None) -> float:
        """The floor on a per-question CRPS denominator, as a share of the
        metric's global scale.

        analyze_continuous.py's local and baseline normalizations divide by a
        number the question itself supplies, which on the scenarios where a
        metric cannot move at all goes to 0 — or near enough that the cell
        would swamp any mean it entered. Both floor the denominator at this
        fraction of continuous_eval.GLOBAL_SCALES, so no question is dropped
        and no denominator is unbounded. `override` is --norm-global-frac.
        """
        value = (
            override
            if override is not None
            else self.get_float_or("norm_global_frac", DEFAULT_NORM_GLOBAL_FRAC)
        )
        if not 0 <= value < 1:
            raise ConfigError(
                f"'norm_global_frac' must be in [0, 1), got {value!r}: it is a "
                "share of the metric's global scale, and at 1 every question "
                "would be floored onto that scale"
            )
        return value

    def get_seed(self, override: int | None = None) -> int:
        """The 'seed', or override when one was passed on the command line."""
        return override if override is not None else self.get_int("seed")

    def get_label(self, override: str | None = None) -> str:
        """The 'label', or the config file's stem if it has none.

        Names the continuous eval's per-run output directory, so every config
        gets a distinct one even without setting the key explicitly. `override`
        is a --label from the command line, which wins over both.
        """
        if override is not None:
            return override
        value = self.data.get("label")
        if value is None:
            return self.path.stem
        if not isinstance(value, str):
            raise ConfigError(
                f"parameter 'label' in {self.path} must be a string, got {value!r}"
            )
        return value


def _bool_arg(value: str) -> bool:
    """Parse a --disasters entry, so `--disasters true false` reads as a list.

    argparse's own bool() would make every non-empty string True, silently
    turning `--disasters false` into a disasters-on run.
    """
    if value.lower() in ("true", "yes", "on", "1"):
        return True
    if value.lower() in ("false", "no", "off", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def add_config_args(
    ap: argparse.ArgumentParser,
    default: Path | None = None,
    many: bool = False,
) -> None:
    """Add the config-file argument and the parameter overrides every script takes.

    The overrides — --seed, --cities, --disasters, --models, --label — are the
    parameters that most often vary run to run, so they are worth a flag even
    though everything else comes from the config file. A script that reads none
    of them still gets the flags; they are simply unused.

    `default` names the config to use when the positional argument is omitted,
    for a script whose natural default is not default.json5. Passed here rather
    than applied by the caller after parsing, so the --help text names the file
    the script will actually read.

    `many` takes a list of config files rather than one, for a script that
    reports over several runs at once; the overrides then apply to each of
    them. Read the list with load_configs(), which also covers the single-config
    case so a script can switch without its caller changing.
    """
    if many:
        ap.add_argument(
            "config",
            nargs="*",
            # The same fallback the single-config form has, as a one-element
            # list, so naming no config still reports on the default one.
            default=[default or DEFAULT_CONFIG_PATH],
            metavar="CONFIG",
            help="One or more JSON5 config files "
            f"(default: {default or DEFAULT_CONFIG_PATH})",
        )
    else:
        ap.add_argument(
            "config",
            nargs="?",
            default=default,
            help=f"JSON5 config file (default: {default or DEFAULT_CONFIG_PATH})",
        )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to run at, overriding the config's 'seed'",
    )
    ap.add_argument(
        "--cities",
        nargs="+",
        metavar="CITY",
        default=None,
        help="City names to run, overriding the config's 'cities' list. "
        f"One or more of: {', '.join(g.CITY_CHOICES)}",
    )
    ap.add_argument(
        "--disasters",
        nargs="+",
        metavar="BOOL",
        type=_bool_arg,
        default=None,
        help="Disaster settings to run each city under, overriding the config's "
        "'disasters' list. 'true false' runs both variants of every city",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="Model ids to prompt, overriding the config's 'models' list. "
        "Ids are in provider/name form; see data/micropolis/available_models.md",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="Name of the output directory under data/micropolis/continuous/, "
        "overriding the config's 'label'",
    )


def load_config(args: argparse.Namespace) -> Config:
    """Load the config named by parsed args, exiting with a message on failure."""
    try:
        return Config.load(args.config)
    except ConfigError as e:
        msg.error(str(e))
        sys.exit(1)


def load_configs(args: argparse.Namespace) -> list[Config]:
    """Every config named by parsed args, exiting with a message on failure.

    For a script that took add_config_args(many=True). Naming no config at all
    leaves the argument's default in place — the same single file load_config()
    would have read — so running the script bare keeps behaving as before.
    """
    paths = args.config if isinstance(args.config, list) else [args.config]
    configs = []
    for path in paths:
        try:
            configs.append(Config.load(path))
        except ConfigError as e:
            msg.error(str(e))
            sys.exit(1)
    return configs


def main_with_config(main: Callable[[], None]) -> Callable[[], None]:
    """Wrap a script's main() so a bad parameter exits with a message.

    Parameters are read lazily as the script needs them, so a missing or
    ill-typed key surfaces well after load_config() returned; without this the
    script would die with a traceback instead of a one-line error.
    """

    @functools.wraps(main)
    def wrapper() -> None:
        try:
            main()
        except ConfigError as e:
            msg.error(str(e))
            sys.exit(1)

    return wrapper


def scenarios_from(
    cfg: Config,
    cities: list[str] | None = None,
    disasters: list[bool] | None = None,
) -> list[tuple[str, bool]]:
    """The (city, disasters) pairs a run covers: cities x disasters.

    `cities` and `disasters` override the config's lists, for a --cities or
    --disasters on the command line.
    """
    return [
        (city, dis)
        for city in cfg.get_cities(cities)
        for dis in cfg.get_disasters(disasters)
    ]
