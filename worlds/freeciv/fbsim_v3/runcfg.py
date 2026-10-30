"""runcfg.py — one FreeCiv v3 run's configuration, read from data/freeciv/fbsim_v3/runs/<run>/run.env.

The run is named by `--run NAME` or FBSIM_RUN; with neither, the paper's settings apply (5 players, fork turn 60,
horizons 90..210). fleet/1_new_run.sh writes run.env; fleet/paths.sh sources the same file.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULTS = {"FBSIM_N_PLAYERS": "5", "FBSIM_BARBARIANS": "", "FBSIM_FORK_TURN": "60",
            "FBSIM_HORIZONS": "90,120,150,180,210", "FBSIM_RNG": "5001-6000", "FBSIM_NATCOND": "1", "FBSIM_SEEDS": ""}


def runs_root() -> Path:
    """<repo>/data/freeciv/fbsim_v3/runs, located from this file as fleet/paths.sh does (no fbsim_core needed)."""
    return Path(__file__).resolve().parents[3] / "data" / "freeciv" / "fbsim_v3" / "runs"


def parse_env(path) -> dict:
    """KEY=VALUE lines; blank lines, comments and surrounding quotes ignored."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("\"'")
    return out


@dataclass(frozen=True)
class Run:
    name: str | None
    dir: Path | None
    n_players: int
    barbarians: str
    fork_turn: int
    horizons: tuple
    rng: str
    natcond: bool
    seeds: str

    @property
    def start(self):        # first turn played after the snapshot
        return self.fork_turn + 1

    @property
    def reveal_end(self):   # natural conditionals reveal turns start..reveal_end
        return self.horizons[0]

    @property
    def end_turn(self):     # replays are played to here
        return self.horizons[-1]

    @property
    def report_name(self):
        return f"turn_{self.fork_turn:03d}_report.txt"


def load(name=None) -> Run:
    name = name or os.environ.get("FBSIM_RUN") or None
    cfg, d = dict(DEFAULTS), None
    if name:
        d = runs_root() / name
        if not (d / "run.env").is_file():
            raise SystemExit(f"run {name!r}: no {d / 'run.env'} (create it with fleet/1_new_run.sh)")
        cfg.update(parse_env(d / "run.env"))
    return Run(name=name, dir=d, n_players=int(cfg["FBSIM_N_PLAYERS"]), barbarians=cfg["FBSIM_BARBARIANS"],
               fork_turn=int(cfg["FBSIM_FORK_TURN"]), horizons=tuple(int(h) for h in cfg["FBSIM_HORIZONS"].split(",")),
               rng=cfg["FBSIM_RNG"], natcond=cfg["FBSIM_NATCOND"] == "1", seeds=cfg["FBSIM_SEEDS"])


# ---- elicitation: the paper's prompt sentences, rewritten for a run's fork turn and civilization count
PAPER_BACKGROUND = "observed at turn 60. Five AI civilizations are competing."
PAPER_NO_NEWS = "No new information about turns 61 through 90 is available."
CIVS_LINES = {1: "One AI civilization is playing alone.", 2: "Two AI civilizations are competing.",
              3: "Three AI civilizations are competing.", 4: "Four AI civilizations are competing.",
              5: "Five AI civilizations are competing."}


def n_civs(run: Run) -> int:
    """FBSIM_FREECIV_N_CIVS if set, else the run's player count."""
    n = int(os.environ.get("FBSIM_FREECIV_N_CIVS") or run.n_players)
    if n not in CIVS_LINES:
        raise SystemExit(f"civilization count {n}: expected one of 1-5")
    return n


def background(text: str, run: Run) -> str:
    assert PAPER_BACKGROUND in text, "prompt lacks the paper's background sentence"
    return text.replace(PAPER_BACKGROUND, f"observed at turn {run.fork_turn}. {CIVS_LINES[n_civs(run)]}")


def no_news(text: str, run: Run) -> str:
    assert PAPER_NO_NEWS in text, "prompt lacks the paper's no-news sentence"
    return text.replace(PAPER_NO_NEWS, f"No new information about turns {run.start} through {run.reveal_end} is available.")


def eval_data(run: Run, default) -> Path:
    """Folder with sets/, reports/ and run/models_v*.csv: FBSIM_FREECIV_V3_DATA, else the run folder, else `default`."""
    return Path(os.environ.get("FBSIM_FREECIV_V3_DATA") or run.dir or default)


def pop_run(argv=sys.argv) -> Run:
    """Remove `--run NAME` from argv (in place) and load that run, else FBSIM_RUN, else the paper's defaults."""
    name = None
    if "--run" in argv:
        i = argv.index("--run")
        name = argv[i + 1]
        del argv[i:i + 2]
    return load(name)


def positional(run: Run, defaults, argv=sys.argv, usage=""):
    """The leading positional arguments, each missing one taken from `defaults` (paths inside the run folder)."""
    pos = []
    for a in argv[1:]:
        if a.startswith("--"):
            break
        pos.append(a)
    if len(pos) < len(defaults) and run.dir is None:
        raise SystemExit(f"usage: {usage or argv[0]} (or name a run with --run / FBSIM_RUN)")
    return pos[:len(defaults)] + [str(run.dir / d) for d in defaults[len(pos):]]
