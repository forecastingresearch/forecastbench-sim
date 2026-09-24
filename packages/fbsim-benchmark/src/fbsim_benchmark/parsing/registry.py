"""Select a native parser explicitly; never infer or repair response format."""
from . import freeciv_single, freeciv_batch, micropolis, starsim_binary, starsim_continuous, starsim_strict
PARSERS = {
    ('freeciv','binary-single'): freeciv_single.parse_prob,
    ('freeciv','continuous-single'): freeciv_single.parse_pct,
    ('freeciv','batch'): freeciv_batch.parse_block,
    ('micropolis','binary-single'): micropolis.parse_probability,
    ('micropolis','binary-batch'): micropolis.parse_batch_probabilities,
    ('micropolis','continuous-single'): micropolis.parse_percentiles,
    ('micropolis','continuous-batch'): micropolis.parse_batch_percentiles,
    ('micropolis','continuous-semantic'): micropolis.parse_batch_percentiles_semantic,
    ('starsim','binary'): starsim_strict.parse,
    ('starsim','binary-historical'): starsim_binary.parse,
    ('starsim','continuous'): starsim_continuous.parse,
}
def parse_response(world, format, text, **options):
    """Select strict Starsim binary parsing or an explicitly documented native parser.

    See docs/NATIVE_INTERFACES.md for required options and scientific differences.
    This parses already obtained text; it does not load truth or score a forecast.
    """
    try: parser = PARSERS[(world, format)]
    except KeyError: raise ValueError('Unsupported world/format; consult the parser catalogue') from None
    return parser(text, **options)
