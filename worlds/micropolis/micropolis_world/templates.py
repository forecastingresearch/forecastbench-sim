"""Micropolis question templates + the world's registry."""

from fbsim_core.questions.registry import TemplateRegistry
from fbsim_core.questions.schema import QuestionTemplate

from . import module_globals as g


def _get_continuous_template(metric: str) -> QuestionTemplate:
    metric_label = g.METRIC_LABELS[metric]
    return QuestionTemplate(
        template_id=f"{metric}_continuous",
        signal_name=metric,
        # Note we assume there is only one region/player in the Micropolis world, so we don't need to specify a region_id in the question.
        question_template=f"What will the {metric_label} be at turn {{resolution_turn}}?",
        resolution_type="continuous",
        # Turn-major, matching fbsim_core.questions.timeseries and what
        # city_sim.to_world() writes. Descriptive only: the resolver reads the
        # series through signal_at() and never parses this string.
        data_path=f"time_series.{metric}.{{resolution_turn}}.{{player_id}}",
        comparison_op="value",
        required_params=["player_id", "resolution_turn"],
    )


Q_METRICS = [
    #    "cityScore",
    "cityPop",
    "totalFunds",
    "trafficAverage",
    "pollutionAverage",
    "crimeAverage",
    "landValueAverage",
]

ALL_TEMPLATES: list[QuestionTemplate] = [
    _get_continuous_template(metric) for metric in Q_METRICS
]

REGISTRY = TemplateRegistry(ALL_TEMPLATES)


def asked_templates(censor_city_funds: bool) -> list[QuestionTemplate]:
    """ALL_TEMPLATES, minus the city funds question when censoring.

    Censoring hides the balance from the world report, so asking for it would
    be a forecast from nothing the report states; the metric is dropped from
    the corpus instead. REGISTRY keeps every template either way, so a funds
    question still resolves if something else asks one.
    """
    if not censor_city_funds:
        return ALL_TEMPLATES
    return [t for t in ALL_TEMPLATES if t.signal_name != g.FUNDS_METRIC]
