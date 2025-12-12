"""
Predefined question templates for B1, B2, and B3 signals.

B1 (High Base Rate): Deterministic/well-documented mechanics
B2 (Medium Base Rate): Observable trends with stochastic elements
B3 (Low Base Rate): Opaque AI behavior, emergent dynamics, rare events
"""

from .schema import QuestionTemplate


# =============================================================================
# B1 Templates: High Base Rate Availability
# =============================================================================

TECH_COUNT_GTE = QuestionTemplate(
    template_id="tech_count_gte",
    signal_type="B1",
    signal_name="techs_known",
    question_template="Will {civ} have ≥{threshold} technologies at turn {resolution_turn}?",
    resolution_type="threshold",
    data_path="time_series.techs_known.{player_id}.{resolution_turn}",
    comparison_op=">=",
    required_params=["civ", "player_id", "threshold", "resolution_turn"],
)

TECH_DISCOVERED = QuestionTemplate(
    template_id="tech_discovered",
    signal_type="B1",
    signal_name="techs_known",
    question_template="Will {civ} have discovered {tech_name} by turn {resolution_turn}?",
    resolution_type="state_check",
    data_path="tech_state.{player_id}.{resolution_turn}",
    comparison_op="contains",
    required_params=["civ", "player_id", "tech_name", "tech_id", "resolution_turn"],
)

POPULATION_GTE = QuestionTemplate(
    template_id="population_gte",
    signal_type="B1",
    signal_name="population",
    question_template="Will {civ}'s population exceed {threshold} at turn {resolution_turn}?",
    resolution_type="threshold",
    data_path="time_series.population.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["civ", "player_id", "threshold", "resolution_turn"],
)

SCORE_RANK_1 = QuestionTemplate(
    template_id="score_rank_1",
    signal_type="B1",
    signal_name="scores",
    question_template="Will {civ} be ranked #1 at turn {resolution_turn}?",
    resolution_type="comparison",
    data_path="snapshots.{resolution_turn}.rankings",
    comparison_op="rank==1",
    required_params=["civ", "player_id", "resolution_turn"],
)

SCORE_GTE = QuestionTemplate(
    template_id="score_gte",
    signal_type="B1",
    signal_name="scores",
    question_template="Will {civ}'s score exceed {threshold} at turn {resolution_turn}?",
    resolution_type="threshold",
    data_path="snapshots.{resolution_turn}.scores.{player_id}",
    comparison_op=">",
    required_params=["civ", "player_id", "threshold", "resolution_turn"],
)


# =============================================================================
# B2 Templates: Medium Base Rate Availability
# =============================================================================

TERRITORY_GTE = QuestionTemplate(
    template_id="territory_gte",
    signal_type="B2",
    signal_name="territory_size",
    question_template="Will {civ} control ≥{threshold} tiles at turn {resolution_turn}?",
    resolution_type="threshold",
    data_path="time_series.territory_size.{player_id}.{resolution_turn}",
    comparison_op=">=",
    required_params=["civ", "player_id", "threshold", "resolution_turn"],
)

TERRITORY_GAIN = QuestionTemplate(
    template_id="territory_gain",
    signal_type="B2",
    signal_name="territory_size",
    question_template="Will {civ} have gained ≥{threshold} tiles between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="threshold",
    data_path="time_series.territory_size.{player_id}",
    comparison_op=">=",
    required_params=["civ", "player_id", "threshold", "snapshot_turn", "resolution_turn"],
)

TREASURY_GTE = QuestionTemplate(
    template_id="treasury_gte",
    signal_type="B2",
    signal_name="treasury",
    question_template="Will {civ}'s treasury exceed {threshold} gold at turn {resolution_turn}?",
    resolution_type="threshold",
    data_path="time_series.treasury.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["civ", "player_id", "threshold", "resolution_turn"],
)

CITIES_GTE = QuestionTemplate(
    template_id="cities_gte",
    signal_type="B2",
    signal_name="cities_count",
    question_template="Will {civ} have ≥{threshold} cities at turn {resolution_turn}?",
    resolution_type="threshold",
    data_path="time_series.cities_count.{player_id}.{resolution_turn}",
    comparison_op=">=",
    required_params=["civ", "player_id", "threshold", "resolution_turn"],
)

CITY_FOUNDED = QuestionTemplate(
    template_id="city_founded",
    signal_type="B2",
    signal_name="events",
    question_template="Will {civ} found a new city between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["civ", "player_id", "snapshot_turn", "resolution_turn"],
)


# =============================================================================
# B3 Templates: Low Base Rate Availability
# =============================================================================

AT_WAR_DYAD = QuestionTemplate(
    template_id="at_war_dyad",
    signal_type="B3",
    signal_name="diplomacy",
    question_template="Will {civ_a} and {civ_b} be at war at turn {resolution_turn}?",
    resolution_type="state_check",
    data_path="diplomacy.relations.{player_id_a}_{player_id_b}.{resolution_turn}.state",
    comparison_op="==",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)

AT_WAR_ANY = QuestionTemplate(
    template_id="at_war_any",
    signal_type="B3",
    signal_name="diplomacy",
    question_template="Will {civ} be at war with any civilization at turn {resolution_turn}?",
    resolution_type="state_check",
    data_path="diplomacy.relations",
    comparison_op="any",
    required_params=["civ", "player_id", "resolution_turn"],
)

CITY_CONQUERED_ANY = QuestionTemplate(
    template_id="city_conquered_any",
    signal_type="B3",
    signal_name="events",
    question_template="Will any city be conquered between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["snapshot_turn", "resolution_turn"],
)

CITY_LOST = QuestionTemplate(
    template_id="city_lost",
    signal_type="B3",
    signal_name="events",
    question_template="Will {civ} lose a city between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["civ", "player_id", "snapshot_turn", "resolution_turn"],
)

GOVERNMENT_AT = QuestionTemplate(
    template_id="government_at",
    signal_type="B3",
    signal_name="government",
    question_template="Will {civ} be in {government_type} at turn {resolution_turn}?",
    resolution_type="state_check",
    data_path="government_state.{player_id}.{resolution_turn}",
    comparison_op="==",
    required_params=["civ", "player_id", "government_type", "resolution_turn"],
)

ANARCHY_EVENT = QuestionTemplate(
    template_id="anarchy_event",
    signal_type="B3",
    signal_name="events",
    question_template="Will {civ} experience anarchy between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["civ", "player_id", "snapshot_turn", "resolution_turn"],
)

ALLIANCE_DYAD = QuestionTemplate(
    template_id="alliance_dyad",
    signal_type="B3",
    signal_name="diplomacy",
    question_template="Will {civ_a} and {civ_b} have an alliance at turn {resolution_turn}?",
    resolution_type="state_check",
    data_path="diplomacy.relations.{player_id_a}_{player_id_b}.{resolution_turn}.state",
    comparison_op="==",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)

WONDER_COMPLETED = QuestionTemplate(
    template_id="wonder_completed",
    signal_type="B3",
    signal_name="events",
    question_template="Will {wonder_name} be completed by turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["wonder_name", "wonder_id", "snapshot_turn", "resolution_turn"],
)

WONDER_FIRST = QuestionTemplate(
    template_id="wonder_first",
    signal_type="B3",
    signal_name="events",
    question_template="Will {civ} complete {wonder_name} before any other civilization?",
    resolution_type="event",
    data_path="events",
    comparison_op="first",
    required_params=["civ", "player_id", "wonder_name", "wonder_id", "snapshot_turn", "resolution_turn"],
)


# =============================================================================
# Template Registry
# =============================================================================

B1_TEMPLATES = [
    TECH_COUNT_GTE,
    TECH_DISCOVERED,
    POPULATION_GTE,
    SCORE_RANK_1,
    SCORE_GTE,
]

B2_TEMPLATES = [
    TERRITORY_GTE,
    TERRITORY_GAIN,
    TREASURY_GTE,
    CITIES_GTE,
    CITY_FOUNDED,
]

B3_TEMPLATES = [
    AT_WAR_DYAD,
    AT_WAR_ANY,
    CITY_CONQUERED_ANY,
    CITY_LOST,
    GOVERNMENT_AT,
    ANARCHY_EVENT,
    ALLIANCE_DYAD,
    WONDER_COMPLETED,
    WONDER_FIRST,
]

ALL_TEMPLATES = B1_TEMPLATES + B2_TEMPLATES + B3_TEMPLATES

TEMPLATES_BY_ID: dict[str, QuestionTemplate] = {t.template_id: t for t in ALL_TEMPLATES}

TEMPLATES_BY_SIGNAL_TYPE: dict[str, list[QuestionTemplate]] = {
    "B1": B1_TEMPLATES,
    "B2": B2_TEMPLATES,
    "B3": B3_TEMPLATES,
}


def get_template(template_id: str) -> QuestionTemplate:
    """Get a template by its ID."""
    if template_id not in TEMPLATES_BY_ID:
        raise ValueError(f"Unknown template_id: {template_id}")
    return TEMPLATES_BY_ID[template_id]


def get_templates_by_signal_type(signal_type: str) -> list[QuestionTemplate]:
    """Get all templates for a given signal type (B1, B2, B3)."""
    if signal_type not in TEMPLATES_BY_SIGNAL_TYPE:
        raise ValueError(f"Unknown signal_type: {signal_type}")
    return TEMPLATES_BY_SIGNAL_TYPE[signal_type]
