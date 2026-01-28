"""
Predefined question templates for CivBench forecasting questions.

Each template defines a reusable question pattern with:
- signal_name: The metric being measured
- question_template: Text with placeholders
- resolution_type: How the question is resolved
- data_path: Where to find the data
- comparison_op: How to compare values
"""

from .schema import QuestionTemplate


# =============================================================================
# Comparative Templates
# =============================================================================

TECH_COMPARATIVE = QuestionTemplate(
    template_id="tech_comparative",
    signal_name="techs_known",
    question_template="Will {civ_a} have more technologies than {civ_b} at turn {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.techs_known.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)

SCORE_COMPARATIVE = QuestionTemplate(
    template_id="score_comparative",
    signal_name="scores",
    question_template="Will {civ_a} have a higher score than {civ_b} at turn {resolution_turn}?",
    resolution_type="comparative",
    data_path="snapshots.{resolution_turn}.scores.{player_id}",
    comparison_op=">",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)

POPULATION_COMPARATIVE = QuestionTemplate(
    template_id="population_comparative",
    signal_name="population",
    question_template="Will {civ_a} have a larger total population than {civ_b} at turn {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.population.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)

CITY_COUNT_COMPARATIVE = QuestionTemplate(
    template_id="city_count_comparative",
    signal_name="cities_count",
    question_template="Will {civ_a} have more cities than {civ_b} at turn {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.cities_count.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)

TERRITORY_COMPARATIVE = QuestionTemplate(
    template_id="territory_comparative",
    signal_name="territory_size",
    question_template="Will {civ_a} control more tiles than {civ_b} at turn {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.territory_size.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)

TREASURY_COMPARATIVE = QuestionTemplate(
    template_id="treasury_comparative",
    signal_name="treasury",
    question_template="Will {civ_a} have a larger treasury than {civ_b} at turn {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.treasury.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)


# =============================================================================
# Rank Templates
# =============================================================================

SCORE_RANK_1 = QuestionTemplate(
    template_id="score_rank_1",
    signal_name="scores",
    question_template="Will {civ} be ranked #1 at turn {resolution_turn}?",
    resolution_type="rank",
    data_path="snapshots.{resolution_turn}.rankings",
    comparison_op="rank==1",
    required_params=["civ", "player_id", "resolution_turn"],
)


# =============================================================================
# Milestone Templates
# =============================================================================

TECH_DISCOVERED = QuestionTemplate(
    template_id="tech_discovered",
    signal_name="techs_known",
    question_template="Will {civ} have discovered {tech_name} by turn {resolution_turn}?",
    resolution_type="milestone",
    data_path="events",
    comparison_op="contains",
    required_params=["civ", "player_id", "tech_name", "tech_id", "resolution_turn"],
)


# =============================================================================
# Event Templates
# =============================================================================

CITY_FOUNDED = QuestionTemplate(
    template_id="city_founded",
    signal_name="events",
    question_template="Will {civ} found a new city between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["civ", "player_id", "snapshot_turn", "resolution_turn"],
)

TREASURY_ZERO = QuestionTemplate(
    template_id="treasury_zero",
    signal_name="treasury",
    question_template="Will {civ}'s treasury fall to 0 at any point between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="event",
    data_path="time_series.treasury.{player_id}",
    comparison_op="exists",
    required_params=["civ", "player_id", "snapshot_turn", "resolution_turn"],
)

BORDER_CONTACT = QuestionTemplate(
    template_id="border_contact",
    signal_name="territory",
    question_template="Will {civ_a}'s borders touch {civ_b}'s borders by turn {resolution_turn}?",
    resolution_type="event",
    data_path="territory_snapshots",
    comparison_op="exists",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "snapshot_turn", "resolution_turn"],
)

CITY_CONQUERED_ANY = QuestionTemplate(
    template_id="city_conquered_any",
    signal_name="events",
    question_template="Will any city be conquered between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["snapshot_turn", "resolution_turn"],
)

CITY_LOST = QuestionTemplate(
    template_id="city_lost",
    signal_name="events",
    question_template="Will {civ} lose a city between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["civ", "player_id", "snapshot_turn", "resolution_turn"],
)

ANARCHY_EVENT = QuestionTemplate(
    template_id="anarchy_event",
    signal_name="events",
    question_template="Will {civ} experience anarchy between turn {snapshot_turn} and turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["civ", "player_id", "snapshot_turn", "resolution_turn"],
)

WONDER_COMPLETED = QuestionTemplate(
    template_id="wonder_completed",
    signal_name="events",
    question_template="Will {wonder_name} be completed by any civilization by turn {resolution_turn}?",
    resolution_type="event",
    data_path="events",
    comparison_op="exists",
    required_params=["wonder_name", "wonder_id", "snapshot_turn", "resolution_turn"],
)

WONDER_FIRST = QuestionTemplate(
    template_id="wonder_first",
    signal_name="events",
    question_template="Will {civ} complete {wonder_name} before any other civilization?",
    resolution_type="event",
    data_path="events",
    comparison_op="first",
    required_params=["civ", "player_id", "wonder_name", "wonder_id", "snapshot_turn", "resolution_turn"],
)


# =============================================================================
# State Check Templates
# =============================================================================

AT_WAR_DYAD = QuestionTemplate(
    template_id="at_war_dyad",
    signal_name="diplomacy",
    question_template="Will {civ_a} and {civ_b} be at war at turn {resolution_turn}?",
    resolution_type="state_check",
    data_path="diplomacy.relations.{player_id_a}_{player_id_b}.{resolution_turn}.state",
    comparison_op="==",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)

AT_WAR_ANY = QuestionTemplate(
    template_id="at_war_any",
    signal_name="diplomacy",
    question_template="Will {civ} be at war with any civilization at turn {resolution_turn}?",
    resolution_type="state_check",
    data_path="diplomacy.relations",
    comparison_op="any",
    required_params=["civ", "player_id", "resolution_turn"],
)

ALLIANCE_DYAD = QuestionTemplate(
    template_id="alliance_dyad",
    signal_name="diplomacy",
    question_template="Will {civ_a} and {civ_b} have an alliance at turn {resolution_turn}?",
    resolution_type="state_check",
    data_path="diplomacy.relations.{player_id_a}_{player_id_b}.{resolution_turn}.state",
    comparison_op="==",
    required_params=["civ_a", "civ_b", "player_id_a", "player_id_b", "resolution_turn"],
)

GOVERNMENT_AT = QuestionTemplate(
    template_id="government_at",
    signal_name="government",
    question_template="Will {civ} be in {government_type} at turn {resolution_turn}?",
    resolution_type="state_check",
    data_path="government_state.{player_id}.{resolution_turn}",
    comparison_op="==",
    required_params=["civ", "player_id", "government_type", "resolution_turn"],
)


# =============================================================================
# Template Registry
# =============================================================================

ALL_TEMPLATES = [
    # Comparative
    TECH_COMPARATIVE,
    SCORE_COMPARATIVE,
    POPULATION_COMPARATIVE,
    CITY_COUNT_COMPARATIVE,
    TERRITORY_COMPARATIVE,
    TREASURY_COMPARATIVE,
    # Rank
    SCORE_RANK_1,
    # Milestone
    TECH_DISCOVERED,
    # Event
    # NOTE: The following templates are commented out because they use "between X and Y"
    # event-window framing that doesn't translate well to H0 comprehension questions.
    # CITY_FOUNDED,
    # TREASURY_ZERO,
    # BORDER_CONTACT,
    # CITY_CONQUERED_ANY,
    # CITY_LOST,
    # ANARCHY_EVENT,
    WONDER_COMPLETED,
    # WONDER_FIRST,  # Competitive framing doesn't work for H0
    # State check
    # NOTE: Diplomacy templates commented out due to asymmetric/incorrect data in game_data.json files.
    # The savegame parser stores both directions (X_Y and Y_X) which can have inconsistent states.
    # See verify_ground_truth.py output for details on the 18 mismatches found.
    # AT_WAR_DYAD,
    # AT_WAR_ANY,
    # ALLIANCE_DYAD,
    GOVERNMENT_AT,
]

TEMPLATES_BY_ID: dict[str, QuestionTemplate] = {t.template_id: t for t in ALL_TEMPLATES}


def get_template(template_id: str) -> QuestionTemplate:
    """Get a template by its ID."""
    if template_id not in TEMPLATES_BY_ID:
        raise ValueError(f"Unknown template_id: {template_id}")
    return TEMPLATES_BY_ID[template_id]
