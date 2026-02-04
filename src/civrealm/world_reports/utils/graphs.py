"""Graph generation utilities for world reports

This module provides functions to create time-series graphs and charts.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Import player colors from visualizations module
from .visualizations import PLAYER_COLORS


def create_time_series_graph(
    data: Dict[int, Dict[int, float]],
    title: str,
    ylabel: str,
    player_names: Dict[int, str],
    xlabel: str = 'Turn',
    dpi: int = 150,
    figsize: Tuple[int, int] = (10, 6),
    style: Optional[str] = None
) -> BytesIO:
    """Create a multi-line time-series graph for civilizations

    Args:
        data: Nested dict mapping {turn: {player_id: value}}
        title: Graph title
        ylabel: Y-axis label
        player_names: Dict mapping player_id to civilization name
        xlabel: X-axis label (default: 'Turn')
        dpi: Image DPI (default: 150)
        figsize: Figure size as (width, height) tuple
        style: Optional matplotlib style (e.g., 'ggplot', 'bmh')

    Returns:
        BytesIO buffer containing PNG image
    """
    # Set matplotlib style if provided
    if style:
        try:
            plt.style.use(style)
        except Exception:
            pass  # Ignore style errors, use default

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Get all player IDs and sort turns
    all_player_ids = set()
    for turn_data in data.values():
        all_player_ids.update(turn_data.keys())
    player_ids = sorted(all_player_ids)
    turns = sorted(data.keys())

    # Plot line for each player
    for idx, player_id in enumerate(player_ids):
        values = []
        turn_list = []

        for turn in turns:
            if player_id in data[turn]:
                values.append(data[turn][player_id])
                turn_list.append(turn)

        if values:  # Only plot if we have data
            color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]
            name = player_names.get(player_id, f'Player {player_id}')
            ax.plot(turn_list, values, marker='o', markersize=4, linewidth=2,
                   label=name, color=color)

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Ensure x-axis shows integer turns
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def create_stacked_bar_chart(
    data: Dict[int, Dict[str, int]],
    title: str,
    categories: List[str],
    xlabel: str = 'Turn',
    ylabel: str = 'Count',
    dpi: int = 150,
    figsize: Tuple[int, int] = (12, 6),
    style: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None
) -> BytesIO:
    """Create a stacked bar chart for categorical data over time

    Args:
        data: Nested dict mapping {turn: {category: value}}
        title: Graph title
        categories: List of category names in order (bottom to top)
        xlabel: X-axis label (default: 'Turn')
        ylabel: Y-axis label (default: 'Count')
        dpi: Image DPI (default: 150)
        figsize: Figure size as (width, height) tuple
        style: Optional matplotlib style (e.g., 'ggplot', 'bmh')
        colors: Optional dict mapping category names to colors

    Returns:
        BytesIO buffer containing PNG image
    """
    # Set matplotlib style if provided
    if style:
        try:
            plt.style.use(style)
        except Exception:
            pass  # Ignore style errors, use default

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Sort turns
    turns = sorted(data.keys())

    # Default colors if not provided
    if colors is None:
        color_palette = ['#90EE90', '#FFD700', '#FFA500', '#FF6347']  # Green, Yellow, Orange, Red
        colors = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}

    # Prepare data arrays
    category_data = {cat: [] for cat in categories}
    for turn in turns:
        turn_data = data.get(turn, {})
        for cat in categories:
            category_data[cat].append(turn_data.get(cat, 0))

    # Create stacked bars
    bottom = np.zeros(len(turns))
    for cat in categories:
        values = category_data[cat]
        ax.bar(turns, values, label=cat, bottom=bottom, color=colors.get(cat))
        bottom += np.array(values)

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, axis='y', alpha=0.3)

    # Ensure x-axis shows integer turns
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def create_diplomacy_chart(
    relations: Dict[str, Dict[int, Dict[str, any]]],
    from_player_id: int,
    player_names: Dict[int, str],
    title: str,
    dpi: int = 150,
    figsize: Tuple[int, int] = (12, 6),
    style: Optional[str] = None,
    use_love: bool = False
) -> BytesIO:
    """Create a FOCUS-style diplomacy chart showing one player's relations with others

    Creates horizontal bar chart where each row is a civilization and color represents
    the diplomatic state or attitude over time.

    Args:
        relations: Dict mapping "{from}_{to}" keys to {turn: {state, love, ...}}
        from_player_id: The player whose perspective we're showing
        player_names: Dict mapping player_id to civilization name
        title: Chart title
        dpi: Image DPI (default: 150)
        figsize: Figure size as (width, height) tuple
        style: Optional matplotlib style
        use_love: If True, color by love/attitude; if False, by diplomatic state

    Returns:
        BytesIO buffer containing PNG image
    """
    if style:
        try:
            plt.style.use(style)
        except Exception:
            pass

    # Color schemes
    # For diplomatic states
    state_colors = {
        'Alliance': '#228B22',      # Forest Green
        'Peace': '#90EE90',         # Light Green
        'Cease-fire': '#FFFF99',    # Light Yellow
        'Armistice': '#FFFACD',     # Lemon Chiffon
        'War': '#DC143C',           # Crimson
        'Never met': '#D3D3D3',     # Light Gray
        'No contact': '#D3D3D3',    # Light Gray
        'Team': '#4169E1',          # Royal Blue
        'Unknown': '#E0E0E0'        # Gray
    }

    # For attitude (love values) - gradient from red to green
    def love_to_color(love_value: int) -> str:
        """Convert love value (-1000 to 1000) to color"""
        # Normalize to 0-1 range
        normalized = (love_value + 1000) / 2000
        normalized = max(0, min(1, normalized))

        # Gradient from red (0) through yellow (0.5) to green (1)
        if normalized < 0.5:
            # Red to yellow
            r = 255
            g = int(255 * (normalized * 2))
            b = 0
        else:
            # Yellow to green
            r = int(255 * (1 - (normalized - 0.5) * 2))
            g = 255
            b = 0

        return f'#{r:02x}{g:02x}{b:02x}'

    # Find all relations for this player
    target_players = []
    all_turns = set()

    for key, turn_data in relations.items():
        parts = key.split('_')
        if len(parts) != 2:
            continue
        from_p, to_p = int(parts[0]), int(parts[1])
        if from_p == from_player_id:
            target_players.append(to_p)
            # Convert turn keys to integers
            all_turns.update(int(t) for t in turn_data.keys())

    if not target_players or not all_turns:
        # Return empty plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, 'No diplomatic data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    target_players = sorted(target_players)
    turns = sorted(all_turns)
    min_turn, max_turn = min(turns), max(turns)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot each player's relationship as a horizontal bar
    for row_idx, to_player in enumerate(target_players):
        key = f"{from_player_id}_{to_player}"
        turn_data = relations.get(key, {})

        # Create segments for each turn
        prev_turn = min_turn
        prev_value = None

        # Sort turns we have data for (keys might be strings or ints)
        data_turns = sorted([int(t) for t in turn_data.keys()])

        for turn in data_turns:
            # Try both int and string keys
            data = turn_data.get(turn) or turn_data.get(str(turn)) or {}

            if use_love:
                love = data.get('love', 0)
                color = love_to_color(love)
            else:
                state = data.get('state', 'Unknown')
                color = state_colors.get(state, state_colors['Unknown'])

            # Draw bar from prev_turn to this turn
            if prev_value is not None:
                ax.barh(row_idx, turn - prev_turn, left=prev_turn, height=0.8,
                       color=prev_value, edgecolor='none')

            prev_turn = turn
            prev_value = color

        # Draw final segment to max_turn
        if prev_value is not None:
            ax.barh(row_idx, max_turn - prev_turn + 1, left=prev_turn, height=0.8,
                   color=prev_value, edgecolor='none')

    # Y-axis labels (civilization names)
    y_labels = [player_names.get(p, f'Player {p}') for p in target_players]
    ax.set_yticks(range(len(target_players)))
    ax.set_yticklabels(y_labels)

    # X-axis (turns)
    ax.set_xlim(min_turn, max_turn + 1)
    ax.set_xlabel('Turn', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Create legend
    if use_love:
        # Gradient legend for love
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        # Create a custom colormap
        colors_list = ['#ff0000', '#ffff00', '#00ff00']  # red -> yellow -> green
        cmap = LinearSegmentedColormap.from_list('love', colors_list)
        norm = Normalize(vmin=-1000, vmax=1000)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Opinion (Love)', fontsize=10)
    else:
        # Create legend patches for states
        from matplotlib.patches import Patch
        legend_elements = []
        shown_states = set()

        # Determine which states are actually used
        for key in relations:
            parts = key.split('_')
            if len(parts) != 2:
                continue
            from_p = int(parts[0])
            if from_p != from_player_id:
                continue
            for turn_data in relations[key].values():
                if isinstance(turn_data, dict):
                    shown_states.add(turn_data.get('state', 'Unknown'))

        # Order: good to bad
        state_order = ['Alliance', 'Team', 'Peace', 'Cease-fire', 'Armistice',
                      'Never met', 'No contact', 'War']
        for state in state_order:
            if state in shown_states:
                legend_elements.append(Patch(facecolor=state_colors[state],
                                            edgecolor='gray', label=state))

        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                 frameon=True, shadow=True)

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def create_multi_player_stacked_bars(
    data: Dict[int, Dict[int, Dict[str, int]]],
    title: str,
    categories: List[str],
    player_names: Dict[int, str],
    xlabel: str = 'Turn',
    ylabel: str = 'Count',
    dpi: int = 150,
    figsize: Tuple[int, int] = (14, 8),
    style: Optional[str] = None
) -> BytesIO:
    """Create stacked bars for multiple players side by side

    Args:
        data: Nested dict mapping {turn: {player_id: {category: value}}}
        title: Graph title
        categories: List of category names
        player_names: Dict mapping player_id to civilization name
        xlabel: X-axis label (default: 'Turn')
        ylabel: Y-axis label (default: 'Count')
        dpi: Image DPI (default: 150)
        figsize: Figure size as (width, height) tuple
        style: Optional matplotlib style (e.g., 'ggplot', 'bmh')

    Returns:
        BytesIO buffer containing PNG image
    """
    # Set matplotlib style if provided
    if style:
        try:
            plt.style.use(style)
        except Exception:
            pass  # Ignore style errors, use default

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Get all player IDs and sort turns
    all_player_ids = set()
    for turn_data in data.values():
        all_player_ids.update(turn_data.keys())
    player_ids = sorted(all_player_ids)
    turns = sorted(data.keys())

    # Bar positioning
    num_players = len(player_ids)
    bar_width = 0.8 / num_players
    turn_positions = np.arange(len(turns))

    # Category colors
    color_palette = ['#90EE90', '#FFD700', '#FFA500', '#FF6347']  # Green, Yellow, Orange, Red
    cat_colors = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}

    # Plot for each player
    for player_idx, player_id in enumerate(player_ids):
        offset = (player_idx - num_players / 2) * bar_width + bar_width / 2
        positions = turn_positions + offset

        # Prepare data
        category_data = {cat: [] for cat in categories}
        for turn in turns:
            turn_data = data.get(turn, {})
            player_data = turn_data.get(player_id, {})
            for cat in categories:
                category_data[cat].append(player_data.get(cat, 0))

        # Create stacked bars for this player
        bottom = np.zeros(len(turns))
        for cat_idx, cat in enumerate(categories):
            values = np.array(category_data[cat])
            label = f'{cat}' if player_idx == 0 else None  # Only label once
            ax.bar(positions, values, bar_width, label=label, bottom=bottom,
                  color=cat_colors[cat], alpha=0.8)
            bottom += values

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(turn_positions)
    ax.set_xticklabels([str(t) for t in turns])
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf
