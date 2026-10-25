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
