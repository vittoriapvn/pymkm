def choose_horizontal_subplot_layout(n: int, max_cols_per_fig: int = 3) -> list[tuple[int, int]]:
    """
    Divide n plots in multiple figures, each with at most `max_cols_per_fig` columns (1 row).
    Returns a list of (n_rows, n_cols) tuples, one per figure.
    
    Example:
    n = 5 → [(1, 3), (1, 2)]
    n = 7 → [(1, 3), (1, 3), (1, 1)]
    """
    layout = []
    remaining = n
    while remaining > 0:
        cols = min(remaining, max_cols_per_fig)
        layout.append((1, cols))
        remaining -= cols
    return layout
