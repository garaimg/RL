"""Text rendering for PushBox."""

Position = tuple[int, int]


def render_ansi(
    *,
    width: int,
    height: int,
    agent_position: Position,
    box_position: Position,
    goal_position: Position,
) -> str:
    """Build a text view from a discrete environment state."""
    rows: list[str] = []

    for y in range(height):
        row: list[str] = []
        for x in range(width):
            position = (x, y)

            if position == box_position and position == goal_position:
                character = "*"
            elif position == agent_position:
                character = "A"
            elif position == box_position:
                character = "B"
            elif position == goal_position:
                character = "G"
            else:
                character = "."

            row.append(character)
        rows.append("".join(row))

    return "\n".join(rows) + "\n"
