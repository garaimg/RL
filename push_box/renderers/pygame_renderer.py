"""PyGame visualization for the discrete PushBox state."""

import pygame

Position = tuple[int, int]


class PygameRenderer:
    CELL_SIZE = 80

    BACKGROUND_COLOR = (245, 245, 240)
    GRID_COLOR = (190, 190, 185)
    AGENT_COLOR = (50, 120, 220)
    BOX_COLOR = (190, 120, 45)
    GOAL_COLOR = (70, 170, 90)
    COMPLETED_BOX_COLOR = (60, 145, 75)

    def __init__(self, width: int, height: int, fps: int):
        pygame.init()
        self.fps = fps
        self.window = pygame.display.set_mode(
            (width * self.CELL_SIZE, height * self.CELL_SIZE)
        )
        pygame.display.set_caption("PushBox")
        self.clock = pygame.time.Clock()
        self.closed = False

    def render(
        self,
        *,
        agent_position: Position,
        box_position: Position,
        goal_position: Position,
    ) -> None:
        if self.closed:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.window.fill(self.BACKGROUND_COLOR)
        self._draw_grid()
        self._draw_goal(goal_position)
        self._draw_box(box_position, box_position == goal_position)
        self._draw_agent(agent_position)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_grid(self) -> None:
        width, height = self.window.get_size()
        for x in range(0, width + 1, self.CELL_SIZE):
            pygame.draw.line(self.window, self.GRID_COLOR, (x, 0), (x, height))
        for y in range(0, height + 1, self.CELL_SIZE):
            pygame.draw.line(self.window, self.GRID_COLOR, (0, y), (width, y))

    def _draw_goal(self, position: Position) -> None:
        center = self._cell_center(position)
        pygame.draw.circle(self.window, self.GOAL_COLOR, center, self.CELL_SIZE // 4)

    def _draw_box(self, position: Position, on_goal: bool) -> None:
        x, y = position
        margin = self.CELL_SIZE // 6
        rectangle = pygame.Rect(
            x * self.CELL_SIZE + margin,
            y * self.CELL_SIZE + margin,
            self.CELL_SIZE - 2 * margin,
            self.CELL_SIZE - 2 * margin,
        )
        color = self.COMPLETED_BOX_COLOR if on_goal else self.BOX_COLOR
        pygame.draw.rect(self.window, color, rectangle, border_radius=6)

    def _draw_agent(self, position: Position) -> None:
        center = self._cell_center(position)
        pygame.draw.circle(self.window, self.AGENT_COLOR, center, self.CELL_SIZE // 3)

    def _cell_center(self, position: Position) -> Position:
        x, y = position
        return (
            x * self.CELL_SIZE + self.CELL_SIZE // 2,
            y * self.CELL_SIZE + self.CELL_SIZE // 2,
        )

    def close(self) -> None:
        if not self.closed:
            pygame.display.quit()
            self.closed = True
