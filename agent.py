from mesa import Agent
from fire_spread import FireCell

class HeuristicDroneAgent(Agent):
    def __init__(self, model, battery_life=100, sensor_range=5):
        super().__init__(model)
        self.battery_life = battery_life
        self.sensor_range = sensor_range
        self.current_task = None  # Holds the target position of a burning cell

    def step(self):
        if self.battery_life <= 0:
            # Drone is out of battery and cannot act.
            return

        # Use sensor range to search for burning cells.
        # The grid method 'get_neighbors' can be used with a radius.
        nearby_agents = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.sensor_range
        )
        # Filter to only burning FireCells.
        burning_cells = [agent for agent in nearby_agents if isinstance(agent, FireCell) and agent.state == "burning"]

        if burning_cells:
            # Choose the closest burning cell using Euclidean distance.
            target = min(
                burning_cells,
                key=lambda cell: ((cell.pos[0] - self.pos[0]) ** 2 + (cell.pos[1] - self.pos[1]) ** 2) ** 0.5
            )
            self.current_task = target.pos
        else:
            self.current_task = None

        # If a target is set, attempt to move one step toward it.
        if self.current_task:
            new_position = self.move_towards(self.current_task)
            # Move the drone if the new position is determined.
            if new_position:
                self.model.grid.move_agent(self, new_position)
            # If the drone has reached the target cell, extinguish the fire.
            if self.pos == self.current_task:
                agents_here = self.model.grid.get_cell_list_contents([self.pos])
                for agent in agents_here:
                    if isinstance(agent, FireCell) and agent.state == "burning":
                        agent.state = "burnt"  # Extinguish the fire.
                        # Optionally, you could log this event or collect data.
                        self.current_task = None

        # Decrement battery life each step.
        self.battery_life -= 1

    def move_towards(self, target_pos):
        """Compute a new position one step toward the target while avoiding collisions with other drones."""
        current_x, current_y = self.pos
        target_x, target_y = target_pos

        # Determine the step in each direction (simple heuristic).
        step_x = 1 if target_x > current_x else -1 if target_x < current_x else 0
        step_y = 1 if target_y > current_y else -1 if target_y < current_y else 0
        new_pos = (current_x + step_x, current_y + step_y)

        # Check for collision: if another drone is already at the intended new position, try an alternative.
        contents = self.model.grid.get_cell_list_contents(new_pos)
        if any(isinstance(agent, HeuristicDroneAgent) for agent in contents):
            # Look for alternative positions in the immediate neighborhood.
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            for pos in possible_steps:
                if not any(isinstance(agent, HeuristicDroneAgent) for agent in self.model.grid.get_cell_list_contents(pos)):
                    return pos
            # If no alternative is free, remain in place.
            return self.pos
        else:
            return new_pos