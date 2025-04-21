from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random


class FireCell(Agent):
    def __init__(self, model, fuel, moisture, terrain_type):
        super().__init__(model)
        self.fuel = fuel
        self.moisture = moisture
        self.terrain_type = terrain_type
        self.state = "unburnt"  # States: "unburnt", "burning", "burnt"
        self.assigned = False

    def step(self):
        if self.state == "burning":
            self.fuel -= 1
            if self.fuel <= 0:
                self.state = "burnt"
            else:
                # Filter neighbors to only include FireCell instances.
                neighbors = [agent for agent in self.model.grid.get_neighbors(
                    self.pos, moore=True, include_center=False) if isinstance(agent, FireCell)]
                for neighbor in neighbors:
                    if neighbor.state == "unburnt" and self.should_ignite(neighbor):
                        neighbor.state = "burning"

    def should_ignite(self, neighbor):
        base_prob = 0.3
        dx = neighbor.pos[0] - self.pos[0]
        dy = neighbor.pos[1] - self.pos[1]
        wind_dx, wind_dy = self.model.wind_direction
        
        if (dx, dy) == (wind_dx, wind_dy):
            base_prob += 0.2

        terrain_factor = {"forest": 0.2, "grassland": 0.1, "urban": -0.1}
        base_prob += terrain_factor.get(neighbor.terrain_type, 0)
        base_prob *= (1 - neighbor.moisture)
        base_prob = max(0.0, min(1.0, base_prob))
        return self.random.random() < base_prob


class WildfireModel(Model):
    def __init__(self, width, height, initial_fire_positions, urban_start_positions, urban_shape,
                 wind_direction=(1, 0), grassland_max_distance=5, num_drones=5,drone_type="heuristic"):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.wind_direction = wind_direction
        self.running = True
        self.drone_type = drone_type

        # Generate urban cells based on parameters.
        urban_cells = self.generate_urban_cells(width, height, urban_start_positions, urban_shape)

        # Calculate grassland region around urban cells.
        grassland_region = set()
        for (u_x, u_y) in urban_cells:
            for dx in range(-grassland_max_distance, grassland_max_distance + 1):
                for dy in range(-grassland_max_distance, grassland_max_distance + 1):
                    x = u_x + dx
                    y = u_y + dy
                    if 0 <= x < width and 0 <= y < height:
                        distance = (dx**2 + dy**2)**0.5
                        if distance <= grassland_max_distance:
                            grassland_region.add((x, y))

        # Create FireCell agents with appropriate terrain type.
        for x in range(width):
            for y in range(height):
                fuel = random.randint(3, 10)
                moisture = random.random()
                if (x, y) in urban_cells:
                    terrain_type = "urban"
                else:
                    terrain_type = "grassland" if (x, y) in grassland_region else "forest"
                cell = FireCell(self, fuel, moisture, terrain_type)
                self.grid.place_agent(cell, (x, y))

        # Ignite initial fires.
        for pos in initial_fire_positions:
            contents = self.grid.get_cell_list_contents(pos)
            if contents:
                # We assume the first agent in the cell is the FireCell.
                contents[0].state = "burning"

        # Create DroneAgents and place them at random positions.
         # Create drone agents based on chosen type.
        for _ in range(num_drones):
            if self.drone_type == "heuristic":
                drone = HeuristicDroneAgent(self, battery_life=100, sensor_range=100)
            elif self.drone_type == "auction":
                drone = AuctionDroneAgent(self, battery_life=100, sensor_range=100)
            else:
                raise ValueError("Unknown drone_type. Choose 'heuristic' or 'auction'.")
            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(drone, (x, y))

        # If using auction-based drones, create one AuctioneerAgent.
        if self.drone_type == "auction":
            auctioneer = AuctioneerAgent(self)
            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(auctioneer, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"BurningCells": self.count_burning_cells}
        )

    def generate_urban_cells(self, width, height, urban_start_positions, urban_shape):
        urban_cells = set()
        if urban_shape == "circle":
            radius = 5  # Increased radius for thicker shape
            falloff_radius = radius + 5
            for (x0, y0) in urban_start_positions:
                for x in range(max(0, x0 - falloff_radius), min(width, x0 + falloff_radius + 1)):
                    for y in range(max(0, y0 - falloff_radius), min(height, y0 + falloff_radius + 1)):
                        dx = x - x0
                        dy = y - y0
                        dist_sq = dx**2 + dy**2
                        if dist_sq <= radius**2:
                            prob = 0.95
                        elif dist_sq <= (radius + 2)**2:
                            falloff = (dist_sq - radius**2) / ((radius + 2)**2 - radius**2)
                            prob = 0.5 * (1 - falloff)
                        else:
                            continue
                        if random.random() < prob:
                            urban_cells.add((x, y))
        elif urban_shape == "line":
            length = 5
            wind_dx, wind_dy = self.wind_direction
            major_radius = 4
            minor_radius = 2
            for (x0, y0) in urban_start_positions:
                current_x, current_y = x0, y0
                for _ in range(length + 1):
                    for dx in range(-major_radius, major_radius + 1):
                        for dy in range(-minor_radius, minor_radius + 1):
                            major_component = dx * wind_dx + dy * wind_dy
                            minor_component = dx * (-wind_dy) + dy * wind_dx
                            if (major_component**2)/(major_radius**2) + (minor_component**2)/(minor_radius**2) <= 1:
                                x_new = current_x + dx
                                y_new = current_y + dy
                                if 0 <= x_new < width and 0 <= y_new < height:
                                    if random.random() < 0.7:
                                        urban_cells.add((x_new, y_new))
                    if _ < length:
                        if random.random() < 0.5:
                            step_dx, step_dy = wind_dx, wind_dy
                        else:
                            perp = random.choice([(-wind_dy, wind_dx), (wind_dy, -wind_dx)])
                            step_dx = wind_dx + perp[0]
                            step_dy = wind_dy + perp[1]
                        current_x += step_dx
                        current_y += step_dy
        return urban_cells

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    def count_burning_cells(self):
        return sum(1 for agent in self.agents if isinstance(agent, FireCell) and agent.state == "burning")
    
    @property
    def space(self):
        return self.grid

class HeuristicDroneAgent(Agent):
    def __init__(self, model, battery_life=100, sensor_range=100):
        super().__init__(model)
        self.battery_life = battery_life
        self.sensor_range = sensor_range
        self.current_task = None  # Holds the target position of a burning cell

    def step(self):
        if self.battery_life <= 0:
            # Drone is out of battery and cannot act.
            return

        # Gather all positions in the drone's sensor range.
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.sensor_range
        )
        nearby_agents = []
        for pos in neighborhood:
            nearby_agents.extend(self.model.grid.get_cell_list_contents(pos))
            
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

        if self.current_task:
            # Move one step towards the target.
            new_position = self.move_towards(self.current_task)
        else:
            # If no burning cell is detected, perform a random walk.
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = random.choice(possible_moves) if possible_moves else self.pos

        # Move the drone if the new position is different from current.
        if new_position and new_position != self.pos:
            self.model.grid.move_agent(self, new_position)

        # If the drone has reached its target cell, try to extinguish the fire.
        if self.current_task and self.pos == self.current_task:
            agents_here = self.model.grid.get_cell_list_contents([self.pos])
            for agent in agents_here:
                if isinstance(agent, FireCell) and agent.state == "burning":
                    agent.state = "burnt"  # Extinguish the fire.
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

        # Check for collisions: if another drone is already at new_pos, try an alternative.
        contents = self.model.grid.get_cell_list_contents(new_pos)
        if any(isinstance(agent, HeuristicDroneAgent) for agent in contents):
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            for pos in possible_steps:
                if not any(isinstance(agent, HeuristicDroneAgent) for agent in self.model.grid.get_cell_list_contents(pos)):
                    return pos
            # If no alternative is free, remain in place.
            return self.pos
        else:
            return new_pos

class AuctionDroneAgent(Agent):
    def __init__(self, model, battery_life=100, sensor_range=100):
        super().__init__(model)
        self.battery_life = battery_life
        self.sensor_range = sensor_range
        self.current_task = None  # Holds the target position of an assigned burning cell

    def get_bid_cost(self, fire_cell):
        # Compute Euclidean distance from the drone to the fire cell.
        dx = fire_cell.pos[0] - self.pos[0]
        dy = fire_cell.pos[1] - self.pos[1]
        distance = (dx**2 + dy**2) ** 0.5
        # Battery cost: lower battery implies higher cost.
        battery_cost = (100 - self.battery_life) * 0.1
        # You can add more cost metrics here (e.g., workload)
        return distance + battery_cost

    def step(self):
        if self.battery_life <= 0:
            # Drone is out of battery and cannot act.
            return

        # If a task is assigned, move one step toward it.
        if self.current_task:
            new_position = self.move_towards(self.current_task)
        else:
            # No task: perform a random walk.
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = random.choice(possible_moves) if possible_moves else self.pos

        if new_position and new_position != self.pos:
            self.model.grid.move_agent(self, new_position)

        # If reached target, attempt to extinguish fire.
        if self.current_task and self.pos == self.current_task:
            agents_here = self.model.grid.get_cell_list_contents([self.pos])
            for agent in agents_here:
                if isinstance(agent, FireCell) and agent.state == "burning":
                    agent.state = "burnt"  # Extinguish fire.
                    # Mark the fire cell as no longer needing assignment.
                    agent.assigned = True
                    self.current_task = None

        # Decrement battery life each step.
        self.battery_life -= 1

    def move_towards(self, target_pos):
        """Compute a new position one step toward the target while avoiding collisions with other drones."""
        current_x, current_y = self.pos
        target_x, target_y = target_pos

        # Simple heuristic: take one step in the direction of the target.
        step_x = 1 if target_x > current_x else -1 if target_x < current_x else 0
        step_y = 1 if target_y > current_y else -1 if target_y < current_y else 0
        new_pos = (current_x + step_x, current_y + step_y)

        # Collision check: if another drone occupies the target cell, try an alternative.
        contents = self.model.grid.get_cell_list_contents(new_pos)
        if any(isinstance(agent, AuctionDroneAgent) for agent in contents):
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            for pos in possible_steps:
                if not any(isinstance(agent, AuctionDroneAgent) for agent in self.model.grid.get_cell_list_contents(pos)):
                    return pos
            return self.pos
        else:
            return new_pos

class AuctioneerAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        # Identify available tasks: burning fire cells that are not yet assigned.
        tasks = [agent for agent in self.model.agents 
                 if isinstance(agent, FireCell) and agent.state == "burning" and not agent.assigned]
        # Get all auction-based drones.
        drones = [agent for agent in self.model.agents if isinstance(agent, AuctionDroneAgent)]
        for task in tasks:
            bids = []
            for drone in drones:
                # Check if the drone's sensor range can detect the fire.
                dx = task.pos[0] - drone.pos[0]
                dy = task.pos[1] - drone.pos[1]
                distance = (dx**2 + dy**2) ** 0.5
                if distance <= drone.sensor_range:
                    bid_cost = drone.get_bid_cost(task)
                    bids.append((drone, bid_cost))
            if bids:
                # Select the drone with the lowest bid cost.
                winning_drone, winning_cost = min(bids, key=lambda x: x[1])
                # Assign the task to the winning drone.
                winning_drone.current_task = task.pos
                # Mark the fire cell as assigned.
                task.assigned = True

if __name__ == "__main__":
    grid_width, grid_height = 50, 50
    initial_fires = [(10, 30), (30, 10)]
    urban_starts = [(20, 20), (40, 40)]

    # Change drone_type to either "heuristic" or "auction"
    chosen_drone_type = "auction" 

    model = WildfireModel(
        width=grid_width, 
        height=grid_height,
        initial_fire_positions=initial_fires,
        urban_start_positions=urban_starts,
        urban_shape="circle",  
        wind_direction=(1, 0),
        grassland_max_distance=10,
        num_drones=20,
        drone_type=chosen_drone_type
    )

    for i in range(7):
        model.step()
        burning = model.count_burning_cells()
        print(f"\033[1mStep {i+1}\033[0m: Burning cells = {burning}")
        for y in reversed(range(grid_height)):
            row = []
            for x in range(grid_width):
                agents = model.grid.get_cell_list_contents((x, y))
                cell_char = " "  # default
                # Priority: display drone based on chosen type.
                if chosen_drone_type == "heuristic":
                    if any(isinstance(agent, HeuristicDroneAgent) for agent in agents):
                        cell_char = "\033[94mD\033[0m"
                else:
                    if any(isinstance(agent, AuctionDroneAgent) for agent in agents):
                        cell_char = "\033[94mD\033[0m"
                # If cell_char is still the default, check for FireCells.
                if cell_char == " ":
                    fire_cells = [agent for agent in agents if isinstance(agent, FireCell)]
                    if fire_cells:
                        fire_cell = fire_cells[0]
                        if fire_cell.state == "burning":
                            cell_char = "\033[91m█\033[0m"  # Red for burning
                        elif fire_cell.state == "burnt":
                            cell_char = "\033[30m█\033[0m"  # Black for burnt
                        else:
                            if fire_cell.terrain_type == "forest":
                                cell_char = "\033[32m█\033[0m"  # Green
                            elif fire_cell.terrain_type == "grassland":
                                cell_char = "\033[92m█\033[0m"  # Light green
                            elif fire_cell.terrain_type == "urban":
                                cell_char = "\033[90m█\033[0m"  # Gray
                row.append(cell_char)
            print("".join(row))
