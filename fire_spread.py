from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
from pettingzoo.utils.env import ParallelEnv
import supersuit as ss
from stable_baselines3 import PPO
import numpy as np
from gymnasium.spaces import Discrete, Box

class FireCell(Agent):
    def __init__(self, model, fuel, moisture, terrain_type):
        super().__init__(model)
        self.fuel = fuel
        self.moisture = moisture
        self.terrain_type = terrain_type
        self.state = "unburnt"  # Possible States: "unburnt", "burning", "burnt"
        self.assigned = False
        self.owner = None  # Track which drone currently owns this fire, for auction

    # step function for fire, if a tile is in burning state,
    # reduce fuel by 1 each step, if fuel is 0, change tile 
    # to burnt. Also check for neighbor unburnt FireCell if
    # it is possible to ignite, change tile to burning if 
    # possible
    def step(self):
        if self.state == "burning":
            self.fuel -= 1 
            if self.fuel <= 0:
                self.state = "burnt"
                self.model.natural_burnouts += 1 # counter for natural burnout
            else:
                # Filter neighbors to only include FireCell instances.
                neighbors = [
                    agent for agent in self.model.grid.get_neighbors(
                        self.pos, moore=True, include_center=False
                    ) if isinstance(agent, FireCell)
                ]
                for neighbor in neighbors:
                    if neighbor.state == "unburnt" and self.should_ignite(neighbor):
                        neighbor.state = "burning"

    # value for tile ingition probability
    TERRAIN_FACTOR = {
        "forest": 0.2,
        "grassland": 0.1,
        "urban": -0.1
    }

    #  function to check tile ingition probability, account for
    # base probability, wind direction, terrain and moisture
    def should_ignite(self, neighbor):
        base_prob = 0.3
        dx = neighbor.pos[0] - self.pos[0]
        dy = neighbor.pos[1] - self.pos[1]
        wind_dx, wind_dy = self.model.wind_direction
        
        if (dx, dy) == (wind_dx, wind_dy):
            base_prob += 0.2

        base_prob += self.TERRAIN_FACTOR.get(neighbor.terrain_type, 0)
        base_prob *= (1 - neighbor.moisture)
        base_prob = max(0.0, min(1.0, base_prob))
        return random.random() < base_prob

class WildfireModel(Model):
    def __init__(
            self, width, height, 
            initial_fire_positions, urban_start_positions, 
            urban_shape="circle",wind_direction=(1, 0), 
            grassland_max_distance=5, 
            num_drones=5,drone_type="heuristic",
        ):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.wind_direction = wind_direction
        self.running = True
        self.drone_type = drone_type
        self.initial_fire_positions = initial_fire_positions
        self.urban_start_positions = urban_start_positions
        self.urban_shape = urban_shape
        self.grassland_max_distance = grassland_max_distance
        self.num_drones = num_drones

        self.natural_burnouts = 0
        self.extinguished_count = 0
        self.step_count = 0  # Add step counter

        # Generate urban cells
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
                terrain_type = (
                    "urban" if (x,y) in urban_cells
                    else "grassland" if (x,y) in grassland_region
                    else "forest"
                )
                cell = FireCell(self, fuel, moisture, terrain_type)
                self.grid.place_agent(cell, (x, y))
                #self.agents.add(cell)

        # Ignite initial fires.
        for pos in initial_fire_positions:
            for cell in self.grid.get_cell_list_contents(pos):
                if isinstance(cell, FireCell):
                    cell.state = "burning"
                    break

        # Create DroneAgents and place them at random positions.
        self.rl_drones = [] if self.drone_type == "rl" else None
        for _ in range(num_drones):
            if self.drone_type == "heuristic":
                drone = HeuristicDroneAgent(self)
            elif self.drone_type == "auction":
                drone = AuctionDroneAgent(self)
            elif self.drone_type == "rl":
                drone = RLDroneAgent(self)
                self.rl_drones.append(drone)
            else:
                raise ValueError("Unknown drone_type. Choose 'heuristic', 'auction'")
            self.grid.place_agent(drone, (random.randrange(width), random.randrange(height)))
            #self.agents.add(drone)

        # If using auction-based drones, create one AuctioneerAgent.
        if self.drone_type == "auction":
            auctioneer = AuctioneerAgent(self)
            self.grid.place_agent(auctioneer, (random.randrange(width), random.randrange(height)))
            #self.agents.add(auctioneer)

        self.datacollector = DataCollector(
            model_reporters={"BurningCells": self.count_burning_cells}
        )

        self.previous_extinguished_count = 0  # Track extinguished count for reward diff

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
        self.step_count += 1
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    def count_burning_cells(self):
        return sum(
            1 for a in self.agents
            if isinstance(a, FireCell) and a.state == "burning"
        )
    
    @property
    def space(self):
        return self.grid

class HeuristicDroneAgent(Agent):
    def __init__(self, model, battery_life=100, sensor_range=10):
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
                    self.model.extinguished_count += 1
                    self.current_task = None
                    self.battery_life -= 5

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
    def __init__(self, model, battery_life=100, sensor_range=10):
        super().__init__(model)
        self.battery_life = battery_life
        self.sensor_range = sensor_range
        self.current_task = None  # Holds the target position of an assigned burning cell

    def _cell_at(self, pos):
        if pos is None:
            return None
        for agent in self.model.grid.get_cell_list_contents(pos):
            if isinstance(agent, FireCell):
                return agent
        return None

    def get_bid_cost(self, fire_cell):
        # Compute Euclidean distance from the drone to the fire cell.
        dx = fire_cell.pos[0] - self.pos[0]
        dy = fire_cell.pos[1] - self.pos[1]
        distance = (dx**2 + dy**2) ** 0.5
        # Battery cost: lower battery implies higher cost.
        battery_cost = (100 - self.battery_life) * 0.1
        return distance + battery_cost

    def step(self):
        if self.battery_life <= 0:
            return

        # 1) Gather all burning FireCells in sensor range (ignore assigned flag)
        visible = []
        for pos in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True, radius=self.sensor_range):
            for agent in self.model.grid.get_cell_list_contents(pos):
                if isinstance(agent, FireCell) and agent.state == "burning":
                    visible.append(agent)
        # 2) Pick the fire with minimum bid_cost
        if visible:
            best_fire = min(visible, key=self.get_bid_cost)
            # 3) If it’s different (and better) than current_task, reassign:
            if self.current_task != best_fire.pos:
                # Optionally “free” the old one:
                old_cell = self._cell_at(self.current_task)
                if old_cell and old_cell.owner is self:
                    old_cell.assigned = False
                    old_cell.owner = None
                self.current_task = best_fire.pos
                best_fire.assigned = True
                best_fire.owner = self

        # 4) Then proceed exactly as before: move_towards, extinguish, drain battery…
        if self.current_task:
            new_position = self.move_towards(self.current_task)
        else:
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = random.choice(possible_moves) if possible_moves else self.pos

        if new_position and new_position != self.pos:
            self.model.grid.move_agent(self, new_position)

        if self.current_task and self.pos == self.current_task:
            agents_here = self.model.grid.get_cell_list_contents([self.pos])
            for agent in agents_here:
                if isinstance(agent, FireCell) and agent.state == "burning":
                    agent.state = "burnt"
                    self.model.extinguished_count += 1
                    agent.assigned = True
                    agent.owner = None
                    self.current_task = None
                    self.battery_life -= 5

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
        # Identify all burning fire cells (ignore assigned flag)
        tasks = [agent for agent in self.model.agents 
                 if isinstance(agent, FireCell) and agent.state == "burning"]
        drones = [agent for agent in self.model.agents if isinstance(agent, AuctionDroneAgent)]
        for task in tasks:
            bids = []
            for drone in drones:
                dx = task.pos[0] - drone.pos[0]
                dy = task.pos[1] - drone.pos[1]
                distance = (dx**2 + dy**2) ** 0.5
                if distance <= drone.sensor_range:
                    bid_cost = drone.get_bid_cost(task)
                    bids.append((drone, bid_cost))
            if bids:
                winner, _ = min(bids, key=lambda x: x[1])
                # If this fire was already assigned to someone else:
                if task.assigned and task.owner is not winner and task.owner is not None:
                    task.owner.current_task = None
                # Assign to the new winner:
                winner.current_task = task.pos
                task.assigned = True
                task.owner = winner

class RLDroneAgent(Agent):
    MOVE_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    
    def __init__(self, model, battery=100):
        super().__init__(model)
        self.next_action = 4  # Default: stay
        self.battery = battery
        self.extinguished_count = 0  # Track how many fires this agent extinguished
        self.last_pos = None  # For movement tracking

    def step(self):
        if self.battery <= 0:
            return  # Drone inactive
        self.last_pos = self.pos
        dx, dy = RLDroneAgent.MOVE_DIRS[self.next_action]
        x, y = self.pos
        nx, ny = x + dx, y + dy
        # Move if valid
        if 0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height:
            self.model.grid.move_agent(self, (nx, ny))
            self.battery -= 1  # Deduct battery for movement
        # Extinguish fire at current position
        for obj in self.model.grid.get_cell_list_contents([self.pos]):
            if isinstance(obj, FireCell) and obj.state == "burning":
                obj.state = "burnt"
                self.model.extinguished_count += 1
                self.extinguished_count += 1
                self.battery -= 5  # Higher cost for extinguishing
                break

class RLWildfireEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "is_parallelizable": True}

    def __init__(self, config, render_mode=None):
        super().__init__()
        self.config = config.copy()
        self.model = WildfireModel(**self.config)
        self.agents = [f"drone_{i}" for i in range(self.model.num_drones)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {a: i for i, a in enumerate(self.agents)}
        # Use a local window for observation
        self.window_size = 21
        obs_dim = (self.window_size * self.window_size * 5) + 4  # 3 fire + 1 terrain + 1 moisture + 2 pos + 2 wind
        self.action_spaces = {a: Discrete(9) for a in self.agents}
        self.observation_spaces = {
            a: Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for a in self.agents
        }
        self.max_steps = 70
        self.render_mode = render_mode
        self.step_count = 0
        self.previous_extinguished_count = 0
        self.previous_burning = self.model.count_burning_cells()

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.model = WildfireModel(**self.config)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.previous_extinguished_count = 0
        self.previous_burning = self.model.count_burning_cells()
        obs = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _observe(self, agent):
        fire_vals = []
        terrain_vals = []
        moisture_vals = []
        idx = self.agent_name_mapping[agent]
        agent_pos = self.model.rl_drones[idx].pos
        width = self.model.grid.width
        height = self.model.grid.height
        window = self.window_size
        half_window = window // 2
        ax, ay = agent_pos
        for dy in range(-half_window, half_window + 1):
            for dx in range(-half_window, half_window + 1):
                x = ax + dx
                y = ay + dy
                if 0 <= x < width and 0 <= y < height:
                    cell_contents = self.model.grid.get_cell_list_contents((x, y))
                    burning = any(isinstance(obj, FireCell) and obj.state == "burning" for obj in cell_contents)
                    unburnt = any(isinstance(obj, FireCell) and obj.state == "unburnt" for obj in cell_contents)
                    burnt = any(isinstance(obj, FireCell) and obj.state == "burnt" for obj in cell_contents)
                    fire_vals.extend([1.0 if burning else 0.0, 1.0 if unburnt else 0.0, 1.0 if burnt else 0.0])
                    for obj in cell_contents:
                        if isinstance(obj, FireCell):
                            terrain_vals.append(
                                0.0 if obj.terrain_type == "forest"
                                else 0.5 if obj.terrain_type == "grassland"
                                else 1.0
                            )
                            moisture_vals.append(obj.moisture)
                            break
                    else:
                        terrain_vals.append(0.0)
                        moisture_vals.append(0.0)
                else:
                    fire_vals.extend([0.0, 0.0, 0.0])
                    terrain_vals.append(0.0)
                    moisture_vals.append(0.0)
        norm_pos = np.array([
            agent_pos[0] / (width - 1) if width > 1 else 0,
            agent_pos[1] / (height - 1) if height > 1 else 0
        ], dtype=np.float32)
        wind = np.array(self.model.wind_direction, dtype=np.float32) / 1.0
        return np.concatenate([
            np.array(fire_vals, dtype=np.float32),
            np.array(terrain_vals, dtype=np.float32),
            np.array(moisture_vals, dtype=np.float32),
            norm_pos,
            wind
        ])

    def step(self, actions):
        # Assign actions to drones
        for agent, action in actions.items():
            idx = self.agent_name_mapping[agent]
            self.model.rl_drones[idx].next_action = action
        self.model.step()
        self.step_count += 1
        # Team reward: reduction in burning cells
        current_burning = self.model.count_burning_cells()
        reduction = self.previous_burning - current_burning
        self.previous_burning = current_burning
        rewards = {}
        for agent in self.agents:
            idx = self.agent_name_mapping[agent]
            drone = self.model.rl_drones[idx]
            agents_here = self.model.grid.get_cell_list_contents([drone.pos])
            # Diminishing return for extinguishing
            extinguish_reward = 0
            if any(isinstance(a, FireCell) and a.state == "burnt" and self.model.extinguished_count > self.previous_extinguished_count for a in agents_here):
                extinguish_reward = 10 / (1 + drone.extinguished_count)
            # Proximity reward (as before)
            proximity_reward = 0
            for pos in self.model.grid.get_neighborhood(drone.pos, moore=True, radius=3):
                for neighbor in self.model.grid.get_cell_list_contents(pos):
                    if isinstance(neighbor, FireCell) and neighbor.state == "burning":
                        proximity_reward += 0.1
            # Step penalty for idleness
            moved = (drone.last_pos is not None and drone.last_pos != drone.pos)
            heading_to_fire = False
            for pos in self.model.grid.get_neighborhood(drone.pos, moore=True, radius=10):
                for neighbor in self.model.grid.get_cell_list_contents(pos):
                    if isinstance(neighbor, FireCell) and neighbor.state == "burning":
                        # If the drone is closer to a fire than last step, consider it heading to fire
                        if drone.last_pos is not None:
                            old_dist = max(abs(drone.last_pos[0] - pos[0]), abs(drone.last_pos[1] - pos[1]))
                            new_dist = max(abs(drone.pos[0] - pos[0]), abs(drone.pos[1] - pos[1]))
                            if new_dist < old_dist:
                                heading_to_fire = True
            if not moved or not heading_to_fire:
                step_penalty = -1.5
            else:
                step_penalty = -1
            team_reward = reduction * 0.5
            rewards[agent] = extinguish_reward + proximity_reward + step_penalty + team_reward
        self.previous_extinguished_count = self.model.extinguished_count
        # Termination/truncation
        burning_cells = self.model.count_burning_cells()
        terminations = {a: False for a in self.agents}
        truncations = {a: (self.step_count >= self.max_steps) for a in self.agents}
        if burning_cells == 0:
            terminations = {a: True for a in self.agents}
        # Bonus for leftover battery at episode end
        if all(terminations.values()) or all(truncations.values()):
            for agent in self.agents:
                idx = self.agent_name_mapping[agent]
                drone = self.model.rl_drones[idx]
                rewards[agent] += 0.1 * drone.battery  # 0.1 per battery unit left
        # Remove done agents
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
        obs = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.possible_agents}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        """Return grid representation as a string with color, matching the other agent render style"""
        grid = [[' ' for _ in range(self.model.grid.width)] for _ in range(self.model.grid.height)]
        for cell in self.model.grid.coord_iter():
            contents, (x, y) = cell
            cell_char = ' '
            # Drones
            if any(isinstance(obj, RLDroneAgent) for obj in contents):
                cell_char = "\033[94mD\033[0m"
            else:
                fire_cells = [obj for obj in contents if isinstance(obj, FireCell)]
                if fire_cells:
                    fire_cell = fire_cells[0]
                    if fire_cell.state == "burning":
                        cell_char = "\033[91m█\033[0m"
                    elif fire_cell.state == "burnt":
                        cell_char = "\033[30m█\033[0m"
                    else:
                        if fire_cell.terrain_type == "forest":
                            cell_char = "\033[32m█\033[0m"
                        elif fire_cell.terrain_type == "grassland":
                            cell_char = "\033[92m█\033[0m"
                        elif fire_cell.terrain_type == "urban":
                            cell_char = "\033[90m█\033[0m"
            grid[y][x] = cell_char
        grid_str = '\n'.join(''.join(row) for row in grid)
        return grid_str

width = 30
height = 30
num_fires = 2
num_urban = 2
all_positions = [(x, y) for x in range(width) for y in range(height)]
initial_fire_positions = random.sample(all_positions, num_fires)
urban_start_positions = random.sample(all_positions, num_urban)
orig_config = {
    'width': width,
    'height': height,
    'initial_fire_positions': initial_fire_positions,
    'urban_start_positions': urban_start_positions,
    'urban_shape': 'circle',
    'wind_direction': (1, 0),
    'grassland_max_distance': 3,
    'num_drones': 20,
    'drone_type': 'rl'  # 'heuristic', 'auction', or 'rl'
}

if orig_config['drone_type'] in ["heuristic", "auction"]:
    model = WildfireModel(**orig_config)
    episode_steps = []
    burning_cells_list = []
    natural_burnouts_list = []
    extinguished_list = []

    for i in range(70):
        model.step()
        burning = model.count_burning_cells()
        burning_cells_list.append(burning)
        natural_burnouts_list.append(model.natural_burnouts)
        extinguished_list.append(model.extinguished_count)
        step_grid = []
        for y in reversed(range(orig_config['width'])):
            row = []
            for x in range(orig_config['height']):
                agents = model.grid.get_cell_list_contents((x, y))
                cell_char = " "
                if orig_config['drone_type'] == "heuristic":
                    if any(isinstance(agent, HeuristicDroneAgent) for agent in agents):
                        cell_char = "\033[94mD\033[0m"
                elif orig_config['drone_type'] == "auction":
                    if any(isinstance(agent, AuctionDroneAgent) for agent in agents):
                        cell_char = "\033[94mD\033[0m"
                if cell_char == " ":
                    fire_cells = [agent for agent in agents if isinstance(agent, FireCell)]
                    if fire_cells:
                        fire_cell = fire_cells[0]
                        if fire_cell.state == "burning":
                            cell_char = "\033[91m█\033[0m"
                        elif fire_cell.state == "burnt":
                            cell_char = "\033[30m█\033[0m"
                        else:
                            if fire_cell.terrain_type == "forest":
                                cell_char = "\033[32m█\033[0m"
                            elif fire_cell.terrain_type == "grassland":
                                cell_char = "\033[92m█\033[0m"
                            elif fire_cell.terrain_type == "urban":
                                cell_char = "\033[90m█\033[0m"
                row.append(cell_char)
            row_str = "".join(row)
            step_grid.append(row_str)
        episode_steps.append("\n".join(step_grid))

    output_filename = f"episode_steps_{orig_config['drone_type']}.ans"
    with open(output_filename, 'w', encoding='utf-8') as f:
        for step_num, (grid, burning, natural_burnouts, extinguished) in enumerate(
            zip(episode_steps, burning_cells_list, natural_burnouts_list, extinguished_list), 1):
            f.write(f"\033[1mStep {step_num}\033[0m: Burning cells = {burning}\n")
            f.write(f" Natural burnouts this step: {natural_burnouts}\n")
            f.write(f" Extinguished by agents this step: {extinguished}\n")
            f.write(f"{grid}\n{'-' * orig_config['width']}\n")
    print(f"Episode steps saved to '{output_filename}'.") 

elif orig_config['drone_type'] == 'rl':
    env = RLWildfireEnv(orig_config)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=4, num_cpus=1, base_class='stable_baselines3')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save('ppo_fire_extinguish_1000k')
    # demo rollout using vectorized env
    test_env = RLWildfireEnv(orig_config)
    obs_dict, _ = test_env.reset()
    episode_steps = []  # Store grid states for each step
    burning_cells_list = []
    natural_burnouts_list = []
    extinguished_list = []
    while True:
        agents = test_env.agents
        if not agents:
            break
        obs_list = [obs_dict[agent] for agent in agents]
        obs_vec = np.array(obs_list)
        actions_vec, _ = model.predict(obs_vec)
        actions_dict = {agent: actions_vec[i] for i, agent in enumerate(agents)}
        obs_dict, rewards, terminations, truncations, infos = test_env.step(actions_dict)
        grid_state = test_env.render()
        episode_steps.append(grid_state)
        # Collect stats for this step
        burning_cells_list.append(test_env.model.count_burning_cells())
        natural_burnouts_list.append(test_env.model.natural_burnouts)
        extinguished_list.append(test_env.model.extinguished_count)
        if all(terminations.values()) or all(truncations.values()):
            break
    print("Episode finished.")
    # Save the episode steps to a file
    with open('episode_steps_rl_1000k_1.ans', 'w', encoding='utf-8') as f:   
        for step_num, (grid, burning, natural_burnouts, extinguished) in enumerate(
            zip(episode_steps, burning_cells_list, natural_burnouts_list, extinguished_list), 1):
            f.write(f"\033[1mStep {step_num}\033[0m: Burning cells = {burning}\n")
            f.write(f" Natural burnouts this step: {natural_burnouts}\n")
            f.write(f" Extinguished by agents this step: {extinguished}\n")
            f.write(f"{grid}\n{'-' * test_env.model.grid.width}\n")
    print("Episode steps saved.")
    test_env.close()