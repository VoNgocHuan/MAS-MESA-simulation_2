# Standard library
import random

# External library
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.experimental.devs import ABMSimulator
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import PPO
import supersuit as ss
import numpy as np
from mesa.visualization import (
    CommandConsole,
    Slider,
    SolaraViz,
    make_plot_component,
)
from mesa.visualization.components.matplotlib_components import make_mpl_space_component


# Internal library
from drone_agent import (
    FireCell, 
    HeuristicDroneAgent, 
    RLDroneAgent, 
    AuctionDroneAgent, AuctioneerAgent
    )


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
        self.space = self.grid
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
        self.step_count = 0

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
            # Diminishing reward for extinguishing
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
    model.learn(total_timesteps=100000, progress_bar=True)
    model.save('ppo_fire_extinguish_100k')
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
    with open('episode_steps_rl_100k.ans', 'w', encoding='utf-8') as f:   
        for step_num, (grid, burning, natural_burnouts, extinguished) in enumerate(
            zip(episode_steps, burning_cells_list, natural_burnouts_list, extinguished_list), 1):
            f.write(f"\033[1mStep {step_num}\033[0m: Burning cells = {burning}\n")
            f.write(f" Natural burnouts this step: {natural_burnouts}\n")
            f.write(f" Extinguished by agents this step: {extinguished}\n")
            f.write(f"{grid}\n{'-' * test_env.model.grid.width}\n")
    print("Episode steps saved.")
    test_env.close()