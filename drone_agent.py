from mesa import Agent
import random

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

