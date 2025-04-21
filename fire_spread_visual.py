import mesa
from mesa.visualization import SolaraViz
from mesa.visualization.components.altair_components import make_altair_space
from fire_spread import WildfireModel, FireCell

def firecell_portrayal(agent):
    if agent is None:
        return {}
    portrayal = {}
    if isinstance(agent, FireCell):
        if agent.state == "burning":
            portrayal["color"] = "tab:red"
        elif agent.state == "burnt":
            portrayal["color"] = "tab:black"
        else:  # unburnt
            if agent.terrain_type == "forest":
                portrayal["color"] = "tab:green"
            elif agent.terrain_type == "grassland":
                portrayal["color"] = "tab:yellow"
            elif agent.terrain_type == "urban":
                portrayal["color"] = "tab:lightgrey"
            else:
                portrayal["color"] = "green"
        portrayal["size"] = 10
        portrayal["marker"] = "rect"
    return portrayal


def propertylayer_portrayal(*args, **kwargs):
    return {}

# Define a dummy post-processing function (optional)
def post_process(chart):
    return chart

# Create the visualization component with required parameters
space_component = make_altair_space(
    agent_portrayal=firecell_portrayal,
    propertylayer_portrayal=propertylayer_portrayal,
    post_process=post_process
)

model_params = {
    "width": {"type": "SliderInt", "value": 50, "min": 10, "max": 100, "step": 1},
    "height": {"type": "SliderInt", "value": 50, "min": 10, "max": 100, "step": 1},
    "initial_fire_positions": {"value": [(25, 25)]},
    "wind_direction": {"value": (0, 1)},
    "width": 50,
    "height": 50
}

grid_width, grid_height = 100, 100
initial_fires = [(50, 50), (30, 70)] 
urban_starts = [(20, 20), (80, 80)]  

model = WildfireModel(
        width=grid_width, 
        height=grid_height,
        initial_fire_positions=initial_fires,
        urban_start_positions=urban_starts,
        urban_shape="circle",  # "circle" "line"
        wind_direction=(1, 0),
        grassland_max_distance=15)

page = SolaraViz(
    model,
    components=[space_component],
    model_params = model_params,
    name="Wildfire Model"
)

page
