import yaml


# Function to load yaml configuration file
def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config