from os import path as p
import toml

__THIS_DIR_PATH = p.dirname(p.realpath(__file__))
CACHE_DIR = p.join(__THIS_DIR_PATH, "experiments", "models", "_cache")

def get_version():
    project_file = p.join(__THIS_DIR_PATH, 'pyproject.toml')
    with open(project_file, 'r') as f:
        project_config = toml.load(f)
    return project_config['tool']['poetry']['version']
