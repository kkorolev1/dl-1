import yaml
import logging


CONFIG_PATH = "config.yaml"


def read_config(config_path = CONFIG_PATH):
    """Reads config from yaml file, see config.yaml

    Args:
        config_path (str, optional): path to config file. Defaults to CONFIG_PATH.

    Returns:
        dict: dictionary of configuration
    """
    with open(config_path, "r") as yaml_file:
        try:
            data = yaml.safe_load(yaml_file)
            return data
        except yaml.YAMLError as exc:
            logging.error("Cannot read yaml from {}, because of\n{}".format(config_path, exc))

def config_str(config):
    def parse_node(node, level):
        s = ""
        for key, value in node.items():
            s += "\n" + "\t" * level + key + ": "
            if isinstance(value, dict):
                s += parse_node(value, level + 1)
            else:
                s += str(value)
        return s
    return parse_node(config, 0)
    
