import numpy as np
import yaml

from collections import OrderedDict
from ruamel.yaml import YAML
import numpy as np
import yaml

class CompactNumpyDumper(yaml.SafeDumper):
    def represent_list(self, data):
        # Use compact style for lists
        if all(isinstance(i, list) for i in data):
            return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
        return super().represent_list(data)

    def represent_data(self, data):
        if isinstance(data, np.ndarray):
            return self.represent_list(data.tolist())  # Convert array to list
        return super().represent_data(data)

def yaml_ordered_loader(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)

def yaml_ordered_dumper(data, stream=None, Dumper=yaml.SafeDumper, **kwargs):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwargs)

def Optvars2Yaml(opt_vars, sim_path, time_id):
    yaml = YAML()
    yaml.preserve_quotes = True  # Preserve quotes and formatting

    # Read the existing YAML file
    with open(sim_path + '.yaml', "r") as file:
        yaml_data = yaml.load(file)

    # Update the specific key
    yaml_data['controller']['settings']['orientation_deg'] = opt_vars.tolist()

    # Write to a new YAML file, ensuring it overwrites cleanly
    with open(sim_path + time_id + '.yaml', "w") as file:
        yaml.dump(yaml_data, file)


if __name__ == '__main__':
    yaml_path = "../../02_Examples_and_Cases/02_Example_Cases/dummy_yaml"
    t1 = np.array([270, 270, 285, 285])
    t2 = np.array([270, 270, 270, 270])
    opt_vars = np.array([t1, t2]).T
    Optvars2Yaml(opt_vars, yaml_path, '12342452')