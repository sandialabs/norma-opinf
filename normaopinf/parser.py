import yaml

def open_yaml(file_path):
    with open(file_path) as f:
        yaml_obj = yaml.safe_load(f)
    return yaml_obj

def save_yaml(yaml_obj,file_path):
    with open(file_path,'w') as f:
        yaml.dump(yaml_obj,f, default_flow_style=False)

def set_opinf_model_file(yaml_obj,model_file):
    yaml_obj['model']['model-file'] = model_file

### For saving w/ quotes
class quoted(str):
    pass

def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

