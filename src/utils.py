import os
import yaml

def recursive_mkdirs(folder_path:str)->None:
    """
    Recursively create the folders leading to folder_path

    Arguments
    ---------
    - folder_path: (str)
      path to folder to create as well as its ancestors
    """
    parent_folder_path = os.path.dirname(folder_path)
    if not (os.path.exists(parent_folder_path)) and parent_folder_path != "":
        recursive_mkdirs(parent_folder_path)
        if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)
    else:
        if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)


def locate_in_list(var,list):
    for i in range(len(list)):
        if var == list[i]:
            return i
    return None


def read_yaml(src_path:str) -> dict:
    assert os.path.splitext(src_path)[1] in [".yml",".yaml"], f"Wrong extension for the file {src_path}: should be .yml or .yaml"

    with open(src_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(data:dict,dst_path:str):
    assert os.path.splitext(dst_path)[1] in [".yml",".yaml"], f"Wrong extension for the file {dst_path}: should be .yml or .yaml"
    assert os.path.dirname(dst_path) == "" or os.path.exists(os.path.dirname(dst_path)), f"Directory {os.path.dirname(dst_path)} does not exist"

    with open(dst_path, 'w') as file:
        yaml.dump(data, file)


