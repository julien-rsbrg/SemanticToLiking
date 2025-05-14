import os

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