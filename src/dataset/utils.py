import os
import yaml

def create_yaml_file(output_dir, class_name):
    """Creates the dataset.yaml file required by YOLO."""
    
    # Get absolute paths for the YAML file
    train_path = os.path.abspath(os.path.join(output_dir, 'images', 'train'))
    val_path = os.path.abspath(os.path.join(output_dir, 'images', 'val'))
    
    yaml_content = {
        'train': train_path,
        'val': val_path,
        'nc': 1,  # number of classes
        'names': [class_name]  # list of class names
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nSuccessfully created {yaml_path}")
    print("This file points to your training and validation data.")