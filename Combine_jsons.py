import json
import os

def combine_json_files(input_dir: str, output_file: str) -> None:
    combined_data = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                combined_data.extend(data)
    with open(output_file, 'w') as output:
        json.dump(combined_data, output, indent=4)

input_directory = 'CatA_Simple' 
output_file = 'combined_output.json' 

combine_json_files(input_directory, output_file)
