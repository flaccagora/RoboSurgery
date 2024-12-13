import os
import re

def find_last_checkpoint(directory):
    # Define the regex pattern to match the checkpoint file names
    pattern = re.compile(r"rl_model_(\d+)_steps")

    last_checkpoint = None
    max_steps = -1

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the step number
            steps = int(match.group(1))
            # Check if this is the highest step count seen so far
            if steps > max_steps:
                max_steps = steps
                last_checkpoint = filename

    return os.path.splitext(last_checkpoint)[0] if last_checkpoint else None
