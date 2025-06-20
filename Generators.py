import os
import numpy as np
from Base_Generators import BaseGenerator

class PorePyGenerator(BaseGenerator):
    def __init__(self, name='test1',size=(256, 256)):
        # Initialize BaseGenerator with the provided name
        self.name = "PorePyGenerator/"+name
        self.size = size
        self.data = None
        self.metadata = {
            'generator_type': self.name,  # Use the provided name here
            'size': size
        }
        results_folder = os.path.join("Results", self.name) # Use the provided name here
        os.makedirs(results_folder, exist_ok=True)

        # Set the full path for the generator instance's saved files
        # This assumes the name is used as the directory name within "Results"
        self.full_path = results_folder

    """Example implementation using porePy's texture generator"""
    def generate(self, seed=None):
        if seed is None:
            # Generate a random seed if none is provided
            seed = np.random.randint(0, 2**32 - 1)
        print(f"Using seed: {seed}")
        np.random.seed(seed)
        self.data = np.random.rand(*self.size)
        self.add_metadata('seed', seed)
        return self.data
