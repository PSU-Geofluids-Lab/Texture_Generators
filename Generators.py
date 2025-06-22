import os
import numpy as np
from Base_Generators import BaseGenerator
import porespy as ps


class RandomGenerator(BaseGenerator):
    def __init__(self, name='test1',size=(256, 256)):
        # Initialize BaseGenerator with the provided name
        self.name = "RandomGenerator/"+name
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


# ''' he blobs function works by generating an image of random noise then applying a Gaussian blur. The creates a correleated field with a Gaussian distribution. The function will then normalize the values back to a uniform distribution, which allow direct thresholding of the image to obtain a binary images (i.e. solid and void).

# Let’s start by illustrating the correlated noise field returned by the function. It’s possible to create directional correlation by specifying the blobiness argument as an array with a different value in each direction, as 

# It’s also possible to receive an already thresholded image from the function directly by specifying the porosity argument:

# '''

# im1 = ps.generators.blobs(shape=[300, 300], porosity=None, blobiness=[1, 1])
# im2 = ps.generators.blobs(shape=[300, 300], porosity=None, blobiness=[1, 2])

# im1 = ps.generators.blobs(shape=[300, 300], porosity=0.5, blobiness=1)
# im2 = ps.generators.blobs(shape=[300, 300], porosity=0.5, blobiness=2)
# im3 = ps.generators.blobs(shape=[300, 300], porosity=0.5, blobiness=3)

# '''
# Sometimes it is useful or necessary to study processes across multiple scales. It is possible to generate artificial images with 2 scales of porosity fairly easily.

# '''

# im1 = ps.generators.blobs([500, 500], blobiness=0.5, porosity=0.6)
# fig, ax = plt.subplots(1, 1, figsize=[6, 6])
# ax.imshow(im1, interpolation='none')
# ax.axis(False);


# im1 = ps.generators.blobs([500, 500], blobiness=0.5, porosity=0.6)
# fig, ax = plt.subplots(1, 1, figsize=[6, 6])
# ax.imshow(im1, interpolation='none')
# ax.axis(False);

# '''
# Finally, we can multiply these two image (i.e. arrays) together, which will have the effect of perforating the foreground phase of im1 with holes defined by the background phase of im2. Note the inverting of the image to ensure the foreground and background are maintained:


# '''
# im3 = ~(~im1*im2)
# fig, ax = plt.subplots(1, 1, figsize=[6, 6])
# ax.imshow(im3, interpolation='none')
# ax.axis(False);

# lt = ps.filters.local_thickness(im3)
# psd = ps.metrics.pore_size_distribution(lt)

# ps.visualization.bar(psd);

# '''
# Using random_spheres to insert non-overlapping spheres into blobs
# Another way is to use random_spheres to insert spheres into the foreground of another image. Let’s use blobs to define the superstructure, the perforate it with holes. Note that we invert the image (~) before passing it to random_spheres so that the holes are added to the solid phase, then the result is inverted back.


# '''

# im1 = ps.generators.blobs([300, 300], blobiness=0.5, porosity=0.6)
# im2 = ~ps.generators.random_spheres(im=~im1, r=10, clearance=-2)
# fig, ax = plt.subplots(1, 1, figsize=[6, 6])
# ax.imshow(im2, interpolation='none')
# ax.axis(False);