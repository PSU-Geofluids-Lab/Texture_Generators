import os
import numpy as np
from Base_Generators import BaseGenerator
import porespy as ps
import matplotlib.pyplot as plt

class RandomGenerator(BaseGenerator):
    def __init__(self, name='test1',size=(256, 256)):
        # Initialize BaseGenerator with the provided name
        """
        Initialize a RandomGenerator instance.

        Parameters
        ----------
        name : str, optional
            A name for the generator instance. This is used to create a
            directory name within "Results" to store generated files.
            Defaults to 'test1'.
        size : tuple, optional
            A tuple of two integers specifying the (width, height) of the
            generated image. Defaults to (256, 256).
        """
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

    def generate(self, seed=None):
        """
        Generate a random 2D image.

        Parameters
        ----------
        seed : int, optional
            Random seed for the generated image. If None, a random seed is generated.

        Returns
        -------
        data : ndarray
            The generated image
        """
        if seed is None:
            # Generate a random seed if none is provided
            seed = np.random.randint(0, 2**32 - 1)
        print(f"Using seed: {seed}")
        np.random.seed(seed)
        self.data = np.random.rand(*self.size)
        self.add_metadata('seed', seed)
        return self.data

class PoresPyGenerator(BaseGenerator):
    def __init__(self, name='blobs',size=(256, 256)):
        # Initialize BaseGenerator with the provided name
        """
        Initialize the PoresPyGenerator instance.

        Parameters
        ----------
        name : str, optional
            The name of the generator instance, used to create a directory for results.
            Default is 'test1'.
        size : tuple of int, optional
            The size of the generated 2D image as (width, height). Default is (256, 256).

        Attributes
        ----------
        name : str
            The name of the generator prefixed with 'PoresPyGenerator/'.
        size : tuple of int
            The dimensions of the 2D image to be generated.
        data : NoneType
            Placeholder for the generated image data.
        metadata : dict
            Metadata containing 'generator_type' and 'size'.
        full_path : str
            The full path to the directory where results are saved.
        """
        allowed_names = ['blobs', 'cylindrical_pillars_array',
                         'fractal_noise','random_spheres','overlapping_spheres','polydisperse_spheres',
                         'random_cantor_dust','voronoi_edges']
        if name not in allowed_names:
            raise ValueError(f"Invalid name: {name}. Must be one of: {', '.join(allowed_names)}")
        self.name = "PoresPyGenerator/"+name
        self.name_method = name
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

    def generate(self, seed=None,porosity=None,**kwargs):
        if seed is None:
            # Generate a random seed if none is provided
            seed = np.random.randint(0, 2**32 - 1)
        print(f"Using seed: {seed}")
        if self.name_method == 'blobs':
            # Generate a 2D image using the Blobs generator from PoresPy.
            # This function uses the blobs function works by generating an image of random noise then applying a Gaussian blur. 
            # The creates a correleated field with a Gaussian distribution. The function will then normalize the values back to a uniform distribution, 
            # which allow direct thresholding of the image to obtain a binary images (i.e. solid and void).
            # It’s possible to create directional correlation by specifying the blobiness argument as an array with a different value in each direction '''
            blobiness = kwargs.pop('blobiness', [1,1])
            if porosity is not None:
                self.data_unthreshold = ps.generators.blobs(shape=self.size, seed=seed, porosity=None, blobiness=blobiness,**kwargs)
            self.data = ps.generators.blobs(shape=self.size, seed=seed, porosity=porosity, blobiness=blobiness,**kwargs)
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('blobiness', blobiness)
            print(f'Done! - Use the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            return self.data
        elif self.name_method == 'cylindrical_pillars_array':
            # Generate an image of cylindrical pillars using the cylindrical_pillars_array function from PoreSpy.
            # >>> texture_data =generator.generate(spacing=3,dist='norm', dist_kwargs=dict(loc=1, scale=1))
            # >>> texture_data =generator.generate(spacing=3,dist='uniform', dist_kwargs=dict(loc=1, scale=1),lattice='simple',)
            spacing = kwargs.pop('spacing', 10)
            dist = kwargs.pop('dist', 'norm')
            dist_kwargs = kwargs.pop('dist_kwargs', dict(loc=2, scale=4))
            lattice = kwargs.pop('lattice', 'triangular')
            self.data = ps.generators.cylindrical_pillars_array(shape=self.size, seed=seed, spacing=spacing,lattice=lattice,
                                                    dist=dist, dist_kwargs=dist_kwargs,**kwargs)
            porosity = np.sum(self.data)/(self.data.shape[0]*self.data.shape[1])
            print('Porosity: '+str(porosity))
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('lattice', lattice)
            self.add_metadata('spacing', spacing)
            self.add_metadata('dist', dist)
            self.add_metadata('dist_kwargs', dict(loc=1, scale=1))
            print(f'Done! - Use the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            return self.data
        elif self.name_method == 'fractal_noise':
            frequency = kwargs.pop('frequency', 0.1)
            octaves = kwargs.pop('octaves', 6)
            gain = kwargs.pop('gain', 0.5)
            mode = kwargs.pop('mode', 'simplex')
            self.data_unthreshold = ps.generators.fractal_noise(shape=self.size,seed=seed,frequency=frequency,uniform=False,
                                                                    octaves=octaves,gain=gain,mode=mode,**kwargs)       
            self.data = ps.generators.fractal_noise(shape=self.size,seed=seed,frequency=frequency,uniform=True,
                                                                    octaves=octaves,gain=gain,mode=mode,**kwargs)   
            porosity = np.sum(self.data)/(self.data.shape[0]*self.data.shape[1])
            print('Porosity: '+str(porosity))
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('octaves', octaves)
            self.add_metadata('mode', mode)
            self.add_metadata('gain', gain)
            self.add_metadata('frequency', frequency)
            print(f'Done! - Use the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            return self.data
        elif self.name_method == 'random_cantor_dust':
            n = kwargs.pop('n', 8)
            f = kwargs.pop('f', 1)
            p = kwargs.pop('p', 3)
            self.data = ps.generators.random_cantor_dust(shape=self.size,seed=seed,n=n,**kwargs) 
            porosity = np.sum(self.data)/(self.data.shape[0]*self.data.shape[1])
            print('Porosity: '+str(porosity))
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('f', f)
            self.add_metadata('n', n)
            self.add_metadata('p', p)
            print(f'Done! - Use the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            return self.data
        elif self.name_method == 'polydisperse_spheres':
            import scipy.stats as spst
            if porosity is None:
                porosity = 0.5
            dist = kwargs.pop('dist', spst.norm(loc=10, scale=5))
            nbins = kwargs.pop('nbins', 10)
            r_min = kwargs.pop('r_min', 5)
            self.data = ps.generators.polydisperse_spheres(seed=seed,shape=self.size, porosity=porosity, dist=dist, nbins=nbins,r_min=r_min,**kwargs)
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('dist', dist)
            self.add_metadata('nbins', nbins)
            self.add_metadata('r_min', r_min)
            print(f'Done! - Use the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            return self.data
        elif self.name_method == 'random_spheres':
            import scipy.stats as spst
            r = kwargs.pop('r', 5)
            clearance = kwargs.pop('clearance', 0.1)
            protrusion = kwargs.pop('protrusion', 3)
            self.data = ps.generators.random_spheres(seed=seed,im_or_shape=self.size, r=r,edges='contained',clearance=clearance,protrusion=protrusion,
                                                    *kwargs)
            porosity = np.sum(self.data)/(self.data.shape[0]*self.data.shape[1])
            print('Porosity: '+str(porosity))
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('edges', 'contained')
            self.add_metadata('clearance', clearance)
            self.add_metadata('protrusion', protrusion)
            self.add_metadata('r', r)
            print(f'Done! - Use the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            return self.data
        elif self.name_method == 'overlapping_spheres':
            t = 1e-2
            mi=100
            r = kwargs.pop('r', 5)
            if porosity is None:
                porosity = 0.5
            self.data = ps.generators.overlapping_spheres(seed=seed,shape=self.size, r=r,porosity=porosity,maxiter=mi, tol=t, *kwargs)
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('tolerance',t)
            self.add_metadata('maxiert', mi)
            self.add_metadata('r', r)
            print(f'Done! - Use the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            return self.data
        elif self.name_method == 'voronoi_edges':
            ncells = kwargs.pop('ncells', 50)
            r = kwargs.pop('r', 1)
            self.data = ps.generators.voronoi_edges(seed=seed,shape=self.size, ncells=ncells,flat_faces=False,r=r, *kwargs)
            self.add_metadata('seed', seed)
            self.add_metadata('ncells', ncells)
            self.add_metadata('r', r)
            print(f'Done! - Use the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            return self.data
        else:
            raise ValueError(f"Invalid name: {self.name_method}. Must be one of: [blobs,cylindrical_pillars_array,fractal_noise,random_cantor_dust, \\\
                                   random_spheres,overlapping_spheres,polydisperse_spheres,voronoi_edges]")


