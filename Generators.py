import os
import numpy as np
from Base_Generators import BaseGenerator
import porespy as ps
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import networkx as nx
import seaborn as sns

import structify_net.zoo as zoo

def Make_rank_matrix(rank_model,nodeOrder=None,ax=None,make_plot=False,**kwargs):
    """
    Plot a matrix of the graph, ordered by nodePair_order
    graph: a networkx graph
    nodePair_order: a list of node pairs, from the most likely to the less likely
    nodeOrder: a list of nodes, ordered by the order in which they should appear in the matrix
    """
    nodePair_order=rank_model.sortedPairs
    if nodeOrder!=None:
        n=len(nodeOrder)
    else:
        n=len(set([n for e in nodePair_order for n in e]))


    matrix = np.zeros((n,n))
    for i,e in enumerate(nodePair_order):
        e_l=list(e)
        matrix[e_l[0],e_l[1]]=i
        matrix[e_l[1],e_l[0]]=i

    if nodeOrder!=None:
        matrix=matrix[nodeOrder,:]
        matrix=matrix[:,nodeOrder]

    if make_plot:
        heatmap_args={'cmap':"YlGnBu_r",'cbar':False,'xticklabels':False,'yticklabels':False}
        for k,v in kwargs.items():
            heatmap_args[k]=v
        m = sns.heatmap(matrix,ax=ax,**heatmap_args)
        plt.show()
    return matrix

def generate_adjacency_matrix(graph_type_name, n=10, **kwargs):
  """
  Generates an adjacency matrix for a networkx graph of a specified type.

  Args:
    graph_type_name: The name of the networkx graph generation function (string).
                     Must be a function available in networkx.
    n: The number of nodes in the graph (integer).
    **kwargs: Additional keyword arguments to pass to the networkx graph
              generation function.

  Returns:
    A numpy array representing the adjacency matrix of the generated graph.
    Returns None if the graph type is not found.
  """
  try:
    graph_generator = getattr(nx, graph_type_name)
    G = graph_generator(n=n, **kwargs)
    adj_matrix = nx.to_numpy_array(G)
    if graph_type_name == 'margulis_gabber_galil_graph':
        print('Actual graph size in margulis_gabber_galil_graph is',G.number_of_nodes())
    return adj_matrix,G
  except AttributeError:
    print(f"Error: Graph type '{graph_type_name}' not found in networkx.")
    return None

class Graph_Generators(BaseGenerator):
    def __init__(self,size=128,p=0.4,epsilon=0.1):
        """
        Initialize a Graph_Generators instance.

        Parameters
        ----------
        size : int, optional
            The size of the generated texture. Defaults to 128.
        p : float, optional
            The probability of having a connection between two nodes. Defaults to 0.4.
        epsilon : float, optional
            The maximum distance between two points to consider them connected. Defaults to 0.1.

        Notes
        -----
        The generated texture is a graph, where each node is a pixel in the image.
        The edges in the graph are the connections between pixels, which are determined
        by the probability `p` and the maximum distance `epsilon`.
        The generated texture is saved in a directory named after the generator instance,
        within the "Results" directory.

        The generator instance is initialized with the following metadata:
            - 'probability': p
            - 'epsilon': epsilon
            - 'size' : size
            - 'generator_type': self.name
            - 'generator_reference': 'Structify-Net - https://structify-net.readthedocs.io/en/latest/index.html'
        """
        self.name = "Graph_Generators/"
        self.data = None
        self.p=p
        self.epsilon = epsilon
        self.size = size
        self.metadata = {
            'probability': p,
            'epsilon': epsilon,
            'm' : (size*p)**2.,
            'size' : size,
            'generator_type': self.name,  # Use the provided name here
            'generator_reference': 'Structify-Net - https://structify-net.readthedocs.io/en/latest/index.html',  # Use the provided name here
        }
        results_folder = os.path.join("Results", self.name) # Use the provided name here
        self.full_path = results_folder
        self.m = (size*p)**2.
        self.name_models = ["ER","spatial","spatialWS","blocks_assortative","overlapping_communities", "nestedness","maximal_stars","core_distance","fractal_leaves", "fractal_root","fractal_hierarchy","fractal_star","perlin_noise","disconnected_cliques"]
        # Print the list of files
        print('Number of Models : '+str(len(self.name_models)) + '\n Model Names :')
        print('\t'.join(self.name_models))

    def generate(self, file_name,**kwargs):
        self.name = self.name+'/'+file_name+f'/p{self.p}_ep{self.epsilon}/'
        self.model_name = file_name
        self.full_path  = self.full_path+file_name 
        os.makedirs(self.full_path, exist_ok=True)
        # Set the full path for the generator instance's saved files
        # This assumes the name is used as the directory name within "Results"
        k = kwargs.get('dimensions', 10) #Number of nearest neighbors. Defaults to 10.
        blocks = kwargs.get('blocks', None) # Blocks or communities. Can be either a list of lists, where each list is a block, or an integer, in which case the nodes are randomly assigned to the corresponding number of equal size blocks. Defaults to None.
        dimensions = kwargs.get('dimensions', 1) # Number of dimensions to embed the graph in. Defaults to 1.
        d = kwargs.get('d', 3) # Degree of the binary tree. Defaults to 3.
        octaves = kwargs.get('octaves', 6) # Number of octaves to use in the fractal noise generation (Perlin). Defaults to 6.
        if file_name != 'disconnected_cliques':
            rank_model = zoo.all_models_no_param[file_name](self.size,**kwargs)
        else : 
            rank_model = zoo.all_models_with_m[file_name](self.size,self.m)
        matrix = Make_rank_matrix(rank_model,nodeOrder = rank_model.node_order)
        self.data_prob = matrix/np.max(matrix)
        self.make_plots(self.data_prob,filename_save=f'Prob_Density_plot_p{self.p}_ep{self.epsilon}.png')
        self.gpx = rank_model.generate_graph(epsilon=self.epsilon,density=self.p)
        self.data = nx.to_numpy_array(self.gpx)
        self.data  = self.data/np.max(self.data)
        self.make_plots(self.data,filename_save=f'Realization_plot_p{self.p}_ep{self.epsilon}.png')
        # Print the image shape and data type
        self.add_metadata('model_name', file_name)
        self.add_metadata('k', k)
        self.add_metadata('blocks', blocks)
        self.add_metadata('dimensions', dimensions)
        self.add_metadata('octaves', octaves)
        self.add_metadata('d', d)
        return self.data

    def make_spatial_realization(self,scaling_ratio=10,linlog=False,plot_me=False,target_porosity=0.1):
        """
        Make a spatial realization of the graph, using the ForceAtlas2 and Spring embedding methods.
        
        Parameters
        ----------
        scaling_ratio : float, optional
            Determines the scaling of attraction and repulsion forces. Defaults to 2.0.
        linlog : bool, optional
            Uses logarithmic attraction instead of linear. Defaults to False.
        plot_me : bool, optional
            Plot the two embeddings. Defaults to False.
        target_porosity : float, optional
            Target porosity of the generated image. Defaults to 0.1.
        Notes
        -----
        The generated image is saved in a directory named after the generator instance,
        within the "Results" directory.

        The generator instance is initialized with the following metadata:
            - 'porosity': porosity
            - 'generator_type': self.name
            - 'generator_reference': 'Structify-Net - https://structify-net.readthedocs.io/en/latest/index.html'
        """
        self.data = np.zeros([self.size,self.size])
        self.data_spring = np.zeros([self.size,self.size])
        self.porosity =  np.sum(self.data)/(self.data.shape[0]*self.data.shape[1])
        iteration_num = 0
        while self.porosity < target_porosity:
            pos_forceatlas2 = nx.forceatlas2_layout(self.gpx,max_iter=1000,scaling_ratio=scaling_ratio,linlog=linlog)
            pos_spring = nx.spring_layout(self.gpx,iterations=1000,method="energy", scale=1)
            joint_array = np.vstack(list(pos_forceatlas2.values()))
            joint_array_forceatlas2 = (nx.rescale_layout(joint_array,scale=self.size/2) + self.size/2-1).astype(int)
            del joint_array
            joint_array = np.vstack(list(pos_spring.values()))
            joint_array_spring = (nx.rescale_layout(joint_array,scale=self.size/2) + self.size/2-1).astype(int)
            del joint_array
            self.data_adjacency = self.data.copy()
            self.data[joint_array_forceatlas2[:,0],joint_array_forceatlas2[:,1]] = 1
            self.data_spring[joint_array_spring[:,0], joint_array_spring[:,1]] = 1
            self.porosity =  np.sum(self.data)/(self.data.shape[0]*self.data.shape[1])
            print(f'Iteration - porosity target : {self.porosity}, Iteration {iteration_num}')
            iteration_num+=1
        if plot_me:
            plt.figure(figsize=(10,10))
            plt.plot(joint_array_forceatlas2[:,0],joint_array_forceatlas2[:,1],'o',label='ForceAtlas2')
            plt.plot(joint_array_spring[:,0],joint_array_spring[:,1],'o',label='Spring')
            plt.title('Physical Space representation of the generated graph')
            plt.legend()
            plt.show()
        self.add_metadata('porosity', self.porosity)
        print('Done with the graph Spatial representation generation')

    def make_plots(self,matrix,filename_save='Prob_Density_plot.png'):
        plt.figure(figsize=(10,10))
        plt.imshow(matrix,cmap='RdYlBu_r')
        plt.title(f'Model Name : {self.model_name}')
        plt.gca().set_xlabel("x-axis", fontsize=35)
        plt.gca().set_ylabel("y-axis", fontsize=35)
        plt.colorbar()
        plt.savefig(self.full_path + '/'+filename_save)
        plt.close()

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
        self.name = "RandomGenerator/"+name+"/"+str(size[0])+'_'+str(size[1])
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
    def __init__(self, name='blobs',name_append='',size=(256, 256)):
        """
        Initialize a PoresPyGenerator instance.

        Parameters
        ----------
        name : str, optional
            The name of the generator function to use. Must be one of:
            'blobs', 'cylindrical_pillars_array', 'fractal_noise',
            'random_spheres', 'overlapping_spheres', 'polydisperse_spheres',
            'random_cantor_dust', 'voronoi_edges'
        name_append : str, optional
            An optional string to append to the name of the generator.
        size : tuple of int, optional
            The size of the generated image.

        Notes
        -----
        This class is a wrapper around the PoresPy generators. It provides a
        standardized interface for generating 2D images and saving them to
        disk. The generated image is stored in the `data` attribute, and the
        metadata is stored in the `metadata` attribute.
        """
        allowed_names = ['blobs', 'cylindrical_pillars_array',
                         'fractal_noise','random_spheres','overlapping_spheres','polydisperse_spheres',
                         'random_cantor_dust','voronoi_edges']
        if name not in allowed_names:
            raise ValueError(f"Invalid name: {name}. Must be one of: {', '.join(allowed_names)}")
        self.name = "PoresPyGenerator/"+name+'_'+name_append+'/'+str(size[0])+'_'+str(size[1])
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
        """
        Generate a 2D image using the provided PoresPy generator.

        Parameters
        ----------
        seed : int, optional
            Random seed for the generated image. If None, a random seed is generated.
        porosity : float, optional
            Porosity target for the generated image. If None, the porosity is not set.
        **kwargs
            Additional keyword arguments to pass to the PoresPy generator.

        Returns
        -------
        data : ndarray
            The generated image.
        """
        if seed is None:
            # Generate a random seed if none is provided
            seed = np.random.randint(0, 2**32 - 1)
        print(f"Using seed: {seed}")
        if self.name_method == 'blobs':
            """
            Generate a 2D image using the Blobs generator from PoresPy.

            This function generates an image of random noise and applies a Gaussian blur
            to create a correlated field with a Gaussian distribution. The values are then 
            normalized back to a uniform distribution, allowing direct thresholding to obtain 
            binary images (i.e., solid and void). Directional correlation can be created by 
            specifying the blobiness argument as an array with different values in each direction.
            """
            blobiness = kwargs.pop('blobiness', [1, 1])
            if porosity is not None:
                self.data_unthreshold = ps.generators.blobs(
                    shape=self.size, seed=seed, porosity=None, blobiness=blobiness, **kwargs)
            self.data = ps.generators.blobs(
                shape=self.size, seed=seed, porosity=porosity, blobiness=blobiness, **kwargs)
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('blobiness', blobiness)
            print(f'Done! - Used the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            print(f'metadata: {self.metadata}')
            return self.data

        elif self.name_method == 'cylindrical_pillars_array':
            """
            Generate an image of cylindrical pillars using the cylindrical_pillars_array function from PoreSpy.

            Parameters such as spacing, distribution, and lattice type can be provided to customize
            the generated array of pillars.
            """
            spacing = kwargs.pop('spacing', 10)
            dist = kwargs.pop('dist', 'norm')
            dist_kwargs = kwargs.pop('dist_kwargs', dict(loc=2, scale=4))
            lattice = kwargs.pop('lattice', 'triangular')
            self.data = ps.generators.cylindrical_pillars_array(
                shape=self.size, seed=seed, spacing=spacing, lattice=lattice,
                dist=dist, dist_kwargs=dist_kwargs, **kwargs)
            porosity = np.sum(self.data) / (self.data.shape[0] * self.data.shape[1])
            print('Porosity: ' + str(porosity))
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('lattice', lattice)
            self.add_metadata('spacing', spacing)
            self.add_metadata('dist', dist)
            self.add_metadata('dist_kwargs', dist_kwargs)
            print(f'Done! - Used the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            print(f'metadata: {self.metadata}')
            return self.data

        elif self.name_method == 'fractal_noise':
            """
            Generate a 2D image using the Fractal Noise generator from PoreSpy.

            The frequency parameter controls the size of the blobs relative to the image size,
            while octaves control the levels of noise overlaid to create texture. Gain controls
            the intensity of each noise layer, and mode specifies the noise type (perlin, simplex, etc.).
            """
            frequency = kwargs.pop('frequency', 0.1)
            octaves = kwargs.pop('octaves', 6)
            gain = kwargs.pop('gain', 0.5)
            mode = kwargs.pop('mode', 'simplex')
            self.data_unthreshold = ps.generators.fractal_noise(
                shape=self.size, seed=seed, frequency=frequency, uniform=False,
                octaves=octaves, gain=gain, mode=mode, **kwargs)
            self.data = ps.generators.fractal_noise(
                shape=self.size, seed=seed, frequency=frequency, uniform=True,
                octaves=octaves, gain=gain, mode=mode, **kwargs)
            porosity = np.sum(self.data) / (self.data.shape[0] * self.data.shape[1])
            print('Porosity: ' + str(porosity))
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('octaves', octaves)
            self.add_metadata('mode', mode)
            self.add_metadata('gain', gain)
            self.add_metadata('frequency', frequency)
            print(f'Done! - Used the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            print(f'metadata: {self.metadata}')
            return self.data

        elif self.name_method == 'random_cantor_dust':
            """
            Generate a 2D image using the Random Cantor Dust generator from PoreSpy.

            The n parameter controls iterations of the Cantor dust algorithm, f controls the 
            proportion of area removed in each iteration, and p controls the probability of 
            removing a square in each iteration.
            """
            n = kwargs.pop('n', 8)
            f = kwargs.pop('f', 1)
            p = kwargs.pop('p', 3)
            self.data = ps.generators.random_cantor_dust(
                shape=self.size, seed=seed, n=n, **kwargs)
            porosity = np.sum(self.data) / (self.data.shape[0] * self.data.shape[1])
            print('Porosity: ' + str(porosity))
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('f', f)
            self.add_metadata('n', n)
            self.add_metadata('p', p)
            print(f'Done! - Used the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            print(f'metadata: {self.metadata}')
            return self.data

        elif self.name_method == 'polydisperse_spheres':
            """
            Generate a 2D image using the Polydisperse Spheres generator from PoreSpy.

            The dist parameter controls the distribution of the sphere radii, nbins controls 
            the number of bins for discretizing the radius distribution, and r_min controls 
            the minimum radius of the spheres.
            """
            dist = kwargs.pop('dist', spst.norm(loc=10, scale=5))
            nbins = kwargs.pop('nbins', 10)
            r_min = kwargs.pop('r_min', 5)
            self.data = ps.generators.polydisperse_spheres(
                seed=seed, shape=self.size, porosity=porosity, dist=dist, nbins=nbins, r_min=r_min, **kwargs)
            self.add_metadata('seed', seed)
            self.add_metadata('porosity', porosity)
            self.add_metadata('dist', dist)
            self.add_metadata('nbins', nbins)
            self.add_metadata('r_min', r_min)
            print(f'Done! - Used the {self.name_method} function with the following parameters: "shape": {str(self.size)}')
            print(f'metadata: {self.metadata}')
            return self.data

        else:
            raise ValueError(f"Invalid name: {self.name_method}. Must be one of: [blobs, cylindrical_pillars_array, fractal_noise, random_cantor_dust, random_spheres, overlapping_spheres, polydisperse_spheres, voronoi_edges]")

class USC_TextureGenerator(BaseGenerator):
    def __init__(self):
        # Initialize BaseGenerator with the provided name
        """
        Initialize a TextureGenerator instance related to the USC Texture Dataset (both rotated and unrotated images).
        https://sipi.usc.edu/database/database.php?volume=textures&image=10#top
        Parameters
        ----------
        name : str, optional
            A name for the generator instance. This is used to create a
            directory name within "Results" to store generated files.
            Defaults to 'test1'.
        """
        self.name = "USC_TextureGenerator/"
        self.data = None
        self.metadata = {
            'generator_type': self.name,  # Use the provided name here
            'generator_reference': 'https://sipi.usc.edu/database/',  # Use the provided name here
        }
        results_folder = os.path.join("Results", self.name) # Use the provided name here
        self.full_path = results_folder

        folder_path = 'Texture_Files'
        file_extension = '.tiff'  # Change to the desired extension
        # Get a list of files in the folder
        self.files = [file for file in os.listdir(folder_path) if file.endswith(file_extension)]
        # Print the list of files
        print('Number of files : '+str(len(self.files)))
        print('\t'.join(self.files))


    def generate(self, file_name):
        """
        Generate and load a 2D texture image from a specified file.

        Parameters
        ----------
        file_name : str
            The name of the .tiff file to be loaded from the 'Texture_Files/' directory.

        Returns
        -------
        data : ndarray
            The loaded image data, normalized by its maximum value.

        Notes
        -----
        The function updates the instance's name and full_path attributes to include the file name.
        It creates a directory for saving generated files if it does not already exist.
        The image shape and file name are added to the metadata.
        """
        self.name = self.name+'/'+file_name
        self.full_path  = self.full_path+file_name 
        os.makedirs(self.full_path, exist_ok=True)
        # Set the full path for the generator instance's saved files
        # This assumes the name is used as the directory name within "Results"

        folder_path = 'Texture_Files/'
        # Load the .tiff file
        self.data = plt.imread(folder_path+file_name)
        self.data  = self.data/np.max(self.data)
        # Print the image shape and data type
        self.size = self.data.shape
        self.file_name = file_name
        self.add_metadata('size', self.size)
        self.add_metadata('file_name', file_name)
        return self.data

class Graph_NetworkX_Generators(BaseGenerator):
    def __init__(self,size=128):
        self.name = "Graph_NetworkX_Generators/"
        self.data = None
        self.size = size
        self.metadata = {
            'size' : size,
            'generator_type': self.name,  # Use the provided name here
            'generator_reference': 'NetworkX',  # Use the provided name here
        }
        results_folder = os.path.join("Results", self.name) # Use the provided name here
        self.full_path = results_folder
        self.name_models = ['fast_gnp_random_graph','connected_caveman_graph','margulis_gabber_galil_graph', 'newman_watts_strogatz_graph','watts_strogatz_graph','barabasi_albert_graph','gaussian_random_partition_graph']
        # Print the list of files
        print('Number of Models : '+str(len(self.name_models)) + '\n Model Names :')
        print('\t'.join(self.name_models))

    def generate(self, file_name,**kwargs):
        self.name = self.name+'/'+file_name+'/'
        self.model_name = file_name
        self.full_path  = self.full_path+file_name 
        os.makedirs(self.full_path, exist_ok=True)
        # Set the full path for the generator instance's saved files
        # This assumes the name is used as the directory name within "Results"
        if file_name == 'margulis_gabber_galil_graph':
            self.size = int(np.sqrt(self.size))
            self.add_metadata('size', self.size**2.)
            print('New Size (margulis_gabber_galil_graph): '+str(self.size**2.0))
            self.data,self.gpx = generate_adjacency_matrix(file_name,n=self.size,**kwargs)
        elif file_name == 'fast_gnp_random_graph':
            p = kwargs.get('p', .1) # Probability for edge creation.
            self.add_metadata('p', p)
            self.data,self.gpx = generate_adjacency_matrix(file_name,n=self.size,p=p)
        elif file_name == 'newman_watts_strogatz_graph':
            k = kwargs.get('k', 10) # Each node is joined with its k nearest neighbors in a ring topology.
            p = kwargs.get('p', .1) # The probability of adding a new edge for each edge.
            self.add_metadata('p', p)
            self.add_metadata('k', k)
            self.data,self.gpx = generate_adjacency_matrix(file_name,n=self.size,k=k,p=p)
        elif file_name == 'watts_strogatz_graph':
            k = kwargs.get('k', 10) # Each node is joined with its k nearest neighbors in a ring topology.
            p = kwargs.get('p', .1) # The probability of adding a new edge for each edge.
            self.add_metadata('p', p)
            self.add_metadata('k', k)
            self.data,self.gpx = generate_adjacency_matrix(file_name,n=self.size,k=k,p=p)
        elif file_name == 'barabasi_albert_graph':
            m = kwargs.get('m', 10) # Number of edges to attach from a new node to existing nodes
            self.add_metadata('m', m)
            self.data,self.gpx = generate_adjacency_matrix(file_name,n=self.size,m=m)
        elif file_name == 'gaussian_random_partition_graph':
            s = kwargs.get('s', 20) # Mean cluster size
            v = kwargs.get('v', 2) # Variance of the Gaussian distribution is s/v
            p_in = kwargs.get('p_in', .4) # Probability of intra cluster connection.
            p_out = kwargs.get('p_out', .1) # Probability of inter cluster connection.
            self.add_metadata('s', s)
            self.add_metadata('v', v)
            self.add_metadata('p_in', p_in)
            self.add_metadata('p_out', p_out)
            self.data,self.gpx = generate_adjacency_matrix(file_name,n=self.size,s=s,v=v,p_in=p_in,p_out=p_out)
        elif file_name == 'connected_caveman_graph':
            l = kwargs.get('l', int(self.size/4)) # number of cliques
            k = kwargs.get('k', 4) # size of cliques (k at least 2 or NetworkXError is raised)
            self.size = l*k
            self.add_metadata('size', self.size)
            self.add_metadata('l', l)
            self.add_metadata('k', k)
            self.gpx = nx.connected_caveman_graph(l, k)
            self.data = nx.to_numpy_array(self.gpx)
        else :
            raise ValueError(f"Model {file_name} not found in {self.name_models}")
        self.data  = self.data/np.max(self.data)
        # Print the image shape and data type
        self.porosity =  np.sum(self.data)/(self.data.shape[0]*self.data.shape[1])
        self.add_metadata('model_name', file_name)
        self.add_metadata('porosity', self.porosity)
        print('Metadata : '+str(self.metadata))
        return self.data

    def make_spatial_realization(self,scaling_ratio=2,linlog=False,plot_me=False,target_porosity=0.1):
        """
        Make a spatial realization of the graph, using the ForceAtlas2 and Spring embedding methods.
        
        Parameters
        ----------
        scaling_ratio : float, optional
            Determines the scaling of attraction and repulsion forces. Defaults to 2.0.
        linlog : bool, optional
            Uses logarithmic attraction instead of linear. Defaults to False.
        plot_me : bool, optional
            Plot the two embeddings. Defaults to False.
        target_porosity : float, optional
            Target porosity of the generated image. Defaults to 0.1.
        Notes
        -----
        The generated image is saved in a directory named after the generator instance,
        within the "Results" directory.

        The generator instance is initialized with the following metadata:
            - 'porosity': porosity
            - 'generator_type': self.name
            - 'generator_reference': 'Structify-Net - https://structify-net.readthedocs.io/en/latest/index.html'
        """
        self.data = np.zeros([self.size,self.size])
        self.data_spring = np.zeros([self.size,self.size])
        self.porosity =  np.sum(self.data)/(self.data.shape[0]*self.data.shape[1])
        iteration_num = 0
        while self.porosity < target_porosity:
            pos_forceatlas2 = nx.forceatlas2_layout(self.gpx,max_iter=1000,scaling_ratio=scaling_ratio,linlog=linlog)
            pos_spring = nx.spring_layout(self.gpx,iterations=1000,method="energy", scale=1)
            joint_array = np.vstack(list(pos_forceatlas2.values()))
            joint_array_forceatlas2 = (nx.rescale_layout(joint_array,scale=self.size/2) + self.size/2-1).astype(int)
            del joint_array
            joint_array = np.vstack(list(pos_spring.values()))
            joint_array_spring = (nx.rescale_layout(joint_array,scale=self.size/2) + self.size/2-1).astype(int)
            del joint_array
            self.data_adjacency = self.data.copy()
            self.data[joint_array_forceatlas2[:,0],joint_array_forceatlas2[:,1]] = 1
            self.data_spring[joint_array_spring[:,0], joint_array_spring[:,1]] = 1
            self.porosity =  np.sum(self.data)/(self.data.shape[0]*self.data.shape[1])
            print(f'Iteration - porosity target : {self.porosity}, Iteration {iteration_num}')
            iteration_num+=1
        if plot_me:
            plt.figure(figsize=(10,10))
            plt.plot(joint_array_forceatlas2[:,0],joint_array_forceatlas2[:,1],'o',label='ForceAtlas2')
            plt.plot(joint_array_spring[:,0],joint_array_spring[:,1],'o',label='Spring')
            plt.title('Physical Space representation of the generated graph')
            plt.legend()
            plt.show()
        self.add_metadata('porosity', self.porosity)
        print('Done with the graph Spatial representation generation')

