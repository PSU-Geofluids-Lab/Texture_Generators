import os
import numpy as np
import h5py
import csv
import vtk
from abc import ABC, abstractmethod
from stl import mesh
from Plotting import ImagePlotter
import metrics_images as mtr
import matplotlib.pyplot as plt
import dill

class BaseGenerator(ABC):
    """Abstract base class for 2D image generators"""
    def __init__(self, size=(256, 256)):
        self.size = size
        self.data = None
        self.name = self.__class__.__name__
        self.metadata = {
            'generator_type': self.__class__.__name__,
            'size': size
        }
        results_folder = os.path.join("Results", self.name)
        os.makedirs(results_folder, exist_ok=True)
        self.full_path = results_folder

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generate image data (must be implemented by subclasses)"""
        pass

    def add_metadata(self, key, value):
        """Add custom metadata"""
        self.metadata[key] = value

    def binarize_data(self,percentile_val=50,invert=False):
        percentile = np.percentile(self.data, percentile_val)
        if invert:
            self.data = np.where(self.data > percentile, 1, 0)
        else : 
            self.data = np.where(self.data > percentile, 0, 1)

    def to_csv(self,extra_tag=''):
        filename = f"{self.full_path}/Generated_Data{extra_tag}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# ' + f"{k}: {v}" for k, v in self.metadata.items()])
            np.savetxt(f, self.data, delimiter=',')

    def to_png(self,extra_tag=''):
        filename = f"{self.full_path}/Generated_Data{extra_tag}.png"
        ImagePlotter.plot(self.data, save_path=filename)
        filename = f"{self.full_path}/Generated_Data{extra_tag}_NoFrills.png"
        ImagePlotter.plot_Nofrills(self.data, save_path=filename)

    def to_hdf5(self,extra_tag=''):
        filename = f"{self.full_path}/Generated_Data{extra_tag}.h5"
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset('texture', data=self.data)
            for k, v in self.metadata.items():
                dset.attrs[k] = v

    def binarize_data_basic(self,data,percentile_val=50,invert=False):
        percentile = np.percentile(data, percentile_val)
        if invert:
            data = np.where(data > percentile, 1, 0)
        else : 
            data = np.where(data > percentile, 0, 1)
        return data
    
    def colorize_data(self,data,distribution='random', num_colors=10,**kwargs):
        data_color = np.zeros_like(data)
        # Sample values from the specified distribution
        if distribution == 'random':
            colors = np.random.randint(1, num_colors, size=np.sum(data == 1))
        elif distribution == 'normal':
            mean_val = kwargs.get('mean_val', num_colors/2)
            sigma_val = kwargs.get('sigma_val', num_colors/4)
            colors = np.random.normal(mean_val, sigma_val, size=np.sum(data == 1))
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'poisson':
            lam = kwargs.get('lam', num_colors/2)
            colors = np.random.poisson(lam=lam,  size=np.sum(data == 1))
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'lognormal':
            mean_val = kwargs.get('mean_val', num_colors/2)
            sigma_val = kwargs.get('sigma_val', num_colors/16)
            colors = np.random.lognormal(mean_val, sigma_val,  size=np.sum(data == 1))
            colors = num_colors*colors/colors.max()/.5
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'powerlaw':
            slope = kwargs.get('slope', 2)
            colors = np.random.power(slope,  size=np.sum(data == 1))*(num_colors-1)+1
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'exponential':
            scale = kwargs.get('scale', num_colors/4)
            colors = np.random.exponential(scale=scale,  size=np.sum(data == 1))+1
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'binomial':
            colors = np.random.binomial(n=num_colors, p=0.5, size=np.sum(data == 1))
            colors = np.clip(colors, 1, num_colors)
        else:
            raise ValueError("Invalid distribution. Must be 'random', 'normal', 'lognormal', 'powerlaw', 'exponential', 'poisson', or 'binomial'.")
        # # Assign colors to pixels
        data_color[data == 1] = colors
        return data_color

    def colorize(self,distribution='random', num_colors=10,**kwargs):
        self.data_color = np.zeros_like(self.data)
        # Sample values from the specified distribution
        if distribution == 'random':
            colors = np.random.randint(1, num_colors, size=np.sum(self.data == 1))
        elif distribution == 'normal':
            mean_val = kwargs.get('mean_val', num_colors/2)
            sigma_val = kwargs.get('sigma_val', num_colors/4)
            colors = np.random.normal(mean_val, sigma_val, size=np.sum(self.data == 1))
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'poisson':
            lam = kwargs.get('lam', num_colors/2)
            colors = np.random.poisson(lam=lam,  size=np.sum(self.data == 1))
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'lognormal':
            mean_val = kwargs.get('mean_val', num_colors/2)
            sigma_val = kwargs.get('sigma_val', num_colors/16)
            colors = np.random.lognormal(mean_val, sigma_val,  size=np.sum(self.data == 1))
            colors = num_colors*colors/colors.max()/.5
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'powerlaw':
            slope = kwargs.get('slope', 2)
            colors = np.random.power(slope,  size=np.sum(self.data == 1))*(num_colors-1)+1
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'exponential':
            scale = kwargs.get('scale', num_colors/4)
            colors = np.random.exponential(scale=scale,  size=np.sum(self.data == 1))+1
            colors = np.clip(colors, 1, num_colors)
        elif distribution == 'binomial':
            colors = np.random.binomial(n=num_colors, p=0.5, size=np.sum(self.data == 1))
            colors = np.clip(colors, 1, num_colors)
        else:
            raise ValueError("Invalid distribution. Must be 'random', 'normal', 'lognormal', 'powerlaw', 'exponential', 'poisson', or 'binomial'.")

        # # Assign colors to pixels
        self.data_color[self.data == 1] = colors
        self.distribution_color = distribution

        ## Convert to uint8
        self.data_color = self.data_color.astype(np.uint8)
        self.add_metadata('distribution_color',self.distribution_color)
        self.add_metadata('number_of_colors',num_colors)

    def to_vtk(selfextra_tag=''):
        filename = f"{self.full_path}/Generated_Data{extra_tag}.vti"
        # Create structured grid
        grid = vtk.vtkImageData()
        grid.SetDimensions(self.data.shape[1], self.data.shape[0], 1)

        # Add scalar data
        arr = vtk.vtkDoubleArray()
        arr.SetName('intensity')
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfValues(self.data.size)

        flat_data = self.data.flatten(order='F')
        for val in flat_data:
            arr.InsertNextValue(val)

        grid.GetPointData().SetScalars(arr)

        # Write to file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()

    def to_stl(self,scale=10,extra_tag=''):
        filename = f"{self.full_path}/Generated_Data{extra_tag}.stl"
        # Create surface mesh from 2D data
        height, width = self.data.shape
        vertices = []
        faces = []

        # Generate vertices
        for y in range(height):
            for x in range(width):
                z = self.data[y, x] * scale
                vertices.append([x, y, z])

        # Generate faces
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x
                faces.append([idx, idx+1, idx+width])
                faces.append([idx+1, idx+width+1, idx+width])

        # Create mesh
        texture_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                texture_mesh.vectors[i][j] = vertices[f[j]]

        texture_mesh.save(filename)

    def save_all(self,extra_tag=''):
      
      self.to_csv(extra_tag=extra_tag)
      self.to_png(extra_tag=extra_tag)
      self.to_hdf5(extra_tag=extra_tag)
      self.to_vtk(extra_tag=extra_tag)  # VTK Image Data format
      self.to_stl(extra_tag=extra_tag)
      print('All Files saved')

    def fit_region_prop(self,connectivity=2):
        if np.unique(self.data).size != 2:
            raise ValueError("Data must be binary!! - use binarize_data function or something else first")

        import skimage.measure as measure
        import matplotlib.patches as patches
        # Assume texture_data is a 2D numpy array
        # Label the connected regions in the texture data
        self.labeled_image = measure.label(self.data,connectivity=connectivity)
        # Find the region properties
        self.regions = measure.regionprops(self.labeled_image)

        # # Print the properties of each region
        # for region in regions:
        #     print(f"Region {region.label}:")
        #     print(f"  Area: {region.area}")
        #     print(f"  Perimeter: {region.perimeter}")
        #     print(f"  Eccentricity: {region.eccentricity}")
        #     print(f"  Solidity: {region.solidity}")
        #     print(f"  Orientation: {region.orientation}")
        #     print(f"  Bounding box: {region.bbox}")

        # Plot the texture data
        fig, ax = plt.subplots(figsize=(20,20))
        ax.imshow(self.data, cmap='gray')

        for i, region in enumerate(self.regions):
            minr, minc, maxr, maxc = region.bbox
            rect = patches.Rectangle((minc, minr), maxc-minc, maxr-minr, edgecolor='red', linewidth=2, fill=False)
            ax.add_patch(rect)
            ax.text(5+minc + (maxc - minc) / 2, minr + (maxr - minr) / 2, str(i), ha='center', va='center', color='red')

        plt.title(f'Texture Data with Region Properties; Total regions : {len(self.regions)}')
        plt.show()

        fig, ax = plt.subplots(figsize=(20,20))
        im1 = ax.imshow(self.labeled_image, cmap='RdYlBu_r',interpolation=None,origin='lower')
        plt.colorbar(im1)
        plt.title('Texture Data with Region labels of connected components')
        plt.show()

    def color_by_regions(self,num_colors = 10,connectivity=2):
        if np.unique(self.data).size != 2:
            raise ValueError("Data must be binary!! - use binarize_data function or something else first")
        import skimage.measure as measure
        self.labeled_image = measure.label(self.data,connectivity=connectivity)
        num_regions = np.max(self.labeled_image)
        # Create a color map with a different color for each region
        colors = np.random.randint(1, num_colors+1, size=num_regions)
        self.colored_image = np.zeros((self.labeled_image.shape[0], self.labeled_image.shape[1]))
        for i in range(1,num_regions):
            self.colored_image[self.labeled_image == i] = colors[i]


        plt.figure(figsize=(20,20))
        plt.imshow(self.colored_image, cmap='RdYlBu_r',interpolation=None,origin='lower')
        plt.colorbar()
        plt.title('Texture Data with Region labels of connected components')
        plt.show()
        self.original_data = self.data.copy()
        self.data  = self.colored_image
        self.save_all(extra_tag='colored')


    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return dill.load(f)

    def make_save_metrics(self,angles_all = np.arange(0,180,10),bins_chrd=2,plot_fig_chrd=False,r_max=1,sigma=0.5,max_phase=3):
        """
        Calculate and save all the metrics for the generated image.

        Parameters
        ----------
        angles_all : array_like
            An array of angles (in degrees) to rotate the image for the angular chord length distribution.
        bins_chrd : int
            Number of bins to use for the chord length distribution histogram.
        plot_fig_chrd : bool
            If True, plots the chord length distribution for each rotation angle.
        r_max : float
            Maximum radius to use for the radial distribution.
        sigma : float
            Standard deviation of the Gaussian filter to use for the radial distribution.
        max_phase : int
            Maximum phase to use for the radial distribution.

        Notes
        -----
        The function calculates the following metrics and saves them to files:
            - Fractal dimension
            - Distance transform (radial distribution)
            - 2-point correlation
            - Lineal path distribution
            - Chord length distribution
            - Segment and region properties
            - Angular chord distribution
        """
        if np.unique(self.data).shape[0] > 2:
            self.binary_data = self.data.copy()
            threshold = np.percentile(self.binary_data, 50)
            self.binary_data[self.binary_data > threshold] = 1
            self.binary_data[self.binary_data <= threshold] = 0
            
            self.fractal_data = mtr.make_plot_fractal(self.binary_data,filepath=self.full_path)
            self.dt = mtr.make_dist_transform(self.binary_data,filepath=self.full_path)
            self.chrd_x,self.sz_x,self.data_x = mtr.make_chords(self.binary_data,filepath=self.full_path)
            self.paths,self.lpf = mtr.make_lineal_path_distribution(self.binary_data,filepath=self.full_path)
            self.data_x_L,self.angles_all,self.all_pdfs = mtr.make_chord_angle_distr(self.binary_data,angles_all,bin_spacing=bins_chrd,
                                                                                     filepath=self.full_path,plot_fig=plot_fig_chrd)
        else :
            self.fractal_data = mtr.make_plot_fractal(self.data,filepath=self.full_path)
            self.dt = mtr.make_dist_transform(self.data,filepath=self.full_path)
            self.chrd_x,self.sz_x,self.data_x = mtr.make_chords(self.data,filepath=self.full_path)
            self.paths,self.lpf = mtr.make_lineal_path_distribution(self.data,filepath=self.full_path)
            self.data_x_L,self.angles_all,self.all_pdfs = mtr.make_chord_angle_distr(self.data,angles_all,filepath=self.full_path,
                                                                                    bin_spacing=bins_chrd,plot_fig=plot_fig_chrd)

        self.two_pt_corr = mtr.two_pt_corr(self.data,filepath=self.full_path)
        self.radial_dist = mtr.make_radial_dist(self.dt,filepath=self.full_path)
        self.snow_segment,self.data_segmented_im_use = mtr.get_regions_segment(self.data,filepath=self.full_path,sigma=sigma,r_max=r_max,max_phase=max_phase)
        self.df_prop_summary = mtr.make_partition_regionprop(self.data_segmented_im_use,self.snow_segment.regions,
                                                        plot_specific_region=False,region_id=10,
                                                        summary_images=True,filepath=self.full_path)
        self.snow_network,self.data_im_segmented = mtr.get_regions_segment_network(self.data,sigma=sigma,r_max=r_max,filepath=self.full_path,max_phase=max_phase)

        print('Done the metrics : fractal, distance transform (radial dist), ' \
                    '2pt correlation, lineal path distribution, chord length distribution, segment& region prop, and angular chord distribution')