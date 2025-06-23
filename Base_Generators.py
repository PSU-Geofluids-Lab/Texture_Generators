import os
import numpy as np
import h5py
import csv
import vtk
from abc import ABC, abstractmethod
from stl import mesh
from Plotting import ImagePlotter
import metrics_images as mtr

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

    def to_csv(self):
        filename = f"{self.full_path}/Generated_Data.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# ' + f"{k}: {v}" for k, v in self.metadata.items()])
            np.savetxt(f, self.data, delimiter=',')

    def to_png(self):
        filename = f"{self.full_path}/Generated_Data.png"
        ImagePlotter.plot(self.data, save_path=filename)
        filename = f"{self.full_path}/Generated_Data_NoFrills.png"
        ImagePlotter.plot_Nofrills(self.data, save_path=filename)

    def to_hdf5(self):
        filename = f"{self.full_path}/Generated_Data.h5"
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset('texture', data=self.data)
            for k, v in self.metadata.items():
                dset.attrs[k] = v

    def to_vtk(self):
        filename = f"{self.full_path}/Generated_Data.vti"
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

    def to_stl(self,scale=10):
        filename = f"{self.full_path}/Generated_Data.stl"
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

    def save_all(self):
      
      self.to_csv()
      self.to_png()
      self.to_hdf5()
      self.to_vtk()  # VTK Image Data format
      self.to_stl()
      print('All Files saved')

    def make_save_metrics(self,angles_all = np.arange(0,180,5),bins=25,plot_fig=False):
        """
        Compute and save various texture metrics for the generated image.

        Parameters
        ----------
        angles_all : array_like, optional
            An array of angles in degrees for rotating the image to compute angular metrics. Default is np.arange(0, 180, 5).
        bins : int, optional
            Number of bins to use for histograms and distributions. Default is 25.
        plot_fig : bool, optional
            If True, plots will be generated for the computed metrics. Default is False.

        Notes
        -----
        This function computes the following metrics for the texture data:
        - Fractal dimension
        - Distance transform
        - Chord length distribution
        - Lineal path distribution
        - Angular chord length distribution
        - Two-point correlation function
        - Radial distribution

        The results, including plots if requested, are saved to the instance's results directory.
        """
        ## First binarize the image, if not done so before
        if np.unique(self.data).shape[0] > 2:
            self.binary_data = self.data.copy()
            threshold = np.percentile(self.binary_data, 50)
            self.binary_data[self.binary_data > threshold] = 1
            self.binary_data[self.binary_data <= threshold] = 0
            self.fractal_data = mtr.make_plot_fractal(self.binary_data,filepath=self.full_path)
            self.dt = mtr.make_dist_transform(self.binary_data,filepath=self.full_path)
            self.chrd_x,self.sz_x,self.data_x = mtr.make_chords(self.binary_data,filepath=self.full_path)
            self.paths,self.lpf = mtr.make_lineal_path_distribution(self.binary_data,filepath=self.full_path)
            self.data_x_L,self.angles_all,self.all_pdfs = mtr.make_chord_angle_distr(self.binary_data,angles_all,filepath=self.full_path,bins=bins,plot_fig=plot_fig)
        else :
            self.fractal_data = mtr.make_plot_fractal(self.data,filepath=self.full_path)
            self.dt = mtr.make_dist_transform(self.data,filepath=self.full_path)
            self.chrd_x,self.sz_x,self.data_x = mtr.make_chords(self.data,filepath=self.full_path)
            self.paths,self.lpf = mtr.make_lineal_path_distribution(self.data,filepath=self.full_path)
            self.data_x_L,self.angles_all,self.all_pdfs = mtr.make_chord_angle_distr(self.data,angles_all,filepath=self.full_path,bins=bins,plot_fig=plot_fig)

        self.two_pt_corr = mtr.two_pt_corr(self.data,filepath=self.full_path)
        self.radial_dist = mtr.make_radial_dist(self.dt,filepath=self.full_path)
        print('Done the metrics : fractal, distance transform (radial dist), 2pt correlation, lineal path distribution, chord length distribution, and angular chord distribution')