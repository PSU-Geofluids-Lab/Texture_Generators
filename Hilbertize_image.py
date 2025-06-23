import numpy as np
from hilbert import decode, encode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_hilbert_curve(im1, num_bits=6,num_dims=2,plot_all=True,plot_reconstructed=True,plot_hilbert=True,plot_hilbert_reconstructed=False,save_to_file=False,filepath=None):
    """
    Function to generate a hilbert curve of an image, and plot various related figures.

    Parameters
    ----------
    im1 : 2D numpy array
        The image to be processed
    num_bits : int, optional
        The number of bits to use for the hilbert curve. Default is 6
    num_dims : int, optional
        The number of dimensions to use for the hilbert curve. Default is 2
    plot_all : bool, optional
        Whether to plot all of the figures. Default is True
    plot_reconstructed : bool, optional
        Whether to plot the reconstructed image. Default is True
    plot_hilbert : bool, optional
        Whether to plot the hilbert curve. Default is True
    plot_hilbert_reconstructed : bool, optional
        Whether to plot the reconstructed hilbert curve. Default is False
    save_to_file : bool, optional
        Whether to save the figures to file. Default is False
    filepath : str, optional
        The filepath to save the figures to. Default is None

    Returns
    -------
    H : 1D numpy array
        The hilbert indices of the image
    intensities : 1D numpy array
        The intensities of the image

    Notes
    -----
    This function will generate a hilbert curve of the image, and plot various related figures.
    The figures that are plotted will depend on the options chosen.
    """
    max_h = 2**(num_dims*num_bits)
    hilberts = np.arange(max_h)
    locs = decode(hilberts, num_dims, num_bits)
    intensities = im1[locs[:,0], locs[:,1]]
    image_128_reconstructed = np.zeros(im1.shape)
    image_128_reconstructed[locs[:, 0], locs[:, 1]] = intensities
    H = encode(locs, num_dims, num_bits)
    if np.sum(image_128_reconstructed-im1) > 1e-3 :
        raise ValueError(f"Reconstructed image does not match original image : {np.sum(image_128_reconstructed-im1)}")

    cmap = plt.get_cmap('RdYlBu_r')
    if plot_all:
        print('Plotting Hilbert Curve with Hilbert Time Series')
        plt.figure(figsize=(12,4))
        plt.plot(H,intensities)
        plt.xlabel('Hilbert Index')
        plt.ylabel('Image Intensity')
        plt.title('Intensity vs Hilbert Index for Hilbert Curve \n (Hilbert Time series)')
        if save_to_file:
            plt.savefig(filepath+'/Hilbert_Time_Series.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    if plot_reconstructed:
        plt.figure(figsize=(12,8))
        plt.scatter(locs[:,1],locs[:,0],c=intensities,s=10,cmap=cmap,vmin=intensities.min(), vmax=intensities.max())
        plt.colorbar()
        plt.title('Hilbert Reconstruction with Hilbert Curve')
        if save_to_file:
            plt.savefig(filepath+'/Hilbert_Curve_Reconstruction.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    if plot_hilbert_reconstructed:
        plt.figure(figsize=(12,8))
        print('Plotting Plotting Hilbert Curve with Hilbert Curve')
        norm = mcolors.Normalize(vmin=intensities.min(), vmax=intensities.max()) # Example: normalizing data from -1 to 1
        for i in range(0,locs.shape[0]-1):
            x1, x2 = locs[i,0], locs[i+1,0]
            y1, y2 = locs[i,1], locs[i+1,1]
            color_val = norm(intensities[i])
            color = cmap(color_val)
            # segment_color = cmap(intensities[i]*10)
            plt.plot([y1, y2],[x1, x2], color=color)  
            plt.title('Hilbert Reconstruction with Hilbert Curve')
        if save_to_file:
            plt.savefig(filepath+'/Hilbert_Curve_Reconstruction_With Shape.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    if plot_hilbert:
        plt.figure(figsize=(12,8))
        # # Create a Normalize object
        norm = mcolors.Normalize(vmin=H.min(), vmax=H.max()) # Example: normalizing data from -1 to 1
        for i in range(0,locs.shape[0]-1):
            x1, x2 = locs[i,0], locs[i+1,0]
            y1, y2 = locs[i,1], locs[i+1,1]
            color_val = norm(H[i])
            color = cmap(color_val)
            plt.plot([x1, x2],[y1, y2], color=color)
            plt.title('Hilbert Curve image')
        if save_to_file:
            plt.savefig(filepath+'/Hilbert_Curve_Shape_Colored.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return H, intensities