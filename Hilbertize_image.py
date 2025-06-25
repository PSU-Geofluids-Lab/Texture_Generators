import numpy as np
from hilbert import decode, encode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_hilbert_curve(im1, num_bits=6,num_dims=2,plot_all=True,plot_reconstructed=True,plot_hilbert=True,plot_hilbert_reconstructed=False,save_to_file=False,filepath=None):
    """
    Plot the Hilbert curve with Hilbert time series and/or reconstructed image.
    Parameters
    ----------
    im1 : 2D array
        The image to be transformed.
    num_bits : int, optional
        The number of bits to use for the Hilbert curve. The default is 6.
    num_dims : int, optional
        The number of dimensions to use for the Hilbert curve. The default is 2.
    plot_all : bool, optional
        Whether to plot all the plots. The default is True.
    plot_reconstructed : bool, optional
        Whether to plot the reconstructed image. The default is True.
    plot_hilbert : bool, optional
        Whether to plot the Hilbert curve. The default is True.
    plot_hilbert_reconstructed : bool, optional
        Whether to plot the Hilbert curve with the reconstructed image. The default is False.
    save_to_file : bool, optional
        Whether to save the plots to a file. The default is False.
    filepath : str, optional
        The path to save the plots to. The default is None.
    Returns
    -------
    H : array
        The Hilbert time series.
    intensities : array
        The intensities of the image.
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
        plt.plot(H,intensities,linewidth=0.1)
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