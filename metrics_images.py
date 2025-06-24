import os
import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
import openpnm as op
from tqdm import tqdm
import numpy as np
from scipy.interpolate import griddata
import Code_Neper_Conversion as cn
from skimage.morphology import binary_dilation

def rotate_image(image, angle,plot_me=False):
    # Get the image dimensions
    """
    Rotate an image by a given angle.

    Parameters
    ----------
    image : numpy.ndarray
        2D image to be rotated
    angle : float
        Angle of rotation in degrees

    Returns
    -------
    rotated_image : numpy.ndarray
        Rotated image

    Notes
    -----
    This function uses the scipy.interpolate.griddata function to rotate the image.
    The rotation is done by first creating a grid of points in the original image,
    then applying the rotation matrix to the points, and finally interpolating the
    values from the original image to the rotated image.
    """
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    (cX, cY) = (w // 2, h // 2)
    
    # Calculate the rotation matrix
    angle_rad = np.deg2rad(angle)
    M = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), (1 - np.cos(angle_rad)) * cX + np.sin(angle_rad) * cY],
        [np.sin(angle_rad), np.cos(angle_rad), -np.sin(angle_rad) * cX + (1 - np.cos(angle_rad)) * cY],
        [0, 0, 1]
    ])
    
    # Calculate the new dimensions of the rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # Update the rotation matrix with the new dimensions
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    # Create a grid of points in the original image
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Apply the rotation matrix to the points
    rotated_points = np.dot(points, M[:2, :2].T) + M[:2, 2]
    
    # Create a grid of points in the rotated image
    xi = np.linspace(0, nW - 1, nW)
    yi = np.linspace(0, nH - 1, nH)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate the values from the original image to the rotated image
    rotated_image = griddata(rotated_points, image.ravel(), (xi, yi), method='cubic',fill_value=-1)
    rotated_image[rotated_image>image.max()] = image.max()
    if plot_me:
        plt.figure(figsize=(10,10))
        plt.imshow(rotated_image,interpolation='None',origin='lower')
        plt.colorbar()
        plt.show()
    return rotated_image

def rotate_image_simple(image, angle,plot_me=False):
    # Get the image dimensions
    """
    Rotate an image by a given angle.

    Parameters
    ----------
    image : numpy.ndarray
        2D image to be rotated
    angle : float
        Angle of rotation in degrees

    Returns
    -------
    rotated_image : numpy.ndarray
        Rotated image
    """
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    (cX, cY) = (w // 2, h // 2)
    
    # Calculate the rotation matrix
    angle_rad = np.deg2rad(angle)
    M = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), (1 - np.cos(angle_rad)) * cX + np.sin(angle_rad) * cY],
        [np.sin(angle_rad), np.cos(angle_rad), -np.sin(angle_rad) * cX + (1 - np.cos(angle_rad)) * cY],
        [0, 0, 1]
    ])
    
    # Calculate the new dimensions of the rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # Update the rotation matrix with the new dimensions
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    # Rotate the image and pad with zeros
    rotated_image = np.zeros((nH, nW))-1
    for y in range(h):
        for x in range(w):
            new_x = int(M[0, 0] * x + M[0, 1] * y + M[0, 2])
            new_y = int(M[1, 0] * x + M[1, 1] * y + M[1, 2])
            if 0 <= new_x < nW and 0 <= new_y < nH:
                rotated_image[new_y, new_x] = image[y, x]
    
    if plot_me:
        plt.figure(figsize=(10,10))
        plt.imshow(rotated_image,interpolation='None')
        plt.colorbar()
        plt.show()
    return rotated_image

def make_plot_fractal(im,filepath=None):
    """
    Compute the box counting dimension of a binary image and plot it.

    Parameters
    ----------
    im : ndarray
        2D image
    filepath : str, optional
        Path to save the figure. If None, the figure is not saved and is shown instead.

    Returns
    -------
    data : named tuple
        Contains size and count of the boxes, as well as the slope of the
        log-log plot.

    Notes
    -----
    The box counting dimension is calculated as the negative slope of the
    log-log plot of the number of boxes spanning the image vs the box edge
    length.

    """
    if np.unique(im).shape[0] > 2:
        raise ValueError('The image must be binary')
    data = ps.metrics.boxcount(im)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('box edge length')
    ax1.set_ylabel('number of boxes spanning phases')
    ax2.set_xlabel('box edge length')
    ax2.set_ylabel('slope')
    ax2.set_xscale('log')
    ax1.plot(data.size, data.count,'-o')
    ax2.plot(data.size, data.slope,'-o')
    if filepath is not None :
        plt.savefig(f'{filepath}/Fractal_dimension.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return data

def two_pt_corr(im,filepath=None):
    """
    Plot the two-point correlation function of an image.

    Parameters
    ----------
    im : ndarray
        A 2D image of the porous material.

    Returns
    -------
    data : tuple
        A tuple containing the distance and probability arrays from the two-point
        correlation function calculation.

    Notes
    -----
    The two-point correlation function is calculated using Porespy's
    two_point_correlation function.
    """
    data = ps.metrics.two_point_correlation(im)
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.plot(data.distance, data.probability, 'r.')
    ax.set_xlabel("distance")
    ax.set_ylabel("two point correlation function")
    if filepath is not None :
        plt.savefig(f'{filepath}/2pt_correlation.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return data

def make_dist_transform(im,filepath=None):
    """
    Calculate the Euclidean distance transform of a binary image.

    Parameters
    ----------
    im : ndarray
        A 2D binary image of the porous material.

    Returns
    -------
    dt : ndarray
        The Euclidean distance transform of the input image.

    Notes
    -----
    The distance transform is calculated using the EDT library, which is
    a Python wrapper for the EDT algorithm implemented in C++.
    """
    if np.unique(im).shape[0] > 2:
        raise ValueError('The image must be binary')
    from edt import edt
    dt = edt(im)
    fig, ax = plt.subplots(1, 1, figsize=[12, 12])
    msh = ax.imshow(dt, origin='lower', interpolation='none')
    plt.colorbar(msh)
    if filepath is not None :
        plt.savefig(f'{filepath}/Distance_Transform.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return dt


def make_radial_dist(dt,bins=20,log_true=False,filepath=None):
    """
    Plot the radial density distribution of a distance transform.

    Parameters
    ----------
    dt : ndarray
        A 2D array of the Euclidean distance transform of a binary image.
    bins : int, optional
        The number of bins to use in the histogram. Default is 20.
    log_true : bool, optional
        If True, the x-axis is logarithmic. Default is False.

    Returns
    -------
    data : ndarray
        A 1D array of the radial density distribution.

    Notes
    -----
    The function uses Porespy's radial_density_distribution function to calculate the
    radial density distribution of the distance transform. The distribution is then
    plotted as a histogram with logarithmic x-axis.
    """
    data = ps.metrics.radial_density_distribution(dt=dt,bins=bins,log=log_true)
    fig, ax = plt.subplots(1, 3, figsize=[10, 4])
    ax[0].plot(data.bin_centers,data.pdf)
    ax[1].plot(data.bin_centers,data.cdf)
    ax[2].bar(data.bin_centers, data.cdf, data.bin_widths, edgecolor='k')
    ax[0].set_title("Probability Density Function")
    ax[1].set_title("Cumulative Density Function")
    ax[2].set_title('Bar Plot');
    ax[0].set_xlabel("Distance")
    ax[1].set_xlabel("Distance")
    ax[2].set_xlabel("Distance")
    if filepath is not None :
        plt.savefig(f'{filepath}/Distance_Transform_Radial_Density.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return data

def make_chords(texture_data,spacing=3,axis=1,plot_fig=True,bin_spacing=1,filepath=None,use_adaptive_bins=True,bins=20):
    """
    Apply chords to an image and calculate the chord length distribution.

    Parameters
    ----------
    texture_data : ndarray
        A 2D or 3D array of the image data.
    spacing : int, optional
        The spacing between chords. Default is 3.
    axis : int, optional
        The axis to apply the chords along. Default is 1.
    plot_fig : bool, optional
        If True, plots the chord image and chord length distribution. Default is True.
    bins : int, optional
        The number of bins to use in the histogram. Default is 100.

    Returns
    -------
    chrd_x : ndarray
        The image with chords applied.
    sz_x : ndarray
        The size of each connected region in the chord image.
    data_x : named tuple
        The chord length distribution.

    Notes
    -----
    The function uses Porespy's apply_chords and region_size functions to apply chords
    to the image and calculate the size of each connected region. The chord length
    distribution is then calculated using Porespy's chord_length_distribution function.
    If plot_fig is True, the function plots the chord image and chord length distribution.
    """
    if np.unique(texture_data).shape[0] > 2:
        raise ValueError('The image must be binary')
    chrd_x = ps.filters.apply_chords(im=texture_data, spacing=spacing,axis=axis,label=True)
    # region_size counts the number of voxels in each connected region of an image, and replaces those voxels with the numerical value of the region size. 
    sz_x = ps.filters.region_size(chrd_x)
    if use_adaptive_bins:
        bins = np.arange(1, sz_x.max().max(), bin_spacing)
        data_x = ps.metrics.chord_length_distribution(chrd_x, bins=bins)
    else :
        data_x = ps.metrics.chord_length_distribution(chrd_x, bins=bins)

    if plot_fig:
        fig, ax = plt.subplots(1, 4, figsize=[18, 6],width_ratios=(1,1,1,.05))
        ax[0].imshow(texture_data, interpolation='none', origin='lower')
        ax[0].axis(False)
        ax[0].set_title(f'Actual Image')
        ax[1].imshow(chrd_x, interpolation='none', origin='lower')
        ax[1].axis(False)
        ax[1].set_title(f'Chords Image, spacing = {spacing}')
        mesh = ax[2].imshow(sz_x, interpolation='none', origin='lower')
        ax[2].axis(False)
        ax[2].set_title(f'spacing = {spacing}')
        plt.colorbar(mesh,cax=ax[3],label='Region Size')
        if filepath is not None :
            plt.savefig(f'{filepath}/Chrord_Image_and_Region_Size_axis_{axis}_spacing_{spacing}.png',bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        fig, ax = plt.subplots(1, 3, figsize=[10, 4])
        ax[0].plot(data_x.L,data_x.pdf)
        ax[1].plot(data_x.L,data_x.cdf)
        ax[2].bar(data_x.L, data_x.cdf, data_x.bin_widths, edgecolor='k')
        ax[0].set_title("Probability Density Function")
        ax[1].set_title("Cumulative Density Function")
        ax[2].set_title('Bar Plot')
        if filepath is not None :
            plt.savefig(f'{filepath}/Chrord_Image_Stats_axis_{axis}_spacing_{spacing}.png',bbox_inches='tight')
            plt.close()
        else:
            plt.show()        
    return chrd_x,sz_x,data_x

def make_chord_angle_distr(texture_data,angles_all,filepath=None,plot_fig=False,bin_spacing=1,bin_max_fac = 0.5):
    """
    Calculate and plot the angular probability density function (PDF) of chord size distribution
    from given texture data by rotating the image.

    Parameters
    ----------
    texture_data : ndarray
        A 2D array representing the image data.
    angles_all : array_like
        An array of angles (in degrees) to rotate the image.
    filepath : str, optional
        Path to save the plots. If None, the plots are displayed instead.
    bins : int, optional
        Number of bins to use for the chord length distribution histogram. Default is 20.
    plot_fig : bool, optional
        If True, plots the chord length distribution for each rotation angle. Default is False.

    Returns
    -------
    L : ndarray
        The chord lengths used in the PDF computation.
    angles_all : array_like
        The input angles used for image rotation.
    all_pdfs : ndarray
        The angular probability density functions for each rotation angle.

    Notes
    -----
    The function first computes the chord length distribution of the input image, then
    rotates the image for each angle in `angles_all`, computes the chord length distribution
    of the rotated image, and finally plots the angular PDF of chord size distribution
    as both a heatmap and a polar plot.
    """

    _,sz_x,data_x = make_chords(texture_data,plot_fig=False,use_adaptive_bins=False,bins=30)
    bins = np.arange(1, sz_x.max().max()*bin_max_fac, bin_spacing)
    all_pdfs   = np.zeros([bins.shape[0]-1,angles_all.shape[0]])
    all_pdfs_freq   = np.zeros([bins.shape[0]-1,angles_all.shape[0]])

    for i,angle in tqdm(enumerate(angles_all)) :
        rotated_image = rotate_image(texture_data,angle,plot_me=False)
        rotated_image[rotated_image<0]=0
        _,_,data_x = make_chords(np.round(rotated_image),plot_fig=plot_fig,use_adaptive_bins=False,bins=bins,axis=1,spacing=1)
        all_pdfs[:,i]=data_x.pdf
        all_pdfs_freq[:,i]=data_x.relfreq

    plt.figure(figsize=(12, 12))
    img = plt.pcolormesh(data_x.L,angles_all,all_pdfs.T,cmap='RdYlBu_r')
    plt.xlabel('Chord Length')
    plt.title('Angular PDF of chord size distribution')
    plt.ylabel('Image rotation Angle')
    plt.colorbar(img)
    if filepath is not None :
        plt.savefig(f'{filepath}/Chord_Angle_Distribution_Rotation.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()        

    # Create a polar plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, polar=True)
    # Convert angles to radians
    angles_rad = np.deg2rad(angles_all+(angles_all[1]-angles_all[0])*0.5)

    # Plot the data
    img = ax.pcolormesh(angles_rad, data_x.L, all_pdfs, cmap='RdYlBu_r',shading='nearest')

    # Set the title and labels
    ax.set_title('Angular PDF of chord size distribution', fontsize=15, pad=-60)
    ax.set_xlabel('Chord Length', labelpad=-80, fontsize=15)

    # Set the theta limits to only show the top half of the polar plot
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Add a colorbar
    cbar = fig.colorbar(img, ax=ax,shrink=0.6)
    cbar.set_label('Probability Density')
    plt.tight_layout()

    if filepath is not None :
        plt.savefig(f'{filepath}/Chord_Angle_Distribution_Rotation_Polar.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()        

    return data_x.L,angles_all,all_pdfs


def make_lineal_path_distribution(texture_data,filepath=None,axis=0):
    """
    Apply the lineal path distribution to an image and calculate the lineal path distribution.

    Parameters
    ----------
    texture_data : ndarray
        A 2D or 3D array of the image data.
    filepath : str, optional
        The path to save the figure. If None, the figure is not saved and is shown instead.
    axis : int, optional
        The axis to apply the chords along. Default is 0.

    Returns
    -------
    paths : ndarray
        The image with the lineal path distribution applied.
    lpf : named tuple
        The lineal path distribution.

    Notes
    -----
    The function uses Porespy's distance_transform_lin and lineal_path_distribution functions
    to apply the lineal path distribution to the image and calculate the lineal path
    distribution. The lineal path distribution is then plotted as a histogram with
    logarithmic x-axis.
    """
    if np.unique(texture_data).shape[0] > 2:
        raise ValueError('The image must be binary')
    paths = ps.filters.distance_transform_lin(texture_data, mode='forward', axis=axis)
    plt.figure(figsize=(8, 6))
    img = plt.imshow(paths*texture_data.astype(int),origin='lower',cmap='Reds');
    plt.colorbar(img,label='Maximum Path Length')
    if filepath is not None :
        plt.savefig(f'{filepath}/Lineal_path_distribution_axis_{axis}_Image.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()        


    lpf = ps.metrics.lineal_path_distribution(paths, bins=range(1, int(paths.max()), 1))
    fig, ax = plt.subplots(1, 3, figsize=[10, 4])
    ax[0].plot(lpf.L,lpf.pdf)
    ax[1].plot(lpf.L,lpf.cdf)
    ax[2].bar(lpf.L, lpf.cdf, lpf.bin_widths, edgecolor='k')
    ax[0].set_title("Probability Density Function")
    ax[1].set_title("Cumulative Density Function")
    ax[2].set_title('Bar Plot');
    if filepath is not None :
        plt.savefig(f'{filepath}/Lineal_path_distribution_Stats_axis_{axis}.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()        
    return paths,lpf

def get_regions_segment(im,sigma=0.5,r_max=1,max_phase=5,filepath=None,print_stats=True):
    if np.unique(im).shape[0] > max_phase:
        cnvrtr = cn.ImageConverter('results_Msc',filename='img1')
        scale_fact = 255/np.max(im)
        cnvrtr.run_setup_filter(im*scale_fact,max_components=max_phase)
        im_use = cnvrtr.img_adaptive_gray
        print(f'Converted to {max_phase} phases')
    else :
        im_use = im.copy()
        del im

    snow = ps.filters.snow_partitioning_n(im_use,sigma=sigma,r_max=r_max)
    if print_stats:
        print(snow)
    # snow = ps.filters.snow_partitioning(im=texture_data)
    dt_peak = snow.dt.copy()
    peaks_dilated = binary_dilation(snow.peaks > 0)
    dt_peak[peaks_dilated == 0] = np.nan

    if print_stats:
        fig, ax = plt.subplots(2, 2, figsize=[12, 12])
        img_plA = ax[0,0].imshow(im_use, cmap=plt.cm.copper, origin='lower', interpolation='none');
        plt.colorbar(img_plA,shrink=0.6)
        ax[0,0].set_title('Original Image')
        img_pl = ax[0,1].imshow(snow.regions, cmap=plt.cm.RdYlBu_r, origin='lower', interpolation='none');
        plt.colorbar(img_pl,shrink=0.6)
        ax[0,1].set_title('Segmented Regions')
        img_pl2A =ax[1,0].imshow(snow.dt/im_use, origin='lower', interpolation='none')
        plt.colorbar(img_pl2A,shrink=0.6)
        img_pl2 = ax[1,0].imshow(dt_peak, origin='lower', interpolation='none',cmap='RdYlBu_r',vmin=0,vmax=1);
        ax[1,0].axis(False)
        ax[1,0].set_title('Distance Transform')

        img_pl2 = ax[1,1].imshow(dt_peak, origin='lower', interpolation='none',cmap='RdYlBu_r',vmin=0,vmax=1);
        plt.colorbar(img_pl2,shrink=0.6)
        ax[1,1].axis(False)
        ax[1,1].set_title('Distance Transform Peaks')
        plt.tight_layout()
        if filepath is not None :
            plt.savefig(f'{filepath}/Image_Segmentation_Snow.png',bbox_inches='tight')
            plt.close()
        else:
            plt.show()        
    return snow,im_use


def get_regions_segment_network(im,sigma=0.5,r_max=10,filepath=None,max_phase=5):
    if np.unique(im).shape[0] > max_phase:
        cnvrtr = cn.ImageConverter('results_Msc',filename='img1')
        scale_fact = 255/np.max(im)
        cnvrtr.run_setup_filter(im*scale_fact,max_components=max_phase)
        im_use = cnvrtr.img_adaptive_gray
        print(f'Converted to {max_phase} phases')
    else :
        im_use = im.copy()
        del im
        
    snow = ps.networks.snow2(im_use,sigma=sigma,r_max=r_max)
    print(snow)
    pn = op.io.network_from_porespy(snow.network)
    print(pn)
    fig, ax = plt.subplots(2, 2, figsize=[12, 12])
    img_plA = ax[0,0].imshow(im_use, cmap=plt.cm.copper, origin='lower', interpolation='none');
    plt.colorbar(img_plA,shrink=0.6)
    ax[0,0].set_title('Original Image')
    img_pl = ax[0,1].imshow(snow.regions, cmap=plt.cm.RdYlBu_r, origin='lower', interpolation='none');
    plt.colorbar(img_pl,shrink=0.6)
    ax[0,1].set_title('Segmented Regions')
    img_pl2A =ax[1,0].imshow(snow.phases, origin='lower', interpolation='none')
    plt.colorbar(img_pl2A,shrink=0.6)
    ax[1,0].set_title('Phases')
    ax[1,1].imshow(im.T, cmap=plt.cm.bone);
    op.visualization.plot_coordinates(ax=ax[1,1],
                                      network=pn,
                                      size_by=pn["pore.inscribed_diameter"],
                                      color_by=pn["pore.surface_area"],
                                      markersize=200)
    op.visualization.plot_connections(network=pn, ax=ax[1,1])
    ax[1,1].axis("off");    
    ax[1,1].set_title('Distance Transform Peaks')
    plt.tight_layout()
    if filepath is not None :
        plt.savefig(f'{filepath}/Image_Segmentation_Snow2_Network.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()        
    return snow,im_use


def make_partition_regionprop(im,regions,plot_specific_region=False,region_id=0,summary_images=False,filepath=None,print_Stats=True):
    props = ps.metrics.regionprops_3D(regions)
    if print_Stats:
        r = props[region_id]
        attrs = [a for a in r.__dir__() if not a.startswith('_')]
        print('region attributes : ',attrs)
    if plot_specific_region:
        r = props[region_id]
        fig, ax = plt.subplots()
        imgg = ax.imshow(r.border + 0.5*r.inscribed_sphere);
        plt.title('Region Border and Inscribed Sphere')
        ax.imshow(r.image,alpha=0.5);
        plt.colorbar(imgg,shrink=0.6)
        if filepath is not None :
            plt.savefig(f'{filepath}/Image_Segmentation_Snow_Region_Border_Inscribed_Sphere_Region{region_id}.png',bbox_inches='tight')
            plt.close()
        else:
            plt.show()        
        fig, ax = plt.subplots()
        imgg = ax.imshow(r.image + 1.0*r.convex_image);
        plt.title('Image and Convex Image')
        plt.colorbar(imgg)
        if filepath is not None :
            plt.savefig(f'{filepath}/Image_Segmentation_Snow_image_Convex_Image_region{region_id}.png',bbox_inches='tight')
            plt.close()
        else:
            plt.show()        

    df = ps.metrics.props_to_DataFrame(props)
    if print_Stats:
        print(df.columns)
    if summary_images:
        # Create an image of maximally inscribed spheres
        sph = ps.metrics.prop_to_image(regionprops=props, shape=im.shape, prop='orientation')*180./np.pi
        fig, ax = plt.subplots()
        imgg =ax.imshow(sph/im, cmap=plt.cm.inferno, origin='lower', interpolation='none',);
        plt.colorbar(imgg)
        imgg =ax.imshow(0.5*~im, cmap=plt.cm.inferno,alpha=0.1, origin='lower', interpolation='none',);
        plt.title('Image of orientation')
        if filepath is not None :
            plt.savefig(f'{filepath}/Image_Segmentation_Snow_Orientation.png',bbox_inches='tight')
            plt.close()
        else:
            plt.show()        
        # # Create an image colorized by solidity
        # sph = ps.metrics.prop_to_image(regionprops=props, shape=im.shape, prop='label')
        # fig, ax = plt.subplots()
        # imgg = ax.imshow(sph/im, cmap=plt.cm.jet, origin='lower', interpolation='none',);
        # plt.colorbar(imgg)
        # plt.title('Image colorized by solidity')
        # if filepath is not None :
        #     plt.savefig(f'{filepath}/Image_Segmentation_Snow_Solidity.png',bbox_inches='tight')
        #     plt.close()
        # else:
        #     plt.show()        
    return df


def get_regions_segment_Iterative(texture_data,num_steps = 5,sigma=0.5,r_max=10,mode='erosion'):
    from scipy import stats
    from skimage.morphology import disk
    import pandas as pd

    im = texture_data.copy()
    for i in range(num_steps):
        snow,im_use = get_regions_segment(im,sigma=sigma,r_max=r_max,max_phase=2,filepath=None,print_stats=False)
        if i == 0:
            df_prop_summary_all = make_partition_regionprop(im_use,snow.regions,
                                                            plot_specific_region=False,region_id=10,
                                                            summary_images=True,print_Stats=False)
            df_prop_summary_all['label'] = i
        else :
            df_prop_summary = make_partition_regionprop(im_use,snow.regions,
                                                            plot_specific_region=False,region_id=10,
                                                            summary_images=True,print_Stats=False)
            df_prop_summary['label'] = i
            df_prop_summary_all = pd.concat([df_prop_summary_all,df_prop_summary])
        im = ps.filters.fftmorphology(im, strel=disk(1), mode=mode)
                                            
    plt.figure(figsize=(8, 6))
    plt.hist2d(df_prop_summary_all['label'].values,np.log10(df_prop_summary_all['area'].values), bins=[num_steps,10],cmap='RdYlBu_r')
    plt.xlabel('Evolution Step')
    plt.ylabel('Area Distribution')
    plt.colorbar(label='Count')
    plt.show()
    return df_prop_summary_all