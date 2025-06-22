import numpy as np
import matplotlib.pyplot as plt
import pywt
import matplotlib.colors as colors

def simple_cwt_spectra_plotting(intensities,lower_freq=0.0001,vmax=.1,vmin=0.01):
    Fs = 1 # Assuming a sampling frequency of 1 for a simple time series
    min_scale = pywt.frequency2scale('morl', lower_freq/Fs)
    max_scale = pywt.frequency2scale('morl', 1/Fs) # Example widths, adjust as needed
    # Logarithmic scale spacing
    num_scales = int(32 * np.log2(min_scale/max_scale))
    widths = np.logspace(np.log2(min_scale), np.log2(max_scale),
                                    num_scales, base=2)

    t = np.arange(0,intensities.shape[0],Fs)  # 10 seconds of data
    [cwtmatr, freqs] = pywt.cwt(intensities, widths, 'morl', sampling_period=1/Fs)

    plot_data = np.abs(cwtmatr)**2.
    plot_data = plot_data/plot_data.max()

    # Plot the spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t,freqs,plot_data, cmap='plasma',vmax=vmax,vmin=vmin)
    ax = plt.gca()
    # Configure axis
    ax.set_ylim(freqs.min(),freqs.max())
    ax.set_yscale('log')
    plt.colorbar(label='Magnitude')
    plt.ylabel('Frequency (Hz)') # The units here depend on the sampling rate of your intensities
    plt.xlabel('Sample Index')
    plt.title('CWT Spectrogram of Normallized Intensities')
    plt.show()


class WaveletSpectrogram:
    """
    Generate wavelet spectrograms (scalograms) from 1D time series data

    Features:
    - Multiple wavelet options with auto-selection of optimal scales
    - Logarithmic frequency axis (low to high)
    - Customizable plotting parameters
    - Optimized scale calculation
    """

    def __init__(self, t, signal, wavelet='morl', nv=32):
        """
        Initialize with time series data

        Parameters:
        t : array_like
            Time values (uniformly sampled)
        signal : array_like
            1D signal values
        wavelet : str, optional
            Wavelet to use (default: 'morl')
        nv : int, optional
            Number of voices per octave (resolution)
        """
        self.t = np.asarray(t)
        self.signal = np.asarray(signal)
        self.wavelet = wavelet
        self.nv = nv
        self.fs = 1/(t[1] - t[0])  # Sampling frequency
        self.scales = None
        self.coeffs = None
        self.freqs = None
        self.power = None
        self.lower_val = 2./(t.shape[0]*self.fs)# Setting the lowest frequency to be one with half the wavelength of the time series


        # Validate wavelet
        self._validate_wavelet()

    def _validate_wavelet(self):
        """Check if wavelet is available"""
        if self.wavelet not in pywt.wavelist(kind='continuous'):
            raise ValueError(f"Wavelet '{self.wavelet}' not available. "
                             f"Choose from: {pywt.wavelist(kind='continuous')}")

    def compute_spectrogram(self, min_freq=None, max_freq=None):
        """
        Compute wavelet spectrogram

        Parameters:
        min_freq : float, optional
            Minimum frequency to display (Hz)
        max_freq : float, optional
            Maximum frequency to display (Hz)
        """
        # Set frequency range
        nyquist = self.fs / 2
        min_freq = min_freq or self.lower_val * nyquist
        max_freq = max_freq or nyquist

        # Calculate scales
        min_scale = pywt.frequency2scale(self.wavelet, min_freq/nyquist)
        max_scale = pywt.frequency2scale(self.wavelet, max_freq/nyquist)

        # Logarithmic scale spacing
        num_scales = int(self.nv * np.log2(min_scale/max_scale))
        self.scales = np.logspace(np.log2(min_scale), np.log2(max_scale),
                                 num_scales, base=2)

        # Compute CWT
        self.coeffs, self.freqs = pywt.cwt(self.signal, self.scales,
                                          self.wavelet, sampling_period=1/self.fs)

        # Compute power (normalized)
        self.power = np.abs(self.coeffs)**2
        self.power /= np.max(self.power)

    def plot_spectrogram(self, ax=None, cmap='viridis', ylabel='Frequency [Hz]',
                         xlabel='Time [s]', title=None, colorbar=True, log_power=True):
        """
        Plot wavelet spectrogram with optimized frequency axis

        Parameters:
        ax : matplotlib axis, optional
            Axis to plot on (default: create new figure)
        cmap : str, optional
            Colormap (default: 'viridis')
        log_power : bool, optional
            Use logarithmic power scale (dB)
        """
        if self.power is None:
            self.compute_spectrogram()

        if ax is None:
            # Recreate the figure and axes with appropriate grid specification
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 5], width_ratios=[1, 0.05], wspace=0.05)
            ax_ts = fig.add_subplot(gs[0, 0])
            ax_img = fig.add_subplot(gs[1, 0])
            ax_cbar = fig.add_subplot(gs[1, 1])


        # Prepare data for plotting
        if log_power:
            plot_data = 10 * np.log10(self.power + 1e-12)  # dB scale
            vmax = np.max(plot_data)
            vmin = np.max([np.percentile(plot_data, 5),vmax-15])
            label = 'Power [dB]'
        else:
            plot_data = self.power/self.power.max()
            vmin = 0
            vmax = 1
            label = 'Normalized Power'

        ax_ts.plot(self.t, self.signal, 'b', linewidth=0.5)
        ax_ts.set_xlim(self.t[0], self.t[-1])
        ax_ts.set_ylabel('Amplitude')
        ax_ts.set_xlabel('Time [s]')
        ax_ts.set_title('Original Signal')

        # Create spectrogram plot
        mesh = ax_img.pcolormesh(self.t, self.freqs, plot_data,
                            cmap=cmap, shading='gouraud',
                            norm=colors.Normalize(vmin=vmin, vmax=vmax))

        # Configure axis
        ax_img.set_yscale('log')
        ax_img.set_ylim(self.freqs.min(), self.freqs.max())
        ax_img.set_ylabel(ylabel)
        ax_img.set_xlabel(xlabel)
        ax_img.set_title(title or f'Wavelet Spectrogram ({self.wavelet} wavelet)')

        # Add colorbar
        if colorbar:
          # Add colorbar that spans the image axis
          cbar = plt.colorbar(mesh, cax=ax_cbar)
          cbar.set_label(label)

        plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for colorbar
        return fig

    @staticmethod
    def optimal_wavelet(signal, wavelets=None):
        """
        Find wavelet with maximum energy concentration in spectrogram

        Parameters:
        signal : array_like
            Input signal
        wavelets : list, optional
            Wavelets to test (default: common wavelets)

        Returns:
        str : Optimal wavelet name
        """
        wavelets = wavelets or ['morl', 'cmor1.0-0.5', 'gaus', 'mexh', 'cgau']
        max_concentration = -1
        best_wavelet = wavelets[0]

        for wavelet in wavelets:
            try:
                cwtmatr, _ = pywt.cwt(signal, np.arange(.8, 100), wavelet)
                power = np.abs(cwtmatr)**2
                total_energy = np.sum(power)

                # Find 95% energy concentration
                sorted_power = np.sort(power.flatten())[::-1]
                cum_energy = np.cumsum(sorted_power)
                idx = np.where(cum_energy >= 0.95 * total_energy)[0][0]
                concentration = idx / len(cum_energy)

                if concentration < max_concentration or max_concentration == -1:
                    max_concentration = concentration
                    best_wavelet = wavelet
            except:
                continue

        return best_wavelet
    

def basic_usage(intensities,t,nv=16,save_to_file=False,filepath=None):
    # Example Usage
    # Create spectrogram with optimal wavelet
    optimal_wavelet = WaveletSpectrogram.optimal_wavelet(intensities)
    print(f"Optimal wavelet: {optimal_wavelet}")

    # Create and plot spectrogram
    ws = WaveletSpectrogram(t, intensities, wavelet=optimal_wavelet,nv=nv)
    fig = ws.plot_spectrogram(cmap='magma', log_power=True)
    print('Done!')
    if save_to_file:
        fig.savefig(filepath+'/Hilbert_Time_Series_Spectrogram.png', dpi=300, bbox_inches='tight')
        fig.close()
    else:
        fig.show()


