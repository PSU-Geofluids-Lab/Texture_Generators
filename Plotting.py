import matplotlib.pyplot as plt

class ImagePlotter:
    """Handles 2D image visualization"""
    @staticmethod
    def plot(data, title='Generated Texture', cmap='viridis', save_path=None):
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap=cmap,interpolation=None,origin='lower')
        plt.title(title)
        plt.colorbar()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_Nofrills(data, cmap='viridis', save_path=None):
        plt.figure(figsize=(8, 8))
        plt.imshow(data, cmap=cmap,interpolation=None,origin='lower')
        plt.axis('off') # Turn off axes
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

