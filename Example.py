from Generators import PorePyGenerator
from Plotting import ImagePlotter

# Usage Example
if __name__ == "__main__":
    # Generate texture
    base_name = "texture"
    generator = PorePyGenerator(size=(128, 128),name=base_name)
    texture_data = generator.generate()

    # Plot the result
    ImagePlotter.plot(texture_data, title='PorePy Texture')

    # Save in multiple formats

    generator.to_csv()
    generator.to_png()
    generator.save_all()