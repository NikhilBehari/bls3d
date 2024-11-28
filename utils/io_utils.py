import numpy as np
import OpenEXR
import Imath

def save_as_exr(image, filename):
    """
    Save a single-channel (HxW) NumPy array as an EXR image file using the OpenEXR library.

    Parameters:
    - image (numpy.ndarray): A NumPy array of shape (height, width) with float32 data type.
                             The array should contain single-channel (grayscale) image data.
    - filename (str): The name of the file to save the image as.

    Returns:
    - None
    """
    if image.dtype != np.float32:
        raise ValueError("Image data type must be float32.")
    
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel with shape (height, width).")

    height, width = image.shape

    # Create an OpenEXR file header with the correct dimensions
    header = OpenEXR.Header(width, height)

    # Create an OpenEXR file
    exr_file = OpenEXR.OutputFile(filename, header)

    # Convert the image to the required format
    channel_data = image.tobytes()

    # Write the single channel to the EXR file
    exr_file.writePixels({'Y': channel_data})  # 'Y' for luminance (grayscale)

    # Close the file
    exr_file.close()

def load_exr_as_numpy(filename):
    """
    Load a single-channel (HxW) EXR image file as a NumPy array using the OpenEXR library.

    Parameters:
    - filename (str): The name of the EXR file to load.

    Returns:
    - numpy.ndarray: A NumPy array of shape (height, width) with float32 data type containing the grayscale image data.
    """
    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get the image dimensions
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the pixel type
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the single channel ('Y')
    channel_data = np.frombuffer(exr_file.channel('Y', pixel_type), dtype=np.float32).reshape(height, width)

    return channel_data

# Example usage:
if __name__ == "__main__":
    # Create a sample single-channel image
    sample_image = np.random.rand(256, 256).astype(np.float32)

    # Save the image as an EXR file
    save_as_exr(sample_image, 'output.exr')

    # Load the EXR image as a NumPy array
    loaded_image = load_exr_as_numpy('output.exr')

    print(loaded_image.shape)  # Should print (256, 256)
    print(loaded_image.dtype)  # Should print float32

