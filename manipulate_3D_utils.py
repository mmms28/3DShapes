import os, vtk 
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from vtk.util import numpy_support
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def read_sitkimage(image_path: str, verbose: bool = False) -> sitk.Image:
    """
    Reads an sitkImage from the given file path.
    """
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
    try:
        image = sitk.ReadImage(image_path)
        if verbose:
            print(f"Image read from: {image_path}")
        return image
    except Exception as e:
        print(f"Error reading image: {e}")

def write_sitkimage(image: sitk.Image, image_path: str, verbose: bool = False) -> None:
    """
    Writes a vtkImageData to a file using an appropriate VTK writer based on the file extension.
    """
    if not isinstance(image, sitk.Image):
        raise TypeError("Input is not a sitk.Image")
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        # Write the image with compression enabled.
        sitk.WriteImage(image, image_path, useCompression=True)
        if verbose:
            print(f"Image saved as: {image_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def convert_image_format(input_file: str, output_file: str, verbose: bool = False) -> None:
    """
    Converts an image from one format to another using SimpleITK.
    """
    try:
        image = read_sitkimage(input_file, verbose)
        write_sitkimage(image, output_file, verbose)
    except Exception as e:
        print(f"Error converting image format: {e}")


def read_vtkimage(image_path: str, verbose: bool = False) -> vtk.vtkImageData:
    """
    Reads a VTK image file from the given file path using an appropriate VTK reader based on the file extension.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    try:
        # Find file extension.
        extension = os.path.splitext(image_path)[1].lower()

        # Mapping file extensions to the corresponding VTK reader classes.
        reader_mapping = {
            '.mhd': vtk.vtkMetaImageReader,
            '.mha': vtk.vtkMetaImageReader,
            '.vti': vtk.vtkXMLImageDataReader,
            '.vtk': vtk.vtkDataSetReader,
            '.nii': vtk.vtkNIFTIImageReader,
            '.nii.gz': vtk.vtkNIFTIImageReader
        }

        reader_class = reader_mapping.get(extension)
        if reader_class is None:
            raise ValueError(f"Unsupported file extension: {extension}")

        reader = reader_class()
        reader.SetFileName(image_path)
        reader.Update()
        image = reader.GetOutput()

        if verbose:
            print(f"Image read from: {image_path}")

        return image

    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def write_vtkimage(image: vtk.vtkImageData, image_path: str, verbose: bool = False) -> None:
    """
    Writes a vtkImageData to a file using an appropriate VTK writer based on the file extension.
    """
    try:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure the output directory exists
        
        # Determine the file extension
        extension = os.path.splitext(image_path)[1].lower()
        
        writer_mapping = {
            '.mhd': vtk.vtkMetaImageWriter,
            '.mha': vtk.vtkMetaImageWriter,
            '.vti': vtk.vtkXMLImageDataWriter,
            '.vtk': vtk.vtkDataSetWriter, 
            '.nii': vtk.vtkNIFTIImageWriter,
            '.nii.gz': vtk.vtkNIFTIImageWriter
        }
        writer_class = writer_mapping.get(extension)
        if writer_class is None:
            raise ValueError(f"Unsupported file extension: {extension}")

        writer = writer_class()
        writer.SetFileName(image_path)
        writer.SetInputData(image)

        # Enable compression if supported by the writer
        if hasattr(writer, 'SetCompression'):
            writer.SetCompression(True)
        
        writer.Write()
        
        if verbose:
            print(f"Image saved as: {image_path}")
    
    except Exception as e:
        print(f"Error saving image: {e}")

def create_vtkimage_from_polydata(dimensions: tuple[int, int, int], spacing: tuple[float, float, float],
                               origin: tuple[float, float, float], vtk_polydata: vtk.vtkPolyData) -> vtk.vtkImageData:
    """
    Converts a closed vtkPolyData into a vtkImageData by using VTK's stencil conversion.
    """
    # Define image extent as (xmin, xmax, ymin, ymax, zmin, zmax)
    extent = (0, dimensions[0] - 1,
              0, dimensions[1] - 1,
              0, dimensions[2] - 1)

    # Convert vtkPolyData to an image stencil.
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(vtk_polydata)
    poly_to_stencil.SetOutputOrigin(origin)
    poly_to_stencil.SetOutputSpacing(spacing)
    poly_to_stencil.SetOutputWholeExtent(extent)
    poly_to_stencil.Update()

    # Convert the image stencil to a vtkImageData.
    stencil_to_image = vtk.vtkImageStencilToImage()
    stencil_to_image.SetInputConnection(poly_to_stencil.GetOutputPort())
    stencil_to_image.SetInsideValue(255)  # Set voxels inside the surface.
    stencil_to_image.SetOutsideValue(0)   # Set voxels outside the surface.
    stencil_to_image.SetOutputScalarType(vtk.VTK_SHORT)
    stencil_to_image.Update()

    return stencil_to_image.GetOutput()

def create_sitkimage_from_array(image_data: np.ndarray, spacing: tuple[float, float, float], origin: tuple[float, float, float]) -> sitk.Image:
    """
    Converts a 3d numpy array to a SimpleITK image.
    """
    sitk_image = sitk.GetImageFromArray(image_data)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    return sitk_image

def create_vtkimage_from_array(image_data: np.ndarray, spacing: tuple[float, float, float],
                               dimensions: tuple[float, float, float], origin: tuple[float, float, float]) -> vtk.vtkImageData:
    """
    Convert a 3D numpy array to a vtkImageData object with short data type.
    """
    # Ensure the data is contiguous and convert to 16-bit integers (short)
    image_data = np.ascontiguousarray(image_data, dtype=np.int16)
    
    # Flatten the data for VTK conversion
    flat_data = image_data.ravel()
    
    # Convert the numpy array to a VTK array with short data type
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_data, deep=True,
                                             array_type=vtk.VTK_SHORT)
    
    # Create vtkImageData and set its properties
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(dimensions)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    
    # Assign the converted data as the scalars of the image
    vtk_image.GetPointData().SetScalars(vtk_array)
    
    return vtk_image

def get_image_array(image):
    """
    Get a numpy array from the image.
    """
    if isinstance(image, sitk.Image):
        image_array = sitk.GetArrayFromImage(image)
    elif isinstance(image, vtk.vtkImageData):
        # Get the image dimensions: (width, height, depth)
        dims = image.GetDimensions()
        vtk_array = image.GetPointData().GetScalars()
        # Convert the flat VTK array to a NumPy array and reshape it to (depth, height, width)
        image_array = numpy_support.vtk_to_numpy(vtk_array).reshape((dims[2], dims[1], dims[0]))
    else:
        raise ValueError("Input image must be a sitk.Image Image or vtk.vtkImageData.")
    
    return image_array

def plot_views(image, image_name: str = "Image", display: bool = True, 
                 save: bool = False, output_dir = None) -> None:
    """
    Plots the mid-slice views (sagittal, coronal, and axial) of a 3D image.
    """
    image_array = get_image_array(image)

    # Get array shape: expect (depth, height, width) for a 3D image.
    num_slices, height, width = image_array.shape

    # Determine the mid-slice index for each view.
    axial_index = num_slices // 2      # Axial view (Z-axis slice)
    coronal_index = height // 2          # Coronal view (Y-axis slice)
    sagittal_index = width // 2          # Sagittal view (X-axis slice)

    # Extract the slices.
    axial = image_array[axial_index, :, :]
    coronal = image_array[:, coronal_index, :]
    sagittal = image_array[:, :, sagittal_index]

    # Create subplots in the order: Sagittal, Coronal, Axial.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the Sagittal view.
    axes[0].imshow(sagittal, cmap='gray')
    axes[0].set_title('Sagittal View (X-axis slice)')
    axes[0].axis('off')

    # Plot the Coronal view.
    axes[1].imshow(coronal, cmap='gray')
    axes[1].set_title('Coronal View (Y-axis slice)')
    axes[1].axis('off')

    # Plot the Axial view.
    axes[2].imshow(axial, cmap='gray')
    axes[2].set_title('Axial View (Z-axis slice)')
    axes[2].axis('off')

    fig.suptitle(image_name, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    if save:
        if output_dir is None:
            output_dir = os.getcwd()
        file_path = os.path.join(output_dir, f"{image_name}.png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        print(f"Plot saved to: {file_path}")

    if display:
        plt.show()
    else:
        plt.close()

def describe_image(image, verbose: bool = False):
    """
    Returns the dimensions, spacing, and origin of an image.
    """
    try:
        if isinstance(image, sitk.Image) or isinstance(image, vtk.vtkImageData):
            # For SimpleITK images, GetSize returns dimensions (int tuple)
            dimensions = image.GetSize()
            spacing = image.GetSpacing()
            origin = image.GetOrigin()
        else:
            raise TypeError("Input must be a sitk.Image or a vtk.vtkImageData.")

        if verbose:
            print("Dimensions:", dimensions)
            print("Spacing:", spacing)
            print("Origin:", origin)

        return [dimensions, spacing, origin]

    except Exception as e:
        print(f"Error describing image: {e}")


def read_object(file_path: str, verbose: bool = False) :
    """
    Reads an VTK polydata from the given file path using an appropriate VTK reader based on the file extension.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Find file extension.
        extension = os.path.splitext(file_path)[1].lower()

        # Mapping file extensions to the corresponding VTK reader classes.
        reader_mapping = {
                        ".obj": vtk.vtkOBJReader,
                        ".stl": vtk.vtkSTLReader,
                        ".vtk": vtk.vtkPolyDataReader
        }

        reader_class = reader_mapping.get(extension)
        if reader_class is None:
            raise ValueError(f"Unsupported file format: {file_path}")

        reader = reader_class()
        reader.SetFileName(file_path)
        reader.Update()
        polydata = reader.GetOutput()

        if verbose:
            print(f"Object read from: {file_path}")
        return polydata

    except Exception as e:
        print(f"Error reading object: {e}")
        return None

def write_object(polydata: vtk.vtkPolyData, file_path: str, verbose: bool = False) -> None:
    """
    Writes vtkPolyData to a file using an appropriate VTK writer based on the file extension.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the output directory exists
        
        # Determine the file extension
        extension = os.path.splitext(file_path)[1].lower()
        
        writer_mapping = {
            ".obj": vtk.vtkOBJWriter,
            ".stl": vtk.vtkSTLWriter,
            ".vtk": vtk.vtkPolyDataWriter
        }
        writer_class = writer_mapping.get(extension)
        if writer_class is None:
            raise ValueError(f"Unsupported file extension: {extension}")

        writer = writer_class()
        writer.SetFileName(file_path)
        writer.SetInputData(polydata)
        writer.Write()

        if verbose:
            print(f"Object saved as: {file_path}")

    except Exception as e:
        print(f"Error writing object: {e}")

def describe_vtkpolydata(polydata: vtk.vtkPolyData, verbose: bool = False):
    """
    Returns the number of points and cells in a vtkPolyData object.
    """
    try:
        if not isinstance(polydata, vtk.vtkPolyData):
            raise TypeError("Input is not vtkPolyData.")

        num_points = polydata.GetNumberOfPoints()
        num_cells = polydata.GetNumberOfCells()

        if verbose:
            print(f"Number of Points: {num_points} \n Number of Cells: {num_cells}")

        return [num_points, num_cells]

    except Exception as e:
        print(f"Error describing polydata: {e}")


def convert_mesh(input_file: str, output_file: str, verbose: bool = False) -> None:
    """
    Converts a mesh from one format to another.
    """
    try:
        polydata = read_object(input_file, verbose)
        write_object(polydata, output_file, verbose)

    except Exception as e:
        print(f"Error converting mesh: {e}")


def vtkPolyData_to_numpy(polydata):
    """
    Convert vtkPolyData points and polygonal faces to numpy arrays.
    """
    num_points = polydata.GetNumberOfPoints()
    points = np.array([polydata.GetPoint(i) for i in range(num_points)])

    faces = []
    # Iterate over each cell in the polydata.
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        num_cell_points = cell.GetNumberOfPoints()
        # Only include cells with three or more points
        if num_cell_points >= 3:
            faces.append([cell.GetPointId(j) for j in range(num_cell_points)])
    
    return points, faces

def plot_vtkPolyData(polydata: vtk.vtkPolyData, polydata_name: str="Object", display: bool=True, save: bool=False, output_dir = None) -> None:
    """
    Plot a vtkPolyData object using matplotlib
    """
    points, faces = vtkPolyData_to_numpy(polydata)

    # Create a new figure and a 3D subplot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a Poly3DCollection from the faces using the extracted points.
    poly_collection = Poly3DCollection([points[face] for face in faces],
                                         alpha=0.1, edgecolor='k')
    ax.add_collection3d(poly_collection)

    # Scatter plot the points.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=1)

    #ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    #ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    #ax.set_zlim(points[:, 2].min(), points[:, 2].max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    fig.suptitle(polydata_name, fontsize=16)    
    plt.tight_layout()

    if save:
        if output_dir is None:
            output_dir = os.getcwd()
        file_path = os.path.join(output_dir, f"{polydata_name}.png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        print(f"Plot saved to: {file_path}")

    if display:
        plt.show()
    else:
        plt.close()

def create_polydata_from_3Dcoordinates(resolution: int, x: 'np.ndarray', y: 'np.ndarray', z: 'np.ndarray'):
    """
    Create a VTK polydata object from x, y, and z coordinates.
    """
    # Create a vtkPoints object and pre-allocate the required number of points.
    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(resolution * resolution)
    
    # Loop through the grid indices and assign coordinates to each point.
    for i in range(resolution):
        for j in range(resolution):
            idx = i * resolution + j
            vtk_points.SetPoint(idx, x[i, j], y[i, j], z[i, j])
    
    # Create a vtkCellArray to store the triangle connectivity.
    triangles = vtk.vtkCellArray()
    
    # Loop through the grid to create two triangles for each grid cell.
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Compute indices for the four corners of the current cell.
            p1 = i * resolution + j
            p2 = p1 + 1
            p3 = (i + 1) * resolution + j
            p4 = p3 + 1

            # Insert the first triangle: vertices p1, p2, p3.
            triangles.InsertNextCell(3)
            triangles.InsertCellPoint(p1)
            triangles.InsertCellPoint(p2)
            triangles.InsertCellPoint(p3)

            # Insert the second triangle: vertices p2, p4, p3.
            triangles.InsertNextCell(3)
            triangles.InsertCellPoint(p2)
            triangles.InsertCellPoint(p4)
            triangles.InsertCellPoint(p3)
    
    vtk_polydata = vtk.vtkPolyData()
    vtk_polydata.SetPoints(vtk_points)
    vtk_polydata.SetPolys(triangles)
    
    return vtk_polydata

def create_polydata_from_vtkImage(image_vtk: vtk.vtkImageData, apply_gaussian: bool = False) -> vtk.vtkPolyData:
    """
    Convert a vtkImageData to vtkPolyData using thresholding and Marching Cubes,
    optionally applying a Gaussian smoothing filter.
    """
    # Create a threshold filter
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(image_vtk)
    threshold.ThresholdByUpper(128)  # Set the threshold value
    threshold.Update() 

    # Determine which filter to use next: either direct threshold output or smoothed version
    if apply_gaussian:
        # Apply Gaussian smoothing to the thresholded image
        gaussian_smooth = vtk.vtkImageGaussianSmooth()
        gaussian_smooth.SetInputConnection(threshold.GetOutputPort())
        gaussian_smooth.SetStandardDeviations(1.0, 1.0, 1.0)  # Sigma values
        gaussian_smooth.Update()
        marching_input = gaussian_smooth.GetOutputPort()
    else:
        marching_input = threshold.GetOutputPort()

    # Create the Marching Cubes algorithm
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputConnection(marching_input)
    marching_cubes.SetValue(0, 128)  # Set the threshold value for surface extraction
    marching_cubes.Update()

    # Return the resulting polydata (surface extracted by Marching Cubes)
    return marching_cubes.GetOutput()
