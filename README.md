
# 3DShapes

3DShapes is a Python project designed for generating, manipulating, and visualizing 3D shapes and volumes. It offers two main functionalities: creating 3D volumes and surface objects from parametric equations (via `create_3D_utils.py`), and manipulating, converting, and visualizing these images and meshes (via `manipulate_3D_utils.py`). The project uses powerful libraries such as NumPy, VTK, and SimpleITK for processing and visualization.

## Table of Contents
- [Features](#features)
- [Usage](#usage)
  - [Running the Scripts](#running-the-scripts)
  - [Jupyter Notebooks](#jupyter-notebooks)
    - [Parametric Volumes and Surfaces Notebook](#parametric-volumes-and-surfaces-notebook)
    - [All Volumes and Surfaces Notebook](#all-volumes-and-surfaces-notebook)
- [Function Documentation](#function-documentation)
  - [Volume and Object Creation](#volume-and-object-creation)
  - [Manipulation Utilities](#manipulation-utilities)
- [Dependencies](#dependencies)

## Features
- **3D Volume Generation:** Create voxel-based volumes for ellipsoids, prisms, cylinders, cones, pyramids, and tori.
- **Surface Object Creation:** Generate 3D surface objects (ellipsoids, cylinders, cones, tori) from parametric equations.
- **Image & Mesh Manipulation:** Read, write, convert, and describe images (SimpleITK and VTK) and 3D objects (VTK polydata).
- **Visualization:** Plot 2D mid-slices from 3D images and display 3D meshes using matplotlib.

## Project Structure

- **create_3D_utils.py:**  
  Contains functions that create various 3D volumes (ellipsoid, prism, cylinder, cone, pyramid, torus) and surface objects using parametric equations. Each function returns a VTK image (for volumes) or VTK polydata (for objects) along with a string name derived from its parameters.

- **manipulate_3D_utils.py:**  
  Provides utilities for reading, writing, and converting images and meshes using both SimpleITK and VTK. It includes helper functions to convert between NumPy arrays and VTK images, extract image slices for visualization, convert polydata to NumPy arrays, and plot 3D data using matplotlib.

## Usage
### Utils
#### Generate Volumes and Objects:
You can run the create_3D_utils.py script (or import its functions) to generate 3D volumes or surface objects. For example:

```
from create_3D_utils import *
import create_3D_utils
```

#### Manipulate and Visualize Data:
Use the functions in manipulate_3D_utils.py to read, write, convert, and plot images and 3D objects.

### Jupyter Notebooks
#### Parametric Volumes and Surfaces Notebook
The parametric_volumes_surfaces.ipynb notebook shows a complete workflow for creating 3D shapes from their parametric equations. In this notebook, you will see examples for:

- **Ellipsoid, Sphere, Cone, Cylinder and Torus**.

For each shape, the notebook illustrates:

- **Polydata Generation**: Creating the surface object (VTK polydata) via parametric equations.
- **Image Conversion**: Converting the polydata into a voxel-based image.
- **Visualization**: Plotting both the 3D polydata and the mid-slice images of the corresponding volume.

#### All Volumes and Surfaces Notebook
The all_volumes_surfaces.ipynb notebook provides a comprehensive demonstration of generating 3D shapes as volumes. It covers:

- **Sphere, Ellipsoid, Cube, Prism, Pyramid, Cylinder, Torus**.

After creating these volumes, the notebook shows how to convert each volume into VTK polydata and then visualize both the original volumes (via mid-slice views) and the corresponding 3D polydata.

### Examples

2 folders of examples where provided, this ones were generated from the 2 jupyter notebooks.

## Function Documentation
### Volume and Object Creation
Functions in create_3D_utils.py include:

- ``` create_ellipsoid_volume(radii, dimensions, center, origin, spacing):```
Creates a 3D ellipsoid (or sphere) within a voxel grid. Returns a VTK image and a descriptive name.

- ``` create_prism_volume(size, dimensions, center, origin, spacing):```
Generates a rectangular prism (or cube) in a voxel grid and returns a VTK image and its name.

- ``` create_cylinder_volume(radius, height, dimensions, center, origin, spacing):```
Creates a cylindrical volume with specified dimensions, returning a VTK image and its name.

- ``` create_cone_volume(base_radius, height, dimensions, center, origin, spacing):```
Constructs a cone volume by tapering the cross-section along its height. Returns a VTK image and a name.

- ``` create_pyramid_volume(base_size, height, dimensions, center, origin, spacing):```
Builds a pyramid volume by reducing the base dimensions with height. Returns a VTK image and its name.

- ``` create_torus_volume(major_radius, minor_radius, dimensions, center, origin, spacing):```
Generates a torus volume using its parametric equation. Returns a VTK image and the shape name.

- ``` create_ellipsoid_object(resolution, radii, center):```
Uses parametric equations to generate surface points for an ellipsoid (or sphere), returning VTK polydata and its name.

- ``` create_cylinder_object(resolution, radius, height, center):```
Generates a cylinder’s surface points via parametric equations, returning VTK polydata and a descriptive name.

- ``` create_cone_object(resolution, radius, height, center):```
Produces a cone’s surface from parametric equations and returns VTK polydata with its name.

- ``` create_torus_object(resolution, major_radius, minor_radius, center):```
Computes a torus surface using parametric equations and returns VTK polydata and its name.

### Manipulation Utilities
Functions in manipulate_3D_utils.py include:

- ``` read_sitkimage(image_path, verbose):```
Reads a SimpleITK image from the specified file path.

- ``` write_sitkimage(image, image_path, verbose):```
Writes a SimpleITK image to file, using compression if available.

- ``` convert_image_format(input_file, output_file, verbose): :```
Converts an image between formats using SimpleITK.

- ``` read_vtkimage(image_path, verbose):```
Reads a VTK image file using the appropriate reader based on the file extension.

- ``` write_vtkimage(image, image_path, verbose):```
Writes a VTK image to file, using a writer determined by the file extension.

- ``` create_vtkimage_from_polydata(dimensions, spacing, origin, vtk_polydata):```
Converts closed VTK polydata into a VTK image using stencil conversion.

- ``` create_sitkimage_from_array(image_data, spacing, origin):```
Converts a 3D NumPy array into a SimpleITK image.

- ``` create_vtkimage_from_array(image_data, spacing, dimensions, origin):```
Converts a 3D NumPy array to a VTK image (with 16-bit data).

- ``` get_image_array(image):```
Retrieves a NumPy array from a SimpleITK image or a VTK image.

- ``` plot_views(image, image_name, display, save, output_dir):```
Plots the mid-slice views (sagittal, coronal, axial) of a 3D image using matplotlib.

- ``` describe_image(image, verbose):```
Returns key information (dimensions, spacing, origin) about an image.

- ``` read_object(file_path, verbose):```
Reads a 3D object (VTK polydata) from a file (supports OBJ, STL, VTK).

- ``` write_object(polydata, file_path, verbose):```
Writes VTK polydata to file using the correct writer.

- ``` describe_vtkpolydata(polydata, verbose):```
Returns the number of points and cells in a VTK polydata object.

- ``` convert_mesh(input_file, output_file, verbose):```
Converts a mesh from one format to another.

- ``` vtkPolyData_to_numpy(polydata):```
Converts VTK polydata points and polygonal faces to NumPy arrays.

- ``` plot_vtkPolyData(polydata, polydata_name, display, save, output_dir):```
Plots a VTK polydata object (mesh) using matplotlib’s 3D plotting.

- ``` create_polydata_from_3Dcoordinates(resolution, x, y, z):```
Generates VTK polydata from 3D coordinate arrays.

- ``` create_polydata_from_vtkImage(image_vtk, apply_gaussian):```
Converts a VTK image to polydata using thresholding and Marching Cubes (with optional Gaussian smoothing).

## Dependencies
- NumPy: For numerical computations.
- SimpleITK: For image processing and volume creation.
- VTK: For 3D visualization and data handling.
- Matplotlib: For plotting 2D and 3D visualizations.
- Jupyter Notebook: For interactive exploration via notebooks.