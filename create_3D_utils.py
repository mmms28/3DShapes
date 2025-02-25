import os
import numpy as np
import SimpleITK as sitk
from manipulate_3D_utils import *


# Hello hello 

def create_ellipsoid_volume(radii: tuple[float, float, float], dimensions: tuple[float, float, float], 
                            center: tuple[float, float, float], 
                            origin: tuple[float, float, float], spacing: tuple[float, float, float]):
    """ Creates a 3D ellipsoid (or sphere) inside a voxel grid and saves it. """
    image_data = np.zeros(dimensions, dtype=np.uint8)
    x, y, z = np.indices(dimensions)
    mask = ((x - center[0])**2 / radii[0]**2 +
            (y - center[1])**2 / radii[1]**2 +
            (z - center[2])**2 / radii[2]**2) <= 1
    image_data[mask] = 255 

    vtk_image = create_vtkimage_from_array(image_data=image_data, dimensions=dimensions, spacing=spacing, origin=origin)
    # Name the object based on its parameters
    shape_name = f"sphere_r{radii[0]}" if radii[0] == radii[1] == radii[2] else f"ellipsoid_x{radii[0]}_y{radii[1]}_z{radii[2]}"
    
    return vtk_image, shape_name



def create_prism_volume(size: tuple[float, float, float], dimensions: tuple[float, float, float], 
                            center: tuple[float, float, float],
                            origin: tuple[float, float, float], spacing: tuple[float, float, float]):
    """ Creates a 3D rectangular prism or cube inside a voxel grid and saves it. """
    image_data = np.zeros(dimensions, dtype=np.uint8)
    x_min, x_max = center[0] - size[0] // 2, center[0] + size[0] // 2
    y_min, y_max = center[1] - size[1] // 2, center[1] + size[1] // 2
    z_min, z_max = center[2] - size[2] // 2, center[2] + size[2] // 2
    image_data[x_min:x_max, y_min:y_max, z_min:z_max] = 255

    vtk_image = create_vtkimage_from_array(image_data=image_data, dimensions=dimensions, spacing=spacing, origin=origin)
    # Name the object based on its parameters
    shape_name = f"cube_{size[0]}" if size[0] == size[1] == size[2] else f"prism_x{size[0]}_y{size[1]}_z{size[2]}"

    return vtk_image, shape_name


def create_cylinder_volume(radius: float, height:float, dimensions: tuple[float, float, float], 
                            center: tuple[float, float, float],
                            origin: tuple[float, float, float], spacing: tuple[float, float, float]):
    """ Creates a 3D cylinder inside a voxel grid and saves it. """
    image_data = np.zeros(dimensions, dtype=np.uint8)
    z_min, z_max = center[2] - height // 2, center[2] + height // 2
    x, y = np.indices((dimensions[0], dimensions[1]))
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    image_data[mask, z_min:z_max] = 255

    vtk_image = create_vtkimage_from_array(image_data=image_data, dimensions=dimensions, spacing=spacing, origin=origin)
    # Name the object based on its parameters
    shape_name = f"cylinder_r{radius}_h{height}"

    return vtk_image, shape_name

def create_cone_volume(base_radius: float, height:float, dimensions: tuple[float, float, float], 
                            center: tuple[float, float, float],
                            origin: tuple[float, float, float], spacing: tuple[float, float, float]):
    """ Creates a 3D cone inside a voxel grid and saves it. """
    image_data = np.zeros(dimensions, dtype=np.uint8)
    z_min, z_max = center[2] - height // 2, center[2] + height // 2
    for z in range(z_min, z_max):
        current_radius = base_radius * (1 - (z - z_min) / height)
        x, y = np.indices((dimensions[0], dimensions[1]))
        mask = (x - center[0])**2 + (y - center[1])**2 <= current_radius**2
        image_data[mask, z] = 255

    vtk_image = create_vtkimage_from_array(image_data=image_data, dimensions=dimensions, spacing=spacing, origin=origin)
    # Name the object based on its parameters
    shape_name = f"cone_br{base_radius}_h{height}"

    return vtk_image, shape_name

def create_pyramid_volume(base_size: tuple[float, float], height: float, dimensions: tuple[float, float, float], 
                            center: tuple[float, float, float],
                            origin: tuple[float, float, float], spacing: tuple[float, float, float]):
    """ Creates a 3D pyramid inside a voxel grid and saves it. """
    image_data = np.zeros(dimensions, dtype=np.uint8)
    z_min, z_max = center[2] - height // 2, center[2] + height // 2
    for z in range(z_min, z_max):
        current_half_x = (base_size[0] // 2) * (1 - (z - z_min) / height)
        current_half_y = (base_size[1] // 2) * (1 - (z - z_min) / height)
        x_min, x_max = int(center[0] - current_half_x), int(center[0] + current_half_x)
        y_min, y_max = int(center[1] - current_half_y), int(center[1] + current_half_y)
        image_data[x_min:x_max, y_min:y_max, z] = 255

    vtk_image = create_vtkimage_from_array(image_data=image_data, dimensions=dimensions, spacing=spacing, origin=origin)
    # Name the object based on its parameters
    shape_name = f"pyramid_bx{base_size[0]}_by{base_size[1]}_h{height}"

    return vtk_image, shape_name

def create_torus_volume(major_radius: float, minor_radius: float,  dimensions: tuple[float, float, float], 
                            center: tuple[float, float, float],
                            origin: tuple[float, float, float], spacing: tuple[float, float, float]):
    """ Creates a 3D torus inside a voxel grid and saves it. """
    image_data = np.zeros(dimensions, dtype=np.uint8)
    x, y, z = np.indices(dimensions)
    dx, dy, dz = x - center[0], y - center[1], z - center[2]
    radial_distance = np.sqrt(dx**2 + dy**2)
    mask = (radial_distance - major_radius)**2 + dz**2 <= minor_radius**2
    image_data[mask] = 255

    vtk_image = create_vtkimage_from_array(image_data=image_data, dimensions=dimensions, spacing=spacing, origin=origin)
    # Name the object based on its parameters
    shape_name = f"torus_MR{major_radius}_mr{minor_radius}"

    return vtk_image, shape_name

def create_ellipsoid_object(resolution: int, radii: tuple[float, float, float], center: tuple[float, float, float]):
    # Create points for the ellipsoid using parametric equations
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radii[0] * np.sin(phi) * np.cos(theta)
    y = center[1] + radii[1] * np.sin(phi) * np.sin(theta)
    z = center[2] + radii[2] * np.cos(phi)

    obj_polydata = create_polydata_from_3Dcoordinates(resolution, x, y, z)
    # Name the object based on its parameters   
    obj_name = f"sphere_r{radii[0]}" if radii[0] == radii[1] == radii[2] else f"ellipsoid_x{radii[0]}_y{radii[1]}_z{radii[2]}"
    
    return obj_polydata, obj_name

def create_cylinder_object(resolution: int, radius: float, height: float, center: tuple[float, float, float]):
    # Create points for the cylinder using parametric equations
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, resolution)
    theta, z = np.meshgrid(theta, z)
    
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = center[2] + z  
    
    obj_polydata = create_polydata_from_3Dcoordinates(resolution, x, y, z)
    # Name the object based on its parameters
    obj_name = f"cylinder_r{radius}_h{height}"
    
    return obj_polydata, obj_name

def create_cone_object(resolution: int, radius: float, height: float, center: tuple[float, float, float]):
    # Create points for the cone using parametric equations
    phi = np.linspace(0, 2 * np.pi, resolution)
    z_values = np.linspace(0, height, resolution)
    phi, z_values = np.meshgrid(phi, z_values)

    x = center[0] + radius * (1 - z_values / height) * np.cos(phi)
    y = center[1] + radius * (1 - z_values / height) * np.sin(phi)
    z = center[2] + z_values
    
    obj_polydata = create_polydata_from_3Dcoordinates(resolution, x, y, z)
    # Name the object based on its parameters
    obj_name = f"cone_r{radius}_h{height}"
    
    return obj_polydata, obj_name

def create_torus_object(resolution: int, major_radius: float, minor_radius: float, center: tuple[float, float, float]):
    # Create points for the torus using parametric equations
    phi = np.linspace(0, 2* np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
    y = center[1] + (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
    z = center[2] + minor_radius * np.sin(theta)

    obj_polydata = create_polydata_from_3Dcoordinates(resolution, x, y, z)
    # Name the object based on its parameters
    obj_name = f"torus_Mr{major_radius}_mr{minor_radius}"
    
    return obj_polydata, obj_name
 