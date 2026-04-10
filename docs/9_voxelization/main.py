import numpy as np
from tools import process_voxel, convert_binvox_to_numpy, export_to_obj

if __name__ == "__main__":
    input_path = "cerebro.stl"
    filename = input_path.split(".")[0]
    output_path = "./"
    process_voxel(input_path, output_path, resolution=128)
    data = convert_binvox_to_numpy(f"{filename}.binvox")

    print("Geometric Analysis")

    # Extract the (x, y, z) coordinates of voxels with mass (where value equals 1)
    points = np.argwhere(data == 1)

    # Calculate the center of mass
    center_of_mass = np.mean(points, axis=0)
    print(f"Center of Mass (x, y, z): {center_of_mass}")

    # Center the object (translate origin to center of mass)
    centered_points = points - center_of_mass

    x = centered_points[:, 0]
    y = centered_points[:, 1]
    z = centered_points[:, 2]

    # Calculate the inertia tensor
    I_xx = np.sum(y**2 + z**2)
    I_yy = np.sum(x**2 + z**2)
    I_zz = np.sum(x**2 + y**2)
    I_xy = -np.sum(x * y)
    I_xz = -np.sum(x * z)
    I_yz = -np.sum(y * z)

    inertia_tensor = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz]
    ])

    print("\nInertia Tensor (3x3 Matrix):\n", inertia_tensor)

    # Compute principal axes (eigenvalues and eigenvectors)
    eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)

    print("\nEigenvalues (Principal inertia moments):\n", eigenvalues)
    print("\nEigenvectors (Principal axes - columns of matrix):\n", eigenvectors)

    # Alignment to ensure rotation invariance
    aligned_points = np.dot(centered_points, eigenvectors)

    print("\nProcess completed!\nThe object has been aligned with its principal axes.")

    export_to_obj(aligned_points, f"{filename}_alineado.obj")