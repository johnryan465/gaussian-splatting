import numpy as np


def sample_unit_sphere():
  """Samples a point on the unit sphere.

  Args:
    n_samples: The number of points to sample.

  Returns:
    A NumPy array of shape (n_samples, 3) containing the sampled points.
  """

  # Generate random angles in spherical coordinates.
  theta = np.random.uniform(0, 2 * np.pi, size=1)
  phi = np.arccos(1 - 2 * np.random.uniform(0, 1, size=1))

  # Convert the spherical coordinates to Cartesian coordinates.
  x = np.cos(theta) * np.sin(phi)
  y = np.sin(theta) * np.sin(phi)
  z = np.cos(phi)

  return np.array([x, y, z]).reshape(3)

def rotation_matrix(axis: np.ndarray, angle: float):
    """Computes the rotation matrix for a rotation around an arbitrary axis.
    
    Args:
        axis: A NumPy array of shape (3,) containing the axis of rotation.
        angle: The angle of rotation in radians.
    
    Returns:
        A NumPy array of shape (3, 3) containing the rotation matrix.
    """
    
    # Normalize the axis.
    axis = axis / np.linalg.norm(axis)
    
    # Compute the skew-symmetric cross product matrix.
    axis_cross = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    # Compute the rotation matrix.
    R = np.eye(3) + np.sin(angle) * axis_cross + (1 - np.cos(angle)) * axis_cross @ axis_cross
    
    return R

def random_rotation(max_angle: float):
  axis = np.array([0,1,0]) # sample_unit_sphere()
  angle = max_angle / 180 * np.pi # np.random.uniform(0, (max_angle / 180) * np.pi)
  # create homogeneous rotation matrix
  R = np.eye(4)
  R[:3, :3] = rotation_matrix(axis, angle)
  return R
  
  

if __name__ == "__main__":
  print(sample_unit_sphere())
  print(np.linalg.norm(sample_unit_sphere()))
  print(rotation_matrix(np.array([0,0,1]), 30))