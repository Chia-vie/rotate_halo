import numpy as np
import warnings

warnings.filterwarnings("error")

class Halo():
    def __init__(self, coordinates, velocities, masses,
                 center_coordinates=None, center_velocities=None):
        """
        :param particleIDs: 1D array with length N, dtype: string. The IDs of the particles.
        :type particleIDs: np.ndarray
        :param coordinates: np.ndarray with shape (3, N), dtype: float.
        The x, y and z coordinates of the particles.
        :type coordinates: np.ndarray
        :param velocities: np.ndarray with shape (3, N), dtype: float.
        The x, y and z velocity components of the particles.
        :type velocities: np.ndarray
        :param masses: 1D array, dtype: float.
        The masses of the particles.
        :type masses: np.ndarray
        :param center_coordinates: np.ndarray with shape (3,), dtype: float.
        The x, y, z coordinates of the center of mass of the halo
        :type center_coordinates: np.ndarray
        :param center_velocities: np.ndarray with shape (3,), dtype: float.
        The x, y, z components of the velocity of the center of mass of the halo
        :type center_coordinates: np.ndarray

        """
        self.masses = masses

        self.coordinates = coordinates
        self.velocities = velocities

        self.check_input_shape('coordinates')
        self.check_input_shape('velocities')

        self.center_coordinates = center_coordinates
        self.center_velocities = center_velocities

    def check_input_shape(self, name):
        """
        Checks if arguments have proper shape.
        Raises Value Error.
        :param name:
        :type name:
        :return:
        :rtype:
        """
        attribute = getattr(self, name)
        if not isinstance(attribute, np.ndarray):
            try:
                attribute = np.array(attribute)
            except:
                raise ValueError(f'{name} must be ndarray of shape (3, N).')
        if not attribute.shape[0] == 3:
            raise ValueError(f'{name} must be an ndarray of shape (3, N).')
        setattr(self, name, attribute)

    def rotate_faceon(self, shift_particles=True, disk_radius=30., h=0.6774):
        """
        :param shift_particles: If this is set to true, the function first re-centers
        all the particles before calculating the face-on matrix
        :type shift_particles: bool
        :param disk_radius: the roughly expected radius of the galaxy's
        disk in physical units (kpc) to be considered for
        the calculation of the angular momentum
        :type disk_radius: float
        :param h: hubble parameter
        :type h: float
        :return: 1) the face-on rotated coordinates 2) the face-on rotated velocities
        :rtype: tuple of two np.ndarrays
        """
        if shift_particles:
            self.shift_particles()
        self.select_disk(self.coordinates_shifted,disk_radius, h)
        self.calculate_J()
        self.calc_faceon_matrix()
        # apply the rotation matrix to the positions of stellar particles
        self.coordinates_faceon = np.dot(self.matr, self.coordinates_shifted)  # rotated x/y/z is now pos[:,0/1/2]
        # apply the rotation matrix to the velocities of stellar particles
        self.velocities_faceon = np.dot(self.matr, self.velocities_shifted)  # rotated vx/vy/vz is now vel[:,0/1/2]
        return self.coordinates_faceon, self.velocities_faceon

    def rotate_angle(self, deg=45., axis='x', values=None, shift_particles=False):
        """
        :param deg: the desired inclination angle in degrees
        :type deg: float
        :param axis: the rotation axis
        :type axis: str
        :param values: the values which should be rotated (e.g. coordinates, velocities)
        :type values:
        :param shift_particles:
        :type shift_particles:
        :return:
        :rtype:
        """
        if values is None:
            if hasattr(self, 'coordinates_faceon'):
                values = self.coordinates_faceon
            else:
                values = self.rotate_faceon(shift_particles=shift_particles)

        matr = self.calc_rotation_matrix(axis, deg)
        self.values_rotated = np.dot(matr, values)
        return self.values_rotated

    def calc_faceon_matrix(self, up=[0.0, 1.0, 0.0]):
        """
        calculates the faceon rotation matrix for the given halo
        This is the same function is adopted from pynbody.analysis.angmom.calc_faceon_matrix()
        :param up: the vector pointing in the
        :type up:
        """
        J_tot_norm = self.J_tot / np.linalg.norm(self.J_tot) # normalize J vector
        vec_p1 = np.cross(up, J_tot_norm)
        vec_p1_norm = vec_p1 / np.linalg.norm(vec_p1) # normalize p1 vector
        vec_p2 = np.cross(J_tot_norm, vec_p1_norm)
        self.matr = np.concatenate((vec_p1_norm, vec_p2, J_tot_norm)).reshape((3, 3))
        # checks if the matrix is orthogonal
        resid = np.dot(self.matr, self.matr.T) - np.eye(3)
        resid = np.sum(resid ** 2)
        if resid > 1.e-8:
            print('Warning: The rotation matrix is not orthogonal')

    def calc_rotation_matrix(self, axis, angle_deg):
        """
        :param axis: can be 'x', 'y' or 'z', the rotation axis
        :type axis: str
        :param angle_deg: angle in degree by which the halo should be rotated
        :type angle_deg: float
        :return: rotation matrix
        :rtype: np.ndarray
        """
        theta = np.radians(angle_deg)
        s = np.sin(theta)
        c = np.cos(theta)

        if axis == 'x':
            return np.array([[1, 0, 0],
                             [0, c, -s],
                             [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s],
                             [0, 1, 0],
                             [-s, 0, c]])
        elif axis == 'z':
            return np.array([[c, -s, 0],
                             [s, c, 0],
                             [0, 0, 1]])

    def shift_particles(self):
        """
        :param subhalo_coords: np.ndarray with lenght 3.
        The x, y and z coordinates of the halo's center of mass.
        In Illustris-TNG these coordinates are provided as SubhaloPos.
        :type subhalo_coords: np.ndarray
        :param subhalo_vel: np.ndarray with lenght 3.
        The x, y and z coordinates of the halo's center of mass
        In Illustris-TNG these coordinates are provided as SubhaloVel
        :type subhalo_vel: np.ndarray
        :return: np.ndarray with shape (3, N).
        The shifted (re-centered) coordinates
        :rtype: np.ndarray
        """
        if self.center_coordinates is None:
            print('No central coordinates given. '
                  'Calculated center of mass coordinates to:')
            self.center_coordinates = self.find_center_of_mass(self.coordinates)
            print(*self.center_coordinates)

        if self.center_velocities is None:
            print('No systemic velocity given.')
            print('Calculated systemic velocities to:')
            self.center_velocities = self.find_center_of_mass(self.velocities)
            print(*self.center_velocities)

        # Create 3D array out of central coordinates
        center_coords_3 = np.array([self.center_coordinates] * self.velocities.shape[1])
        self.coordinates_shifted = self.coordinates - center_coords_3.T

        # Create 3D array out of central coordinates
        center_vel_3 = np.array([self.center_velocities] * self.coordinates.shape[1])
        self.velocities_shifted = self.velocities - center_vel_3.T

        return self.coordinates_shifted, self.velocities_shifted

    def find_center_of_mass(self, values, masses=None):
        """
        Finds the center of mass values for given values.
        E.g. center of mass coordinates or velocities
        of the central region with radius ``radius`` of the halo.

        :return: The x, y, z values of the center of mass
        :rtype: np.ndarray of shape (3,)
        """

        if masses is None:
            masses = self.masses

        m_tot = np.sum(masses)
        central_values = np.sum(values * masses, axis=1)/m_tot

        return central_values

    def select_disk(self, coordinates, physical_radius=30., h=0.6774):
        """
        Creates a mask which includes only the central disk.
        :param coordinates: particle coordinates
        :type coordinates: np.ndarray
        :param physical_radius: radius in kpc which is assumed for the disk extent
        :type physical_radius: float
        :param h: hubble constant
        :type h: float
        """
        # select central disk region to calculate center of mass
        r = np.sqrt(np.sum(coordinates ** 2, axis=0))
        self.disk_mask = r / h <= physical_radius  # conversion from ckpc/h to kpc

    def calculate_J(self):
        """
        Calculates the total angular momentum
        """
        x, y, z = self.coordinates_shifted
        vx, vy, vz = self.velocities_shifted

        Jx = np.sum((y[self.disk_mask] * vz[self.disk_mask] - z[self.disk_mask] * vy[self.disk_mask]) * self.masses[self.disk_mask])
        Jy = np.sum((z[self.disk_mask] * vx[self.disk_mask] - x[self.disk_mask] * vz[self.disk_mask]) * self.masses[self.disk_mask])
        Jz = np.sum((x[self.disk_mask] * vy[self.disk_mask] - y[self.disk_mask] * vx[self.disk_mask]) * self.masses[self.disk_mask])

        self.J_tot = np.array([Jx, Jy, Jz])




