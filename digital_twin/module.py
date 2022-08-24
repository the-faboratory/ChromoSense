"""
Module file to generate and interact with a virtual module for the simulation.

"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.optimize import root
from os.path import exists
import time
from scipy.ndimage.interpolation import rotate


class Module:
    """Module class to use in GUI file
    """

    def __init__(self, length_base_arms, length_base_between_arms, length_lower_arm, length_upper_arm,
                 length_surface_arms, length_surface_between_arms, thickness, joint_radius, translate_x, translate_y):
        """Initialize the module

        For the arguments length, since we take only interest in visualization, the developer advise to keep values around 10 cm with the right proportions.

        Args:
            length_base_arms (float): Length of the base of the module where the arms are linked in cm
            length_base_between_arms (float): Length  of the base of the module between arms in cm
            length_lower_arm (float): Length of the lower part of the arm in cm
            length_upper_arm (float): Length of the upper arm of the arm in cm
            length_surface_arms (float): Length of the surface of the module where the arms are linked in cm
            length_surface_between_arms (float): Length of the surface of the module between arms in cm
            thickness (float): thickness of the material used in cm
            joint_radius (float): radius of the joint between upper and lower arm in cm, cannot be zero
            translate_x (float): translate module on the x axis
            translate_y (float): translate module on the y axis
        """
        self.length_base_arms = length_base_arms
        self.length_base_between_arms = length_base_between_arms
        self.length_lower_arm = length_lower_arm
        self.length_upper_arm = length_upper_arm
        self.length_surface_arms = length_surface_arms
        self.length_surface_between_arms = length_surface_between_arms
        self.thickness = thickness
        self.joint_radius = joint_radius

        # Computation of the minimal distance between the fold between two arms.
        self.dmin_fold = self.compute_dmin_fold()

        self.translate = [translate_x, translate_y, 0]

        # Initialization function of the arms position for forward and backward kinematics.
        self.res_kinematics_arm1 = None
        self.res_kinematics_arm2 = None
        self.res_kinematics_arm3 = None

        self.arm1_junction = None
        self.arm2_junction = None
        self.arm3_junction = None

        # Module generation and workspaces
        self.generate_module()
        self.generate_workspace()

        # Workspace initialization
        self.ws_init = self.ws

        self.set_angles(0, 0, 0)

    def set_angles(self, angle1, angle2, angle3):
        """Set the angles between lower and upper arm of the module.

        Args:
            angle1 (float): Angle of the first arm [radians]
            angle2 (float): Angle of the second arm [radians]
            angle3 (float): Angle of the third arm [radians]
        """
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3

    def get_angle1(self):
        return self.angle1

    def get_angle2(self):
        return self.angle2

    def get_angle3(self):
        return self.angle3

    def compute_dmin_fold(self):
        """Compute minimal distance between the folds of each arm

        Returns:
            float: Minimal distance between two arms.
        """
        return np.sqrt(2) * 0.5 * self.length_base_arms * self.thickness / self.joint_radius

    def generate_module(self):
        """Generate the initial module to interact with.
        """
        self.init_base = np.array([[0, 0, 0],
                                   [self.length_base_arms, 0, 0],
                                   [(self.length_base_arms - self.length_base_between_arms) / 2,
                                    0.5 * (self.length_base_arms + self.length_base_between_arms) * np.sqrt(3), 0],
                                   [-self.length_base_between_arms / 2,
                                    0.5 * self.length_base_between_arms * np.sqrt(3), 0],
                                   [self.length_base_arms + self.length_base_between_arms / 2,
                                    self.length_base_between_arms * np.cos(np.pi / 6), 0],
                                   [self.length_base_arms / 2 + self.length_base_between_arms / 2,
                                    0.5 * (self.length_base_arms + self.length_base_between_arms) * np.sqrt(3), 0], ])
        self.base = self.init_base

        # Reference for lower arm
        self.init_ref_rect1 = np.array([[0, 0, 0],
                                        [self.length_base_arms, 0, 0],
                                        [self.length_base_arms / 2, -self.length_lower_arm, 0],
                                        [0, -self.length_lower_arm + self.length_base_arms / 2, 0],
                                        [self.length_base_arms, -self.length_lower_arm + self.length_base_arms / 2, 0]])

        # Reference for upper arm
        self.init_ref_rect2 = np.array([[0, -(self.length_upper_arm - self.length_lower_arm), 0],
                                        [self.length_base_arms, -(self.length_upper_arm - self.length_lower_arm), 0],
                                        [self.length_base_arms / 2, -self.length_lower_arm, 0],
                                        [0, -self.length_lower_arm + self.length_base_arms, 0],
                                        [self.length_base_arms, -self.length_lower_arm + self.length_base_arms, 0]])

        self.arm1 = {'rect1': self.init_ref_rect1, 'rect2': self.init_ref_rect2}

        # Generate second arm (rotation of the reference of lower and upper arm)
        self.init_ref_arm2 = {'rect1': self.init_ref_rect1, 'rect2': self.init_ref_rect2}
        self.ref_arm2 = self.init_ref_arm2
        self.arm2 = self.generate_arm2()

        # Generate third arm (rotation of the refereence of lower and upper arm)
        self.init_ref_arm3 = {'rect1': self.init_ref_rect1, 'rect2': self.init_ref_rect2}
        self.ref_arm3 = self.init_ref_arm3
        self.arm3 = self.generate_arm3()

        # Initialize angles and module creation.
        self.set_angles(0, 0, 0)

        if (self.length_base_between_arms == 0) and self.length_lower_arm > self.length_upper_arm:
            self.rotate_arms(init=True)
        else:
            self.init_surf = np.array([[0, 0, 11],
                                       [self.length_surface_arms, 0, 11],
                                       [(self.length_surface_arms - self.length_surface_between_arms) / 2,
                                        0.5 * (self.length_surface_arms + self.length_surface_between_arms) * np.sqrt(
                                            3), 11],
                                       [-self.length_surface_between_arms / 2,
                                        0.5 * self.length_surface_between_arms * np.sqrt(3), 11],
                                       [self.length_surface_arms + self.length_surface_between_arms / 2,
                                        self.length_surface_between_arms * np.cos(np.pi / 6), 11],
                                       [self.length_surface_arms / 2 + self.length_surface_between_arms / 2,
                                        0.5 * (self.length_surface_arms + self.length_surface_between_arms) * np.sqrt(
                                            3), 11]])

            self.surf = self.init_surf
        self.set_angles(0, 0, 0)
        self.rotate_arms()
        self.inverse_kinematics(R=[1, 0, 0, 0, 1, 0, 0, 0, 1])

    def generate_arm2(self):
        """Generate the second arm.

        Returns:
            dict: upper and lower arm of the second arm.
        """
        Rmat_arm2 = np.array([[np.cos(120 * np.pi / 180), -np.sin(120 * np.pi / 180), 0],
                              [np.sin(120 * np.pi / 180), np.cos(120 * np.pi / 180), 0], [0, 0, 1]])
        rect1 = np.matmul(self.ref_arm2['rect1'], Rmat_arm2)
        rect2 = np.matmul(self.ref_arm2['rect2'], Rmat_arm2)
        rect1[:, 0] = rect1[:, 0] + self.base[2, 0]
        rect1[:, 1] = rect1[:, 1] + self.base[2, 1]
        rect2[:, 0] = rect2[:, 0] + self.base[2, 0]
        rect2[:, 1] = rect2[:, 1] + self.base[2, 1]
        return {'rect1': rect1, 'rect2': rect2}

    def generate_arm3(self):
        """Generate the third arm.

        Returns:
            dict: upper and lower arm of the third arm.
        """
        Rmat_arm2 = np.array([[np.cos(240 * np.pi / 180), -np.sin(240 * np.pi / 180), 0],
                              [np.sin(240 * np.pi / 180), np.cos(240 * np.pi / 180), 0], [0, 0, 1]])
        rect1 = np.matmul(self.ref_arm3['rect1'], Rmat_arm2)
        rect2 = np.matmul(self.ref_arm3['rect2'], Rmat_arm2)
        rect1[:, 0] = rect1[:, 0] + self.base[4, 0]
        rect1[:, 1] = rect1[:, 1] + self.base[4, 1]
        rect2[:, 0] = rect2[:, 0] + self.base[4, 0]
        rect2[:, 1] = rect2[:, 1] + self.base[4, 1]
        return {'rect1': rect1, 'rect2': rect2}

    def generate_workspace(self):
        """Generate workspace. If the workspace does not exist, it will be generated computing the surface position for each angles combination.
        """
        if not (exists('workspaces/workspace_module.pkl')):
            self.ws = []

            # First computation to help setting right numerical approximation of the surface.
            self.angle1 = 10
            self.angle2 = 10
            self.angle3 = 10
            self.rotate_arms(workspace=True)
            for i in range(0, 90):
                for j in range(0, 90):
                    for k in range(0, 90):
                        self.angle1 = i
                        self.angle2 = j
                        self.angle3 = k
                        print([self.angle1, self.angle2, self.angle3])
                        self.rotate_arms(workspace=True)
                        if not (self.check_impossible()):
                            self.ws.append(self.surf)

            np.array(self.ws).dump('./workspaces/20_20_0.pkl')
        else:
            self.ws = np.load('./workspaces/workspace_module.pkl', allow_pickle=True)

    def plot_module(self, ax):
        """Plot the module. Plot each surface.

        Args:
            ax (matplotlib.pyplot.Axes): Polygons drawn for each surface.
        """
        try:
            self.translate_module()

            ax.plot_trisurf(self.base[:, 0], self.base[:, 1], self.base[:, 2], color='k', alpha=0.8)

            ax.plot_trisurf(self.surf[:, 0], self.surf[:, 1], self.surf[:, 2], color='k')

            ax.plot_trisurf(self.fold_arm1_left_upper[:, 0], self.fold_arm1_left_upper[:, 1],
                            self.fold_arm1_left_upper[:, 2], color='orange', alpha=0.4)
            ax.plot_trisurf(self.fold_arm1_left_lower[:, 0], self.fold_arm1_left_lower[:, 1],
                            self.fold_arm1_left_lower[:, 2], color='orange', alpha=0.4)

            ax.plot_trisurf(self.fold_arm1_right_upper[:, 0], self.fold_arm1_right_upper[:, 1],
                            self.fold_arm1_right_upper[:, 2], color='orange', alpha=0.4)
            ax.plot_trisurf(self.fold_arm1_right_lower[:, 0], self.fold_arm1_right_lower[:, 1],
                            self.fold_arm1_right_lower[:, 2], color='orange', alpha=0.4)

            ax.plot_trisurf(self.fold_arm2_left_upper[:, 0], self.fold_arm2_left_upper[:, 1],
                            self.fold_arm2_left_upper[:, 2], color='orange', alpha=0.4)
            ax.plot_trisurf(self.fold_arm2_left_lower[:, 0], self.fold_arm2_left_lower[:, 1],
                            self.fold_arm2_left_lower[:, 2], color='orange', alpha=0.4)

            ax.plot_trisurf(self.fold_arm2_right_upper[:, 0], self.fold_arm2_right_upper[:, 1],
                            self.fold_arm2_right_upper[:, 2], color='orange', alpha=0.4)
            ax.plot_trisurf(self.fold_arm2_right_lower[:, 0], self.fold_arm2_right_lower[:, 1],
                            self.fold_arm2_right_lower[:, 2], color='orange', alpha=0.4)

            ax.plot_trisurf(self.fold_arm3_left_upper[:, 0], self.fold_arm3_left_upper[:, 1],
                            self.fold_arm3_left_upper[:, 2], color='orange', alpha=0.4)
            ax.plot_trisurf(self.fold_arm3_left_lower[:, 0], self.fold_arm3_left_lower[:, 1],
                            self.fold_arm3_left_lower[:, 2], color='orange', alpha=0.4)

            ax.plot_trisurf(self.fold_arm3_right_upper[:, 0], self.fold_arm3_right_upper[:, 1],
                            self.fold_arm3_right_upper[:, 2], color='orange', alpha=0.4)
            ax.plot_trisurf(self.fold_arm3_right_lower[:, 0], self.fold_arm3_right_lower[:, 1],
                            self.fold_arm3_right_lower[:, 2], color='orange', alpha=0.4)

            ax.plot_trisurf(self.arm1['rect1'][:, 0], self.arm1['rect1'][:, 1], self.arm1['rect1'][:, 2],
                            color='orange', alpha=0.4)
            ax.plot_trisurf(self.arm1['rect2'][:, 0], self.arm1['rect2'][:, 1], self.arm1['rect2'][:, 2],
                            color='orange', alpha=0.4)

            ax.plot_trisurf(self.arm2['rect1'][:, 0], self.arm2['rect1'][:, 1], self.arm2['rect1'][:, 2],
                            color='orange', alpha=0.4)
            ax.plot_trisurf(self.arm2['rect2'][:, 0], self.arm2['rect2'][:, 1], self.arm2['rect2'][:, 2],
                            color='orange', alpha=0.4)

            ax.plot_trisurf(self.arm3['rect1'][:, 0], self.arm3['rect1'][:, 1], self.arm3['rect1'][:, 2],
                            color='orange', alpha=0.4)
            ax.plot_trisurf(self.arm3['rect2'][:, 0], self.arm3['rect2'][:, 1], self.arm3['rect2'][:, 2],
                            color='orange', alpha=0.4)



        except RuntimeError as e:
            pass

    def plot_workspace(self, ax, value = 0):
        """Plot the workspace of the robot.

        Args:
            ax (matplotlib.pyplot.Axes): Scatterplot of the workspace, colored by arm on the z-axis.
            value (int) : Adaptive resolution
        """

        ax.scatter(self.ws[::1000-int(9.75*value)][:, 0:2, 0], self.ws[::1000-int(9.75*value)][:, 0:2, 1], self.ws[::1000-int(9.75*value)][:, 0:2, 2],
                   c=self.ws[::1000-int(9.75*value)][:, :2, 2], cmap='PuRd', alpha=0.2)
        ax.scatter(self.ws[::1000-int(9.75*value)][:, 2:4, 0], self.ws[::1000-int(9.75*value)][:, 2:4, 1], self.ws[::1000-int(9.75*value)][:, 2:4, 2],
                   c=self.ws[::1000-int(9.75*value)][:, 2:4, 2], cmap='PuBu', alpha=0.2)
        ax.scatter(self.ws[::1000-int(9.75*value)][:, 4:6, 0], self.ws[::1000-int(9.75*value)][:, 4:6, 1], self.ws[::1000-int(9.75*value)][:, 4:6, 2],
                   c=self.ws[::1000-int(9.75*value)][:, 4:6, 2], cmap='BuGn', alpha=0.2)

    def rotate_arms(self, init=False):
        """Rotate the arms depending on the angle between lower and upper arm.

        Args:
            init (bool, optional): If init = True, the upper surface of the origami has to be computed. Defaults to False.
        """

        # Reset all arms
        self.base = np.matmul(self.init_base, np.identity(3))

        self.arm1['rect1'] = np.matmul(self.init_ref_rect1, np.identity(3))
        self.arm1['rect2'] = np.matmul(self.init_ref_rect2, np.identity(3))

        self.ref_arm2['rect1'] = np.matmul(self.init_ref_rect1, np.identity(3))
        self.ref_arm2['rect2'] = np.matmul(self.init_ref_rect2, np.identity(3))

        self.ref_arm3['rect1'] = np.matmul(self.init_ref_rect1, np.identity(3))
        self.ref_arm3['rect2'] = np.matmul(self.init_ref_rect2, np.identity(3))

        # Rotate lower arms according to the angles set, adjust position of the upper arm to match the the joint between lower and upper arm.

        self.arm1['rect1'] = self.rotate_rect(self.arm1['rect1'], [self.length_base_arms / 2, 0, 0], self.angle1, 'yaw')
        self.arm1['rect2'][:, 2] += self.arm1['rect1'][2, 2] - self.arm1['rect2'][2, 2]
        self.arm1['rect2'][:, 1] += self.arm1['rect1'][2, 1] - self.arm1['rect2'][2, 1]

        self.ref_arm2['rect1'] = self.rotate_rect(self.ref_arm2['rect1'], [self.length_base_arms / 2, 0, 0],
                                                  self.angle2, 'yaw')
        self.ref_arm2['rect2'][:, 2] += self.ref_arm2['rect1'][2, 2] - self.ref_arm2['rect2'][2, 2]
        self.ref_arm2['rect2'][:, 1] += self.ref_arm2['rect1'][2, 1] - self.ref_arm2['rect2'][2, 1]

        self.ref_arm3['rect1'] = self.rotate_rect(self.ref_arm3['rect1'], [self.length_base_arms / 2, 0, 0],
                                                  self.angle3, 'yaw')
        self.ref_arm3['rect2'][:, 2] += self.ref_arm3['rect1'][2, 2] - self.ref_arm3['rect2'][2, 2]
        self.ref_arm3['rect2'][:, 1] += self.ref_arm3['rect1'][2, 1] - self.ref_arm3['rect2'][2, 1]

        self.arm2 = self.generate_arm2()
        self.arm3 = self.generate_arm3()

        # Apply constraints to the arms according to the surface.
        self.constraints(init)

    def translate_module(self):
        """Translate the module according to the self.translate list.
        """
        self.base += self.translate

        self.arm1['rect1'] += self.translate
        self.fold_arm1_left_lower += self.translate
        self.fold_arm1_left_upper += self.translate
        self.fold_arm1_right_lower += self.translate
        self.fold_arm1_right_upper += self.translate
        self.arm1['rect2'] += self.translate

        self.arm2['rect1'] += self.translate
        self.fold_arm2_left_lower += self.translate
        self.fold_arm2_left_upper += self.translate
        self.fold_arm2_right_lower += self.translate
        self.fold_arm2_right_upper += self.translate
        self.arm2['rect2'] += self.translate

        self.arm3['rect1'] += self.translate
        self.fold_arm3_left_lower += self.translate
        self.fold_arm3_left_upper += self.translate
        self.fold_arm3_right_lower += self.translate
        self.fold_arm3_right_upper += self.translate
        self.arm3['rect2'] += self.translate

        self.surf += self.translate

        self.ws = self.ws_init + self.translate

    def rotate_module(self, angle):
        """Rotate module from angle "angle" radians

        Args:
            angle (float): Angle (rad)
        """
        self.base = np.matmul(self.base,
                              [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

        self.arm1['rect1'] = np.matmul(self.arm1['rect1'],
                                       [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
        self.fold_arm1_left_lower = np.matmul(self.fold_arm1_left_lower,
                                              [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                               [0, 0, 1]])
        self.fold_arm1_left_upper = np.matmul(self.fold_arm1_left_upper,
                                              [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                               [0, 0, 1]])
        self.fold_arm1_right_lower = np.matmul(self.fold_arm1_right_lower,
                                               [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                                [0, 0, 1]])
        self.fold_arm1_right_upper = np.matmul(self.fold_arm1_right_upper,
                                               [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                                [0, 0, 1]])
        self.arm1['rect2'] = np.matmul(self.arm1['rect2'],
                                       [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])

        self.arm2['rect1'] = np.matmul(self.arm2['rect1'],
                                       [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
        self.fold_arm2_left_lower = np.matmul(self.fold_arm2_left_lower,
                                              [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                               [0, 0, 1]])
        self.fold_arm2_left_upper = np.matmul(self.fold_arm2_left_upper,
                                              [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                               [0, 0, 1]])
        self.fold_arm2_right_lower = np.matmul(self.fold_arm2_right_lower,
                                               [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                                [0, 0, 1]])
        self.fold_arm2_right_upper = np.matmul(self.fold_arm2_right_upper,
                                               [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                                [0, 0, 1]])
        self.arm2['rect2'] = np.matmul(self.arm2['rect2'],
                                       [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])

        self.arm3['rect1'] = np.matmul(self.arm3['rect1'],
                                       [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
        self.fold_arm3_left_lower = np.matmul(self.fold_arm3_left_lower,
                                              [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                               [0, 0, 1]])
        self.fold_arm3_left_upper = np.matmul(self.fold_arm3_left_upper,
                                              [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                               [0, 0, 1]])
        self.fold_arm3_right_lower = np.matmul(self.fold_arm3_right_lower,
                                               [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                                [0, 0, 1]])
        self.fold_arm3_right_upper = np.matmul(self.fold_arm3_right_upper,
                                               [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                                [0, 0, 1]])
        self.arm3['rect2'] = np.matmul(self.arm3['rect2'],
                                       [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])

        self.surf = np.matmul(self.surf,
                              [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

        # self.ws = self.ws_init + self.translate

    def Rmat(self, angle, direction):
        """Rotation matrix to turn any part of the module to a certain angle. Pitch and Yaw rotations are implemented.

        Args:
            angle (float): Angle determining the rotation in radians
            direction (string): "yaw" or "pitch"

        Returns:
            np.array: 1x3 array containing the rotation matrix.
        """
        if direction == 'yaw':
            return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        elif direction == 'pitch':
            return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rotate_rect(self, rect, center, angle, direction):
        """Rotate part of an arm.

        Args:
            rect (list): Part of an arm to rotate.
            center (list): rotation center.
            angle (float): Angle determining the rotation in radians.
            direction (string): "yaw" or "pitch"

        Returns:
            list: Rotated rectangle.
        """
        rect_adj = rect - center
        rect_adj_rot = np.matmul(rect_adj, self.Rmat(angle, direction))
        return rect_adj_rot + center

    def constraints(self, init):
        """Physical constraints to apply to the module.

        Args:
            init (bool): If True, computation of the initial surface.
        """
        if init:
            def equations_surf(X):
                """Equations of the constraints of the upper surface of the origami.

                Args:
                    X (list): 1x18 list, coordinates of the surface

                Returns:
                    list: 1x18 list, equations result.
                """
                return [
                    (X[0] - X[15]) ** 2 + (X[1] - X[16]) ** 2 + (X[2] - X[17]) ** 2 - (
                                2 * (self.length_upper_arm - self.length_lower_arm) * np.cos(np.radians(30))) ** 2,
                    (X[3] - X[6]) ** 2 + (X[4] - X[7]) ** 2 + (X[5] - X[8]) ** 2 - (
                                2 * (self.length_upper_arm - self.length_lower_arm) * np.cos(np.radians(30))) ** 2,
                    (X[9] - X[12]) ** 2 + (X[10] - X[13]) ** 2 + (X[11] - X[14]) ** 2 - (
                                2 * (self.length_upper_arm - self.length_lower_arm) * np.cos(np.radians(30))) ** 2,
                    (X[0] - X[3]) ** 2 + (X[1] - X[4]) ** 2 + (X[2] - X[5]) ** 2 - self.length_base_arms ** 2,
                    (X[6] - X[9]) ** 2 + (X[7] - X[10]) ** 2 + (X[8] - X[11]) ** 2 - self.length_base_arms ** 2,
                    (X[12] - X[15]) ** 2 + (X[13] - X[16]) ** 2 + (X[14] - X[17]) ** 2 - self.length_base_arms ** 2,
                    (X[0] - self.arm1['rect1'][2, 0]) ** 2 + (X[1] - self.arm1['rect1'][2, 1]) ** 2 + (
                                X[2] - self.arm1['rect1'][2, 2]) ** 2 - (
                                self.length_upper_arm ** 2 + (self.length_base_arms / 2) ** 2),
                    (X[3] - self.arm1['rect1'][2, 0]) ** 2 + (X[4] - self.arm1['rect1'][2, 1]) ** 2 + (
                                X[5] - self.arm1['rect1'][2, 2]) ** 2 - (
                                self.length_upper_arm ** 2 + (self.length_base_arms / 2) ** 2),
                    (X[6] - self.arm2['rect1'][2, 0]) ** 2 + (X[7] - self.arm2['rect1'][2, 1]) ** 2 + (
                                X[8] - self.arm2['rect1'][2, 2]) ** 2 - (
                                self.length_upper_arm ** 2 + (self.length_base_arms / 2) ** 2),
                    (X[9] - self.arm2['rect1'][2, 0]) ** 2 + (X[10] - self.arm2['rect1'][2, 1]) ** 2 + (
                                X[11] - self.arm2['rect1'][2, 2]) ** 2 - (
                                self.length_upper_arm ** 2 + (self.length_base_arms / 2) ** 2),
                    (X[12] - self.arm3['rect1'][2, 0]) ** 2 + (X[13] - self.arm3['rect1'][2, 1]) ** 2 + (
                                X[14] - self.arm3['rect1'][2, 2]) ** 2 - (
                                self.length_upper_arm ** 2 + (self.length_base_arms / 2) ** 2),
                    (X[15] - self.arm3['rect1'][2, 0]) ** 2 + (X[16] - self.arm3['rect1'][2, 1]) ** 2 + (
                                X[17] - self.arm3['rect1'][2, 2]) ** 2 - (
                                self.length_upper_arm ** 2 + (self.length_base_arms / 2) ** 2),
                    self.coplanarity(X[15:], X[3:6], X[9:12], X[:3]),
                    self.coplanarity(X[15:], X[3:6], X[9:12], X[6:9]),
                    self.coplanarity(X[15:], X[3:6], X[9:12], X[12:15]),
                    np.sqrt((X[3] - X[15]) ** 2 + (X[4] - X[16]) ** 2 + (X[5] - X[17]) ** 2) - np.sqrt(
                        (X[3] - X[9]) ** 2 + (X[4] - X[10]) ** 2 + (X[5] - X[11]) ** 2),
                    np.sqrt((X[3] - X[9]) ** 2 + (X[4] - X[10]) ** 2 + (X[5] - X[11]) ** 2) - np.sqrt(
                        (X[15] - X[9]) ** 2 + (X[16] - X[10]) ** 2 + (X[17] - X[11]) ** 2),
                    np.sqrt((X[3] - X[15]) ** 2 + (X[4] - X[16]) ** 2 + (X[5] - X[17]) ** 2) - np.sqrt(
                        (X[15] - X[9]) ** 2 + (X[16] - X[10]) ** 2 + (X[17] - X[11]) ** 2),
                    np.sqrt((X[3] - X[15]) ** 2 + (X[4] - X[16]) ** 2 + (X[5] - X[17]) ** 2) - np.sqrt(
                        self.length_base_arms ** 2 + (
                                    self.length_lower_arm - self.length_upper_arm) ** 2) - self.length_lower_arm + self.length_upper_arm
                ]

            X0 = self.base
            X0[:, 2] += 1
            # Root function from scipy to satisfy equations_surf.
            res_kinematics_surf = root(equations_surf, X0, method='lm')
            print(equations_surf(res_kinematics_surf.x))

            # Initialize the surface.
            self.init_surf = np.array(
                [res_kinematics_surf.x[:3], res_kinematics_surf.x[3:6], res_kinematics_surf.x[6:9],
                 res_kinematics_surf.x[9:12], res_kinematics_surf.x[12:15], res_kinematics_surf.x[15:]])
            self.surf = self.init_surf
            self.surf[:, 2] += 10

        # Arm kinematics

        self.arm1['rect2'][1] = self.surf[0]
        self.arm1['rect2'][0] = self.surf[1]

        self.arm2['rect2'][1] = self.surf[2]
        self.arm2['rect2'][0] = self.surf[3]

        self.arm3['rect2'][1] = self.surf[4]
        self.arm3['rect2'][0] = self.surf[5]

        def equations_arm1(X):
            """Equations of the constraints of the first arm of the origami.

            Args:
                X (list): 1x6 list, coordinates of the first arm

            Returns:
                list: 1x6 list, equations result.
            """
            return [
                self.coplanarity(self.arm1['rect1'][2], self.arm1['rect2'][0], self.arm1['rect2'][1], X[:3]),
                self.coplanarity(self.arm1['rect1'][2], self.arm1['rect2'][0], self.arm1['rect2'][1], X[3:6]),
                (X[0] - X[3]) ** 2 + (X[1] - X[4]) ** 2 + (X[2] - X[5]) ** 2 - self.length_base_arms ** 2,
                (X[0] - self.arm1['rect2'][1, 0]) ** 2 + (X[1] - self.arm1['rect2'][1, 1]) ** 2 + (
                            X[2] - self.arm1['rect2'][1, 2]) ** 2 - (
                            self.length_upper_arm - self.length_base_arms / 2) ** 2,
                (X[0] - self.arm1['rect2'][2, 0]) ** 2 + (X[1] - self.arm1['rect2'][2, 1]) ** 2 + (
                            X[2] - self.arm1['rect2'][2, 2]) ** 2 - 2 * (self.length_base_arms / 2) ** 2,
                (X[3] - self.arm1['rect2'][0, 0]) ** 2 + (X[4] - self.arm1['rect2'][0, 1]) ** 2 + (
                            X[5] - self.arm1['rect2'][0, 2]) ** 2 - (
                            self.length_upper_arm - self.length_base_arms / 2) ** 2,
                (X[3] - self.arm1['rect2'][2, 0]) ** 2 + (X[4] - self.arm1['rect2'][2, 1]) ** 2 + (
                            X[5] - self.arm1['rect2'][2, 2]) ** 2 - 2 * (self.length_base_arms / 2) ** 2,
            ]

        X0_arm = [-10, -10, -10, -10, -10, -20]

        # If a solution already exists from a previous surface, we use the last approximation to compute the next one to fasten the computation.

        if self.res_kinematics_arm1 == None:
            self.res_kinematics_arm1 = root(equations_arm1, X0_arm, method='lm')
        else:
            self.res_kinematics_arm1 = root(equations_arm1, self.res_kinematics_arm1.x, method='lm')

        self.arm1['rect2'][4] = self.res_kinematics_arm1.x[0:3]
        self.arm1['rect2'][3] = self.res_kinematics_arm1.x[3:]

        def equations_arm2(X):
            """Equations of the constraints of the second arm of the origami.

            Args:
                X (list): 1x6 list, coordinates of the second arm

            Returns:
                list: 1x6 list, equations result
            """
            return [
                self.coplanarity(self.arm2['rect1'][2], self.arm2['rect2'][0], self.arm2['rect2'][1], X[:3]),
                self.coplanarity(self.arm2['rect1'][2], self.arm2['rect2'][0], self.arm2['rect2'][1], X[3:6]),
                (X[0] - X[3]) ** 2 + (X[1] - X[4]) ** 2 + (X[2] - X[5]) ** 2 - self.length_surface_arms ** 2,
                (X[0] - self.arm2['rect2'][1, 0]) ** 2 + (X[1] - self.arm2['rect2'][1, 1]) ** 2 + (
                            X[2] - self.arm2['rect2'][1, 2]) ** 2 - (
                            self.length_upper_arm - self.length_surface_arms / 2) ** 2,
                (X[0] - self.arm2['rect2'][2, 0]) ** 2 + (X[1] - self.arm2['rect2'][2, 1]) ** 2 + (
                            X[2] - self.arm2['rect2'][2, 2]) ** 2 - 2 * (self.length_surface_arms / 2) ** 2,
                (X[3] - self.arm2['rect2'][0, 0]) ** 2 + (X[4] - self.arm2['rect2'][0, 1]) ** 2 + (
                            X[5] - self.arm2['rect2'][0, 2]) ** 2 - (
                            self.length_upper_arm - self.length_surface_arms / 2) ** 2,
                (X[3] - self.arm2['rect2'][2, 0]) ** 2 + (X[4] - self.arm2['rect2'][2, 1]) ** 2 + (
                            X[5] - self.arm2['rect2'][2, 2]) ** 2 - 2 * (self.length_surface_arms / 2) ** 2,
            ]

        # If a solution already exists from a previous surface, we use the last approximation to compute the next one to fasten the computation.

        if self.res_kinematics_arm2 == None:
            self.res_kinematics_arm2 = root(equations_arm2, X0_arm, method='lm')
        else:
            self.res_kinematics_arm2 = root(equations_arm2, self.res_kinematics_arm2.x, method='lm')

        self.arm2['rect2'][4] = self.res_kinematics_arm2.x[0:3]
        self.arm2['rect2'][3] = self.res_kinematics_arm2.x[3:]

        def equations_arm3(X):
            """Equations of the constraints of the third arm of the origami.

            Args:
                X (list): 1x6 list, coordinates of the third arm

            Returns:
                list: 1x6 list, equations result
            """
            return [
                self.coplanarity(self.arm3['rect1'][2], self.arm3['rect2'][0], self.arm3['rect2'][1], X[:3]),
                self.coplanarity(self.arm3['rect1'][2], self.arm3['rect2'][0], self.arm3['rect2'][1], X[3:6]),
                (X[0] - X[3]) ** 2 + (X[1] - X[4]) ** 2 + (X[2] - X[5]) ** 2 - self.length_surface_arms ** 2,
                (X[0] - self.arm3['rect2'][1, 0]) ** 2 + (X[1] - self.arm3['rect2'][1, 1]) ** 2 + (
                            X[2] - self.arm3['rect2'][1, 2]) ** 2 - (
                            self.length_upper_arm - self.length_surface_arms / 2) ** 2,
                (X[0] - self.arm3['rect2'][2, 0]) ** 2 + (X[1] - self.arm3['rect2'][2, 1]) ** 2 + (
                            X[2] - self.arm3['rect2'][2, 2]) ** 2 - 2 * (self.length_surface_arms / 2) ** 2,
                (X[3] - self.arm3['rect2'][0, 0]) ** 2 + (X[4] - self.arm3['rect2'][0, 1]) ** 2 + (
                            X[5] - self.arm3['rect2'][0, 2]) ** 2 - (
                            self.length_upper_arm - self.length_surface_arms / 2) ** 2,
                (X[3] - self.arm3['rect2'][2, 0]) ** 2 + (X[4] - self.arm3['rect2'][2, 1]) ** 2 + (
                            X[5] - self.arm3['rect2'][2, 2]) ** 2 - 2 * (self.length_surface_arms / 2) ** 2,
            ]

        # If a solution already exists from a previous surface, we use the last approximation to compute the next one to fasten the computation.

        if self.res_kinematics_arm3 == None:
            self.res_kinematics_arm3 = root(equations_arm3, X0_arm, method='lm')
        else:
            self.res_kinematics_arm3 = root(equations_arm3, self.res_kinematics_arm3.x, method='lm')

        self.arm3['rect2'][4] = self.res_kinematics_arm3.x[0:3]
        self.arm3['rect2'][3] = self.res_kinematics_arm3.x[3:]

        def equations_fold_arm1_left(X):
            """Equation of the fold between upper and lower arm of first arm, left side

            Args:
                X (list): 1x3 point determining the point of the fold

            Returns:
                list: 1x3 equations result.
            """
            return [
                np.sqrt((X[0] - self.arm1['rect1'][3, 0]) ** 2 + (X[1] - self.arm1['rect1'][3, 1]) ** 2 + (
                            X[2] - self.arm1['rect1'][3, 2]) ** 2) - self.length_surface_arms / 2,
                np.sqrt((X[0] - self.arm1['rect2'][4, 0]) ** 2 + (X[1] - self.arm1['rect2'][4, 1]) ** 2 + (
                            X[2] - self.arm1['rect2'][4, 2]) ** 2) - self.length_surface_arms / 2,
                np.sqrt((X[0] - self.arm1['rect2'][2, 0]) ** 2 + (X[1] - self.arm1['rect2'][2, 1]) ** 2 + (
                            X[2] - self.arm1['rect2'][2, 2]) ** 2) - self.length_surface_arms / 2,
            ]

        X0_fold = np.ones(3)
        self.fold_arm1_left = root(equations_fold_arm1_left, X0_fold, method='lm').x
        self.fold_arm1_left_lower = np.array(
            [list(self.arm1['rect1'][3]), list(self.arm1['rect2'][2]), self.fold_arm1_left])
        self.fold_arm1_left_upper = np.array(
            [list(self.arm1['rect2'][4]), list(self.arm1['rect2'][2]), self.fold_arm1_left])

        def equations_fold_arm1_right(X):
            """Equation of the fold between upper and lower arm of first arm, right side

            Args:
                X (list): 1x3 point determining the point of the fold

            Returns:
                list: 1x3 equations result.
            """
            return [
                np.sqrt((X[0] - self.arm1['rect1'][4, 0]) ** 2 + (X[1] - self.arm1['rect1'][4, 1]) ** 2 + (
                            X[2] - self.arm1['rect1'][4, 2]) ** 2) - self.length_surface_arms / 2,
                np.sqrt((X[0] - self.arm1['rect2'][3, 0]) ** 2 + (X[1] - self.arm1['rect2'][3, 1]) ** 2 + (
                            X[2] - self.arm1['rect2'][3, 2]) ** 2) - self.length_surface_arms / 2,
                np.sqrt((X[0] - self.arm1['rect2'][2, 0]) ** 2 + (X[1] - self.arm1['rect2'][2, 1]) ** 2 + (
                            X[2] - self.arm1['rect2'][2, 2]) ** 2) - self.length_surface_arms / 2,
            ]

        self.fold_arm1_right = root(equations_fold_arm1_right, X0_fold, method='lm').x
        self.fold_arm1_right_lower = np.array(
            [list(self.arm1['rect1'][4]), list(self.arm1['rect2'][2]), self.fold_arm1_right])
        self.fold_arm1_right_upper = np.array(
            [list(self.arm1['rect2'][3]), list(self.arm1['rect2'][2]), self.fold_arm1_right])

        def equations_fold_arm2_left(X):
            """Equation of the fold between upper and lower arm of second arm, left side

            Args:
                X (list): 1x3 point determining the point of the fold

            Returns:
                list: 1x3 equations result.
            """
            return [
                np.sqrt((X[0] - self.arm2['rect1'][3, 0]) ** 2 + (X[1] - self.arm2['rect1'][3, 1]) ** 2 + (
                            X[2] - self.arm2['rect1'][3, 2]) ** 2) - self.length_surface_arms / 2,
                np.sqrt((X[0] - self.arm2['rect2'][4, 0]) ** 2 + (X[1] - self.arm2['rect2'][4, 1]) ** 2 + (
                            X[2] - self.arm2['rect2'][4, 2]) ** 2) - self.length_surface_arms / 2,
                np.sqrt((X[0] - self.arm2['rect2'][2, 0]) ** 2 + (X[1] - self.arm2['rect2'][2, 1]) ** 2 + (
                            X[2] - self.arm2['rect2'][2, 2]) ** 2) - self.length_surface_arms / 2,
            ]

        self.fold_arm2_left = root(equations_fold_arm2_left, X0_fold, method='lm').x
        self.fold_arm2_left_lower = np.array(
            [list(self.arm2['rect1'][3]), list(self.arm2['rect2'][2]), self.fold_arm2_left])
        self.fold_arm2_left_upper = np.array(
            [list(self.arm2['rect2'][4]), list(self.arm2['rect2'][2]), self.fold_arm2_left])

        def equations_fold_arm2_right(X):
            """Equation of the fold between upper and lower arm of second arm, right side

            Args:
                X (list): 1x3 point determining the point of the fold

            Returns:
                list: 1x3 equations result.
            """
            return [
                np.sqrt((X[0] - self.arm2['rect1'][4, 0]) ** 2 + (X[1] - self.arm2['rect1'][4, 1]) ** 2 + (
                            X[2] - self.arm2['rect1'][4, 2]) ** 2) - self.length_surface_arms / 2,
                np.sqrt((X[0] - self.arm2['rect2'][3, 0]) ** 2 + (X[1] - self.arm2['rect2'][3, 1]) ** 2 + (
                            X[2] - self.arm2['rect2'][3, 2]) ** 2) - self.length_surface_arms / 2,
                np.sqrt((X[0] - self.arm2['rect2'][2, 0]) ** 2 + (X[1] - self.arm2['rect2'][2, 1]) ** 2 + (
                            X[2] - self.arm2['rect2'][2, 2]) ** 2) - self.length_surface_arms / 2,
            ]

        self.fold_arm2_right = root(equations_fold_arm2_right, X0_fold, method='lm').x
        self.fold_arm2_right_lower = np.array(
            [list(self.arm2['rect1'][4]), list(self.arm2['rect2'][2]), self.fold_arm2_right])
        self.fold_arm2_right_upper = np.array(
            [list(self.arm2['rect2'][3]), list(self.arm2['rect2'][2]), self.fold_arm2_right])

        def equations_fold_arm3_left(X):
            """Equation of the fold between upper and lower arm of third arm, left side

            Args:
                X (list): 1x3 point determining the point of the fold

            Returns:
                list: 1x3 equations result.
            """
            return [
                np.sqrt((X[0] - self.arm3['rect1'][3, 0]) ** 2 + (X[1] - self.arm3['rect1'][3, 1]) ** 2 + (
                            X[2] - self.arm3['rect1'][3, 2]) ** 2) - self.length_base_arms / 2,
                np.sqrt((X[0] - self.arm3['rect2'][4, 0]) ** 2 + (X[1] - self.arm3['rect2'][4, 1]) ** 2 + (
                            X[2] - self.arm3['rect2'][4, 2]) ** 2) - self.length_base_arms / 2,
                np.sqrt((X[0] - self.arm3['rect2'][2, 0]) ** 2 + (X[1] - self.arm3['rect2'][2, 1]) ** 2 + (
                            X[2] - self.arm3['rect2'][2, 2]) ** 2) - self.length_base_arms / 2,
            ]

        self.fold_arm3_left = root(equations_fold_arm3_left, X0_fold, method='lm').x
        self.fold_arm3_left_lower = np.array(
            [list(self.arm3['rect1'][3]), list(self.arm3['rect2'][2]), self.fold_arm3_left])
        self.fold_arm3_left_upper = np.array(
            [list(self.arm3['rect2'][4]), list(self.arm3['rect2'][2]), self.fold_arm3_left])

        def equations_fold_arm3_right(X):
            """Equation of the fold between upper and lower arm of third arm, right side

            Args:
                X (list): 1x3 point determining the point of the fold

            Returns:
                list: 1x3 equations result.
            """
            return [
                np.sqrt((X[0] - self.arm3['rect1'][4, 0]) ** 2 + (X[1] - self.arm3['rect1'][4, 1]) ** 2 + (
                            X[2] - self.arm3['rect1'][4, 2]) ** 2) - self.length_base_arms / 2,
                np.sqrt((X[0] - self.arm3['rect2'][3, 0]) ** 2 + (X[1] - self.arm3['rect2'][3, 1]) ** 2 + (
                            X[2] - self.arm3['rect2'][3, 2]) ** 2) - self.length_base_arms / 2,
                np.sqrt((X[0] - self.arm3['rect2'][2, 0]) ** 2 + (X[1] - self.arm3['rect2'][2, 1]) ** 2 + (
                            X[2] - self.arm3['rect2'][2, 2]) ** 2) - self.length_base_arms / 2,
            ]

        self.fold_arm3_right = root(equations_fold_arm3_right, X0_fold, method='lm').x
        self.fold_arm3_right_lower = np.array(
            [list(self.arm3['rect1'][4]), list(self.arm3['rect2'][2]), self.fold_arm3_right])
        self.fold_arm3_right_upper = np.array(
            [list(self.arm3['rect2'][3]), list(self.arm3['rect2'][2]), self.fold_arm3_right])

    def coplanarity(self, P1, P2, P3, P4):
        """Check the coplanarity between point 4 (P4) w.r.t the plane composed of points 1,2 and 3 (P1,P2,P3).

        Args:
            P1 (list): 1x3, coordinates x,y,z
            P2 (list): 1x3, coordinates x,y,z
            P3 (list): 1x3, coordinates x,y,z
            P4 (list): 1x3, coordinates x,y,z

        Returns:
            float: if it is 0, points is coplanar.
        """
        a1 = P2[0] - P1[0]
        b1 = P2[1] - P1[1]
        c1 = P2[2] - P1[2]

        a2 = P3[0] - P1[0]
        b2 = P3[1] - P1[1]
        c2 = P3[2] - P1[2]

        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = (- a * P1[0] - b * P1[1] - c * P1[2])

        return a * P4[0] + b * P4[1] + c * P4[2] + d

    def inverse_kinematics(self, R):
        """Inverse kinematics of a module w.r.t the rotation of the surface determined by the rotation matrix R

        Args:
            R (list): 1x9, rotation matrix
        """

        # Norm for each row - Arcsin definition
        norm_row1 = np.linalg.norm(R[0:3])
        norm_row2 = np.linalg.norm(R[3:6])
        norm_row3 = np.linalg.norm(R[6:])

        # Matrix multiplication of the rotation matrix to change the surface orientation.
        self.surf = np.matmul(self.init_surf, [R[0:3] / norm_row1, R[3:6] / norm_row2, R[6:] / norm_row3])

        # Midpoints computations to constitute planes to compute the inverse kinematics.
        midpoint_arm1_surf = [(self.surf[0, 0] + self.surf[1, 0]) / 2, (self.surf[0, 1] + self.surf[1, 1]) / 2,
                              (self.surf[0, 2] + self.surf[1, 2]) / 2]
        midpoint_arm2_surf = [(self.surf[2, 0] + self.surf[3, 0]) / 2, (self.surf[2, 1] + self.surf[3, 1]) / 2,
                              (self.surf[2, 2] + self.surf[3, 2]) / 2]
        midpoint_arm3_surf = [(self.surf[4, 0] + self.surf[5, 0]) / 2, (self.surf[4, 1] + self.surf[5, 1]) / 2,
                              (self.surf[4, 2] + self.surf[5, 2]) / 2]
        midpoint_arm1_base = [(self.base[0, 0] + self.base[1, 0]) / 2, (self.base[0, 1] + self.base[1, 1]) / 2,
                              (self.base[0, 2] + self.base[1, 2]) / 2]
        midpoint_arm2_base = [(self.base[2, 0] + self.base[3, 0]) / 2, (self.base[2, 1] + self.base[3, 1]) / 2,
                              (self.base[2, 2] + self.base[3, 2]) / 2]
        midpoint_arm3_base = [(self.base[4, 0] + self.base[5, 0]) / 2, (self.base[4, 1] + self.base[5, 1]) / 2,
                              (self.base[4, 2] + self.base[5, 2]) / 2]

        def equations_arm1(X):
            """Equations for arm 1 to retrieve the arm junction

            Args:
                X (list): 1x3, arm junction coordinates

            Returns:
                list: equations result.
            """
            return [
                (X[0] - midpoint_arm1_surf[0]) ** 2 + (X[1] - midpoint_arm1_surf[1]) ** 2 + (
                            X[2] - midpoint_arm1_surf[2]) ** 2 - self.length_upper_arm ** 2,
                (X[0] - midpoint_arm1_base[0]) ** 2 + (X[1] - midpoint_arm1_base[1]) ** 2 + (
                            X[2] - midpoint_arm1_base[2]) ** 2 - self.length_lower_arm ** 2,
                (X[0] - self.surf[0, 0]) ** 2 + (X[1] - self.surf[0, 1]) ** 2 + (
                            X[2] - self.surf[0, 2]) ** 2 - self.length_upper_arm ** 2 - (
                            self.length_surface_arms / 2) ** 2,
                (X[0] - self.surf[1, 0]) ** 2 + (X[1] - self.surf[1, 1]) ** 2 + (
                            X[2] - self.surf[1, 2]) ** 2 - self.length_upper_arm ** 2 - (
                            self.length_surface_arms / 2) ** 2,
                (X[0] - self.base[0, 0]) ** 2 + (X[1] - self.base[0, 1]) ** 2 + (
                            X[2] - self.base[0, 2]) ** 2 - self.length_lower_arm ** 2 - (
                            self.length_base_arms / 2) ** 2,
                (X[0] - self.base[1, 0]) ** 2 + (X[1] - self.base[1, 1]) ** 2 + (
                            X[2] - self.base[1, 2]) ** 2 - self.length_lower_arm ** 2 - (self.length_base_arms / 2) ** 2
            ]

        X0_arm1 = self.arm1['rect1'][0]
        X0_arm1[2] -= 1
        self.arm1_junction = root(equations_arm1, X0_arm1, method='lm').x

        # Computation of the angle of the first arm.
        if self.arm1_junction[2] / self.length_lower_arm > 1:
            alpha_arm1 = np.pi / 2
        else:
            alpha_arm1 = np.arcsin(self.arm1_junction[2] / self.length_lower_arm)

        def equations_arm2(X):
            """Equations for arm 2 to retrieve the arm junction

            Args:
                X (list): 1x3, arm junction coordinates

            Returns:
                list: equations result.
            """
            return [
                (X[0] - midpoint_arm2_surf[0]) ** 2 + (X[1] - midpoint_arm2_surf[1]) ** 2 + (
                            X[2] - midpoint_arm2_surf[2]) ** 2 - self.length_upper_arm ** 2,
                (X[0] - midpoint_arm2_base[0]) ** 2 + (X[1] - midpoint_arm2_base[1]) ** 2 + (
                            X[2] - midpoint_arm2_base[2]) ** 2 - self.length_lower_arm ** 2,
                (X[0] - self.surf[2, 0]) ** 2 + (X[1] - self.surf[2, 1]) ** 2 + (
                            X[2] - self.surf[2, 2]) ** 2 - self.length_upper_arm ** 2 - (
                            self.length_surface_arms / 2) ** 2,
                (X[0] - self.surf[3, 0]) ** 2 + (X[1] - self.surf[3, 1]) ** 2 + (
                            X[2] - self.surf[3, 2]) ** 2 - self.length_upper_arm ** 2 - (
                            self.length_surface_arms / 2) ** 2,
                (X[0] - self.base[2, 0]) ** 2 + (X[1] - self.base[2, 1]) ** 2 + (
                            X[2] - self.base[2, 2]) ** 2 - self.length_lower_arm ** 2 - (
                            self.length_base_arms / 2) ** 2,
                (X[0] - self.base[3, 0]) ** 2 + (X[1] - self.base[3, 1]) ** 2 + (
                            X[2] - self.base[3, 2]) ** 2 - self.length_lower_arm ** 2 - (self.length_base_arms / 2) ** 2
            ]

        X0_arm2 = self.arm2['rect1'][0]
        X0_arm2[2] -= 1
        self.arm2_junction = root(equations_arm2, X0_arm2, method='lm').x

        # Computation of the angle of the second arm.
        if self.arm2_junction[2] / self.length_lower_arm > 1:
            alpha_arm2 = np.pi / 2
        else:
            alpha_arm2 = np.arcsin(self.arm2_junction[2] / self.length_lower_arm)

        def equations_arm3(X):
            """Equations for arm 3 to retrieve the arm junction

            Args:
                X (list): 1x3, arm junction coordinates

            Returns:
                list: equations result.
            """
            return [
                (X[0] - midpoint_arm3_surf[0]) ** 2 + (X[1] - midpoint_arm3_surf[1]) ** 2 + (
                            X[2] - midpoint_arm3_surf[2]) ** 2 - self.length_upper_arm ** 2,
                (X[0] - midpoint_arm3_base[0]) ** 2 + (X[1] - midpoint_arm3_base[1]) ** 2 + (
                            X[2] - midpoint_arm3_base[2]) ** 2 - self.length_lower_arm ** 2,
                (X[0] - self.surf[4, 0]) ** 2 + (X[1] - self.surf[4, 1]) ** 2 + (
                            X[2] - self.surf[4, 2]) ** 2 - self.length_upper_arm ** 2 - (
                            self.length_surface_arms / 2) ** 2,
                (X[0] - self.surf[5, 0]) ** 2 + (X[1] - self.surf[5, 1]) ** 2 + (
                            X[2] - self.surf[5, 2]) ** 2 - self.length_upper_arm ** 2 - (
                            self.length_surface_arms / 2) ** 2,
                (X[0] - self.base[4, 0]) ** 2 + (X[1] - self.base[4, 1]) ** 2 + (
                            X[2] - self.base[4, 2]) ** 2 - self.length_lower_arm ** 2 - (
                            self.length_base_arms / 2) ** 2,
                (X[0] - self.base[5, 0]) ** 2 + (X[1] - self.base[5, 1]) ** 2 + (
                            X[2] - self.base[5, 2]) ** 2 - self.length_lower_arm ** 2 - (self.length_base_arms / 2) ** 2
            ]

        X0_arm3 = self.arm3['rect2'][0]
        X0_arm3[2] -= 1
        self.arm3_junction = root(equations_arm3, X0_arm3, method='lm').x

        # Computation of the angle of the third arm.
        if self.arm3_junction[2] / self.length_lower_arm > 1:
            alpha_arm3 = np.pi / 2
        else:
            alpha_arm3 = np.arcsin(self.arm3_junction[2] / self.length_lower_arm)

        self.set_angles(alpha_arm1, alpha_arm2, alpha_arm3)
        self.rotate_arms()
