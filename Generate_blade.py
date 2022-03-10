from typing import Dict, List, Any, Optional
import argparse
from unittest import result
import yaml
import numpy as np
from scipy.interpolate import PchipInterpolator as spline
import windio2cad.geometry_tools as geom
import solid
import subprocess
from numpy.linalg import norm
from math import sin, cos
import matplotlib.pyplot as plt


class Blade:
    """
    This class renders one blade for the rotor.
    """

    def __init__(self, yaml_filename: str):
        """
        The constructor opens the YAML file and extracts the blade
        and airfoil information into instance attributes.

        Parameters
        ----------
        yaml_filename: str
            Filename that contains the geometry for the rotor.
        """
        geometry = yaml.load(open(yaml_filename, "r"), yaml.FullLoader)
        self.outer_shape = geometry["components"]["blade"]["outer_shape_bem"]
        self.airfoils = geometry["airfoils"]

    @staticmethod
    def myinterp(xi, x, f) -> np.array:
        # print(x, f)
        print(len(x), len(f))
        myspline = spline(x, f)
        return myspline(xi)


    def generate_lofted(self, n_span_min=10, n_xy=8) -> np.array:
        """
        Creates the lofted shape of a blade and returns a NumPy array
        of the polygons at each cross section.

        Parameters
        ----------
        n_span_min: int
            Number of cross sections to create across span of
            blade.

        n_xy: int
            The number of x, y points in the polygons at each slice of
            the blade.

        Returns
        -------
        np.array
            An array of the polygons at each cross section of the blade.
        """
        # Use yaml grid points and others that we add
        r_span = np.unique(
            np.r_[
                np.linspace(0.0, 1.0, n_span_min),
                self.outer_shape["chord"]["grid"],
                self.outer_shape["twist"]["grid"],
                self.outer_shape["pitch_axis"]["grid"],
                self.outer_shape["reference_axis"]["x"]["grid"],
                self.outer_shape["reference_axis"]["y"]["grid"],
                self.outer_shape["reference_axis"]["z"]["grid"],
            ]
        )
        n_span = len(r_span)

        # print(r_span)
        

        # Read in blade spanwise geometry values and put on common grid
        chord = self.myinterp(
            r_span,
            self.outer_shape["chord"]["grid"],
            self.outer_shape["chord"]["values"],
        )
        twist = self.myinterp(
            r_span,
            self.outer_shape["twist"]["grid"],
            self.outer_shape["twist"]["values"],
        )
        pitch_axis = self.myinterp(
            r_span,
            self.outer_shape["pitch_axis"]["grid"],
            self.outer_shape["pitch_axis"]["values"],
        )
        ref_axis = np.c_[
            self.myinterp(
                r_span,
                self.outer_shape["reference_axis"]["x"]["grid"],
                self.outer_shape["reference_axis"]["x"]["values"],
            ),
            self.myinterp(
                r_span,
                self.outer_shape["reference_axis"]["y"]["grid"],
                self.outer_shape["reference_axis"]["y"]["values"],
            ),
            self.myinterp(
                r_span,
                self.outer_shape["reference_axis"]["z"]["grid"],
                self.outer_shape["reference_axis"]["z"]["values"],
            ),
        ]

        grid_index = []

        for i, grid in enumerate(self.outer_shape['chord']['grid']):
            for af_pos in self.outer_shape["airfoil_position"]["grid"]:
                if grid == af_pos:
                    grid_index.append(i)

        # Get airfoil names and thicknesses
        af_position = self.outer_shape["airfoil_position"]["grid"]
        af_used = self.outer_shape["airfoil_position"]["labels"]
        n_af_span = len(af_position)
        n_af = len(self.airfoils)
        name = n_af * [""]
        r_thick = np.zeros(n_af)
        for i in range(n_af):
            name[i] = self.airfoils[i]["name"]
            r_thick[i] = self.airfoils[i]["relative_thickness"]

        with plt.style.context('default'):
            fig, ax1 = plt.subplots()

            # Create common airfoil coordinates grid
            coord_xy = np.zeros((n_af, n_xy, 2))
            for i in range(n_af):
                points = np.c_[
                    self.airfoils[i]["coordinates"]["x"],
                    self.airfoils[i]["coordinates"]["y"],
                ]

                # Check that airfoil points are declared from the TE suction side to TE pressure side
                idx_le = np.argmin(points[:, 0])
                if np.mean(points[:idx_le, 1]) > 0.0:
                    print('flip')
                    print(af_used[i])
                    points = np.flip(points, axis=0)
                # if i == 2:
                #     ax1.plot(points[:,0], points[:, 1])
                #     ax1.set_xlabel('x')
                #     ax1.set_ylabel('y')
                
                # Remap points using class AirfoilShape
                af = geom.AirfoilShape(points=points)
                af.redistribute(n_xy, even=False, dLE=True)
                af_points = af.points

                # Add trailing edge point if not defined
                if [1, 0] not in af_points.tolist():
                    af_points[:, 0] -= af_points[np.argmin(af_points[:, 0]), 0]
                c = max(af_points[:, 0]) - min(af_points[:, 0])
                af_points[:, :] /= c

                coord_xy[i, :, :] = af_points

                # if 0 < i < 4:
                #     ax1.plot(coord_xy[i,:,0], coord_xy[i, :,1])
                #     ax1.set_xlabel('x')
                #     ax1.set_ylabel('y')

                # if 25 > ref_axis[i, 2] > 24:
                #     ax1.plot(coord_xy[i,:,0], coord_xy[i, :,1])
                #     ax1.set_xlabel('x')
                #     ax1.set_ylabel('y')

            # Reconstruct the blade relative thickness along span with a pchip
            r_thick_used = np.zeros(n_af_span)
            coord_xy_used = np.zeros((n_af_span, n_xy, 2))
            coord_xy_interp = np.zeros((n_span, n_xy, 2))
            coord_xy_dim = np.zeros((n_span, n_xy, 2))

            for i in range(n_af_span):
                for j in range(n_af):
                    if af_used[i] == name[j]:
                        r_thick_used[i] = r_thick[j]
                        coord_xy_used[i, :, :] = coord_xy[j, :, :]
                
                # if 1 < i < 4:
                ax1.plot(coord_xy_used[i, :, 0], coord_xy_used[i, :, 1], '.')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')

            r_thick_interp = self.myinterp(r_span, af_position, r_thick_used)

            # ax1.plot(r_span, r_thick_interp)

            # Spanwise interpolation of the profile coordinates with a pchip - this is where the kink appears
            r_thick_unique, indices = np.unique(r_thick_used, return_index=True)

            print(r_thick_interp.shape, r_thick_unique.shape, coord_xy_used[indices, :, :].shape)
            for i in range(n_xy):
                for j in range(2):
                    coord_xy_interp[:, i, j] = np.flip(
                        self.myinterp(
                            np.flip(r_thick_interp), r_thick_unique, coord_xy_used[indices, i, j]
                        ),
                        axis=0,
                    )
            for i in range(n_span):
                if 25 > ref_axis[i, 2] > 24:
                    ax1.plot(coord_xy_interp[i,:,0], coord_xy_interp[i,:,1])
                    ax1.set_xlabel('x')
                    ax1.set_ylabel('y')
                # Correction to move the leading edge (min x point) to (0,0)
                af_le = coord_xy_interp[i, np.argmin(coord_xy_interp[i, :, 0]), :]
                coord_xy_interp[i, :, 0] -= af_le[0]
                coord_xy_interp[i, :, 1] -= af_le[1]
                c = max(coord_xy_interp[i, :, 0]) - min(coord_xy_interp[i, :, 0])
                coord_xy_interp[i, :, :] /= c
                # If the rel thickness is smaller than 0.4 apply a trailing ege smoothing step
                if r_thick_interp[i] < 0.4:
                    coord_xy_interp[i, :, :] = geom.trailing_edge_smoothing(
                        coord_xy_interp[i, :, :]
                    )
                
                # ax1.plot(coord_xy_interp[i,:,0], coord_xy_interp[i, :,1])
                # ax1.set_xlabel('x')
                # ax1.set_ylabel('y')

            for i in range(n_af_span):
                # Correction to move the leading edge (min x point) to (0,0)
                af_le = coord_xy_used[i, np.argmin(coord_xy_used[i, :, 0]), :]
                coord_xy_used[i, :, 0] -= af_le[0]
                coord_xy_used[i, :, 1] -= af_le[1]
                c = max(coord_xy_used[i, :, 0]) - min(coord_xy_used[i, :, 0])
                coord_xy_used[i, :, :] /= c
                # If the rel thickness is smaller than 0.4 apply a trailing ege smoothing step
                if r_thick_used[i] < 0.4:
                    coord_xy_used[i, :, :] = geom.trailing_edge_smoothing(
                        coord_xy_used[i, :, :]
                    )
                
                # ax1.plot(coord_xy_interp[i,:,0], coord_xy_interp[i, :,1])
                # ax1.set_xlabel('x')
                # ax1.set_ylabel('y')

            # Offset by pitch axis and scale for chord
            coord_xy_dim = coord_xy_interp.copy()
            coord_xy_dim[:, :, 0] -= pitch_axis[:, np.newaxis]
            coord_xy_dim = coord_xy_dim * chord[:, np.newaxis, np.newaxis]

            pitch_used = [self.outer_shape['pitch_axis']['values'][i] for i in grid_index]

            coord_xy_dim_used = coord_xy_used.copy()
            coord_xy_dim_used[:, :, 0] -= pitch_used[:, np.newaxis]
            coord_xy_dim_used = coord_xy_dim * chord[:, np.newaxis, np.newaxis]


            # Rotate to twist angle
            coord_xy_dim_twisted = np.zeros(coord_xy_interp.shape)
            for i in range(n_span):
                # ax1.plot(coord_xy_dim[i,:,0], coord_xy_dim[i, :,1])
                # ax1.set_xlabel('x')
                # ax1.set_ylabel('y')
                x = coord_xy_dim[i, :, 0]
                y = coord_xy_dim[i, :, 1]
                coord_xy_dim_twisted[i, :, 0] = x * np.cos(twist[i]) - y * np.sin(twist[i])
                coord_xy_dim_twisted[i, :, 1] = y * np.cos(twist[i]) + x * np.sin(twist[i])

                # ax1.plot(coord_xy_dim_twisted[i,:,0], coord_xy_dim_twisted[i, :,1])
                # ax1.set_xlabel('x')
                # ax1.set_ylabel('y')

            # Assemble lofted shape along reference axis
            lofted_shape = np.zeros((n_span, n_xy, 3))
            for i in range(n_span):
                for j in range(n_xy):
                    lofted_shape[i, j, :] = (
                        np.r_[
                            coord_xy_dim_twisted[i, j, 1],
                            coord_xy_dim_twisted[i, j, 0],
                            0.0,
                        ]
                        + ref_axis[i, :]
                    )
                # if 25 > ref_axis[i, 2] > 24:
                #     ax1.plot(lofted_shape[i,:,0], lofted_shape[i, :,1])
                #     ax1.set_xlabel('x')
                #     ax1.set_ylabel('y')

            fig.savefig('aerofoils.png', format='png')

            return lofted_shape

    def blade_hull(self, downsample_z: int = 1) -> solid.OpenSCADObject:
        """
        This creates an OpenSCAD hull object around cross sections of a blade,
        thereby rendering the complete geometry for a single blade.

        Parameters
        ----------
        downsample_z: int
            Skips to every nth sample across the z axis of the blade. For
            example, 10 uses only every tenth cross section.

        Returns
        -------
        solid.OpenSCADObject
            The OpenSCAD object that is ready to render to code.
        """

        # Get the lofted shape and the number of sections across its span
        lofted_shape = self.generate_lofted()
        n_span = lofted_shape.shape[0]

        # Find the distance between each cross section. Find the minimum of
        # these distances and multiply by 0.1. This will be the height of each
        # extrusion for each cross section.

        diff_z = []
        for k in range(n_span - 1):
            diff_z.append(lofted_shape[k + 1, 0, 2] - lofted_shape[k, 0, 2])
        dz = 0.1 * min(diff_z)

        # Make the range to sample the span of the blade. If downsample_z
        # is 1, that means every cross section will be plotted. If it is
        # greater than 1, samples will be skipped. This is reflected in
        # the range to sample the span.

        if downsample_z == 1:
            n_span_range = range(n_span)
        else:
            n_span_range = range(0, n_span, downsample_z)

        # Create one extrusion per cross section.
        extrusions = []
        for k in n_span_range:
            bottom = lofted_shape[k, 0, 2]
            points = tuple((row[0], row[1]) for row in lofted_shape[k, :, :])
            polygon = solid.polygon(points)
            extrusion = solid.linear_extrude(dz)(polygon)
            translated_extrusion = solid.translate((0.0, 0.0, bottom))(extrusion)
            extrusions.append(translated_extrusion)

        # Create a hull around all the cross sections and return it.
        hull_of_extrusions = solid.hull()(extrusions)
        return hull_of_extrusions


def c2(self, r):
    r_support = 20

    e = r / r_support

    if hasattr(e, '__iter__'):
        result = []
        for ei in e:
            if ei < 1:
                result.append((1 - e)**4 + (4 * e + 1))
            else:
                result.append(0)
        result = np.array(result)
        
    if ei < 1:
        result = ((1 - e)**4 + (4 * e + 1))
    else:
        result = (0)

    return result


blade = Blade('IEA-15-240-RWT.yaml')
print(len(blade.outer_shape['airfoil_position']['labels']))
points = blade.generate_lofted(n_span_min=300, n_xy=300)
# points = blade.blade_hull(downsample_z = 10)

print(points.shape)


# f = open('surf_coarse.plt','w')

# f.write('TITLE = \" WINDIO TEST CASE\" \n')
# f.write('VARIABLES = \"X\" \"Y\" \"Z\" \n')
# f.write('ZONE I= {} J = {} F=point \n'.format(points.shape[1] + 1, int(points.shape[0]/6)))
# for i in range(points.shape[0]):
#     if i % 6 == 0 :
#         for j in range(points.shape[1]):
#             f.write('{} \t {} \t {}\n'.format(points[i, j, 0], points[i, j, 1], points[i, j, 2]))
#         f.write('{} \t {} \t {}\n'.format(points[i, 0, 0], points[i, 0, 1], points[i, 0, 2]))

# f.close()


f = open('surf_coarse.dat','w')

f.write('TITLE = \" WINDIO TEST CASE\" \n')
f.write('VARIABLES = \"X\" \"Y\" \"Z\" \n')
f.write('ZONE I= {} J = {} F=point \n'.format(points.shape[1] + 1, (points.shape[0])))
for i in range(points.shape[0]):
    for j in range(points.shape[1]):
        f.write('{} \t {} \t {}\n'.format(points[i, j, 0], points[i, j, 1], points[i, j, 2]))
    f.write('{} \t {} \t {}\n'.format(points[i, 0, 0], points[i, 0, 1], points[i, 0, 2]))

f.close()


f = open('../FLOWSOLVER2018/IEA_15MW/tiny/IEA_15MW_patch.dat','w')
f.write('{} \t {} \n'.format(points.shape[1] + 1, points.shape[0]))
for i in range(points.shape[0]):
    for j in range(points.shape[1]):
        f.write('{} \t {} \t {}\n'.format(points[i, j, 1], points[i, j, 2] + 3, points[i, j, 0]))
    f.write('{} \t {} \t {}\n'.format(points[i, 0, 1], points[i, 0, 2] + 3, points[i, 0, 0]))

f.close()

f = open('surf_coarse.p3d','w')

npts = points.shape[0] * points.shape[1]

f.write('{} \t {} \t {} \n'.format(points.shape[1] + 1, points.shape[0], 1))
for i in range(points.shape[0]):
    for j in range(points.shape[1]):
        f.write('{}\n'.format(points[i, j, 0]))
    f.write('{}\n'.format(points[i, 0, 0]))
for i in range(points.shape[0]):
    for j in range(points.shape[1]):
        f.write('{}\n'.format(points[i, j, 1]))
    f.write('{}\n'.format(points[i, 0, 1]))
for i in range(points.shape[0]):
    for j in range(points.shape[1]):
        f.write('{}\n'.format(points[i, j, 2]))
    f.write('{}\n'.format(points[i, 0, 2]))



f.close()

print(points.shape)
