import abc
import copy
from placement.placers import FlyGrid
from common.grid import RectangularGrid3D
from IPython.core.debugger import set_trace
import numpy as np
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from gsim_conf import use_mayavi
if use_mayavi:
    print("Loading MayaVi...")
    from mayavi import mlab
    print("done")

from gsim.gfigure import GFigure

from common.fields import FunctionVectorField
from common.utilities import natural_to_dB, dbm_to_watt, watt_to_dbm
from channels.channel import Channel


class Building():
    def __init__(self,
                 sw_corner=None,
                 ne_corner=None,
                 height=None,
                 absorption=1):
        assert sw_corner[2] == 0
        assert ne_corner[2] == 0
        assert ne_corner[0] > sw_corner[0]
        assert ne_corner[1] > sw_corner[1]
        self.sw_corner = sw_corner
        self.ne_corner = ne_corner
        self.height = height
        self.absorption = absorption  # dB/m

    @property
    def nw_corner(self):
        return np.array([self.sw_corner[0], self.ne_corner[1], 0])

    @property
    def se_corner(self):
        return np.array([self.ne_corner[0], self.sw_corner[1], 0])

    def plot(self):
        """Adds a rectangle per side of the building to the current figure.  """
        assert use_mayavi

        def high(pt):
            return np.array([pt[0], pt[1], self.height])

        def lateral_face(pt1, pt2):
            return np.array([
                [high(pt1), high(pt2)],
                [pt1, pt2],
            ])

        def plot_face(face):
            mlab.mesh(face[..., 0], face[..., 1], face[..., 2])
            mlab.mesh(face[..., 0],
                      face[..., 1],
                      face[..., 2],
                      representation='wireframe',
                      color=(0, 0, 0))

        # top face
        face = np.array([
            [high(self.nw_corner), high(self.ne_corner)],
            [high(self.sw_corner), high(self.se_corner)],
        ])
        plot_face(face)

        # west face
        face = lateral_face(self.nw_corner, self.sw_corner)
        plot_face(face)

        # north face
        face = lateral_face(self.nw_corner, self.ne_corner)
        plot_face(face)

        # east face
        face = lateral_face(self.ne_corner, self.se_corner)
        plot_face(face)

        # south face
        face = lateral_face(self.sw_corner, self.se_corner)
        plot_face(face)

        # face = np.array([
        #     [self.nw_corner, self.ne_corner],
        #     [self.sw_corner, self.se_corner],
        # ])
        # plot_face(face)

    @property
    def min_x(self):
        return self.sw_corner[0]

    @property
    def max_x(self):
        return self.ne_corner[0]

    @property
    def min_y(self):
        return self.sw_corner[1]

    @property
    def max_y(self):
        return self.ne_corner[1]

    # True for points inside the building, False otherwise
    def indicator(self, coords):
        x, y, z = coords
        return self.absorption * ((x >= self.min_x) & (x <= self.max_x) &
                                  (y >= self.min_y) & (y <= self.max_y) &
                                  (z < self.height))


class SLField(FunctionVectorField):
    pass


class Environment():
    area_len = None

    def __init__(self, area_len=None):
        if area_len:
            self.area_len = area_len


class UrbanEnvironment(Environment):

    l_users = None  # list of points corresponding to the user locations
    l_lines = None  # list of lists of points. A line is plotted for each list
    _fly_grid = None
    # `dl_uavs`: dict whose keys are strings and whose values are lists of
    # points corresponding to the UAV locations. Each list will be represented
    # with a different color and marker.
    dl_uavs = dict()

    def __init__(self,
                 base_fly_grid=None,
                 buildings=None,
                 num_pts_slf_grid=None,
                 **kwargs):
        """ Args: 

            `base_fly_grid`: object of class FlyGrid. The gridpoints inside
            buildings are disabled. The resulting grid can be accessed through
            `self.fly_grid`.

            `num_pts_slf_grid`: vector with 3 entries corresponding to the
            number of points along the X, Y, and Z dimensions
        """

        super().__init__(**kwargs)

        #self.grid = slf_grid
        self.buildings = buildings
        slf_grid = RectangularGrid3D(num_pts=num_pts_slf_grid,
                                     area_len=self.area_len)
        self.slf = SLField(grid=slf_grid, fun=self.f_buildings_agg)
        if base_fly_grid is not None:
            self._fly_grid = base_fly_grid
            self._fly_grid.disable_by_indicator(self.building_indicator)

    def building_indicator(self, coords):
        """Each entry is the indicator for a building scaled by the
        absorption. """
        x, y, z = coords
        return [building.indicator(coords) for building in self.buildings]

    # Combine all possible buildings into a single SLF.
    @property
    def f_buildings_agg(self):
        return lambda coords: np.max(self.building_indicator(coords))[None, ...
                                                                      ]

    @property
    def fly_grid(self):
        return self._fly_grid

    def random_pts_on_street(self, num_pts):
        """ Returns a `num_pts` x 3 matrix whose rows contain the coordinates of
        points drawn uniformly at random on the ground and out of the buildings.

        The limits of the buildings are given by the function self.f_buildings.
        The building boundaries obtained in this way may not coincide with the
        voxel boundaries.
        """

        f_indicator = self.f_buildings_agg

        def filter_street(l_pts):
            return np.array([pt for pt in l_pts if not f_indicator(pt)])

        num_remaining = num_pts
        l_pts = []
        while num_remaining > 0:
            # TODO: generate the points using self.area_len rather than self.slf.grid
            new_pts = self.slf.grid.random_pts(num_pts, z_val=0)
            new_pts = filter_street(new_pts)
            if len(new_pts) > 0:
                l_pts += [new_pts]
            num_remaining -= len(new_pts)

        user_coords = np.concatenate(l_pts, axis=0)[:num_pts, :]

        #plt.plot(user_coords[:,0], user_coords[:,1], ".")
        #plt.show()
        return user_coords

    def disable_flying_gridpts_by_dominated_verticals(self, channel):
        """Disables gridpoints of the fly grid according to the positions of the users"""

        assert self.l_users is not None

        channel = copy.deepcopy(channel)
        channel.disable_gridpts_by_dominated_verticals = True

        map = channel.capacity_map(grid=self.fly_grid,
                                   user_coords=self.l_users)

        self._fly_grid.t_enabled = map.grid.t_enabled

    def plot_buildings(self):
        if not use_mayavi:
            return self.slf.plot_as_blocks()
        else:
            for building in self.buildings:
                building.plot()

    def plot(self, bgcolor=(1., 1., 1.), fgcolor=(0., 0., 0.)):
        if not use_mayavi:
            #fig = plt.figure()
            #ax = fig.gca(projection='3d')
            ax = self.plot_buildings()
            self._plot_pt_dl(self.dl_uavs, ax=ax)
            self._plot_pts(self.l_users, style="o", ax=ax)
            if self._fly_grid is not None:
                self._plot_pts(self._fly_grid.list_pts(),
                               style=".",
                               ax=ax,
                               color=(0, .7, .9),
                               alpha=0.3)
            self._plot_lines(ax)
        else:
            mlab.figure(bgcolor=bgcolor, fgcolor=bgcolor)
            self.plot_buildings()
            self._plot_pt_dl(self.dl_uavs, scale_factor=self.area_len[0] / 20.)
            self._plot_pts(self.l_users,
                           style="2dsquare",
                           scale_factor=self.area_len[0] / 150.,
                           color=(.5, .5, 0.))
            if self._fly_grid is not None:
                self._plot_pts(self._fly_grid.list_pts(),
                               style="cube",
                               scale_factor=self.area_len[0] / 150.,
                               color=(0, 0.7, 0.9),
                               opacity=.3)
            self._plot_lines()
            self._plot_ground()
            return

    def show(self):
        if not use_mayavi:
            plt.show()
        else:
            mlab.show()

    def _plot_ground(self):
        min_x = 0  # g.min_x
        min_y = 0  # g.min_y
        max_x, max_y, max_z = self.area_len
        ground = np.array([
            [[0, max_y, 0], [max_x, max_y, 0]],
            [[0, 0, 0], [max_x, 0, 0]],
        ])
        mlab.mesh(ground[..., 0],
                  ground[..., 1],
                  ground[..., 2],
                  color=(0, 0, 0))
        ax = mlab.axes(extent=[min_x, max_x, min_y, max_y, 0, max_z],
                       nb_labels=4)
        #ax.axes.font_factor = .8
    def _plot_pt_dl(self, dl_pts, **kwargs):
        """`dl_pts` is a dict of lists of points."""
        if len(dl_pts) == 0:
            return
        if use_mayavi:
            l_markers = [
                "2dcircle",
                "2dcross",
                "2ddiamond",
                "2ddash",
                "2dhooked_arrow",
                "2dsquare",
                "2dthick_arrow",
                "2dthick_cross",
                "2dtriangle",
                "2dvertex",
                "arrow",
                "axes",
                "cone",
                "cube",
                "cylinder",
                "point",
                "sphere",
                "2darrow",
            ][:len(dl_pts)]
        else:
            l_markers = ['x', 'o', 'd'][:len(dl_pts)]
        #m_colors = np.random.random((len(dl_pts), 3))
        l_colors = [[.4, .8, .0], [.7, 0, 0], [.7, .3, .1], [.8, .0, .1],
                    [.9, .5, .0]][0:len(dl_pts)]
        print("Legend:")
        for name, marker, color in zip(dl_pts.keys(), l_markers, l_colors):
            self._plot_pts(dl_pts[name],
                           style=marker,
                           color=tuple(color),
                           **kwargs)
            print(f"{marker} --> {name}")

    def _plot_pts(self,
                  l_pts,
                  style,
                  ax=None,
                  scale_factor=3.,
                  color=(.5, .5, .5),
                  **kwargs):
        if l_pts is not None:
            pts = np.array(l_pts)
            if not use_mayavi:
                ax.plot(pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        style,
                        color=color,
                        **kwargs)
            else:
                mlab.points3d(pts[:, 0],
                              pts[:, 1],
                              pts[:, 2],
                              mode=style,
                              scale_factor=scale_factor,
                              color=color,
                              **kwargs)

    def _plot_lines(self, ax=None):
        if self.l_lines is not None:
            for line in self.l_lines:
                pts = np.array(line)
                if not use_mayavi:
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2])
                else:
                    mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2])


class BlockUrbanEnvironment(UrbanEnvironment):
    def __init__(self,
                 num_pts_fly_grid=[5, 5, 3],
                 min_fly_height=20,
                 building_height=20,
                 building_absorption=1,
                 **kwargs):
        """ Args: 

            `num_pts_slf_grid` and `num_pts_fly_grid`: vectors with 3 entries
            corresponding to the number of points along the X, Y, and Z
            dimensions

            `building_height`: if None, then generated randomly for each building.
        """
        assert "area_len" not in kwargs.keys()
        assert "base_fly_grid" not in kwargs.keys()
        base_fly_grid = FlyGrid(area_len=self.area_len,
                                num_pts=num_pts_fly_grid,
                                min_height=min_fly_height)

        super().__init__(base_fly_grid=base_fly_grid,
                         buildings=self._get_buildings(
                             height=building_height,
                             absorption=building_absorption),
                         **kwargs)

    def _get_buildings(self, height, absorption):
        def get_height():
            if height is not None:
                return height
            h = np.random.normal(loc=15, scale=15)
            return min(max(h, 5), 80)

        l_buildings = []
        for block_x in self.block_limits_x:
            for block_y in self.block_limits_y:
                bld = Building(sw_corner=[block_x[0], block_y[0], 0],
                               ne_corner=[block_x[1], block_y[1], 0],
                               height=get_height(),
                               absorption=absorption)
                l_buildings.append(bld)

        return l_buildings


class BlockUrbanEnvironment1(BlockUrbanEnvironment):
    area_len = [100, 80, 50]
    block_limits_x = np.array([[20, 30], [50, 60], [80, 90]])
    block_limits_y = np.array([[15, 30], [50, 60]])


class BlockUrbanEnvironment2(BlockUrbanEnvironment):
    area_len = [1000, 1000, 100]

    street_width = 100
    building_width = 120

    def __init__(self, *args, **kwargs):
        def block_limits(len_axis):
            block_ends = np.arange(self.street_width + self.building_width,
                                   self.area_len[0],
                                   step=self.street_width +
                                   self.building_width)[:, None]

            return np.concatenate(
                [np.maximum(block_ends - self.building_width, 0), block_ends],
                axis=1)

        self.block_limits_x = block_limits(self.area_len[0])
        self.block_limits_y = block_limits(self.area_len[1])

        super().__init__(*args, **kwargs)


class GridBasedBlockUrbanEnvironment(BlockUrbanEnvironment):
    height_over_min_enabled_height = 3

    def __init__(self,
                 area_len=[100, 80, 50],
                 num_pts_fly_grid=[5, 5, 3],
                 min_fly_height=20,
                 building_absorption=1,
                 building_height=None,
                 **kwargs):
        """ Args: 

            `num_pts_slf_grid` and `num_pts_fly_grid`: vectors with 3 entries
            corresponding to the number of points along the X, Y, and Z
            dimensions

            `building_height`: if None, it is set to
            `self.height_over_min_enabled_height` units above the min enabled
            flying height.
        """
        assert "base_fly_grid" not in kwargs.keys()
        assert num_pts_fly_grid[0] % 2 == 1
        assert num_pts_fly_grid[1] % 2 == 1
        base_fly_grid = FlyGrid(area_len=area_len,
                                num_pts=num_pts_fly_grid,
                                min_height=min_fly_height)

        if building_height is None:
            building_height = base_fly_grid.min_enabled_height + self.height_over_min_enabled_height

        def edges_to_block_limits(edges):
            if len(edges) % 2 == 1:
                edges = edges[0:-1]
            return np.reshape(edges, (-1, 2))

        self.block_limits_x = edges_to_block_limits(base_fly_grid.t_edges[0, 0,
                                                                          0,
                                                                          1:])
        self.block_limits_y = edges_to_block_limits(
            np.flip(base_fly_grid.t_edges[1, 0, :-1, 0]))

        super(BlockUrbanEnvironment, self).__init__(
            area_len=area_len,
            base_fly_grid=base_fly_grid,
            buildings=self._get_buildings(height=building_height,
                                          absorption=building_absorption),
            **kwargs)

    # def __init__(self, *args, **kwargs):
    #     def block_limits(len_axis):
    #         block_ends = np.arange(self.street_width+self.building_width,
    #                                      self.area_len[0],
    #                                      step=self.street_width +
    #                                      self.building_width)[:, None]

    #         return np.concatenate(
    #             [np.maximum(block_ends - self.building_width, 0), block_ends], axis=1)

    #     self.block_limits_x = block_limits(self.area_len[0])
    #     self.block_limits_y = block_limits(self.area_len[1])

    #     super().__init__(*args, **kwargs)
