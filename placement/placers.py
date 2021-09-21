import sys
import os
from IPython.core.debugger import set_trace
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import endswith
from sklearn.cluster import KMeans
import scipy
from scipy.optimize import linprog

from gsim import rng
from gsim.utils import xor

from common.fields import FunctionVectorField, VectorField
from common.grid import RectangularGrid3D
from common.solvers import group_sparse_cvx, sparsify, group_sparsify, weighted_group_sparse_cvx, weighted_group_sparse_scipy
from common.runner import Runner
import warnings
import logging

log = logging.getLogger("placers")


class FlyGrid(RectangularGrid3D):
    def __init__(self,
                 *args,
                 f_disable_indicator=None,
                 min_height=None,
                 **kwargs):
        """ Args: `f_disable_indicator` is a function that takes a vector
            `coords` of shape (3,) with the coordinates of a point and returns a
            vector or scalar. A grid point with coordinates `coords` and height
            >= `min_height` is enabled (flying allowed) iff
            `f_disable_indicator(coords)` is False, 0, or a 0 vector. 
        """
        super().__init__(*args, **kwargs)

        if f_disable_indicator is not None:
            self.disable_by_indicator(f_disable_indicator)

        self._min_height = min_height
        if self._min_height is not None:
            self.disable_by_indicator(lambda coords:
                                      (coords[2] < self._min_height))

    @property
    def min_height(self):
        return self._min_height

    # def plot_as_blocks(self):
    # #     return self.enable_field.plot_as_blocks()

    # def plot_allowed_pts(self, ax):
    #     """ `ax` is a pyplot 3D axis"""
    #     allowed_pts = self.list_pts()
    #     ax.plot(allowed_pts[:, 0], allowed_pts[:, 1], allowed_pts[:, 2], ".")


class Placer():
    """ Abstract class for Placers"""
    
    # to be overridden by subclasses
    _name_on_figs = ""

    # This is set at construction time and should not be modified by subclasses
    def __init__(self, num_uavs=None):
        self.num_uavs = num_uavs

    # To be implemented by subclasses
    def place(self, fly_grid, channel, user_coords):
        """
        Args: 

        - `user_coords` is num_users x 3

        Returns:

        - `uav_coords` is num_uav x 3

       
        """
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def name_on_figs(self):
        if self._name_on_figs:
            return self._name_on_figs
        else:  
            return self.name


class CapacityBasedPlacer(Placer):
    def __init__(self, min_user_rate=None, max_uav_total_rate=None, **kwargs):
        super().__init__(**kwargs)

        self.min_user_rate = min_user_rate
        self.max_uav_total_rate = max_uav_total_rate

    def plot_capacity_maps(self, fly_grid, channel, user_coords, znum=4):
        """ Returns one GFigure for each user with the corresponding capacity map."""
        maps = channel.capacity_maps(
            grid=fly_grid,
            user_coords=user_coords,
        )
        return [map.plot_z_slices(znum=4) for map in maps]

    # def get_capacity_map(self, user_coords):
    #     return


class SingleUAVPlacer(CapacityBasedPlacer):
    def __init__(self, criterion="sum_rate", num_uavs=1, **kwargs):

        assert num_uavs == 1
        super().__init__(num_uavs=1, **kwargs)

        self.criterion = criterion

    def place(self, fly_grid, channel, user_coords=None):
        """
        See parent.
        """
        # TODO: change the following line to use channel.capacity_map. This
        # would unify GroupSparseUAVPlacer.place and SingleUAVPlacer.place but
        # requires the rest of this function.
        maps = channel.capacity_maps(grid=fly_grid, user_coords=user_coords)

        if self.criterion == "sum_rate":
            sum_rate_map = VectorField.sum_fields(maps)
            debug = 0
            if debug:
                F = sum_rate_map.plot_z_slices(znum=4)
                F.plot()
                plt.show()

            uav_coords = sum_rate_map.arg_coord_max()
        elif self.criterion == "max_min_rate":
            # maximize the minimum rate across grid points
            min_rate_map = VectorField.min_over_fields(maps)
            uav_coords = min_rate_map.arg_coord_max()
        else:
            raise ValueError("unrecognized self.criterion = ", self.criterion)

        return np.array([uav_coords])


class SparseUAVPlacer(CapacityBasedPlacer):    
    """
    A subset of grid pts is selected by solving:

    minimize_{v_alpha}    || v_alpha ||_1

    s.t.                  m_C @ v_alpha >= min_user_rate * ones(num_users,)

                          0 <= v_alpha <= 1

    where m_C is num_users x num_pts and contains the capacity of each link.
    Those points associated with a positive entry of the optimal v_alpha
    indicate the positions of the UAVs.

    """
    _name_on_figs = "Sparse Placer (proposed)"
    def __init__(self,
                 sparsity_tol=1e-2,
                 num_max_reweight_iter=4,
                 epsilon_reweighting=1e-2,
                 **kwargs):
        """ Args:

            `sparsity_tol`: tolerance for sparsifying the groups. 

            `num_max_reweight_iter`: maximum number of times that reweighting is 
            applied when solving the placement opt. problem.
        """

        super().__init__(**kwargs)
        assert self.max_uav_total_rate is None, "this placer cannot guarantee a max total rate per UAV"
        assert self.num_uavs is None, "This placer does not enforce a fixed number of UAVs"

        self.sparsity_tol = sparsity_tol
        self.num_max_reweight_iter = num_max_reweight_iter
        self.epsilon_reweighting = epsilon_reweighting

    def place(self, fly_grid, channel, user_coords=None):
        """
        See parent.
        """

        #     max_rate_map = VectorField.clip(map, upper=self.min_user_rate)
        #     max_rate_map.disable_gridpts_by_dominated_verticals()

        #     return self._sparse_placement(max_rate_map)

        # def _sparse_placement(self, map):

        def get_pt_soft_indicators(gridpt_weights, m_capacity):

            #gridpt_weights = gridpt_weights/np.sum(gridpt_weights)
            bounds = np.zeros((num_pts, 2))
            bounds[:, 1] = 1
            A_ub = -m_capacity
            b_ub = -self.min_user_rate * np.ones((num_users, ))
            with warnings.catch_warnings():
                # Suppress deprecation warning inside SciPy
                warnings.simplefilter("ignore")

                num_failures = 0
                while True:
                    res = linprog(gridpt_weights,
                                  A_ub=A_ub,
                                  b_ub=b_ub,
                                  bounds=bounds)
                    v_weights = res.x
                    if any(np.isnan(v_weights)):
                        log.warning(
                            "SparseUAVPlacer: Linprog returned a NaN solution")
                        gridpt_weights = 0.1 * gridpt_weights
                        num_failures += 1
                        if num_failures >= 5:
                            set_trace()
                            print("linprog failed Nan")

                    else:
                        break
            status = "success" if res.success else res["message"]

            #debug
            # res2= linprog(gridpt_weights/np.sum(gridpt_weights),
            #                   A_ub=A_ub,
            #                   b_ub=b_ub,
            #                   bounds=bounds)
            # v_weights2 = res2.x
            # diff = np.linalg.norm(v_weights-v_weights2)
            # print("diff = ", diff)
            # if diff>0.1:
            #     print("different")

            return v_weights, status

        def weights_to_rates_and_coords(v_pt_weights, m_capacity):
            v_sp_pt_weights = sparsify(v_pt_weights, self.sparsity_tol)
            v_inds = np.nonzero(v_sp_pt_weights)[0]
            m_rates = m_capacity[:, v_inds]
            uav_coords = map.grid.list_pts()[v_inds, :]
            return uav_coords, m_rates

        np.set_printoptions(precision=2, suppress=True, linewidth=2000)

        map = channel.capacity_map(
            grid=fly_grid,
            user_coords=user_coords,
        )
        m_capacity = map.list_vals().T
        # The following improves the results
        m_capacity = np.minimum(m_capacity, self.min_user_rate)
        num_users, num_pts = m_capacity.shape

        # Reweighting [Candes, Wakin, and Boyd, 2008]
        # TODO: experiment with the initial weights
        gridpt_weights = 1 / (1 + np.sum(m_capacity >= self.min_user_rate) +
                              1e-4 * np.random.random((num_pts, )))
        num_uavs_prev = None
        for ind in range(self.num_max_reweight_iter):
            v_pt_soft_indicators, status = get_pt_soft_indicators(
                gridpt_weights, m_capacity)
            m_uav_coords, m_rates = weights_to_rates_and_coords(
                v_pt_soft_indicators, m_capacity)
            num_uavs = m_rates.shape[1]

            log.debug(f"Number of UAVs after {ind+1} iterations = {num_uavs}")
            log.debug("UAV rates=")
            log.debug(m_rates)

            if (num_uavs == 1) or (num_uavs == num_uavs_prev):
                break
            num_uavs_prev = num_uavs

            gridpt_weights = 1 / (v_pt_soft_indicators +
                                  self.epsilon_reweighting)

        return m_uav_coords


class FromFixedNumUavsPlacer(CapacityBasedPlacer):
    """ If self.num_uavs is not provided, then the number of uavs is gradually
        increased until the minimum user rate exceeds `self.min_user_rate`.

     """

    # If `_place_num_uavs` fails, the following functions are tried. It must be a list of functions.
    _last_resort_place_num_uavs = None

    def __init__(self, **kwargs):
        """ Args:
            
        """
        super().__init__(**kwargs)
        assert self.max_uav_total_rate is None, "this placer cannot guarantee a max total rate per UAV"
        assert xor(self.min_user_rate is None, self.num_uavs is None)

    def place(self, fly_grid, channel, user_coords, *args, **kwargs):
        map = channel.capacity_map(
            grid=fly_grid,
            user_coords=user_coords,
        )
        if self.min_user_rate is not None:
            ind_uavs = self._place_by_increasing_num_uavs(
                *args, map=map, user_coords=user_coords, **kwargs)
        else:
            ind_uavs = self._place_num_uavs(map=map,
                                            user_coords=user_coords,
                                            num_uavs=self.num_uavs)
        if ind_uavs is None:
            return None
        return map.grid.list_pts()[ind_uavs]

    def _place_by_increasing_num_uavs(self, *args, **kwargs):
        """Runs self._place_num_uavs with an increasing num_uavs until all users
        receive the minimum rate. The latter condition holds when each user
        receives `self.min_user_rate` from all UAVs combined. If you want this
        condition to be satisfied when each user connects only to the strongest
        UAV, then set `channel.min_link_capacity=self.min_user_rate`."""

        inds_uavs = self._place_by_increasing_num_uavs_from_fun(
            self._place_num_uavs, *args, **kwargs)

        if inds_uavs is None:
            log.warning(
                f"Impossible to guarantee minimum rate in {self.__class__.__name__}"
            )

        if self._last_resort_place_num_uavs is not None:
            for fun in self._last_resort_place_num_uavs:
                log.warning("   Using last resort procedure")
                inds_uavs = self._place_by_increasing_num_uavs_from_fun(
                    fun, *args, **kwargs)

                if inds_uavs is None:
                    log.warning(
                        f"    Last resort procedure also failed in {self.__class__.__name__}"
                    )
        return inds_uavs

    def _place_by_increasing_num_uavs_from_fun(
        self,
        fun,
        map,
        user_coords=None,
        max_num_uavs=30,
    ):

        m_capacity = map.list_vals().T
        max_num_uavs = np.minimum(max_num_uavs, len(user_coords))

        for num_uavs in range(1, max_num_uavs + 1):
            inds = fun(map=map, user_coords=user_coords, num_uavs=num_uavs)
            if (inds is not None) and all(
                    np.sum(m_capacity[:, inds], axis=1) >= self.min_user_rate):
                return inds

        #raise ValueError("Infeasible")
        return None

    # Abstract methods
    def _place_num_uavs(self, map, user_coords, num_uavs):
        """ Returns a vector of indices of the grid points where UAVs need to be
        placed. 
        """
        raise NotImplementedError


class KMeansPlacer(FromFixedNumUavsPlacer):
    _name_on_figs = "Galkin et al."
    @staticmethod
    def _place_num_uavs(map, user_coords, num_uavs):
        assert user_coords.shape[1] == 3
        kmeans = KMeans(n_clusters=num_uavs)
        kmeans.fit(user_coords)
        centers = kmeans.cluster_centers_
        return map.grid.nearest_inds(centers)


class SpaceRateKMeans(FromFixedNumUavsPlacer):
    """ This would be inspired by the modified K-means algorithm in
        [hammouti2019mechanism].

        This placer aims at maximizing the sum rate. However, the 
        discretization imposed by the fly grid and the shadowing hinder this purpose.

        The placement assumes that each user connects to one and only one UAV.

        At each stage:

        1 - each user associates with the UAV from which it receives the
        greatest rate. 

        2- each UAV is moved to the grid point that lies closest to the arithmetic 
        mean of the coordinates of associated users.
    """
    _name_on_figs = "Hammouti et al."
    def __init__(self,
                 num_max_iter=200,
                 use_kmeans_as_last_resort=False,
                 **kwargs):
        """ Args:
    
        """
        super().__init__(**kwargs)
        self.num_max_iter = num_max_iter

        if use_kmeans_as_last_resort:
            self._last_resort_place_num_uavs = [KMeansPlacer._place_num_uavs]

    def _place_num_uavs(self, map, user_coords, num_uavs):
        def _associate_users(v_uav_inds):
            """Returns a list of length num_users where the n-th entry indicates the
            index of the UAV to which user n must be associated. The entries of
            this list are therefore taken from v_uav_inds. """

            v_ass_inds_rel_to_v_uav_inds = np.argmax(m_capacity[:, v_uav_inds],
                                                     axis=1)
            v_ass_inds = v_uav_inds[v_ass_inds_rel_to_v_uav_inds]
            return v_ass_inds

        def _place_uavs_at_centroids(v_ass_inds):
            """Returns:

                `m_uav_coords`: num_uavs x 3 matrix with the coordinates of the
                UAVs. The coordinates of each UAV are the arithmetic means of
                the coordinates of the UAVs associated with that UAV. The rows
                do not follow any specific order. 

            """
            m_uav_coords = np.array([
                np.mean(user_coords[np.where(v_ass_inds == ind_uav)[0], :],
                        axis=0) for ind_uav in set(v_ass_inds)
            ])
            return m_uav_coords

        m_capacity = map.list_vals().T
        m_capacity = np.minimum(m_capacity, self.min_user_rate)
        num_users = len(user_coords)
        assert num_uavs <= num_users

        # Initial random set of UAV coordinates
        v_rnd_user_inds = rng.choice(num_users, (num_uavs, ), replace=False)
        m_uav_coords = user_coords[v_rnd_user_inds, :]
        v_uav_inds = np.sort(map.grid.nearest_inds(m_uav_coords))

        for _ in range(self.num_max_iter):
            v_ass_inds = _associate_users(v_uav_inds)
            m_uav_coords = _place_uavs_at_centroids(v_ass_inds)
            v_uav_inds_new = np.sort(map.grid.nearest_inds(m_uav_coords))

            if np.all(v_uav_inds_new == v_uav_inds):
                #print(f"Finishing after {_+1} iterations")
                return v_uav_inds_new
            v_uav_inds = v_uav_inds_new

        log.warning("Maximum number of iterations reached at SpaceRateKMeans")
        return v_uav_inds
        #raise ValueError("Maximum number of iterations reached at SpaceRateKMeans")


class GridRatePlacer(FromFixedNumUavsPlacer):
    """ The placement assumes that each user connects to one and only one UAV.
    The goal is to maximize the sum rate. 

    At each stage, 

    1 - each user associates with the UAV from which it receives the greatest
    rate. 

    2- each UAV is moved to the grid point that maximizes the rate to its
    associated users. 

    In this way, the sum rate should never decrease, which implies that the
    algorithm  eventually converges.
    """
    def __init__(self, num_max_iter=200, num_initializations=20, **kwargs):
        """ Args:
    
        """
        super().__init__(**kwargs)
        self.num_max_iter = num_max_iter
        self.num_initializations = num_initializations

    def _place_num_uavs(self, map, user_coords, num_uavs):
        m_capacity = map.list_vals().T

        def sum_rate(v_inds):
            # Each user connects to only 1 UAV.
            return np.sum(np.max(m_capacity[:, v_inds], axis=1))

        v_uav_inds = None
        for _ in range(self.num_initializations):
            v_uav_inds_new = self._place_num_uavs_one_initialization(
                m_capacity, user_coords, num_uavs)
            #print("new rate = ", sum_rate(v_uav_inds_new))
            if (v_uav_inds is None) or (sum_rate(v_uav_inds_new) >
                                        sum_rate(v_uav_inds)):
                v_uav_inds = v_uav_inds_new
        return v_uav_inds

    def _place_num_uavs_one_initialization(self,
                                           m_capacity,
                                           user_coords,
                                           num_uavs,
                                           debug=0):
        def _associate_users(v_uav_inds):
            """Returns a list of length num_users where the n-th entry indicates the
            index of the UAV to which user n must be associated. The entries of
            this list are therefore taken from v_uav_inds. """

            v_ass_inds_rel_to_v_uav_inds = np.argmax(m_capacity[:, v_uav_inds],
                                                     axis=1)
            v_ass_inds = v_uav_inds[v_ass_inds_rel_to_v_uav_inds]
            return v_ass_inds

        def _place_uavs_to_maximize_sum_rate_of_associated_users(v_ass_inds):
            """Returns:

                `v_gridpt_inds`: list of length `num_uavs` indicating the indices
                of the grid points where UAVs need to be placed. Each UAV is placed
                at the grid point that maximizes the sum rate of the associated users.
                """
            l_gridpt_inds = []
            for ind_uav in set(v_ass_inds):
                sum_rate_per_gridpt = np.sum(
                    m_capacity[np.where(v_ass_inds == ind_uav)[0], :], axis=0)
                new_gridpt = np.argmax(sum_rate_per_gridpt)
                l_gridpt_inds.append(new_gridpt)

            return np.array(l_gridpt_inds)

        num_users, num_gridpts = m_capacity.shape
        assert num_uavs <= num_users

        # Initial random set of UAV indices
        v_uav_inds = rng.choice(num_gridpts, (num_uavs, ), replace=False)
        sum_rate = None
        if debug:
            print("New initialization ----------")
        for _ in range(self.num_max_iter):
            v_ass_inds = _associate_users(v_uav_inds)
            v_uav_inds = _place_uavs_to_maximize_sum_rate_of_associated_users(
                v_ass_inds)

            sum_rate_new = np.sum(m_capacity[range(0, num_users), v_ass_inds])

            if (sum_rate is not None) and (sum_rate == sum_rate_new):
                #if np.all(v_uav_inds_new == v_uav_inds): # this check apparently leads to oscillations --> better check sum rate
                #print(f"Finishing after {_+1} iterations")
                return v_uav_inds
            sum_rate = sum_rate_new

            if debug:
                print("sum rate = ", sum_rate)

        raise ValueError("Maximum number of iterations reached")


class FromMinRadiusPlacer(CapacityBasedPlacer):
    """ The number of UAVs is determined so that all users lie within a certain
        distance from the UAVs. The distance is initially determined from
        self.min_user_rate and reduced by `self.distance_reduce_factor`
        iteratively until the minimum rate is guaranteed. 

        Each user associates with the strongest UAV.
     """
    def __init__(self, num_radius=4, radius_discount=0.9, **kwargs):
        """ Args:

            `num_radius`: number of values for the radius to try between the
            radius determined by the rate without grid quantization and with
            grid quantization.

            `radius_discount`: if the minimum radius that guarantees the
            existence of a solution in free space is reached, then the
            subsequent attempted radii are obtained by multiplying the previous
            one by `radius_discount`.

        """
        super().__init__(**kwargs)
        assert self.max_uav_total_rate is None, "this placer cannot guarantee a max total rate per UAV"
        assert self.num_uavs is None
        assert self.min_user_rate is not None
        self.num_radius = num_radius
        self.radius_discount = radius_discount

    def place(self,
              fly_grid,
              channel,
              user_coords,
              *args,
              delta_radius=.1,
              debug=0,
              **kwargs):
        """ See parent. 

            A set of radii is used determined by the distance to guarantee a
            certain capacity and the error due to the grid quantization. A
            solution is guaranteed in the worst case in free space, but larger
            radii are tried first just in case the number of UAVs can be
            reduced. 

            Args:

            `delta_radius`: the radius guaranteeing coverage is reduced by this
            amount to avoid numerical problems. 
        """

        map = channel.capacity_map(
            grid=fly_grid,
            user_coords=user_coords,
        )
        m_capacity = map.list_vals().T

        # Determine values of the radius to use
        assert self.min_user_rate > 0
        max_dist = channel.max_distance_for_rate(self.min_user_rate)
        radius_bf_quantization = np.sqrt(max_dist**2 -
                                         fly_grid.min_enabled_height**2)
        # The following is the "critical radius". Always feasible in free space.
        radius_after_quantization = radius_bf_quantization - map.grid.max_herror
        if radius_after_quantization < 0:
            raise ValueError(
                "Either the rate is too low or the grid to coarse")
        supercritical_radii = np.linspace(radius_bf_quantization,
                                          radius_after_quantization,
                                          num=self.num_radius)
        # Determine the number of subcritical radii until we reach a radius = map.grid.max_herror
        num_subc_radii = np.ceil(
            np.log(map.grid.max_herror / radius_after_quantization) /
            np.log(self.radius_discount))
        subcritical_radii = radius_after_quantization * (
            self.radius_discount**np.arange(1, num_subc_radii + 1))
        v_radius = np.concatenate([supercritical_radii, subcritical_radii])
        v_radius -= delta_radius
        # plt.plot(v_radius, "-x")
        # plt.show()

        for radius in v_radius:
            inds_uavs = self._place_given_radius(map=map,
                                                 user_coords=user_coords,
                                                 radius=radius,
                                                 debug=debug)
            if debug:
                print("radius = ", radius)
                print(
                    "non-covered users = ",
                    np.sum(
                        np.max(m_capacity[:, inds_uavs], axis=1) <
                        self.min_user_rate))
                print("num_uavs=", len(inds_uavs))

                ind_non_covered_users = np.where(
                    np.max(m_capacity[:, inds_uavs], axis=1) <
                    self.min_user_rate)[0]
                if len(ind_non_covered_users):
                    ind_user = ind_non_covered_users[0]
                    uc = user_coords[ind_user]
                    uav_coords = map.grid.list_pts()[inds_uavs]
                    hdists = np.linalg.norm(uav_coords[:, 0:2] - uc[None, 0:2],
                                            axis=1)
                    ind_nearest_uav = inds_uavs[np.argmin(hdists)]
                    coords_nearest_uav = map.grid.list_pts()[ind_nearest_uav]
                    min_hdist = np.min(hdists)
                    print("hdist to closest UAV", min_hdist)
                    print("capacity to closest UAV",
                          m_capacity[ind_user, ind_nearest_uav])
                    dbgain = channel.dbgain(uc, coords_nearest_uav)
                    channel.dbgain_to_capacity(dbgain)
                    channel.dist_to_dbgain_free_space(
                        np.linalg.norm(coords_nearest_uav - uc))

            if all(
                    np.max(m_capacity[:, inds_uavs], axis=1) >=
                    self.min_user_rate):
                return map.grid.list_pts()[inds_uavs]

        # No solution found.
        log.warning(
            f"Maximum number of iterations reached at {self.__class__.__name__} without guaranteeing a minimum rate"
        )
        return None

    # Abstract methods
    def _place_given_radius(self, map, user_coords, radius):
        """ Returns a vector of indices of the grid points where UAVs need to be
        placed. 
        """
        raise NotImplementedError


class SpiralPlacer(FromMinRadiusPlacer):
    """ Implements lyu2017mounted. MATLAB code provided by the authors."""
    _name_on_figs = "Lyu et al."
    def _place_given_radius(self, map, user_coords, radius, debug):
        """ Returns a vector of indices of the grid points where UAVs need to be
        placed. 
        """
        r = Runner("placement", "Spiral.m")
        data_in = OrderedDict()
        data_in["m_users"] = user_coords[:, 0:2].T
        data_in["radius"] = radius
        # 2 x num_uavs
        #uav_coords_2d = r.run("save", data_in)[0]
        uav_coords_2d = r.run("place", data_in)[0]

        # num_uavs x 3
        uav_coords_3d = np.concatenate(
            (uav_coords_2d.T, np.zeros((uav_coords_2d.shape[1], 1))), axis=1)

        v_uav_inds = np.sort(map.grid.nearest_inds(uav_coords_3d))

        if debug:

            def dist_to_nearest_uav(uav_coords, uc):
                dists = np.linalg.norm(uav_coords[:, 0:2] - uc[None, 0:2],
                                       axis=1)
                #dists = np.linalg.norm(uav_coords_2d.T - uc[None, 0:2], axis=1)
                return np.min(dists)

            max_min_dist = np.max(
                [dist_to_nearest_uav(uav_coords_3d, uc) for uc in user_coords])
            print("max hmin distance bf discretization: ", max_min_dist)

            uav_coords_grid = map.grid.list_pts()[v_uav_inds, :]
            max_min_dist = np.max([
                dist_to_nearest_uav(uav_coords_grid, uc) for uc in user_coords
            ])
            print("max hmin distance after discretization: ", max_min_dist)

            for uavc in uav_coords_3d:
                uavc_grid = map.grid.nearest_pt(uavc)
                print(
                    f"uav at {uavc} mapped to {uavc_grid}. HError: {np.linalg.norm(uavc[0:2]-uavc_grid[0:2])}"
                )

        return v_uav_inds


class SparseRecoveryPlacer(FromMinRadiusPlacer):
    """ Implements huang2020sparse. """
    _name_on_figs = "Huang et al."
    max_num_users = 15  # above that, it just returns None, since it is computationally too complex

    def place(self, user_coords, *args, **kwargs):
        num_users = len(user_coords)
        if num_users > self.max_num_users:
            return None
        return super().place(user_coords=user_coords, *args, **kwargs)

    def _place_given_radius(self, map, user_coords, radius, debug):
        """ Returns a vector of indices of the grid points where UAVs need to be
        placed. 
        """
        r = Runner("placement", "SparseRecoveryPlacer.m")
        data_in = OrderedDict()
        data_in["m_users"] = user_coords[:, 0:2]
        data_in["radius"] = radius
        # 2 x num_uavs
        #uav_coords_2d = r.run("save", data_in)[0]
        uav_coords_2d = r.run("place", data_in)[0]

        # num_uavs x 3
        uav_coords_3d = np.concatenate(
            (uav_coords_2d, np.zeros((uav_coords_2d.shape[0], 1))), axis=1)

        v_uav_inds = np.sort(map.grid.nearest_inds(uav_coords_3d))

        if debug:

            def dist_to_nearest_uav(uav_coords, uc):
                dists = np.linalg.norm(uav_coords[:, 0:2] - uc[None, 0:2],
                                       axis=1)
                #dists = np.linalg.norm(uav_coords_2d.T - uc[None, 0:2], axis=1)
                return np.min(dists)

            max_min_dist = np.max(
                [dist_to_nearest_uav(uav_coords_3d, uc) for uc in user_coords])
            print("max hmin distance bf discretization: ", max_min_dist)

            uav_coords_grid = map.grid.list_pts()[v_uav_inds, :]
            max_min_dist = np.max([
                dist_to_nearest_uav(uav_coords_grid, uc) for uc in user_coords
            ])
            print("max hmin distance after discretization: ", max_min_dist)

            for uavc in uav_coords_3d:
                uavc_grid = map.grid.nearest_pt(uavc)
                print(
                    f"uav at {uavc} mapped to {uavc_grid}. HError: {np.linalg.norm(uavc[0:2]-uavc_grid[0:2])}"
                )

        return v_uav_inds