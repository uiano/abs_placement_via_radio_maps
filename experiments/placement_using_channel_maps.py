#from numpy.lib.arraysetops import isin
#from common.fields import FunctionVectorField
from collections import OrderedDict
from common.runner import Runner
import time
import numpy as np
from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import cvxopt as co
import scipy

from common.solvers import group_sparse_cvx, weighted_group_sparse_scipy

import gsim
from gsim.gfigure import GFigure
from common.utilities import dB_to_natural, dbm_to_watt, empty_array, natural_to_dB, watt_to_dbW, watt_to_dbm
from common.grid import RectangularGrid3D
from channels.channel import Channel, FreeSpaceChannel
from channels.tomographic_channel import TomographicChannel
from common.environment import BlockUrbanEnvironment1, BlockUrbanEnvironment2, GridBasedBlockUrbanEnvironment, UrbanEnvironment, Building

from placement.placers import FlyGrid, SingleUAVPlacer, \
    SparseUAVPlacer, KMeansPlacer, \
        SpaceRateKMeans, GridRatePlacer, SpiralPlacer,\
            SparseRecoveryPlacer
from simulators.PlacementSimulator import metrics_vs_min_user_rate, \
    metrics_vs_num_users,     place_and_plot, mean_num_uavs, user_loc_mc,\
        metrics_vs_environments_and_channels, metrics_vs_placers



class ExperimentSet(gsim.AbstractExperimentSet):
    def experiment_1000(l_args):
        print("Test experiment")

        return

    """###################################################################
    10. Preparatory experiments
    ###################################################################

    EXPERIMENT -------------------------------------------

    Channel map associated with a single source in free space.

    """

    def experiment_1001(l_args):

        # Grid
        area_len = [100, 80, 50]
        grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])

        # Free-space channel
        channel = FreeSpaceChannel(freq_carrier=3e9)
        pt_tx = grid.random_pts(z_val=0)[0]
        print(f"pt_tx = {pt_tx}")
        fl_path_loss = channel.dbgain_from_pt(grid=grid, pt_1=pt_tx)

        # Map at different heights
        F = fl_path_loss.plot_z_slices(zvals=[1, 7.5, 20, 40])

        return F

    """ EXPERIMENT -------------------------------------------

    Plot of two buildings.

    """

    def experiment_1002(l_args):

        area_len = [100, 80, 50]
        fly_grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[10, 10, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])

        env.plot()
        env.show()

    """ EXPERIMENT -------------------------------------------

    Approximation of a line integral.

    """

    def experiment_1003(l_args):
        area_len = [100, 80, 50]
        fly_grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[10, 10, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])        
        
        pt_tx = np.array([50, 60, 37])        
        pt_rx = np.array([19, 1, 0])
        print("points = ", [pt_tx, pt_rx])

        li = env.slf.line_integral(pt_tx, pt_rx, mode="python")
        print("line integral (Python) = ", li)

        li = env.slf.line_integral(pt_tx, pt_rx, mode="c")
        print("line integral (C) = ", li)

        env.dl_uavs = {'tx-rx': [pt_tx, pt_rx]}
        env.l_lines = [[pt_tx, pt_rx]]
        env.plot()
        env.show()

    """ EXPERIMENT -------------------------------------------

    Absorption and channel gain vs. position of the UAV for a single ground user.

    """

    def experiment_1004(l_args):
        area_len = [100, 80, 50]
        fly_grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[10, 10, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            #max_link_capacity=10,
        )
        #channel = TomographicChannel(grid=grid, slf=env.slf)

        pt_rx = np.array([80, 40, 0])
        pt_tx_start = np.array([1, 1, 1])
        pt_tx_end = np.array([1, 70, 40])
        v_t = np.linspace(0, 1, 1000)

        # Path loss vs. position --> the transmitter moves
        l_pt_tx = [pt_tx_start + t * (pt_tx_end - pt_tx_start) for t in v_t]
        absorption_loss = [
            channel.dbabsorption(pt_tx, pt_rx) for pt_tx in l_pt_tx
        ]
        free_space_gain = [
            channel.dbgain_free_space(pt_tx, pt_rx) for pt_tx in l_pt_tx
        ]
        total_gain = [channel.dbgain(pt_tx, pt_rx) for pt_tx in l_pt_tx]

        env.dl_uavs = {'rx': [pt_rx]}
        env.l_lines = [[pt_tx_start, pt_tx_end]]

        env.plot()
        env.show()

        F = GFigure(xaxis=v_t,
                    yaxis=absorption_loss,
                    xlabel="t",
                    ylabel="Absorption Loss [dB]")
        F.next_subplot(xaxis=v_t,
                       yaxis=free_space_gain,
                       xlabel="t",
                       ylabel="Free Space Gain [dB]")
        F.next_subplot(xaxis=v_t,
                       yaxis=total_gain,
                       xlabel="t",
                       ylabel="Total Gain [dB]")
        return F

    """ EXPERIMENT -------------------------------------------

    Channel gain map for a single ground user.

    """

    def experiment_1005(l_args):

        area_len = [100, 80, 50]
        fly_grid = RectangularGrid3D(area_len=area_len, num_pts=[20, 30, 5])
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[10, 10, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            #max_link_capacity=10,
        )

        pt_rx = np.array([19, 1, 0])
        map = channel.dbgain_from_pt(grid=env.slf.grid, pt_1=pt_rx)
        print("number of grid points = ", map.t_values.size)

        env.l_users = [pt_rx]
        env.plot()
        env.show()

        return map.plot_z_slices(zvals=[0, 20, 30, 40])

    """ EXPERIMENT -------------------------------------------

    Optimal placement of a single UAV for communicating with two users on the
    ground. 

    Good illustration of the objects involved in these simulations.

    """

    def experiment_1006(l_args):

        area_len = [100, 80, 50]
        fly_grid = FlyGrid(area_len=area_len,
                           num_pts=[10, 11, 7],
                           min_height=10)
        env = UrbanEnvironment(area_len=area_len,
                               num_pts_slf_grid=[20, 30, 5],
                               base_fly_grid=fly_grid,
                               buildings=[
                                   Building(sw_corner=[30, 50, 0],
                                            ne_corner=[50, 70, 0],
                                            height=70),
                                   Building(sw_corner=[20, 20, 0],
                                            ne_corner=[30, 30, 0],
                                            height=20),
                               ])
        channel = TomographicChannel(slf=env.slf)

        env.l_users = np.array([[10, 55, 2], [60, 60, 2]])

        pl = SingleUAVPlacer(criterion="max_min_rate")

        env.dl_uavs = {
            pl.name:
            pl.place(fly_grid=env.fly_grid,
                     channel=channel,
                     user_coords=env.l_users)
        }

        l_F = pl.plot_capacity_maps(fly_grid=fly_grid,
                                    channel=channel,
                                    user_coords=env.l_users)
        #map = channel.dbgain_from_pt(pt_1 = pt_rx_2)
        #print("number of grid points = ", map.t_values.size)

        env.plot()
        env.show()

        return l_F

    """ EXPERIMENT -------------------------------------------

    Tests with specific UrbanEnvironments.

    """

    def experiment_1007(l_args):

        # Base environment
        if False:
            area_len = [100, 80, 50]
            fly_grid = FlyGrid(area_len=area_len,
                               num_pts=[10, 11, 7],
                               min_height=10)
            env = UrbanEnvironment(area_len=area_len,
                                   num_pts_slf_grid=[20, 30, 5],
                                   base_fly_grid=fly_grid,
                                   buildings=[
                                       Building(sw_corner=[30, 50, 0],
                                                ne_corner=[50, 70, 0],
                                                height=70),
                                       Building(sw_corner=[20, 20, 0],
                                                ne_corner=[30, 30, 0],
                                                height=20),
                                   ])
        if True:
            env = BlockUrbanEnvironment1(num_pts_slf_grid=[20, 30, 5],
                                         num_pts_fly_grid=[8, 8, 3],
                                         min_fly_height=10,
                                         building_height=None,
                                         building_absorption=1)

        if False:
            env = BlockUrbanEnvironment2(num_pts_slf_grid=[20, 30, 5],
                                         num_pts_fly_grid=[10, 10, 3],
                                         min_fly_height=10,
                                         building_height=50,
                                         building_absorption=1)

        if False:
            env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                                 num_pts_slf_grid=[20, 30, 5],
                                                 num_pts_fly_grid=[9, 9, 3],
                                                 min_fly_height=50,
                                                 building_absorption=1)

        # Test to determine the dimensions of the area and comm parameters
        if True:
            freq_carrier = 2.4e9
            bandwidth = 20e6
            target_rate = 5e6
            min_snr = natural_to_dB(2**(target_rate / bandwidth) - 1)
            tx_dbpower = watt_to_dbW(.1)
            dbgain = Channel.dist_to_dbgain_free_space(500,
                                                       wavelength=3e8 /
                                                       freq_carrier)
            max_noise_dbpower = tx_dbpower + dbgain - min_snr

            channel = TomographicChannel(
                slf=env.slf,
                freq_carrier=freq_carrier,
                tx_dbpower=tx_dbpower,
                noise_dbpower=max_noise_dbpower,
                bandwidth=bandwidth,
                min_link_capacity=2,
                max_link_capacity=7,
            )

            max_dist = channel.max_distance_for_rate(min_rate=15e6)
            ground_radius = np.sqrt(max_dist**2 -
                                    env.fly_grid.min_enabled_height**2)
            print(f"ground_radius = {ground_radius}")

        env.plot()
        env.show()

        return

    
    """###################################################################
    20. Placement of multiple UAVs
    ###################################################################
    """
    """ EXPERIMENT -------------------------------------------

    Playground to run tests.

    """

    def experiment_2001(l_args):
        #np.random.seed(2021)

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        # Set to None one of the following
        min_user_rate = 15e6
        num_uavs = None

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_user_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")

        # channel = TomographicChannel(
        #     slf=env.slf,
        #     tx_dbpower=90,
        #     min_link_capacity=2,
        #     max_link_capacity=min_user_rate,
        # )

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_gr = GridRatePlacer(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate)
        pl_srec = SparseRecoveryPlacer(min_user_rate=min_user_rate)

        # # Choose:
        place_and_plot(environment=env,
                       channel=channel,
                       min_user_rate=min_user_rate,
                       l_placers=[pl_s, pl_km, pl_sr, pl_sp],
                       num_users=40)
        #d_out = mean_num_uavs(environment=env, channel=channel, min_user_rate=min_user_rate, l_placers=[pl_sp, pl_gr], num_users=135, num_mc_iter=3)
        #
        # d_out = user_loc_mc(env,
        #                     channel,
        #                     l_placers=[pl_sr, pl_km],
        #                     num_users=12,
        #                     min_user_rate=min_user_rate,
        #                     num_mc_iter=3)
        # print("output=", d_out)

    # beautiful illustration of placement
    # Conf. PAPER
    def experiment_2002(l_args):
        #np.random.seed(2021)

        env = BlockUrbanEnvironment1(num_pts_slf_grid=[20, 30, 5],
                                     num_pts_fly_grid=[8, 8, 3],
                                     min_fly_height=10,
                                     building_height=None,
                                     building_absorption=3)

        # Set to None one of the following
        min_user_rate = 15e6
        num_uavs = None

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
            disable_gridpts_by_dominated_verticals=False,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_user_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")

        # channel = TomographicChannel(
        #     slf=env.slf,
        #     tx_dbpower=90,
        #     min_link_capacity=2,
        #     max_link_capacity=min_user_rate,
        # )

        pl_gs = GroupSparseUAVPlacer(sparsity_tol=1e-2,
                                     criterion="min_uav_num",
                                     min_user_rate=min_user_rate,
                                     max_uav_total_rate=100)
        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_gr = GridRatePlacer(min_user_rate=min_user_rate, num_uavs=num_uavs)
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate)
        pl_srec = SparseRecoveryPlacer(min_user_rate=min_user_rate)

        # # Choose:
        place_and_plot(
            environment=env,
            channel=channel,
            min_user_rate=min_user_rate,
            #l_placers=[pl_s, pl_km, pl_sr, pl_sp],
            l_placers=[pl_s],
            num_users=90,
            disable_flying_gridpts_by_dominated_verticals=False,
            no_axes=True)
        #d_out = mean_num_uavs(environment=env, channel=channel, min_user_rate=min_user_rate, l_placers=[pl_sp, pl_gr], num_users=135, num_mc_iter=3)
        #
        # d_out = user_loc_mc(env,
        #                     channel,
        #                     l_placers=[pl_sr, pl_km],
        #                     num_users=12,
        #                     min_user_rate=min_user_rate,
        #                     num_mc_iter=3)
        # print("output=", d_out)

    """ EXPERIMENT -------------------------------------------

    Num UAVs to guarantee a minimum rate vs. num users.

    """

    def experiment_2010(l_args):
        #np.random.seed(2021)

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_user_rate = 5e6  # 15e6

        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=1e6,
            max_link_capacity=min_user_rate,
        )

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate)
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate)
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )
        pl_srec = SparseRecoveryPlacer(min_user_rate=min_user_rate)

        v_num_users = [10, 15, 30, 50, 70, 90]
        d_out = metrics_vs_num_users(
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s, pl_srec],
            #l_placers=[pl_sp],
            v_num_users=v_num_users,
            min_user_rate=min_user_rate,
            num_mc_iter=30 * 2)

        G = GFigure(
            xlabel="Number of users",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_num_users,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    # scaling by the sqrt of the distance (NeSh scaling) and greater absorption
    # -> Conf. PAPER
    def experiment_2011(l_args):
        #np.random.seed(2021)

        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=3)

        min_user_rate = 5e6  # 15e6

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=2.4e9,
                                     bandwidth=20e6,
                                     tx_dbpower=watt_to_dbW(.1),
                                     noise_dbpower=-96,
                                     min_link_capacity=1e6,
                                     max_link_capacity=min_user_rate,
                                     nesh_scaling=True)

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate)
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate)
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )
        pl_srec = SparseRecoveryPlacer(min_user_rate=min_user_rate)

        v_num_users = [10, 15, 30, 50, 70, 90]
        d_out = metrics_vs_num_users(
            environment=env,
            channel=channel,
            l_placers=[pl_srec, pl_km, pl_sp, pl_sr, pl_s],
            #l_placers=[pl_sp],
            v_num_users=v_num_users,
            min_user_rate=min_user_rate,
            num_mc_iter=60)  # 15/hour

        G = GFigure(
            xlabel="Number of users",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            legend_loc="upper left",
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_num_users,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    """ EXPERIMENT -------------------------------------------

    Num UAVs to guarantee a minimum rate vs. building height.

    """

    def experiment_2020(l_args):
        #np.random.seed(2021)

        l_heights = np.linspace(0, 60, 8)
        l_envs = [
            GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                           num_pts_slf_grid=[20, 30, 5],
                                           num_pts_fly_grid=[9, 9, 3],
                                           min_fly_height=50,
                                           building_absorption=1,
                                           building_height=height)
            for height in l_heights
        ]

        min_user_rate = 5e6

        l_channels = [
            TomographicChannel(
                slf=env.slf,
                freq_carrier=2.4e9,
                bandwidth=20e6,
                tx_dbpower=watt_to_dbW(.1),
                noise_dbpower=-96,
                #min_link_capacity=1e6,
                min_link_capacity=1e6,
                max_link_capacity=min_user_rate,
            ) for env in l_envs
        ]

        print(
            f"ground_radius = ", l_channels[0].max_ground_radius_for_height(
                min_rate=min_user_rate,
                height=l_envs[0].fly_grid.min_enabled_height))

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, )
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, )
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )

        d_out = metrics_vs_environments_and_channels(
            environments=l_envs,
            channels=l_channels,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            #l_placers=[pl_sp],
            num_users=10,
            min_user_rate=min_user_rate,
            num_mc_iter=600)

        G = GFigure(
            xlabel="Height of the buildings",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=l_heights,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    # higher rate
    def experiment_2021(l_args):
        #np.random.seed(2021)

        l_heights = np.linspace(0, 60, 8)
        l_envs = [
            GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                           num_pts_slf_grid=[20, 30, 5],
                                           num_pts_fly_grid=[9, 9, 3],
                                           min_fly_height=50,
                                           building_absorption=1,
                                           building_height=height)
            for height in l_heights
        ]

        min_user_rate = 20e6

        l_channels = [
            TomographicChannel(
                slf=env.slf,
                freq_carrier=2.4e9,
                bandwidth=20e6,
                tx_dbpower=watt_to_dbW(.1),
                noise_dbpower=-96,
                #min_link_capacity=1e6,
                min_link_capacity=1e6,
                max_link_capacity=min_user_rate,
            ) for env in l_envs
        ]

        print(
            f"ground_radius = ", l_channels[0].max_ground_radius_for_height(
                min_rate=min_user_rate,
                height=l_envs[0].fly_grid.min_enabled_height))

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, )
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, )
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )

        d_out = metrics_vs_environments_and_channels(
            environments=l_envs,
            channels=l_channels,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            #l_placers=[pl_sp],
            num_users=10,
            min_user_rate=min_user_rate,
            num_mc_iter=1800)

        G = GFigure(
            xlabel="Height of the buildings",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=l_heights,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    # denser slf grid along z
    def experiment_2022(l_args):
        #np.random.seed(2021)

        l_heights = np.linspace(0, 60, 8)
        l_envs = [
            GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                           num_pts_slf_grid=[20, 30, 150],
                                           num_pts_fly_grid=[9, 9, 3],
                                           min_fly_height=50,
                                           building_absorption=1,
                                           building_height=height)
            for height in l_heights
        ]

        min_user_rate = 20e6

        l_channels = [
            TomographicChannel(
                slf=env.slf,
                freq_carrier=2.4e9,
                bandwidth=20e6,
                tx_dbpower=watt_to_dbW(.1),
                noise_dbpower=-96,
                #min_link_capacity=1e6,
                min_link_capacity=1e6,
                max_link_capacity=min_user_rate,
            ) for env in l_envs
        ]

        print(
            f"ground_radius = ", l_channels[0].max_ground_radius_for_height(
                min_rate=min_user_rate,
                height=l_envs[0].fly_grid.min_enabled_height))

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, )
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, )
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )

        d_out = metrics_vs_environments_and_channels(
            environments=l_envs,
            channels=l_channels,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            #l_placers=[pl_sp],
            num_users=10,
            min_user_rate=min_user_rate,
            num_mc_iter=300)

        G = GFigure(
            xlabel="Height of the buildings",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=l_heights,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    # scaling by the sqrt of the distance (NeSh scaling) and greater absorption
    # -> Conf. PAPER
    def experiment_2023(l_args):
        #np.random.seed(2021)

        l_heights = np.linspace(0, 45, 8)
        l_envs = [
            GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                           num_pts_slf_grid=[20, 30, 150],
                                           num_pts_fly_grid=[9, 9, 5],
                                           min_fly_height=50,
                                           building_absorption=3,
                                           building_height=height)
            for height in l_heights
        ]

        min_user_rate = 20e6

        l_channels = [
            TomographicChannel(
                slf=env.slf,
                freq_carrier=2.4e9,
                bandwidth=20e6,
                tx_dbpower=watt_to_dbW(.1),
                noise_dbpower=-96,
                #min_link_capacity=1e6,
                min_link_capacity=1e6,
                max_link_capacity=min_user_rate,
                nesh_scaling=True,
            ) for env in l_envs
        ]

        print(
            f"ground_radius = ", l_channels[0].max_ground_radius_for_height(
                min_rate=min_user_rate,
                height=l_envs[0].fly_grid.min_enabled_height))

        pl_s = SparseUAVPlacer(min_user_rate=min_user_rate, sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=min_user_rate, )
        pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate, )
        pl_sp = SpiralPlacer(min_user_rate=min_user_rate, )

        d_out = metrics_vs_environments_and_channels(
            environments=l_envs,
            channels=l_channels,
            l_placers=[pl_km, pl_sp, pl_sr, pl_s],
            #l_placers=[pl_sp],
            num_users=10,
            min_user_rate=min_user_rate,
            num_mc_iter=180)  # 100/hour

        G = GFigure(
            xlabel="Height of the buildings",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee min. rate = {min_user_rate/1e6} Mb/s",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=l_heights,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        print("output=", d_out)
        return G

    """ EXPERIMENT -------------------------------------------

    Num UAVs to guarantee a minimum rate vs. the minimum rate.

    """

    def experiment_2030(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )
        #pl_srec = SparseRecoveryPlacer(min_user_rate=v_min_user_rate[0])

        num_users = 40
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # More users
    def experiment_2032(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[20, 30, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # Denser SLF grid and less noise
    def experiment_2033(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-100,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # Even less noise
    def experiment_2034(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # Less absorption
    def experiment_2035(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=.1)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # More absorption -> Iterate this for 400 MC (5 h)
    def experiment_2036(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=.5)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # More absorption, and NeSh scaling -> Iterate this for 400 MC (5 h)
    def experiment_2037(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 3],
                                             min_fly_height=50,
                                             building_absorption=3)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96,  #-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
            nesh_scaling=True)

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 60
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=80)

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G

    # More absorption, more gridpts, and NeSh scaling -> Iterate this for 400 MC (5 h)
    # Conf. PAPER
    def experiment_2038(l_args):
        env = GridBasedBlockUrbanEnvironment(area_len=[500, 400, 150],
                                             num_pts_slf_grid=[48, 40, 5],
                                             num_pts_fly_grid=[9, 9, 5],
                                             min_fly_height=50,
                                             building_absorption=3)

        min_rate = 1e6
        max_rate = 15e6
        channel = TomographicChannel(
            slf=env.slf,
            freq_carrier=2.4e9,
            bandwidth=20e6,
            tx_dbpower=watt_to_dbW(.1),
            noise_dbpower=-96, #-110,  #-96,
            min_link_capacity=min_rate,
            max_link_capacity=max_rate,
            nesh_scaling=True
        )

        max_dist = channel.max_distance_for_rate(min_rate=min_rate)
        ground_radius = np.sqrt(max_dist**2 -
                                env.fly_grid.min_enabled_height**2)
        print(f"ground_radius = {ground_radius}")
        # env.plot()
        # env.show()
        # return

        v_min_user_rate = np.linspace(min_rate, max_rate, 6)
        pl_s = SparseUAVPlacer(min_user_rate=v_min_user_rate[0],
                               sparsity_tol=1e-2)
        pl_km = KMeansPlacer(min_user_rate=v_min_user_rate[0], )
        pl_sr = SpaceRateKMeans(min_user_rate=v_min_user_rate[0],
                                use_kmeans_as_last_resort=True)
        pl_sp = SpiralPlacer(min_user_rate=v_min_user_rate[0], )

        num_users = 80
        d_out = metrics_vs_min_user_rate(  #
            environment=env,
            channel=channel,
            l_placers=[pl_km, pl_sr, pl_sp, pl_s],
            num_users=num_users,
            l_min_user_rates=v_min_user_rate,
            num_mc_iter=100) # 20/35 min

        print("output=", d_out)
        G = GFigure(
            xlabel="Minimum rate [Mb/s]",
            ylabel="Mean number of ABSs",
            title=
            f"Minimum number of ABSs to guarantee a min. rate for {num_users} users",
            legend=list(d_out.keys()),
            styles=['-o', '-x', '-*', '-v', '-^'],
            xaxis=v_min_user_rate / 1e6,
            yaxis=[
                d_out[placer_name]['num_uavs'] for placer_name in d_out.keys()
            ])
        return G


def m(A):
    if isinstance(A, list):
        return [m(Am) for Am in A]
    return co.matrix(A)


def um(M):  #"unmatrix"
    if isinstance(M, list):
        return [um(Mm) for Mm in M]
    return np.array(M)


def sparsify(M, tol=0.01):
    n = np.linalg.norm(np.ravel(M), ord=1)
    M[M < tol * n] = 0
    return M


def group_sparsify(M, tol=0.01):
    n = np.linalg.norm(np.ravel(M), ord=1)
    for ind_col in range(M.shape[1]):
        if np.linalg.norm(M[:, ind_col]) < tol * n:
            M[:, ind_col] = 0
    return M
