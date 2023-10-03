import hj_reachability as hj
import jax.numpy as jnp
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
from refine_cbfs import HJControlAffineDynamics
import numpy as np
import rospy


class Config:
    def __init__(self, hj_setup=True):
        self.dynamics_class = rospy.get_param("~/env/dynamics_class")
        self.dynamics = self.setup_dynamics()
        self.control_space = rospy.get_param("~/env/control_space")  # These need to be box spaces
        self.disturbance_space = rospy.get_param("~/env/disturbance_space")
        self.state_domain = rospy.get_param("~/env/state_domain")
        self.grid = self.setup_grid()

        if hj_setup:
            self.safe_set = rospy.get_param("~/env/safe_set")
            self.obstacles = rospy.get_param("~/env/obstacles")
            control_space_hj = hj.sets.Box(lo=jnp.array(self.control_space["lo"]), 
                                           hi=jnp.array(self.control_space["hi"]))
            dist_space_hj = hj.sets.Box(lo=jnp.array(self.disturbance_space["lo"]), 
                                        hi=jnp.array(self.disturbance_space["hi"]))
            self.hj_dynamics = HJControlAffineDynamics(self.dynamics, control_space=control_space_hj, 
                                                       disturbance_space=dist_space_hj)
        
        self.assert_valid(hj_setup)

    def assert_valid(self, hj_setup):
        assert len(self.control_space["lo"]) == self.dynamics.control_dims
        assert len(self.control_space["hi"]) == self.dynamics.control_dims
        assert self.dynamics.n_dims == self.grid.ndim
        assert self.dynamics.periodic_dims == np.where(self.grid._is_periodic_dim)[0].tolist()

        if hj_setup:
            assert len(self.safe_set["lo"]) == self.grid.ndim
            assert len(self.safe_set["hi"]) == self.grid.ndim
            for obstacle in self.obstacles:
                assert len(obstacle["lo"]) == self.grid.ndim
                assert len(obstacle["hi"]) == self.grid.ndim
        
        
    def setup_dynamics(self):
        if self.dynamics_class == "quad_near_hover":            
            return QuadNearHoverPlanarDynamics(params={"dt": 0.1, "g": 9.81}) 
        elif self.dynamics_class == "dubins_car":
            return DubinsCarDynamics(params={"dt": 0.1})
        else:
            raise ValueError("Invalid dynamics type: {}".format(self.dynamics_class))

    def setup_grid(self):
        bounding_box = hj.sets.Box(lo=jnp.array(self.state_domain["lo"]), 
                                   hi=jnp.array(self.state_domain["hi"]))
        grid_resolution = self.state_domain["resolution"]
        p_dims = self.state_domain["periodic_dims"]
        return hj.Grid.from_lattice_parameters_and_boundary_conditions(bounding_box, grid_resolution,
                                                                       periodic_dims=p_dims)

class QuadNearHoverPlanarDynamics(ControlAffineDynamics):
    """
    Simplified dynamics, and we need to convert controls from phi to tan(phi)"""
    STATES = ["y", "z", "v_y", "v_z"]
    CONTROLS = ["tan(phi)", "T"]
    def __init__(self, params, **kwargs):
        self.g = params.get("g", 9.81)
        super().__init__(params, kwargs)
    
    def open_loop_dynamics(self, state, time: float = 0.0):
        return jnp.array([state[2], state[3], 0.0, -self.g])
    
    def control_matrix(self, state, time: float = 0.0):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [self.g, 0.0], [0.0, 1.0]])
    
    def disturbance_jacobian(self, state, time: float = 0.0):
        return jnp.expand_dims(jnp.zeros(4), axis=-1)
    
class DubinsCarDynamics(ControlAffineDynamics):
    """
    Dubins Car Dynamics for Turtlebot
    """
    STATES = ["x","y","theta"]
    CONTROLS = ["w","v"]
    def __init__(self, params, **kwargs):
        super().__init__(params, kwargs)

    def open_loop_dynamics(self,state,time: float = 0.0):
        return jnp.array([0.0,0.0,0.0])
    
    def control_matrix(self, state, time: float = 0.0):
        return jnp.array([[0.0,jnp.cos(state[2])],[0.0,jnp.sin(state[2])],[1.0,0.0]])
    
    def disturbance_jacobian(self, state, time: float = 0.0):
        return jnp.array([[1.0,0.0],[0.0,1.0],[0.0,0.0]])

    
