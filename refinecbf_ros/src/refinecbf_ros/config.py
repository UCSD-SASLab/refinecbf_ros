import hj_reachability as hj
import jax.numpy as jnp
import jax
from cbf_opt import ControlAffineDynamics
from refine_cbfs import HJControlAffineDynamics
import numpy as np
import rospy


class Config:
    def __init__(self, hj_setup=False):
        self.dynamics_class = rospy.get_param("~/env/dynamics_class")
        self.dynamics = self.setup_dynamics()
        self.control_space = rospy.get_param("~/env/control_space")  # These need to be box spaces
        self.disturbance_space = rospy.get_param("~/env/disturbance_space")
        self.state_domain = rospy.get_param("~/env/state_domain")
        self.grid = self.setup_grid()

        if hj_setup:
            self.safe_set = rospy.get_param("~/env/safe_set")
            self.obstacle_list = rospy.get_param("~/env/obstacles")
            self.boundary_env = rospy.get_param("~/env/boundary")
            (
                self.detection_obstacles,
                self.service_obstacles,
                self.update_obstacles,
                self.active_obstacles,
                self.boundary,
            ) = self.setup_obstacles()
            if self.control_space["n_dims"] == 0:
                control_space_hj = hj.sets.Box(lo=jnp.array([]), hi=jnp.array([]))
            else:
                control_space_hj = hj.sets.Box(
                    lo=jnp.array(self.control_space["lo"]), hi=jnp.array(self.control_space["hi"])
                )
            if self.disturbance_space["n_dims"] == 0:
                dist_space_hj = hj.sets.Box(lo=jnp.array([]), hi=jnp.array([]))
            else:
                dist_space_hj = hj.sets.Box(
                    lo=jnp.array(self.disturbance_space["lo"]), hi=jnp.array(self.disturbance_space["hi"])
                )
            self.hj_dynamics = HJControlAffineDynamics(
                self.dynamics, control_space=control_space_hj, disturbance_space=dist_space_hj
            )

        self.assert_valid(hj_setup)

    def assert_valid(self, hj_setup):
        assert len(self.control_space["lo"]) == self.dynamics.control_dims
        assert len(self.control_space["hi"]) == self.dynamics.control_dims
        assert self.dynamics.n_dims == self.grid.ndim
        assert self.dynamics.periodic_dims == np.where(self.grid._is_periodic_dim)[0].tolist()

        if hj_setup:
            assert len(self.safe_set["lo"]) == self.grid.ndim
            assert len(self.safe_set["hi"]) == self.grid.ndim
            for obstacle in self.obstacle_list.values():
                if obstacle["type"] == "Circle":
                    assert len(obstacle["center"]) == len(obstacle["indices"])
                if obstacle["type"] == "Rectangle":
                    assert len(obstacle["minVal"]) == len(obstacle["indices"])
                    assert len(obstacle["maxVal"]) == len(obstacle["indices"])
            assert len(self.boundary_env["minVal"]) == len(self.boundary_env["indices"])
            assert len(self.boundary_env["maxVal"]) == len(self.boundary_env["indices"])

    def setup_obstacles(self):
        detection_obstacles = []  # Obstacles that are "detected" by the robot when in close enough range
        service_obstacles = []  # Obstalces that are activated by a service
        update_obstacles = []  # Obstacles that become activated after a specified amount of time
        active_obstacles = [] # Obstacles that are always active
        for obstacle in self.obstacle_list.values():
            if obstacle["mode"] == "Detection":
                if obstacle["type"] == "Circle":
                    detection_obstacles.append(
                        Circle(
                            stateIndices=obstacle["indices"],
                            radius=obstacle["radius"],
                            center=obstacle["center"],
                            updateRule="Detection",
                            padding=obstacle["padding"],
                            detectionRadius=obstacle["detectionradius"],
                        )
                    )
                elif obstacle["type"] == "Rectangle":
                    detection_obstacles.append(
                        Rectangle(
                            stateIndices=obstacle["indices"],
                            minVal=obstacle["minVal"],
                            maxVal=obstacle["maxVal"],
                            updateRule="Detection",
                            padding=obstacle["padding"],
                            detectionRadius=obstacle["detectionradius"],
                        )
                    )
                else:
                    raise ValueError("Invalid Obstacle Type: {}".format(obstacle["type"]))
            elif obstacle["mode"] == "Update":
                if obstacle["type"] == "Circle":
                    update_obstacles.append(
                        Circle(
                            stateIndices=obstacle["indices"],
                            radius=obstacle["radius"],
                            center=obstacle["center"],
                            updateRule="Update",
                            padding=obstacle["padding"],
                            updateTime=obstacle["updatetime"],
                        )
                    )
                elif obstacle["type"] == "Rectangle":
                    update_obstacles.append(
                        Rectangle(
                            stateIndices=obstacle["indices"],
                            minVal=obstacle["minVal"],
                            maxVal=obstacle["maxVal"],
                            updateRule="Update",
                            padding=obstacle["padding"],
                            updateTime=obstacle["updatetime"],
                        )
                    )
                else:
                    raise ValueError("Invalid Obstacle Type: {}".format(obstacle["type"]))
            elif obstacle["mode"] == "Service":
                if obstacle["type"] == "Circle":
                    service_obstacles.append(
                        Circle(
                            stateIndices=obstacle["indices"],
                            radius=obstacle["radius"],
                            center=obstacle["center"],
                            updateRule="Service",
                            padding=obstacle["padding"],
                        )
                    )
                elif obstacle["type"] == "Rectangle":
                    service_obstacles.append(
                        Rectangle(
                            stateIndices=obstacle["indices"],
                            minVal=obstacle["minVal"],
                            maxVal=obstacle["maxVal"],
                            updateRule="Service",
                            padding=obstacle["padding"],
                        )
                    )
            elif obstacle["mode"] == "Active":
                if obstacle["type"] == "Circle":
                    active_obstacles.append(
                        Circle(
                            stateIndices=obstacle["indices"],
                            radius=obstacle["radius"],
                            center=obstacle["center"],
                            updateRule="Active",
                            padding=obstacle["padding"],
                        )
                    )
                elif obstacle["type"] == "Rectangle":
                    active_obstacles.append(
                        Rectangle(
                            stateIndices=obstacle["indices"],
                            minVal=obstacle["minVal"],
                            maxVal=obstacle["maxVal"],
                            updateRule="Active",
                            padding=obstacle["padding"],
                        )
                    )
                else:
                    raise ValueError("Invalid Obstacle Type: {}".format(obstacle["type"]))
            else:
                raise ValueError("Invalid Obstacle Activation Type: {}".format(obstacle["mode"]))

        boundary = Boundary(
            stateIndices=self.boundary_env["indices"],
            minVal=self.boundary_env["minVal"],
            maxVal=self.boundary_env["maxVal"],
            padding=self.boundary_env["padding"],
        )

        return detection_obstacles, service_obstacles, update_obstacles, active_obstacles, boundary

    def setup_dynamics(self):
        if self.dynamics_class == "quad_near_hover":
            return QuadNearHoverPlanarDynamics(params={"g": 9.81}, dt=0.05, test=False)
        elif self.dynamics_class == "dubins_car":
            return DubinsCarDynamics(params={"g": 9.81}, dt=0.05, test=False)
        else:
            raise ValueError("Invalid dynamics type: {}".format(self.dynamics_class))

    def setup_grid(self):
        bounding_box = hj.sets.Box(lo=jnp.array(self.state_domain["lo"]), hi=jnp.array(self.state_domain["hi"]))
        grid_resolution = self.state_domain["resolution"]
        p_dims = self.state_domain["periodic_dims"]
        return hj.Grid.from_lattice_parameters_and_boundary_conditions(
            bounding_box, grid_resolution, periodic_dims=p_dims
        )


# Obstacle Classes
class Obstacle:
    def __init__(self, type, stateIndices, updateRule, padding, updateTime, detectionRadius) -> None:
        self.type = type
        self.stateIndices = stateIndices
        self.updateRule = updateRule
        self.padding = padding
        self.updateTime = updateTime
        self.detectionRadius = detectionRadius


class Circle(Obstacle):
    def __init__(
        self, stateIndices, radius, center, updateRule="Time", padding=0, updateTime=None, detectionRadius=None
    ) -> None:
        super().__init__("Circle", stateIndices, updateRule, padding, updateTime, detectionRadius)
        self.radius = radius
        self.center = jnp.reshape(np.array(center), (-1, 1))

    def obstacle_sdf(self, x):
        obstacle_sdf = (
            jnp.linalg.norm(jnp.array([self.center - jnp.reshape(x[..., self.stateIndices],(-1,1))])) - self.radius - self.padding
        )
        return obstacle_sdf

    def distance_to_obstacle(self, state):
        point = state[self.stateIndices]
        distance = np.linalg.norm(self.center - point) - self.radius - self.padding
        return max(distance, 0.0)


class Rectangle(Obstacle):
    def __init__(
        self, stateIndices, minVal, maxVal, updateRule="Time", padding=0, updateTime=None, detectionRadius=None
    ) -> None:
        super().__init__("Rectangle", stateIndices, updateRule, padding, updateTime, detectionRadius)
        self.minVal = jnp.reshape(np.array(minVal), (-1, 1))
        self.maxVal = jnp.reshape(np.array(maxVal), (-1, 1))

    def obstacle_sdf(self, x):
        max_dist_per_dim = jnp.max(
            jnp.array([self.minVal - jnp.reshape(x[..., self.stateIndices],(-1,1)), jnp.reshape(x[..., self.stateIndices],(-1,1)) - self.maxVal]), axis=0
        )

        def outside_obstacle(_):
            return jnp.linalg.norm(jnp.maximum(max_dist_per_dim, 0))

        def inside_obstacle(_):
            return jnp.max(max_dist_per_dim)

        obstacle_sdf = (
            jax.lax.cond(jnp.all(max_dist_per_dim < 0.0), inside_obstacle, outside_obstacle, operand=None)
            - self.padding
        )
        return obstacle_sdf

    def distance_to_obstacle(self, state):
        point = state[self.stateIndices]
        if np.all(np.logical_and(self.minVal <= point, point <= self.maxVal)):
            return 0.0
        distances = np.abs(point - np.maximum(self.minVal, np.minimum(point, self.maxVal)))
        return np.linalg.norm(distances)


class Boundary(Obstacle):
    def __init__(self, stateIndices, minVal, maxVal, padding=0) -> None:
        super().__init__("Boundary", stateIndices, None, padding, updateTime=None, detectionRadius=None)
        self.minVal = jnp.reshape(np.array(minVal), (-1, 1))
        self.maxVal = jnp.reshape(np.array(maxVal), (-1, 1))

    def boundary_sdf(self, x):
        max_dist_per_dim = jnp.max(
            jnp.array([self.minVal - jnp.reshape(x[..., self.stateIndices],(-1,1)), jnp.reshape(x[..., self.stateIndices],(-1,1)) - self.maxVal]), axis=0
        )

        def outside_boundary(_):
            return -jnp.linalg.norm(jnp.maximum(max_dist_per_dim, 0))

        def inside_boundary(_):
            return -jnp.max(max_dist_per_dim)

        obstacle_sdf = (
            jax.lax.cond(jnp.all(max_dist_per_dim < 0.0), inside_boundary, outside_boundary, operand=None)
            - self.padding
        )
        return obstacle_sdf


# Dynamics Classes
class QuadNearHoverPlanarDynamics(ControlAffineDynamics):
    """
    Simplified dynamics, and we need to convert controls from phi to tan(phi)"""

    STATES = ["y", "z", "v_y", "v_z"]
    CONTROLS = ["tan(phi)", "T"]

    def open_loop_dynamics(self, state, time: float = 0.0):
        return jnp.array([state[2], state[3], 0.0, -self.params["g"]])

    def control_matrix(self, state, time: float = 0.0):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [-self.params["g"], 0.0], [0.0, 1.0]])


class DubinsCarDynamics(ControlAffineDynamics):
    """
    Dubins Car Dynamics for the Turtlebot
    """

    STATES = ["x", "y", "theta"]
    CONTROLS = ["omega", "v"]
    # DISTURBANCES = ["dx", "dy"]

    def open_loop_dynamics(self, state, time: float = 0):
        return jnp.array([0.0, 0.0, 0.0])

    def control_matrix(self, state, time: float = 0.0):
        return jnp.array([[jnp.cos(state[2]), 0.0], [jnp.sin(state[2]), 0.0], [0.0, 1.0]])

    # def disturbance_jacobian(self, state, time: float = 0.0):
    #     return jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
