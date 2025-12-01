import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CELLO_CONFIG = ArticulationCfg(
    # TODO: remove hardcoded nucleus address
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"omniverse://nucleus.fortableau.com/Users/tfoldi/Cello_v1.usd"
    ),
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"], stiffness=None, damping=None
        )
    },
)
