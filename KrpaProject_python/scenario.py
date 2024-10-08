# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np

from omni.isaac.core import *#PhysicsContext, SimulationContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import distance_metrics
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.motion_generation import ArticulationMotionPolicy, RmpFlow
from omni.isaac.motion_generation.interface_config_loader import load_supported_motion_policy_config

# cloth requirements
from omni.physx.scripts import physicsUtils, particleUtils
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, UsdShade, PhysxSchema

import math
import omni.usd
import omni.kit.commands
import omni.physxdemos as demo
import carb.settings

import time

# camera

from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
import matplotlib.pyplot as plt

# positioning

from omni.isaac.dynamic_control import _dynamic_control
from pxr import Usd

# pddl

import unified_planning as up
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *

import re

# orientation

from scipy.spatial.transform import Rotation as R

# constants

min_height = 0.01 #min height for franka not to collide with floor
cloth_main_resolution = 90

class FrankaRmpFlowExampleScript:
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

        self._script_generator = None

    def load_example_assets(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """
        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim(
            "/World/target",
            scale=[0.04, 0.04, 0.04],
            position=np.array([0.4, 0, 0.25]),
            orientation=euler_angles_to_quats([0, np.pi, 0]),
        )
        
        self._camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.5, 0.0, 4]),
            frequency=20,
            resolution=(256, 256),
            orientation=rot_utils.euler_angles_to_quats(np.array([90, 90, -180]), degrees=True),
        )
        
        self._ground_plane = GroundPlane("/World/Ground")

        # cloth import function
        self._stage = omni.usd.get_context().get_stage()
        self._default_prim_path = Sdf.Path("/World")
        self._scene = UsdPhysics.Scene.Define(self._stage, "/World/physicsScene")

        # doesnt work
        # physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(self._scene.GetPrim())
        # physx_scene_api.CreateEnableGPUDynamicsAttr().Set(True)
        # gpu_dynamics_enabled = physx_scene_api.GetEnableGPUDynamicsAttr().Get()
        #print("Current GPU Dynamics Enabled:", gpu_dynamics_enabled)

        # test
       # PhysxSchema.PhysxSceneAPI.Apply(self._stage.GetPrimAtPath("/World/physicsScene"))
        
       # physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(self._stage, "/World/physicsScene")
       # physxSceneAPI.CreateEnableGPUDynamicsAttr(True)


        # simulation_context = SimulationContext()
        # kontekst = simulation_context.get_physics_context()
        # print(kontekst)
        # kontekst.enable_gpu_dynamics(True)
        #print("sta: ", kontekst.device)
        #############

        self.add_cloth(self._stage)

        # camera functions
        self._camera.initialize()
        self._i = 0
        self._camera.add_motion_vectors_to_frame()

        # corners positioning
        #self._ic = 0
        self._target_points = np.array([[0.6, 0, min_height], [0,0,0], [0,0,0], [0,0,0]])

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target, self._ground_plane, self._camera
        #*self._obstacles, self._goal_block

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        # Set a camera view that looks good
        set_camera_view(eye=[2, 0.8, 1], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        # Loading RMPflow can be done quickly for supported robots
        rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(**rmp_config)

        # defining the obstacles
        #for obstacle in self._obstacles:
        #    self._rmpflow.add_obstacle(obstacle)

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        # Create a script generator to execute my_script().
        self._script_generator = self.my_script()

    def reset(self):
        """
        This function is called when the reset button is pressed.
        In this example the core.World takes care of all necessary resetting
        by putting everything back in the position it was in when loaded.

        In more complicated scripts, e.g. scripts that modify or create USD properties
        or attributes at runtime, the user will need to implement necessary resetting
        behavior to ensure their script runs deterministically.
        """
        self._i = 0
        # Start the script over by recreating the generator.
        self._script_generator = self.my_script()

    """
    The following two functions demonstrate the mechanics of running code in a script-like way
    from a UI-based extension.  This takes advantage of Python's yield/generator framework.  

    The update() function is tied to a physics subscription, which means that it will be called
    one time on every physics step (usually 60 frames per second).  Each time it is called, it
    queries the script generator using next().  This makes the script generator execute until it hits
    a yield().  In this case, no value need be yielded.  This behavior can be nested into subroutines
    using the "yield from" keywords.
    """

    def update(self, step: float):
        try:
            #result0 = next(self.camera_script())
            #result1 = next(self.particle_script())
            result = next(self._script_generator)
        except StopIteration:
            return True
    
##############################################################################################
#                           PROCEDURE SCRIPTS
##############################################################################################
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def grip_handler(self, current_point, parsed_action, orientation_target, want_orientation = False):
        # position before movement
        translation_target0, orientation_target0 = self._target.get_world_pose()
        
        # orientation target calculation
        P2 = self._close_points[1]
        P3 = self._close_points[0]
        theta = math.pi - math.atan2(P3[1]-current_point[1],P3[0]-current_point[0])-math.atan2(P2[1]-current_point[1],P2[0]-current_point[0])
        rotation_quaternion = R.from_euler('x', theta).as_quat()
        new_orient = self.quaternion_multiply(orientation_target0, rotation_quaternion)
       
        if want_orientation:
            orientation_target = new_orient
            print("Orientation target changed | New OT: ", new_orient)
        
        print(f"Going to {parsed_action['parameters'][1]} - {current_point}")
        self._target.set_world_pose(current_point, orientation_target)

        success = yield from self.goto_position(
            current_point, orientation_target, self._articulation, self._rmpflow, timeout=250
        )

        yield from self.close_gripper_franka(self._articulation)

    def p2p_handler(self, current_point, target_point_id, parsed_action, orientation_target):
        print(f"Going from {parsed_action['parameters'][1]} to {parsed_action['parameters'][2]} - {current_point}")
        
        
        # for better path
        middle_point = (current_point+self._target_points[target_point_id])/2 + np.array([0, 0, 0.1])

        print("Going up middle: ", middle_point)
        self._target.set_world_pose(middle_point, orientation_target)

        success = yield from self.goto_position(
            middle_point, orientation_target, self._articulation, self._rmpflow, timeout=250
        )

        # final target (next point pX)
        yield from self.particle_script()
        final_point = self._target_points[target_point_id] + np.array([0, 0, 0.05])
        print("Going to destination", final_point)
        self._target.set_world_pose(final_point, orientation_target)

        success = yield from self.goto_position(
            final_point, orientation_target, self._articulation, self._rmpflow, timeout=2500     
        )     

    def understanding_script(self):
        plan_str = str(self._result.plan)
        actions = plan_str.split('\n')  
        parsed_actions = []
        action_pattern = re.compile(r'(\w+)\(([^)]+)\)')

        # Parsing each action
        for action in actions:
            match = action_pattern.match(action.strip())
            if match:
                action_name = match.group(1)
                parameters = [param.strip() for param in match.group(2).split(',')]
                parsed_actions.append({'action': action_name, 'parameters': parameters})

        # Printing the parsed actions
        #for parsed_action in parsed_actions:
        #    print(f"Action: {parsed_action['action']}, Parameters: {parsed_action['parameters']}")
        

        # handling the acquired actions
        current_point, orientation_target = self._target.get_world_pose()
        
        for parsed_action in parsed_actions:
            #refreshing the positions
            time.sleep(0.5)
            yield from self.particle_script()


            action_name = parsed_action['action']
            parameters = parsed_action['parameters']
            if action_name == "grip":
                if parameters == ['g', 'p1']:
                    print("Grip action with parameters g, p1")
                    current_point = self._target_points[0]
                    yield from self.grip_handler(current_point, parsed_action, orientation_target)
                    
                elif parameters == ['g', 'p3']:
                    print("Grip action with parameters g, p3")
                    current_point = self._target_points[2]
                    yield from self.grip_handler(current_point, parsed_action, orientation_target, True)

            elif action_name == "p2p":
                if parameters == ['g', 'p1', 'p2']:
                    print("p2p action with parameters g, p1, p2")
                    current_point = self._target_points[0]
                    #target_point = self._target_points[1]
                    yield from self.p2p_handler(current_point, 1, parsed_action, orientation_target)
                elif parameters == ['g', 'p3', 'p4']:
                    print("p2p action with parameters g, p3, p4")
                    current_point = self._target_points[2]
                    #target_point = self._target_points[3]
                    yield from self.p2p_handler(current_point, 3, parsed_action, orientation_target)

            elif action_name == "release":
                if parameters == ['g', 'p1']:
                    print("Release action with parameters g, p1")
                    yield from self.open_gripper_franka(self._articulation)
                    #time.sleep(0.5)
                elif parameters == ['g', 'p3']:
                    print("Release action with parameters g, p3")
                    yield from self.open_gripper_franka(self._articulation)
                    #time.sleep(0.5)

                # go up after releasing
                translation_target, orientation_target = self._target.get_world_pose()
                self._target.set_world_pose(translation_target+np.array([0, 0, 0.1]), orientation_target)

                success = yield from self.goto_position(
                    translation_target+np.array([0, 0, 0.1]), orientation_target, self._articulation, self._rmpflow, timeout=2500     
                ) 
            else:
                print(f"Unknown action: {action_name} with parameters {parameters}")
    
        yield True

    def planning_script(self):

        domain_file = "/home/student/aldin/isaac_sim_ws/extensions/KrpaProject/KrpaProject_python/domain.pddl"
        problem_file = "/home/student/aldin/isaac_sim_ws/extensions/KrpaProject/KrpaProject_python/problem.pddl"

        reader = PDDLReader()

        pddl_problem = reader.parse_problem(domain_file, problem_file)
        #print(pddl_problem)

        # disabling the credits
        up.shortcuts.get_environment().credits_stream = None

        # solving the problem
        with OneshotPlanner(problem_kind=pddl_problem.kind) as planner:
            self._result = planner.solve(pddl_problem)
            print("%s returned: %s" % (planner.name, self._result.plan))
        
        yield True

    def particle_script(self):

        # visually, cloth looks like this with regard to indices:
        # 0, 51, 102, 153, 204 ........ 2346, 2397, 2448, 2499, 2550
        # 1 ................................................... 2551
        # . .................................................... .
        # 50, 101, 152, 203 ........... 2396, 2447, 2498, 2549, 2600
        p1_id = int((cloth_main_resolution+1)*((cloth_main_resolution+1)**2/((cloth_main_resolution+1)*10))) #260.1
        #p2_id = int(p1_id+cloth_main_resolution-1.5*((cloth_main_resolution+1)**2/((cloth_main_resolution+1)*10))) 
        p2_id = int(cloth_main_resolution)
        p4_id = int((cloth_main_resolution+1)**2-p1_id)
        p3_id = int(p4_id-cloth_main_resolution+1.5*((cloth_main_resolution+1)**2/((cloth_main_resolution+1)*10)))
        p4_id = int((cloth_main_resolution+1)**2-1)
        print("-----------------------------------------------------------")
        print("Corners IDs: ", p1_id, "/", p2_id, "/", p3_id, "/", p4_id)
        
        localToWorldTransform = self._xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        # absolute position of cloth
        position = localToWorldTransform.ExtractTranslation()

        # relative (local) positions of all particles
        geom_pts = UsdGeom.Points(self._stage.GetPrimAtPath("/Plane"))
        pos = geom_pts.GetPointsAttr().Get()
        np_pos = np.array(pos)

        self._abs_pos = np_pos+position # absolute positions of all particles

        # reconfig the z value because of the gripper
        if(self._abs_pos[p1_id][2] < min_height): #208, 250, 2350, 2392
            self._abs_pos[p1_id][2] = min_height 
        if(self._abs_pos[p2_id][2] < min_height):
            self._abs_pos[p2_id][2] = min_height 
        if(self._abs_pos[p3_id][2] < min_height):
            self._abs_pos[p3_id][2] = min_height 
        if(self._abs_pos[p4_id][2] < min_height):
            self._abs_pos[p4_id][2] = min_height    
        # points close to corners (not exactly on the corners because of the grasp possibilites)
        self._target_points = np.array([self._abs_pos[p1_id], self._abs_pos[p2_id], self._abs_pos[p3_id], self._abs_pos[p4_id]])
        #print("target points: ", self._target_points)
        self._close_points = np.array([self._abs_pos[p3_id+15], self._abs_pos[p3_id-(cloth_main_resolution+1)*15]])
        """"
        #######################################################################################
        # nice script to visually check all particles positions :
        #   comment the >> result = next(self._script_generator)
        #   in the update function before using this chunk of code
        #       for iterating positions, initialize self._ic in load_example_assets
        #       or just use hard-coded value
        #   also, uncomment >> result1 = next(self.particle_script())

        translation_target, orientation_target = self._target.get_world_pose()
        lower_translation_target = self._abs_pos[2350]
        self._target.set_world_pose(lower_translation_target, orientation_target)
            

        self._ic += 5
        if(self._ic > self._abs_pos.shape[0]):
            self._ic = 0
        #######################################################################################
        """
        
        yield True
        
    def camera_script(self):
        # camera_script is designed for cv2 methods usage
        if self._i == 120:
            
            imgplot = plt.imshow(self._camera.get_rgba()[:, :, :3])
            plt.show()
            plt.savefig('/home/student/aldin/isaac_sim_ws/extensions/KrpaProject/KrpaProject_python/proba.png', dpi=200)
            print("Picture saved")

        if(self._i < 121):
            self._i += 1       
        
        yield True

    def my_script(self):
        print("[MAIN] Executing grab and drop script...")
        translation_target, orientation_target = self._target.get_world_pose()

        yield from self.close_gripper_franka(self._articulation)

        success = yield from self.goto_position(
            translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        )

        if not success:
            print("[MAIN] Could not reach target position")
            return
        
        yield from self.open_gripper_franka(self._articulation)

        yield from self.particle_script()

        yield from self.planning_script()

        yield from self.understanding_script()
        
        print("[MAIN] Going up")
        yield from self.open_gripper_franka(self._articulation)
        high_translation_target = np.array([0.6, 0, 0.2])
        self._target.set_world_pose(high_translation_target, orientation_target)

        success = yield from self.goto_position(
            high_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        )
        print("[MAIN] Grab and drop procedure finished.")

######################################################################################################################
#                                   CLOTH IMPORT AND FUNCTIONS
######################################################################################################################

    def add_cloth(self, stage):
        # create a mesh that is turned into a cloth
        plane_mesh_path = Sdf.Path(omni.usd.get_stage_next_free_path(stage, "Plane", True))
        plane_resolution = cloth_main_resolution
        plane_width = 40

        # reset u/v scale for cube u/v being 1:1 the mesh resolution:
        SETTING_U_SCALE = "/persistent/app/mesh_generator/shapes/plane/u_scale"
        SETTING_V_SCALE = "/persistent/app/mesh_generator/shapes/plane/v_scale"
        SETTING_HALF_SCALE = "/persistent/app/mesh_generator/shapes/plane/object_half_scale"
        u_backup = carb.settings.get_settings().get(SETTING_U_SCALE)
        v_backup = carb.settings.get_settings().get(SETTING_V_SCALE)
        hs_backup = carb.settings.get_settings().get(SETTING_HALF_SCALE)
        carb.settings.get_settings().set(SETTING_U_SCALE, 1)
        carb.settings.get_settings().set(SETTING_V_SCALE, 1)
        carb.settings.get_settings().set(SETTING_HALF_SCALE, 0.5 * plane_width)

        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type="Plane",
            u_patches=plane_resolution,
            v_patches=plane_resolution,
        )

        # restore u/v scale backup
        carb.settings.get_settings().set(SETTING_U_SCALE, u_backup)
        carb.settings.get_settings().set(SETTING_V_SCALE, v_backup)
        carb.settings.get_settings().set(SETTING_HALF_SCALE, hs_backup)

        plane_mesh = UsdGeom.Mesh.Define(stage, plane_mesh_path)
        self._cloth = plane_mesh
        physicsUtils.setup_transform_as_scale_orient_translate(plane_mesh)
        physicsUtils.set_or_add_translate_op(plane_mesh, Gf.Vec3f(0.5, 0, 0.05)) #  y=1 , z = 0.3
        physicsUtils.set_or_add_orient_op(plane_mesh, Gf.Quatf(0, Gf.Vec3f(0, 0, 0))) #Gf.Quatf(0.965925826, Gf.Vec3f( 0.258819045, 0.0, 0.2588190451)))
        physicsUtils.set_or_add_scale_op(plane_mesh, Gf.Vec3f(1.0))

        # configure and create particle system
        # we use all defaults, so the particle contact offset will be 5cm / 0.05m
        # so the simulation determines the other offsets from the particle contact offset
        particle_system_path = self._default_prim_path.AppendChild("particleSystem")

        # size rest offset according to plane resolution and width so that particles are just touching at rest
        radius = 0.012 * (plane_width / plane_resolution) #0.5
        restOffset = radius
        contactOffset = restOffset * 0.001 #1.5
        print("Scene_path:"+str(self._scene.GetPath()))
        print("particle_system_path:"+str(particle_system_path))
        particleUtils.add_physx_particle_system(
            stage=stage,
            particle_system_path=particle_system_path,
            contact_offset=contactOffset,
            rest_offset=restOffset,
            particle_contact_offset=contactOffset,
            solid_rest_offset=restOffset,
            fluid_rest_offset=0.0,
            solver_position_iterations=16,
            simulation_owner=self._scene.GetPath(),
        )

        # create material and assign it to the system:
        particle_material_path = self._default_prim_path.AppendChild("particleMaterial")
        particleUtils.add_pbd_particle_material(stage, particle_material_path)
        # add some drag and lift to get aerodynamic effects
        particleUtils.add_pbd_particle_material(stage, particle_material_path, drag=0.1, lift=0.1, friction=0.3) #friction = 0.6
        # last working lift = 0.3, friction=1
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(particle_system_path), particle_material_path
        )

        # for particle positions
        self._xformable = UsdGeom.Xformable(plane_mesh)

        # configure as cloth
        stretchStiffness = 2500.0 #10000
        bendStiffness = 20.0 #200
        shearStiffness = 100.0 #100
        damping = 0.2 #0.2
        particleUtils.add_physx_particle_cloth(
            stage=stage,
            path=plane_mesh_path,
            dynamic_mesh_path=None,
            particle_system_path=particle_system_path,
            spring_stretch_stiffness=stretchStiffness,
            spring_bend_stiffness=bendStiffness,
            spring_shear_stiffness=shearStiffness,
            spring_damping=damping,
            self_collision=True,
            self_collision_filter=True,
        )

        # configure mass:
        particle_mass = 0.005 * (plane_width / (4*plane_resolution)) #0.02 default mass constant
        num_verts = len(plane_mesh.GetPointsAttr().Get())
        mass = particle_mass * num_verts
        massApi = UsdPhysics.MassAPI.Apply(plane_mesh.GetPrim())
        massApi.GetMassAttr().Set(mass)

        # add render material:
        material_path = self.create_pbd_material("OmniPBR")
        omni.kit.commands.execute(
            "BindMaterialCommand", prim_path=plane_mesh_path, material_path=material_path, strength=None
        )

    def create_pbd_material(self, mat_name: str, color_rgb: Gf.Vec3f = Gf.Vec3f(0.2, 0.2, 1)) -> Sdf.Path:
        # create material for extras
        create_list = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=create_list,
            bind_selected_prims=False,
        )
        target_path = "/World/Looks/" + mat_name
        if create_list[0] != target_path:
            omni.kit.commands.execute("MovePrims", paths_to_move={create_list[0]: target_path})
        shader = UsdShade.Shader.Get(self._stage, target_path + "/Shader")
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color_rgb)
        return Sdf.Path(target_path)

######################################################################################################################
#                                   OPERATIONAL FUNCTIONS
######################################################################################################################
    def goto_position(
        self,
        translation_target,
        orientation_target,
        articulation,
        rmpflow,
        translation_thresh=0.01,
        orientation_thresh=0.1,
        timeout=500,
    ):
        """
        Use RMPflow to move a robot Articulation to a desired task-space position.
        Exit upon timeout or when end effector comes within the provided threshholds of the target pose.
        """

        articulation_motion_policy = ArticulationMotionPolicy(articulation, rmpflow, 1 / 60)
        rmpflow.set_end_effector_target(translation_target, orientation_target)

        for i in range(timeout):
            ee_trans, ee_rot = rmpflow.get_end_effector_pose(
                articulation_motion_policy.get_active_joints_subset().get_joint_positions()
            )

            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, translation_target)
            rotation_target = quats_to_rot_matrices(orientation_target)
            rot_dist = distance_metrics.rotational_distance_angle(ee_rot, rotation_target)

            done = trans_dist < translation_thresh and rot_dist < orientation_thresh

            if done:
                return True

            rmpflow.update_world()
            action = articulation_motion_policy.get_next_articulation_action(1 / 60)
            articulation.apply_action(action)

            # If not done on this frame, yield() to pause execution of this function until
            # the next frame.
            yield ()

        return False

    def open_gripper_franka(self, articulation):
        open_gripper_action = ArticulationAction(np.array([0.04, 0.04]), joint_indices=np.array([7, 8]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully opened.
        while not np.allclose(articulation.get_joint_positions()[7:], np.array([0.04, 0.04]), atol=0.001):
            yield ()

        return True

    def close_gripper_franka(self, articulation, close_position=np.array([0, 0]), atol=0.001):
        # To close around the cube, different values are passed in for close_position and atol
        open_gripper_action = ArticulationAction(np.array(close_position), joint_indices=np.array([7, 8]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully closed.
        while not np.allclose(articulation.get_joint_positions()[7:], np.array(close_position), atol=atol):
            yield ()

        return True
