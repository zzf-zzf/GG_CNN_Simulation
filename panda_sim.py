"""
original file come from pybullet
"""

import pybullet as pb
import numpy as np
import math

UseNullSpace = 1
IkSolver = 0
PandaEndEffector = 11
PandaDofs = 9 #we can change the Dof to optimize the inverse kinematic

lower_limit = [-PandaDofs] * PandaDofs
Up_Null = [PandaDofs] * PandaDofs
Joint_Range = [PandaDofs] * PandaDofs
#restposes for null space
JointPositions = (0.8045609285966308, 0.525471701354679, -0.02519566900946519, -1.3925086098003587,
                  0.013443782914225877, 1.9178323512245277, -0.007207024243406651, 0.01999436579245478,
                  0.019977024051412193)
R_P = JointPositions

class PandaSim(object):
    def __init__(self, client, offset):
        """

        :param client:
        :param offset:
        """
        self.zzf = client
        self.zzf.setPhysicsEngineParameter(solverResidualThreshold=0) #set the velocity threshold
        self.offset = np.array(offset) #the start position of robot arm

        flags = self.zzf.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        orien = [0, 0, 0, 1]
        self.PandaId = self.zzf.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0])+self.offset, orien,
                                         useFixedBase=True, flags=flags)
        Index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2

        #create a constraint to keep the fingers centered
        Cen = self.zzf.createConstraint(self.PandaId, 9, self.PandaId, 10, jointType=self.zzf.JOINT_GEAR,
                                        jointAxis=[1, 0, 0], parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])
        self.zzf.changeConstraint(Cen, gearRatio=-1, erp=0.1, maxForce=50)

        for j in range(self.zzf.getNumJoints(self.PandaId)):
            self.zzf.changeDynamics(self.PandaId, j, linearDamping=0.1, angularDamping=0.1) #damping affect the smooth of grasp
            info = self.zzf.getJointInfo(self.PandaId, j)
            Joint_Name = info[1]
            Joint_Type = info[2]
            if (Joint_Type == self.zzf.JOINT_PRISMATIC): #arthrodia
                self.zzf.resetJointState(self.PandaId, j, JointPositions[Index])
                Index += 1

            if (Joint_Type == self.zzf.JOINT_REVOLUTE): #rotate
                self.zzf.resetJointState(self.PandaId, j, JointPositions[Index])
                Index += 1

        self.t = 0


    def Cal_Joint_Loc(self, pos, orien):
        """
        calculate the position of joints according pos and orn by inverse kinematic
        :param pos:
        :param orien:
        :return:
        """
        Joint_Pose = self.zzf.calculateInverseKinematics(self.PandaId, PandaEndEffector, pos, orien, lower_limit,
                                                         Up_Null, Joint_Range, R_P, maxNumIterations=50)
        return Joint_Pose

    def Set_ArmPose(self, pos):
        """

        :param pos:
        :return:
        """
        orien = self.zzf.getQuaternionFromEuler([math.pi, 0., math.pi/2]) # the direction of mechanical arm
        Joint_Pose = self.Cal_Joint_Loc(pos, orien)
        self.Set_Arm(Joint_Pose)

    def Set_Arm(self, Joint_Pose, maxVelocity=10):
        """

        :param Joint_Pose:
        :param maxVelocity:
        :return:
        """
        for i in range(PandaDofs):
            self.zzf.setJointMotorControl2(self.PandaId, i, self.zzf.POSITION_CONTROL, Joint_Pose[i],
                                           force=10000, maxVelocity=maxVelocity)

    def Set_Gripper(self, Finger_Target):
        """

        :param Finger_Target:
        :return:
        """
        for i in  [9, 10]: #the index of gripper
            self.zzf.setJointMotorControl2(self.PandaId, i, self.zzf.POSITION_CONTROL, Finger_Target, force=300)



    def process(self, pos, angle, gripper_width): #grasping processes
        """

        :param pos:
        :param angle:
        :param gripper_width:
        :return:
        """
        # update state
        self.u_s()

        pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)

        pos[2] += 0.001
        if self.state == 0:
            pos[0] = 0
            pos[1] = 0
            pos[2] = 0.2
            orien = self.zzf.getQuaternionFromEuler([math.pi, 0., angle+math.pi/2])
            Joint_Pose = self.Cal_Joint_Loc(pos, orien)
            self.Set_Arm(Joint_Pose)
            self.Set_Gripper(gripper_width)
            return False

        elif self.state == 1:
            pos[2] = 0.15
            orien = self.zzf.getQuaternionFromEuler([math.pi, 0., angle+math.pi/2])
            Joint_Pose = self.Cal_Joint_Loc(pos, orien)
            self.Set_Arm(Joint_Pose)
            return False

        elif self.state == 2:
            orien = self.zzf.getQuaternionFromEuler([math.pi, 0., angle+math.pi/2])
            Joint_Pose = self.Cal_Joint_Loc(pos, orien)
            self.Set_Arm(Joint_Pose, maxVelocity=4)
            return False

        elif self.state == 3:
            self.Set_Gripper(0)
            return False

        elif self.state == 4:
            pos[2] += 0.01
            orien = self.zzf.getQuaternionFromEuler([math.pi, 0., angle+math.pi/2])
            Joint_Pose = self.Cal_Joint_Loc(pos, orien)
            self.Set_Arm(Joint_Pose, maxVelocity=1)
            return False

        elif self.state == 5:
            pos[2] = 0.35
            orien = self.zzf.getQuaternionFromEuler([math.pi, 0., angle+math.pi/2])
            Joint_Pose = self.Cal_Joint_Loc(pos, orien)
            self.Set_Arm(Joint_Pose, maxVelocity=1)
            return False

        elif self.state == 6:
            pos[0] = 0
            pos[1] = -1.2 #finally location of grasped object
            pos[2] = 0.3
            orien = self.zzf.getQuaternionFromEuler([0., -math.pi, angle + math.pi / 2])
            Joint_Pose = self.Cal_Joint_Loc(pos, orien)
            self.Set_Arm(Joint_Pose, maxVelocity=2)
            return False

        elif self.state == 7:
            pos[0] = 0
            pos[1] = -1.2
            pos[2] = 0.1
            orien = self.zzf.getQuaternionFromEuler([0., -math.pi, angle + math.pi / 2])
            Joint_Pose = self.Cal_Joint_Loc(pos, orien)
            self.Set_Arm(Joint_Pose, maxVelocity=2)
            return False

        elif self.state == 8:
            self.Set_Gripper(10)
            return False

        elif self.state == 9:
            pos[0] = 0
            pos[1] = -1.2
            pos[2] = 0.3
            orien = self.zzf.getQuaternionFromEuler([0., -math.pi, angle + math.pi / 2])
            Joint_Pose = self.Cal_Joint_Loc(pos, orien)
            self.Set_Arm(Joint_Pose, maxVelocity=2)
            return False

        elif self.state == 10:
            pos[0] = 0.5
            pos[1] = -0.6
            pos[2] = 0.2
            orien = self.zzf.getQuaternionFromEuler([math.pi, 0., 0.])
            Joint_Pose = self.Cal_Joint_Loc(pos, orien)
            self.Set_Arm(Joint_Pose, maxVelocity=2)
            return False

        elif self.state == 11:
            """
            reset the state
            """
            self.reset()
            return True

    def reset(self):
        self.state = 0
        self.state_t = 0
        self.Current = 0


class PandaSimAuto(PandaSim):
    def __init__(self, client, offset):
        PandaSim.__init__(self, client, offset)
        self.state_time = 0
        self.Current = 0
        self.States = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.Dur_State = [1, 0.15, 0.3, 0.3, 0.3, 1.5, 1.25, 0.5, 0.5, 0.5, 0.6, 0.6]

    def u_s(self):
        self.state_time += self.control_dt
        if self.state_time > self.Dur_State[self.Current]:
            self.Current += 1
            if self.Current >= len(self.States):
                self.Current = 0
            self.state_time = 0
            self.state = self.States[self.Current]


        






















