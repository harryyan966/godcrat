import numpy as np
import pandas as pd
import cratutils as u


########## Top Level Variables ##########


# TODO write comment for all variables
SELF_FRAMES_BEFORE = 0
OTHERS_FRAMES_BEFORE = 2
FRAMES_BEFORE = max(SELF_FRAMES_BEFORE, OTHERS_FRAMES_BEFORE)
SELF_FRAMES_AFTER = 5
OTHERS_FRAMES_AFTER = 2
FRAMES_AFTER = max(SELF_FRAMES_AFTER, OTHERS_FRAMES_AFTER)


########## Agent: Utility Abstraction Class ##########


class Agent():
    def __init__(self, agentRow):
        self.x, self.y, self.vx, self.vy, self.ax, self.ay, self.yaw, self.id, self.type = \
        agentRow['X'], agentRow['Y'], agentRow['V_X'], agentRow['V_Y'], agentRow['A_X'], agentRow['A_Y'], 
        (agentRow['YAW']+90)%(360)*np.pi/180,      # converting to radian where straight to right is zero
        agentRow['TRACK_ID'], agentRow['OBJECT_TYPE']

        self.ar = np.sqrt(self.ax**2, self.ay**2)
        self.atheta = np.arctan2(self.ay, self.ax)

        self.vr = np.sqrt(self.vx**2, self.vy**2)
        self.vtheta = np.arctan2(self.vy, self.vx)

        self.r = np.sqrt(self.x**2, self.y**2)
        self.theta = np.arctan2(self.y, self.x)
        
        # projection of acceleration vector to speed vector, "l" stands for "leading"
        self.lar = self.ar * np.cos(self.vtheta - self.atheta)

    
    def laSmall(self) -> bool:      # TODO: tune the number (currently 0.5) for these four functions 
        '''
        Stands for "as (L)eading, (A)cceleration is (Small)"

        Same applies for the three functions below named like this
        '''
        return abs(self.lar) <= 0.5
    

    def laNeg(self) -> bool:
        return self.lar > -0.5


    def laPos(self) -> bool:
        return self.lar > 0.5
    

    def vSmall(self) -> bool:
        return self.vr <= 0.5
    

    def isInFrontOf(self, another: 'Agent') -> bool:            # TODO: tune the number
        deltaX = another.x - self.x
        deltaY = another.y - self.y
        angle = np.arctan2(deltaY, deltaX)
        angleRelToFacingDirection = self.yaw - angle
        return abs(angleRelToFacingDirection) < np.pi/3


    def isAround(self, another: 'Agent') -> bool:               # TODO: tune the number
        deltaX = another.x - self.x
        deltaY = another.y - self.y
        distance = np.sqrt(deltaX**2 + deltaY**2)
        direction = np.arctan2(deltaY, deltaX)  # Could use with vtheta or yaw?
        return distance <= 0.5
        

    def canBeCuttingInInFront(self, another: 'Agent') -> bool:  # TODO: tune the numbers
        deltaX = another.x - self.x
        deltaY = another.y - self.y
        angle = np.arctan2(deltaY, deltaX)
        angleRelToFacingDirection = self.yaw - angle
        facingDirectionDifference = self.yaw - another.yaw
        return abs(angleRelToFacingDirection) < 0.2 and \
            abs(facingDirectionDifference) > 0.2 and \
            angleRelToFacingDirection * facingDirectionDifference < 0.03   # opposite
    

    def canBeCuttingOutInFront(self, another: 'Agent') -> bool: # TODO: tune the numbers
        deltaX = another.x - self.x
        deltaY = another.y - self.y
        angle = np.arctan2(deltaY, deltaX)
        angleRelToFacingDirection = self.yaw - angle
        facingDirectionDifference = self.yaw - another.yaw
        return abs(angleRelToFacingDirection) < 0.2 and \
            abs(facingDirectionDifference) > 0.2 and \
            angleRelToFacingDirection * facingDirectionDifference > -0.03   # same
    

    def canBeDirectlyLeading(self, another: 'Agent') -> bool:   # TODO: tune the numbers
        '''
        Test for simple leading, when "self" is just in front of "another" at a very near distance

        Used for classifying lead when the lead vehicle and, consequently, the ego, has stopped
        '''

        deltaX = another.x - self.x
        deltaY = another.y - self.y
        distance = np.sqrt(deltaX**2 + deltaY**2)
        angle = np.arctan2(deltaY, deltaX)
        angleRelToFacingDirection = self.yaw - angle
        facingDirectionDifference = self.yaw - another.yaw
        return distance < 15 and \
            abs(angleRelToFacingDirection) < 0.1 and \
            abs(facingDirectionDifference) < 0.1
    


########## Trajectories: Utility Abstraction Class ##########



class Trajectories():
    def __init__(self, video):
        self.frames = video    # List<DataFrame>: list of frames (each frame contains rows of agents)
        self.allFrames = pd.concat(
            [ frame.apply(ti=i) for i, frame in enumerate(video) ],
            ignore_index=True,
        )


    def getFrame(self, ti) -> pd.DataFrame:
        return self.frames[ti].copy()
        # return self.allFrames[self.allFrames['ti'] == ti].copy()
    

    def getAgent(self, ti, agentId) -> Agent:
        frame = self.getFrame(ti)
        
        agentRows = frame[frame['TRACK_ID'] == agentId]

        if len(agentRows) == 0:
            return None
        
        elif len(agentRows) == 1:
            return Agent(agentRows.iloc[0])

        else:
            raise Exception('Multiple agents have the same id.')
    
    
    def getTrajectory(self, tiStart, maxDelta, agentId):
        '''
        Returns generator showing probable states of the given agent in the future.

        Currently simply returns known future states (positions). 
        TODO: implement state extrapolation with kinematic features (v, a, etc.)
        '''

        for ti in range(tiStart, tiStart+maxDelta):
            frame = self.getFrame(ti)

            if len(frame) == 0:
                # TODO: implement extrapolation
                raise Exception('Unimplemented: extrapolation required.')
            
            else:
                # similar to adding this to a list that's defined out of the loop and returning that list in the end
                yield self.getAgent(ti, agentId)

    
    def getSecondClassName(self, ti) -> int:
        ego = self.getFrame(ti).iloc[0]; assert(ego['TRACK_ID'] == 'ego')
        return u.secondClassNames[ego['second_class']]


    def getEgo(self, ti) -> Agent:
        ego = self.getFrame(ti).iloc[0]; assert(ego['TRACK_ID'] == 'ego')
        return Agent(ego)
    

    def getNonEgoAgents(self, ti):
        frame = self.getFrame(ti)
        return [Agent(agent) for agent in frame.iloc[1:]]
    


########## Highest Level Classification Functions ##########



def ClassifyVideo(video):
    '''
    Takes in a list of DataFrames, each DataFrame contains rows of agents, each with its kinematic features and ids and more.
    Outputs a list of label indices (ints) determined for each frame in this video.
    '''

    labels = []

    trajectories = Trajectories(video)

    # Only these timestamps are classified, labels in the beginning and end of the video are inferred
    concernedTimeIndices = range(
        FRAMES_BEFORE,                      # exactly FRAMES_BEFORE elements are before index FRAMES_BEFORE
        u.FRAMES_PER_VID - FRAMES_AFTER     # exactly FRAMES_AFTER elements are after index FRAMES_PER_VID - FRAMES_AFTER - 1
    )

    if len(concernedTimeIndices) == 0:
        raise Exception('Too many frames truncated from the beginning and end.')

    # Classify the frames in the middle
    for ti in concernedTimeIndices:
        label = ClassifyFrame(ti, trajectories)
        labels.append(label)

    # Infer labels for the beginning and end of the video
    firstLabel = labels[0]
    lastLabel = labels[-1]
    labels = [firstLabel] * FRAMES_BEFORE + labels + [lastLabel] * FRAMES_AFTER

    assert(len(labels) == u.FRAMES_PER_VID)
    
    return labels


def ClassifyFrame(ti, trajectories: Trajectories):
    '''
    Classify the frame in the provided timestamp index (ti), given trajectories of all agents in the video as context.
    '''

    secondClass = trajectories.getSecondClassName(ti)

    if secondClass == '1.1 InLane':
        return ClassifyInLaneFrame(ti, trajectories)
    elif secondClass == '2.1 StopAndWait':
        return ClassifyStopAndWaitFrame(ti, trajectories)
    elif secondClass == '2.4 GoStraight':
        return ClassifyGoStraightFrame(ti, trajectories)
    elif secondClass == '2.5 TurnLeft':
        return ClassifyTurnLeftFrame(ti, trajectories)
    elif secondClass == '2.6 TurnRight':
        return ClassifyTurnRightFrame(ti, trajectories)
    elif secondClass == '2.7 UTurn':
        return ClassifyUTurnFrame(ti, trajectories)
    else:
        return None
    


########## Classification Functions for Specific "second_class" Labels ##########



def ClassifyInLaneFrame(ti, trajectories: Trajectories):

    # Labels and high-level variables
    # '1.1.1 LeadVehicleConstant'           haslead, a_small
    # '1.1.2 LeadVehicleCutOut'             haslead, leadcut
    # '1.1.3 VehicleCutInAhead'             cutin
    # '1.1.4 LeadVehicleDecelerating'       haslead, a_neg
    # '1.1.5 LeadVehicleStppoed'            haslead, v_small
    # '1.1.6 LeadVehicleAccelerating'       haslead, a_pos
    # '1.1.7  LeadVehicleWrongDriveway'     whatever (TODO)


    # Set initial state of high-level variables (prevent unassigned variables)
    cutIn = False
    hasLead = False

    # Set future agent as anchor
    for anchor in trajectories.getTrajectory(ti, maxDelta=SELF_FRAMES_AFTER, agentId='ego'):

        # Get agents that are probably cutting in front of the currentEgo based on its relationship with the anchor
        cutInAgents = GetPossibleCutInAgentsAroundAnchor(anchor, ti, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if len(cutInAgents) > 0:
            cutIn = True
            hasLead = False
            break

        # Get the agent that is most probably leading the currentEgo based on its relationship with the anchor
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(anchor, ti, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if leadingAgent is not None:
            cutIn = False
            hasLead = True
            LaNeg, LaSmall, LaPos, LvSmall = \
            leadingAgent.laNeg(), leadingAgent.laSmall(), \
            leadingAgent.laPos(), leadingAgent.vSmall()

            currentEgo = trajectories.getEgo(ti)
            leadCut = leadingAgent.canBeCuttingOutInFront(currentEgo)
            
            break


    # Return a label based on the high-level variables
    if cutIn:
        return u.secondClasses['1.1.3 VehicleCutInAhead']
    if hasLead:
        if leadCut:
            return u.secondClasses['1.1.2 LeadVehicleCutOut']
        if LvSmall:
            return u.secondClasses['1.1.5 LeadVehicleStppoed']
        if LaNeg:
            return u.secondClasses['1.1.4 LeadVehicleDecelerating']
        if LaPos:
            return u.secondClasses['1.1.6 LeadVehicleAccelerating']
        if LaSmall:
            return u.secondClasses['1.1.1 LeadVehicleConstant']
    
    return '9.9.9 Invalid'      # well, whatever?


def ClassifyStopAndWaitFrame(ti, trajectories: Trajectories):

    # '2.1.4 LeadVehicleStppoed'        haslead
    # '2.1.5 PedestrianCrossing'        has pedestrians


    currentEgo = trajectories.getEgo(ti)
    
    canBePedestrainsCrossing = False

    for agent in trajectories.getNonEgoAgents(ti):
        if agent.canBeDirectlyLeading(currentEgo) and agent.type == 'Vehicle':
            return u.secondClasses['2.1.4 LeadVehicleStppoed']
        if agent.isInFrontOf(currentEgo) and agent.type == 'Pedestrian':
            canBePedestrainsCrossing = True

    return u.secondClasses['2.1.5 PedestrianCrossing'] if canBePedestrainsCrossing else '9.9.9 Invalid'


'''Below are a few repetitive functions to leave room for individual optimization (turnleft, turnright, and gostraight)'''
# TODO: implement those properly (they currently do not make sense)


def ClassifyGoStraightFrame(ti, trajectories: Trajectories):

    # '2.4.1 NoVehiclesAhead'           !haslead
    # '2.4.2 WithLeadVehicle'           haslead
    # '2.4.3 VehiclesCrossing'          hascross


    # Set initial state of high-level variables
    hasCross = False
    hasLead = False

    # Keep track of current ego as reference point
    currentEgo = trajectories.getEgo(ti)

    # Set future agent as anchor
    for anchor in trajectories.getTrajectory(ti, maxDelta=SELF_FRAMES_AFTER, agentId='ego'):

        # Get the agent that most probably is crossing the currentEgo based on its relationship with the virtual agent
        crossingAgents = GetPossibleCrossingAgentsAroundAnchor(anchor, ti, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if len(crossingAgents) > 0:
            hasCross = True
            hasLead = False
            break

        # Get the agent that most probably is leading the currentEgo based on its relationship with the virtual agent
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(anchor, currentEgo, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if leadingAgent is not None:
            hasCross = False
            hasLead = True
            break

    
    # Return a label based on the high-level variables
    if hasCross:
        return u.secondClasses['2.4.3 VehiclesCrossing']
    if hasLead:
        return u.secondClasses['2.4.2 WithLeadVehicle']
    else:
        return u.secondClasses['2.4.1 NoVehiclesAhead']


def ClassifyTurnLeftFrame(ti, trajectories: Trajectories):
    # '2.5.1 NoVehiclesAhead'           !haslead
    # '2.5.2 WithLeadVehicle'           haslead
    # '2.5.3 VehiclesCrossing'          hascross


    # Set initial state of high-level variables
    hasCross = False
    hasLead = False

    # Keep track of current ego as reference point
    currentEgo = trajectories.getEgo(ti)

    # Set future agent as anchor
    for anchor in trajectories.getTrajectory(ti, maxDelta=SELF_FRAMES_AFTER, agentId='ego'):

        # Get the agent that most probably is crossing the currentEgo based on its relationship with the virtual agent
        crossingAgents = GetPossibleCrossingAgentsAroundAnchor(anchor, ti, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if len(crossingAgents) > 0:
            hasCross = True
            hasLead = False
            break

        # Get the agent that most probably is leading the currentEgo based on its relationship with the virtual agent
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(anchor, currentEgo, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if leadingAgent is not None:
            hasCross = False
            hasLead = True
            break

    
    # Return a label based on the high-level variables
    if hasCross:
        return u.secondClasses['2.5.3 VehiclesCrossing']
    if hasLead:
        return u.secondClasses['2.5.2 WithLeadVehicle']
    else:
        return u.secondClasses['2.5.1 NoVehiclesAhead']


def ClassifyTurnRightFrame(ti, trajectories: Trajectories):
    # '2.6.1 NoVehiclesAhead'           !haslead
    # '2.6.2 WithLeadVehicle'           haslead
    # '2.6.3 VehiclesCrossing'          hascross


    # Set initial state of high-level variables
    hasCross = False
    hasLead = False

    # Keep track of current ego as reference point
    currentEgo = trajectories.getEgo(ti)

    # Set future agent as anchor
    for anchor in trajectories.getTrajectory(ti, maxDelta=SELF_FRAMES_AFTER, agentId='ego'):

        # Get the agent that most probably is crossing the currentEgo based on its relationship with the virtual agent
        crossingAgents = GetPossibleCrossingAgentsAroundAnchor(anchor, ti, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if len(crossingAgents) > 0:
            hasCross = True
            hasLead = False
            break

        # Get the agent that most probably is leading the currentEgo based on its relationship with the virtual agent
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(anchor, currentEgo, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if leadingAgent is not None:
            hasCross = False
            hasLead = True
            break

    
    # Return a label based on the high-level variables
    if hasCross:
        return u.secondClasses['2.6.3 VehiclesCrossing']
    if hasLead:
        return u.secondClasses['2.6.2 WithLeadVehicle']
    else:
        return u.secondClasses['2.6.1 NoVehiclesAhead']


def ClassifyUTurnFrame(ti, trajectory: Trajectories):
    return u.secondClasses['2.7.1 NoVehiclesAhead']



########## Helper Functions for "second_class" Classification Functions ##########



def GetPossibleCutInAgentsAroundAnchor(egoAnchor: Agent, ti, trajectories: Trajectories) -> list:
    currentEgo = trajectories.getEgo(ti)

    possibleCutInAgents = []

    startTi = ti                        # agents can be in the future route at least zero frames in the future
    maxDeltas = OTHERS_FRAMES_AFTER     # agents can be in the future route at most OTHERS_FRAMES_AFTER frames in the future
    minDeltas = maxDeltas               # everytime we find a cutin agent, we update the minDeltas, this way the most recent cutin agents are found

    for agent in trajectories.getNonEgoAgents(ti):
        for deltas, agentAnchor in enumerate(trajectories.getTrajectory(startTi, minDeltas, agent.id)):
            # "agentAnchor" means the agent's position in the future
            # this function basically tests whether any agent in the future will be on the
            # future route of "currentEgo", and it specifically considers a specific point on the ego route 
            # which is known as "egoAnchor"

            if agentAnchor.isAround(egoAnchor) and agent.canBeCuttingInInFront(currentEgo):
                possibleCutInAgents.append(agent)

                if deltas < minDeltas:
                    possibleCutInAgents.clear()
                    minDeltas = deltas

                # this break means this agent is classified as "possible to cut in" already
                break
    
    return possibleCutInAgents


def GetLeadingAgentAroundAnchor(egoAnchor: Agent, ti, trajectories: Trajectories) -> Agent | None:
    currentEgo = trajectories.getEgo(ti)

    for agent in trajectories.getNonEgoAgents(ti):
        if agent.isAround(egoAnchor) or agent.canBeDirectlyLeading(currentEgo):
            # the "or" in this statement ensures the case where currentEgo has stopped and does not have 
            # a trajectory that can be used to decide on leading agents. 
            return agent
    return None


def GetPossibleCrossingAgentsAroundAnchor(egoAnchor: Agent, ti, trajectories: Trajectories) -> list:
    currentEgo = trajectories.getEgo(ti)

    possibleCrossingAgents = []

    startTi = ti-OTHERS_FRAMES_BEFORE
    maxDeltas = ti+OTHERS_FRAMES_AFTER

    for agent in trajectories.getNonEgoAgents(ti):
        for agentAnchor in trajectories.getTrajectory(startTi, maxDeltas, agent.id):
            # "agentAnchor" means the agent's position in the future
            # this function basically tests whether any agent in the past or future will be on the
            # future route of "currentEgo", and it specifically considers a specific point on the ego route 
            # which is known as "egoAnchor"

            if agentAnchor.isAround(egoAnchor) and agent.canBeCrossing(currentEgo):
                possibleCrossingAgents.append(agent)

                # this break means this agent is classified as "possible to cut in" already
                break