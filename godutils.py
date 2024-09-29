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
    

    def isInFrontOf(self, another: 'Agent') -> bool:
        deltaX = another.x - self.x
        deltaY = another.y - self.y
        relAngle = np.arctan2(deltaY, deltaX)
        facingDirectionDifference = abs(self.yaw - relAngle)
        return deltaY > 1 and \
            facingDirectionDifference < 0.1 and \
            abs(deltaX) < 3                                 # TODO: tune the numbers
    

    def isAround(self, another: 'Agent') -> bool:
        deltaX = another.x - self.x
        deltaY = another.y - self.y
        distance = np.sqrt(deltaX**2 + deltaY**2)
        direction = np.arctan2(deltaY, deltaX)  # Could use with vtheta or yaw?
        return distance <= 0.5                              # TODO: tune the number
    

    def canBeCuttingOutInFront(self, another: 'Agent') -> bool:
        deltaY = another.y - self.y
        facingDirectionDifference = abs(another.yaw - self.yaw)
        return deltaY > 1 and \
            facingDirectionDifference > 0.2                 # TODO: tune the numbers
    

    def canBeDirectlyLeading(self, another: 'Agent') -> bool:
        '''
        Test for simple leading, when "self" is just in front of "another" at a very near distance

        Used for classifying lead when the lead vehicle and, consequently, the ego, has stopped
        '''

        deltaX = another.x - self.x
        deltaY = another.y - self.y
        relAngle = np.arctan2(deltaY, deltaX)
        deviationFromFront = abs(relAngle-self.yaw - np.pi/2)
        facingDirectionDifference = abs(another.yaw - self.yaw)
        return deltaX < 2 and \
            deltaY < 10 and \
            deviationFromFront < 0.1 and \
            facingDirectionDifference < 0.1                 # TODO: tune the numbers
    


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

            leadCut = CheckIfLeadingAgentIsCuttingOut(leadingAgent, currentEgo, trajectories)
            
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


    front: Agent = GetAgentInFrontOfEgo(ti, trajectories)

    if front is not None:
        if front.type == 'Vehicle':
            return u.secondClasses['2.1.4 LeadVehicleStppoed']
        else:
            return u.secondClasses['2.1.5 PedestrianCrossing']

    return '9.9.9 Invalid'      # well, whatever?


'''Below are a few repetitive functions to leave room for individual optimization (turnleft, turnright, and gostraight)'''


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
        crossingAgent: Agent = GetCrossingAgentAroundAnchor(anchor, currentEgo, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if crossingAgent is not None:
            hasCross = True
            hasLead = False
            break

        # Get the agent that most probably is leading the currentEgo based on its relationship with the virtual agent
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(anchor, currentEgo, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if leadingAgent is not None:
            cutIn = False
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
        crossingAgent: Agent = GetCrossingAgentAroundAnchor(anchor, currentEgo, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if crossingAgent is not None:
            hasCross = True
            hasLead = False
            break

        # Get the agent that most probably is leading the currentEgo based on its relationship with the virtual agent
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(anchor, currentEgo, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if leadingAgent is not None:
            cutIn = False
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
        crossingAgent: Agent = GetCrossingAgentAroundAnchor(anchor, currentEgo, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if crossingAgent is not None:
            hasCross = True
            hasLead = False
            break

        # Get the agent that most probably is leading the currentEgo based on its relationship with the virtual agent
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(anchor, currentEgo, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if leadingAgent is not None:
            cutIn = False
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

    startTi = ti + OTHERS_FRAMES_AFTER
    maxDeltas = SELF_FRAMES_AFTER-OTHERS_FRAMES_AFTER
    minDeltas = maxDeltas

    for agent in trajectories.getNonEgoAgents(ti):
        for deltas, agentAnchor in enumerate(trajectories.getTrajectory(startTi, minDeltas, agent.id)):
            if agentAnchor.isAround(egoAnchor) and agent.canBeCuttingInFrontOf(currentEgo):
                possibleCutInAgents.append(agent)
                minDeltas = deltas
                break
    
    return possibleCutInAgents

        
def GetLeadingAgentAroundAnchor(egoAnchor: Agent, ti, trajectories: Trajectories) -> Agent:
    currentEgo = trajectories.getEgo(ti)

    for agent in trajectories.getNonEgoAgents(ti):
        if agent.isAround(egoAnchor) or agent.canBeDirectlyLeading(currentEgo):
            return agent


def GetAgentInFrontOfEgo(ti, trajectories: Trajectories) -> Agent:
    raise Exception('Unimplemented')


def CheckIfLeadingAgentIsCuttingOut(leadingAgent: Agent, currentEgo: Agent, trajectories: Trajectories) -> bool:
    raise Exception('Unimplemented')

    
def GetCrossingAgentAroundAnchor(anchor: Agent, currentEgo: Agent, trajectories: Trajectories) -> Agent:
    raise Exception('Unimplemented')