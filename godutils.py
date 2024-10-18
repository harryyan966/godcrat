import numpy as np
import pandas as pd
import cratutils as u
import helpers as h
import matplotlib.pyplot as plt


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
    def __init__(self, agentRow: pd.Series):

        self.x, self.y, self.vx, self.vy, self.ax, self.ay, self.yaw, self.id, self.type = (
            agentRow['X'],
            agentRow['Y'],
            agentRow['V_X'],
            agentRow['V_Y'],
            agentRow['A_X'],
            agentRow['A_Y'],
            h.transformRawYaw(agentRow['YAW']),
            agentRow['TRACK_ID'],
            agentRow['OBJECT_TYPE']
        )

        self.ar = h.dist(self.ax, self.ay)
        self.atheta = h.arctan(self.ax, self.ay)

        self.vr = h.dist(self.vx, self.vy)
        self.vtheta = h.arctan(self.vx, self.vy)

        self.r = h.dist(self.x, self.y)
        self.rtheta = h.arctan(self.x, self.y)
        
        # projection of acceleration vector to speed vector, "l" stands for "leading"
        self.f_ar = self.ar * np.cos(self.vtheta - self.atheta)

    
    def faSmall(self) -> bool:      # TODO: tune the number (currently 0.5) for these four functions 
        '''
        Stands for "(F)orward (A)cceleration (Small)"

        Same applies for the three functions below named like this
        '''
        return abs(self.f_ar) <= 0.5
    

    def faNeg(self) -> bool:
        return self.f_ar > -0.5


    def faPos(self) -> bool:
        return self.f_ar > 0.5
    

    def vSmall(self) -> bool:
        return self.vr <= 0.5
    

    def isInFrontOf(self, me: 'Agent') -> bool:
        # dx, dy are "self"'s distance relative to "me"

        dx = self.x - me.x
        dy = self.y - me.y

        return dy >= dx ** 2
    

    def matchAnchorAsLeading(self, anchor: 'Agent') -> bool:
        dx = self.x - anchor.x
        dy = self.y - anchor.y

        distance = h.dist(dx, dy)
        direction = h.to180scale(h.arctan(dx, dy) - anchor.yaw)

        fd = distance * h.cos(direction)    # see DEVNOTES for explanation
        cd = distance * h.sin(direction)
        vdd = h.to180scale(self.vtheta - anchor.vtheta)

        return abs(fd) < 5 \
        and abs(cd) < 3 \
        and abs(vdd) < 30


    def isLeadingDirectly(self, ego: 'Agent') -> bool:
        dx = self.x - ego.x
        dy = self.y - ego.y

        distance = h.dist(dx, dy)
        direction = h.to180scale(h.arctan(dx, dy) - ego.yaw)

        fd = distance * h.cos(direction)    # see DEVNOTES for explanation
        cd = distance * h.sin(direction)
        yd = h.to180scale(self.yaw - ego.yaw)         # Yaw distance

        return abs(cd) < 2 \
        and 1 < fd < 20 \
        and abs(yd) < 10


    def matchAnchorAsCrossing(self, anchor: 'Agent') -> bool:
        dx = self.x - anchor.x
        dy = self.y - anchor.y

        distance = h.dist(dx, dy)
        direction = h.to180scale(h.arctan(dx, dy) - anchor.yaw)

        fd = distance * h.cos(direction)    # see DEVNOTES for explanation
        cd = distance * h.sin(direction)
        vdd = h.to180scale(self.vtheta - anchor.vtheta)

        return abs(fd) < 4 \
        and abs(cd) < 8 \
        and 60 < abs(vdd) < 120
    

    def canBeCuttingInInFront(self, ego: 'Agent') -> bool:
        dx = self.x - ego.x
        dy = self.y - ego.y

        distance = h.dist(dx, dy)
        direction = h.to180scale(h.arctan(dx, dy) - ego.yaw)

        fd = distance * h.cos(direction)
        cd = distance * h.sin(direction)            # positive when at left and negative when at right
        vdd = h.to180scale(self.vtheta - ego.vtheta)    # positive when going lefter and negative when going righter
        yd = h.to180scale(self.yaw - ego.yaw)       # positive when going left and negative when going right

        # basically means "self" must be cutting into "ego"'s "front" and in front of "ego"
        return yd * cd < 0 \
        and vdd * cd < 0 \
        and fd > 1 \
    

    def canBeCuttingOutInFront(self, ego: 'Agent') -> bool:
        dx = self.x - ego.x
        dy = self.y - ego.y

        distance = h.dist(dx, dy)
        direction = h.to180scale(h.arctan(dx, dy) - ego.yaw)

        fd = distance * h.cos(direction)
        cd = distance * h.sin(direction)            # positive when at left and negative when at right
        vdd = h.to180scale(self.vtheta - ego.vtheta)    # positive when going lefter and negative when going righter
        yd = h.to180scale(self.yaw - ego.yaw)       # positive when going left and negative when going right

        # basically means "self" must be cutting out of "ego"'s "front" and in front of "ego"
        return yd * cd > 0 \
        and vdd * cd > 0 \
        and fd > 1 \
        


########## Agent Helper ##########



def PlotAgents(agents, greenIf=None, showA=False, egoInd=0, xsize=50, ysize=50, ystart=0, dpi=80):
    if greenIf is None:
        greenIf = lambda agent, ego: agent.isInFrontOf(ego)

    plt.figure(figsize=(ysize/4,xsize/4), dpi=dpi)

    # Draw cross axes
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)

    ego = agents[egoInd]

    def getColor(agent: Agent):
        if agent.id == 'ego':
            return 'red'
        if u.objectTypeNames[agent.type] != 'Vehicle':
            return 'cyan'
        if greenIf(agent, ego):
            return 'green'
        else:
            return 'purple'
    
    for agent in agents:
        x = agent.x - ego.x
        y = agent.y - ego.y

        # Plot speed as an arrow
        color = getColor(agent)
        plt.arrow(x, y, agent.vx, agent.vy,
                  head_width=1, head_length=2, 
                  fc=color, 
                  ec=color)
        
        if (showA):
            # Plot acceleration as an arrow
            accelerationColor = 'blue'
            plt.arrow(x, y, agent.ax, agent.ay,
                    head_width=1, head_length=2, 
                    fc=accelerationColor, 
                    ec=accelerationColor)
        
        # Plot starting point
        plt.plot(x, y, marker='s', color=color, markerfacecolor='white', markersize=13)

        plt.text(x, y, f'{agent.id.split("-")[-1][:2]}', ha='center', va='center', fontsize=9, color=color)
    

    plt.xlim(-xsize, xsize)
    plt.ylim(-ysize+ystart, ysize+ystart)

    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Agents')
    plt.show()



########## Trajectories: Utility Abstraction Class ##########



class Trajectories():

    def __init__(self, video):
        assert(len(video) == u.FRAMES_PER_VID)
        self.frames = video    # List<DataFrame>: list of frames (each frame contains rows of agents)
        self.allFrames = pd.concat(
            [ frame.assign(ti=i) for i, frame in enumerate(video) ],
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
        

    def getAgentByCode(self, ti, agentId) -> Agent:
        frame = self.getFrame(ti)
        agentRows = frame[frame['TRACK_ID'].apply(lambda x : x.split("-")[-1] == str(agentId))]
        return Agent(agentRows.iloc[0])
    
    
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
        secondClassIndex = ego['second_class']
        if secondClassIndex == -1: return '9.9 Invalid'
        return u.secondClassNames[secondClassIndex]


    def getEgo(self, ti) -> Agent:
        ego = self.getFrame(ti).iloc[0]; assert(ego['TRACK_ID'] == 'ego')
        return Agent(ego)
    

    def getNonEgoAgents(self, ti):
        frame = self.getFrame(ti)

        return [Agent(agent) for _, agent in frame.iloc[1:].iterrows()]
    

    def plot(self, ti, *args, **kwargs):
        PlotAgents([self.getEgo(ti), *self.getNonEgoAgents(ti)], *args, **kwargs)
    


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
        return -1       # 9.9.9 Invalid
    


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

    # Set future ego as anchor
    for egoAnchor in trajectories.getTrajectory(ti, maxDelta=SELF_FRAMES_AFTER, agentId='ego'):

        # Get agents that are probably cutting in front of the currentEgo based on its relationship with the anchor
        cutInAgents = GetPossibleCutInAgentsAroundAnchor(egoAnchor, ti, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if len(cutInAgents) > 0:
            cutIn = True
            hasLead = False
            break

        # Get the agent that is most probably leading the currentEgo based on its relationship with the anchor
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(egoAnchor, ti, trajectories)

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
    

    # print("BAD Stuff")
    # print(trajectories.getFrame(ti))
    return -1      # '9.9.9 Invalid'


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

    # Set future ego as anchor
    for egoAnchor in trajectories.getTrajectory(ti, maxDelta=SELF_FRAMES_AFTER, agentId='ego'):

        # Get the agent that most probably is crossing the currentEgo based on its relationship with the virtual agent
        crossingAgents = GetPossibleCrossingAgentsAroundAnchor(egoAnchor, ti, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if len(crossingAgents) > 0:
            hasCross = True
            hasLead = False
            break

        # Get the agent that most probably is leading the currentEgo based on its relationship with the virtual agent
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(egoAnchor, currentEgo, trajectories)

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

    # Set future ego as anchor
    for egoAnchor in trajectories.getTrajectory(ti, maxDelta=SELF_FRAMES_AFTER, agentId='ego'):

        # Get the agent that most probably is crossing the currentEgo based on its relationship with the virtual agent
        crossingAgents = GetPossibleCrossingAgentsAroundAnchor(egoAnchor, ti, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if len(crossingAgents) > 0:
            hasCross = True
            hasLead = False
            break

        # Get the agent that most probably is leading the currentEgo based on its relationship with the virtual agent
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(egoAnchor, currentEgo, trajectories)

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

    # Set future ego as anchor
    for egoAnchor in trajectories.getTrajectory(ti, maxDelta=SELF_FRAMES_AFTER, agentId='ego'):

        # Get the agent that most probably is crossing the currentEgo based on its relationship with the virtual agent
        crossingAgents = GetPossibleCrossingAgentsAroundAnchor(egoAnchor, ti, trajectories)

        # Set high-level variables based on the result and stop searching further into the future.
        if len(crossingAgents) > 0:
            hasCross = True
            hasLead = False
            break

        # Get the agent that most probably is leading the currentEgo based on its relationship with the virtual agent
        leadingAgent: Agent = GetLeadingAgentAroundAnchor(egoAnchor, currentEgo, trajectories)

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

            # the agent is not concerned in this particular delta
            if agentAnchor is None:
                continue

            if agentAnchor.quiteSameAs(egoAnchor) and agent.canBeCuttingInInFront(currentEgo):
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
        if agent.quiteSameAs(egoAnchor) or agent.canBeDirectlyLeading(currentEgo):
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

            # the agent is not concerned in this particular delta
            if agentAnchor is None:
                continue

            if agentAnchor.crossingDirectly(egoAnchor) and agent.canBeCrossing(currentEgo):
                possibleCrossingAgents.append(agent)

                # this break means this agent is classified as "possible to cut in" already
                break