import numpy as np
import pandas as pd
import cratutils as u

SELF_FRAMES_BEFORE = 0
OTHERS_FRAMES_BEFORE = 2
FRAMES_BEFORE = max(SELF_FRAMES_BEFORE, OTHERS_FRAMES_BEFORE)
SELF_FRAMES_AFTER = 5
OTHERS_FRAMES_AFTER = 2
FRAMES_AFTER = max(SELF_FRAMES_AFTER, OTHERS_FRAMES_AFTER)


class Trajectories():
    def __init__(self, frames, startTi):
        self.frames = frames    # List<DataFrame>: list of frames (each frame contains rows of agents)
        self.startTi = startTi  # int: timestamp index of the first element of the frames list

        self.allFrames = pd.concat(frames, ignore_index=True)
    

    def agents(self, ti):
        ind = ti - self.startTi # ind: the index corresponding to the stored frames

        # If index is in bounds
        if 0 <= ind < len(self.frames):
            return self.frames[ti]      # get the dataframe representing agents
        
        return None
    

    def trackAgent(self, agentId):
        agentTrack = self.allFrames[self.allFrames['TRACK_ID'] == agentId].copy()

        return agentTrack


# class AgentState():
#     def __init__(self, x, y, vx, vy, yaw):
#         self.x = x
#         self.y = y
#         self.vr = np.sqrt(vx**2+vy**2)
#         self.vtheta = np.arctan2(vy, vx)/np.pi*180
#         self.yaw = (yaw+90)%360

# myFullTraj: List<AgentState>
# trajs: Dict<AgentId, Dict<TimeDelta: int, AgentState>>
# ti: int

# 1-1
def GetLeadAgentId(myFullTraj, trajs, ti):
    '''Returns: Tuple<TDelta, AgentId>
    Maybe replace single point determining with F-score of two points?
    '''

    leadAgentId = None

    for tiDelta in range(1, SELF_FRAMES_AFTER+1):
        leadAgentId = GetAgentIdClosestToPoint(myFullTraj[ti+tiDelta], trajs, tiDelta)
        
        if leadAgentId != None:
            return minDistAgentId, ti+tiDelta

    return None, None


def GetAgentIdClosestToPoint(myState, trajs, tiDelta):
    distThreshold = 2     # 2 is the maximum distance we tolerate for leading

    for agentId, agentTraj in trajs.items():
        agentDist = DistanceBetween(
            myState,
            agentTraj[tiDelta],
        )
        if agentDist is not None and agentDist < distThreshold:
            minDistAgentId = agentId
            distThreshold = agentDist


def GetCrossAgentId(myFullTraj, trajs, ti):
    '''Returns: Tuple<TDelta, AgentId>
    Maybe replace single point determining with F-score of two points?
    '''

    # TODO two for loops for classification

    distThreshold = 2     # 2 is the maximum distance we tolerate for leading
    minDistAgentId = None

    for myTiDelta in range(1, SELF_FRAMES_AFTER+1):
        for agentId, agentTraj in trajs.items():
            for crossTiDelta in range(OTHERS_FRAMES_BEFORE, min(OTHERS_FRAMES_AFTER+1, SELF_FRAMES_AFTER+1-myTiDelta)):
                agentDist = DistanceBetween(
                    myFullTraj[ti+myTiDelta],
                    agentTraj[crossTiDelta],
                )
                if agentDist is not None and agentDist < distThreshold:
                    minDistAgentId = agentId
                    distThreshold = agentDist
        
        if minDistAgentId != None:
            return minDistAgentId, ti+tiDelta

    return None, None



def ClassifyInLaneFrame(myFullTraj, trajs, agents: pd.DataFrame, ti):
    # '1.1.1 LeadVehicleConstant'           haslead, a_small
    # '1.1.2 LeadVehicleCutOut'             haslead, leadcut
    # '1.1.3 VehicleCutInAhead'             cutin
    # '1.1.4 LeadVehicleDecelerating'       haslead, a_neg
    # '1.1.5 LeadVehicleStppoed'            haslead, v_small
    # '1.1.6 LeadVehicleAccelerating'       haslead, a_pos
    # '1.1.7  LeadVehicleWrongDriveway'     whatever, don't care
    leadAgentId, leadAgentTimeDelta = GetLeadAgentId(myFullTraj, trajs, ti)

    if leadAgentId is None:
        leadAgent = None
    else:
        leadAgent = agents[agents['TRACK_ID'] == leadAgentId].iloc[0]
        v = DistanceBetween((0,0), (leadAgent['V_X'], leadAgent['V_Y']))    # perhaps calc component tangent to direction?
        a = 0    # TODO calc component tangent to direction
    
    cutInAgentId, cutInAgentTimeDelta = GetCutInAgentId(myFullTraj, trajs, ti)

    if cutInAgentId is None:
        cutInAgent = None
    elif cutInAgentTimeDelta > leadAgentTimeDelta:
        cutInAgent = None
    else:
        cutInAgent = agents[agents['TRACK_ID'] == cutInAgentId].iloc[0]

    hasLead = leadAgent is not None
    cutIn = cutInAgent is not None
    if hasLead:
        a_pos = a > 2
        a_neg = a < -2
        a_const = not a_pos and not a_neg
        v_small = v < 1
        leadCut = 0                         # TODO
        

    if cutIn:
        return u.secondClasses['1.1.3 VehicleCutInAhead']
    if leadCut:
        return u.secondClasses['1.1.2 LeadVehicleCutOut']
    if a_neg:
        return u.secondClasses['1.1.4 LeadVehicleDecelerating']
    if v_small:
        return u.secondClasses['1.1.5 LeadVehicleStppoed']
    if a_pos:
        return u.secondClasses['1.1.6 LeadVehicleAccelerating']
    if a_const:
        return u.secondClasses['1.1.1 LeadVehicleConstant']
    
    return '9.9.9 Invalid'      # well, whatever?


def ClassifyStopAndWaitFrame(myFullTraj, trajs, agents):
    # '2.1.4 LeadVehicleStppoed'        haslead
    # '2.1.5 PedestrianCrossing'        has pedestrians
    pass


def ClassifyGoStraightFrame(myFullTraj, trajs, agents):
    # '2.4.1 NoVehiclesAhead'           !haslead
    # '2.4.2 WithLeadVehicle'           haslead
    # '2.4.3 VehiclesCrossing'          hascross
    pass


def ClassifyTurnLeftFrame(myFullTraj, trajs, agents):
    # '2.5.1 NoVehiclesAhead'           !haslead
    # '2.5.2 WithLeadVehicle'           haslead
    # '2.5.3 VehiclesCrossing'          hascross

    pass


def ClassifyTurnRightFrame(myFullTraj, trajs, agents):
    # '2.6.1 NoVehiclesAhead'           !haslead
    # '2.6.2 WithLeadVehicle'           haslead
    # '2.6.3 VehiclesCrossing'          hascross

    pass


def ClassifyUTurnFrame(myFullTraj, trajs, agents):
    return u.secondClasses['2.7.1 NoVehiclesAhead']


labelRelationships = {
    '1.1 InLane': {
        '1.1.1 LeadVehicleConstant',
        '1.1.2 LeadVehicleCutOut',
        '1.1.3 VehicleCutInAhead',
        '1.1.4 LeadVehicleDecelerating',
        '1.1.5 LeadVehicleStppoed',
        '1.1.6 LeadVehicleAccelerating',
        '1.1.7  LeadVehicleWrongDriveway',
    },
    '2.1 StopAndWait': {
        '2.1.4 LeadVehicleStppoed',
        '2.1.5 PedestrianCrossing',
    },
    '2.4 GoStraight': {
        '2.4.1 NoVehiclesAhead',
        '2.4.2 WithLeadVehicle',
        '2.4.3 VehiclesCrossing',
    },
    '2.5 TurnLeft': {
        '2.5.1 NoVehiclesAhead',
        '2.5.2 WithLeadVehicle',
        '2.5.3 VehiclesCrossing',
    },
    '2.6 TurnRight': {
        '2.6.1 NoVehiclesAhead',
        '2.6.2 WithLeadVehicle',
        '2.6.3 VehiclesCrossing',
    },
    '2.7 UTurn': {
        '2.7.1 NoVehiclesAhead',
    },
}


# 1
def ClassifyVideo(video):
    labels = []

    myFullTraj = GetMyFullTrajectory(video)

    classifiedTimeIndices = range(FRAMES_BEFORE, u.FRAMES_PER_VID-FRAMES_AFTER-1)
    for ti in classifiedTimeIndices:
        trajs = Trajs(video[ti-OTHERS_FRAMES_BEFORE : ti+OTHERS_FRAMES_AFTER])
        agents = video[ti]

        # Get trajectories in a few frames before and after for each agent
        for agent in agents:
            agentId = agent['TRACK_ID']
            trajs[agentId] = GetRecentTrajectory(video, ti, agentId)

        labels.append(ClassifyFrame(myFullTraj, trajs, agents, ti))

    firstLabel = labels[0]
    lastLabel = labels[-1]
    labels = [firstLabel, firstLabel] + labels + [lastLabel, lastLabel]
    
    return labels


# 2
def GetMyFullTrajectory(video):
    result = []

    for agents in video:
        me = agents.iloc[0]
        result.append(AgentState(me['X'], me['Y'], me['V_X'], me['V_Y']))
    
    return result


# 3
def GetRecentTrajectory(video, timestampIndex, trackId):
    if timestampIndex < FRAMES_BEFORE or timestampIndex > u.FRAMES_PER_VID-FRAMES_AFTER:
        raise Exception('Invalid timestampIndex received.')

    result = {}

    for ti in range(timestampIndex-FRAMES_BEFORE, timestampIndex+FRAMES_AFTER+1):
        agents = video[ti]

        agent = agents[agents['TRACK_ID'] == trackId]
        if len(agent) == 1:
            result[ti-timestampIndex] = AgentState(agent['X'], agent['Y'], agent['V_X'], agent['V_Y'])
        elif len(agent) == 0:
            result[ti-timestampIndex] = None
        else:
            raise Exception(f'Unexpected amount of agents with trackid: {trackId}')
    
    return result


# 4
def ClassifyFrame(myFullTraj, trajs, agents, ti):
    second_class = u.secondClasses[agents.iloc[0]['second_class']]

    if second_class == '1.1 InLane':
        return ClassifyInLaneFrame(myFullTraj, trajs, agents, ti)
    elif second_class == '2.1 StopAndWait':
        return ClassifyStopAndWaitFrame(myFullTraj, trajs, agents, ti)
    elif second_class == '2.4 GoStraight':
        return ClassifyGoStraightFrame(myFullTraj, trajs, agents, ti)
    elif second_class == '2.5 TurnLeft':
        return ClassifyTurnLeftFrame(myFullTraj, trajs, agents, ti)
    elif second_class == '2.6 TurnRight':
        return ClassifyTurnRightFrame(myFullTraj, trajs, agents, ti)
    elif second_class == '2.7 UTurn':
        return ClassifyUTurnFrame(myFullTraj, trajs, agents, ti)
    else:
        return None
    

# 5
def DistanceBetween(pos1, pos2):
    if pos1 is None or pos2 is None:
        return None
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)