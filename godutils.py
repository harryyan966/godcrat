import pandas as pd
import cratutils as u
import helpers as h
import matplotlib.pyplot as plt


########## Top Level Variables ##########


FORWARD_EGO_ANCHOR_COUNT = 5
FORWARD_CUTIN_ANCHOR_COUNT = 3
FORWARD_CROSSING_ANCHOR_COUNT = 3
BACKWARD_CROSSING_ANCHOR_COUNT = 2

TIME_PADDING_FRONT = 2
TIME_PADDING_END = 5


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
        self.f_ar = self.ar * h.cos(self.vtheta - self.atheta)

    
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
        and abs(vdd) < 30 \
        and u.objectTypeNames[self.type] in ['Vehicle']


    def isLeadingDirectly(self, ego: 'Agent') -> bool:
        dx = self.x - ego.x
        dy = self.y - ego.y

        distance = h.dist(dx, dy)
        direction = h.to180scale(h.arctan(dx, dy) - ego.yaw)

        fd = distance * h.cos(direction)    # see DEVNOTES for explanation
        cd = distance * h.sin(direction)
        yd = h.to180scale(self.yaw - ego.yaw)         # Yaw distance

        return self.isInFrontOf(ego) \
        and abs(cd) < 1 \
        and 1 < fd < 20 \
        and abs(yd) < 10 \
        and u.objectTypeNames[self.type] in ['Vehicle']


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
        and 60 < abs(vdd) < 120 \
        and u.objectTypeNames[self.type] in ['Vehicle']
    

    def matchAnchorAsCutIn(self, anchor: 'Agent') -> bool:
        dx = self.x - anchor.x
        dy = self.y - anchor.y

        distance = h.dist(dx, dy)
        direction = h.to180scale(h.arctan(dx, dy) - anchor.yaw)

        fd = distance * h.cos(direction)    # see DEVNOTES for explanation
        cd = distance * h.sin(direction)
        vdd = h.to180scale(self.vtheta - anchor.vtheta)

        return abs(fd) < 5 \
        and 1 < abs(cd) < 7 \
        and 10 < abs(vdd) < 50 \
        and vdd * cd < 0 \
        and u.objectTypeNames[self.type] in ['Vehicle']
    

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
        and u.objectTypeNames[self.type] in ['Vehicle']
        

    def matchAnchorAsCutOut(self, anchor: 'Agent') -> bool:
        dx = self.x - anchor.x
        dy = self.y - anchor.y

        distance = h.dist(dx, dy)
        direction = h.to180scale(h.arctan(dx, dy) - anchor.yaw)

        fd = distance * h.cos(direction)    # see DEVNOTES for explanation
        cd = distance * h.sin(direction)
        vdd = h.to180scale(self.vtheta - anchor.vtheta)

        return abs(fd) < 5 \
        and 1 < abs(cd) < 7 \
        and 10 < abs(vdd) < 50 \
        and vdd * cd > 0 \
        and u.objectTypeNames[self.type] in ['Vehicle']
    

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
        and u.objectTypeNames[self.type] in ['Vehicle']
        


########## Agent Helper ##########



def PlotAgents(agents,
               greenIf=None,
               greenOnly=False,
               a=False,
               bigA=True,
               xsize=50,
               ysize=50,
               ystart=0,
               dpi=80,
               onlyCars=False,
               extra=[]):
    '''
    Plots a list of agents onto a 2d plane

    Parameters:
        agents: list<Agent>
        greenIf: Agent, Agent -> bool: a function that takes in agent and ego and outputs if the agent will be green on the display. Default behavior is showing agents in front of ego as green
        greenOnly: bool: show green vehicles only (green vehicles can be filtered via greenIf)
        a: bool: whether to show an acceleration vector (arrow)
        bigA: bool: whether to 10x the acceleration vector
        xsize: int: horizontal size of the image
        ysize: int: vertical size of the image
        ystart: int: moves the image down this much (adds this value to minimum y)
        onlyCars: show the ego and "Vehicles" only (instead of "Bicycles" and stuff)
        extra: list<Agent>: extra agents to show
    '''

    if greenIf is None:
        greenIf = lambda agent, ego: agent.isInFrontOf(ego)

    plt.figure(figsize=(ysize/4,xsize/4), dpi=dpi)

    # Draw cross axes
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)

    ego = agents[0]

    def getColor(agent: Agent):
        if agent.id == 'ego':
            return 'red'
        if u.objectTypeNames[agent.type] != 'Vehicle':
            return 'cyan'
        if greenIf(agent, ego):
            return 'green'
        else:
            return 'purple'

    def getAccelerationColor(agent: Agent):
        if agent.f_ar < 0:
            return '#770000'    # dark red
        return '#000077'    # dark blue
    
    def putAgent(agent:Agent, color):
        x = agent.x - ego.x
        y = agent.y - ego.y

        # Plot speed as an arrow
        plt.arrow(x, y, agent.vx, agent.vy,
                  head_width=1, head_length=2, 
                  fc=color, 
                  ec=color)
        
        if a:
            # Plot acceleration as an arrow
            accelerationColor = getAccelerationColor(agent)
            factor = 10 if bigA else 1
            plt.arrow(x, y, agent.ax*factor, agent.ay*factor,
                    head_width=1, head_length=2, 
                    fc=accelerationColor, 
                    ec=accelerationColor)
        
        # Plot starting point
        plt.plot(x, y, marker='s', color=color, markerfacecolor='white', markersize=13)

        plt.text(x, y, f'{agent.id.split("-")[-1][:2]}', ha='center', va='center', fontsize=9, color=color)
    

    
    for agent in agents:
        if onlyCars and u.objectTypeNames[agent.type] not in ['AV', 'Vehicle']:
            continue

        if greenOnly and not greenIf(agent, ego):
            continue

        color = getColor(agent)
        putAgent(agent, color)
    
    for agent in extra:
        putAgent(agent, '#886600')
        

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
        try: return self.frames[ti].copy()
        except: return pd.DataFrame()
        # return self.allFrames[self.allFrames['ti'] == ti].copy()
    

    def getAgent(self, ti, agentId) -> Agent:
        frame = self.getFrame(ti)
        
        agentRows = frame[frame['TRACK_ID'] == agentId]

        if len(agentRows) == 0:
            return Exception(f'No agent has the given id "{agentId}".')
        
        elif len(agentRows) == 1:
            return Agent(agentRows.iloc[0])

        else:
            raise Exception(f'Multiple agents have the given id "{agentId}".')
        

    def getAgentByCode(self, ti, agentCode) -> Agent:
        frame = self.getFrame(ti)

        agentRows = frame[frame['TRACK_ID'].apply(lambda x : x.split("-")[-1] == str(agentCode))]

        if len(agentRows) == 0:
            raise Exception(f'No agent has the given code "{agentCode}".')
        
        elif len(agentRows) == 1:
            return Agent(agentRows.iloc[0])

        else:
            raise Exception(f'Multiple agents have the given code "{agentCode}".')
    
    
    def getTrajectory(self, agentId, tiStart, tiEnd):
        '''
        Returns generator showing probable states of the given agent in the future.

        Currently simply returns known future states (positions). 

        TODO: implement state extrapolation with kinematic features (v, a, etc.)
        '''

        for ti in range(tiStart, tiEnd):
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

    (i.e. each dataframe corresponds to an int)
    '''

    labels = []

    trajectories = Trajectories(video)

    # Only these timestamps are classified, labels in the beginning and end of the video are inferred
    concernedTimeIndices = range(TIME_PADDING_FRONT, u.FRAMES_PER_VID - TIME_PADDING_END)

    if len(concernedTimeIndices) == 0:
        raise Exception('Cannot infer labels in the front and end: no frames are classified in the middle.')

    # Classify the frames in the middle
    for ti in concernedTimeIndices:
        label = ClassifyFrame(ti, trajectories)
        labels.append(label)

    # Infer labels for the beginning and end of the video
    labels = ([labels[0]]*TIME_PADDING_FRONT) + labels + ([labels[-1]]*TIME_PADDING_END)

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

    # Labels and high-level variables (The manual decision Tree)

    # '1.1.1 LeadVehicleConstant'           haslead && a_small
    # '1.1.2 LeadVehicleCutOut'             cutout
    # '1.1.3 VehicleCutInAhead'             cutin
    # '1.1.4 LeadVehicleDecelerating'       haslead && a_neg
    # '1.1.5 LeadVehicleStppoed'            haslead && v_small
    # '1.1.6 LeadVehicleAccelerating'       haslead && a_pos
    # '1.1.7  LeadVehicleWrongDriveway'     whatever (TODO)


    # Set initial state of high-level variables (prevent unassigned variables)
    cutIn = False
    cutOut = False
    hasLead = False
    leading = None

    ego = trajectories.getEgo(ti)

    # check if there are agents leading the current vehicle directly.
    for agent in trajectories.getNonEgoAgents(ti):
        if agent.isLeadingDirectly(ego):    # TODO: tighten the function to avoid flase positives
            hasLead = True
            leading = agent
            break

    # "anchor" means the agent's position in the future
    # it basically help determine if two cars may meet in the future
    egoAnchorTi = ti
    for egoAnchor in trajectories.getTrajectory('ego', ti+1, ti+FORWARD_EGO_ANCHOR_COUNT):

        egoAnchorTi += 1

        if cutIn or hasLead:
            break

        # Check for cutIn
        for agent in trajectories.getNonEgoAgents(ti):

            if cutIn:
                break   # we've already found a cutIn vehicle (TODO: add logic to ensure the thing cutting in is really a "vehicle")

            if not agent.canBeCuttingInInFront(ego):        # TODO loosen the function to prevent false negatives
                continue    # if it's downright impossible for the agent to cut in front of ego, just ignore it

            for agentAnchor in trajectories.getTrajectory(agent.id, egoAnchorTi+1, ti+FORWARD_CUTIN_ANCHOR_COUNT):
                if agentAnchor.matchAnchorAsCutIn(egoAnchor):
                    cutIn = True
                    break

        if cutIn:
            break   # the scene will be classified as cutin instead of leading, so leading wouldn't matter

        # Check for leading and cutout
        for agent in trajectories.getNonEgoAgents(ti):

            if hasLead:
                break   # we've already found a leading vehicle

            if agent.matchAnchorAsCutOut(egoAnchor) and agent.canBeCuttingOutInFront(ego):
                cutOut = True
                break

            if agent.matchAnchorAsLeading(egoAnchor):
                hasLead = True
                leading = agent
                break

    # Define variables needed for the decision tree
    if hasLead:
        LaNeg, LaSmall, LaPos, LvSmall = \
        leading.faNeg(), leading.faSmall(), \
        leading.faPos(), leading.vSmall()
        print('leading', leading.id)

    # Return a label based on the high-level variables
    if cutIn:
        return u.thirdClasses['1.1.3 VehicleCutInAhead']
    if cutOut:
        return u.thirdClasses['1.1.2 LeadVehicleCutOut']
    if hasLead:
        if LvSmall:
            return u.thirdClasses['1.1.5 LeadVehicleStppoed']
        if LaNeg:
            return u.thirdClasses['1.1.4 LeadVehicleDecelerating']
        if LaPos:
            return u.thirdClasses['1.1.6 LeadVehicleAccelerating']
        if LaSmall:
            return u.thirdClasses['1.1.1 LeadVehicleConstant']

    return -1      # '9.9.9 Invalid'


def ClassifyStopAndWaitFrame(ti, trajectories: Trajectories):

    # '2.1.4 LeadVehicleStppoed'        haslead
    # '2.1.5 PedestrianCrossing'        has pedestrians

    ego = trajectories.getEgo(ti)
    minVehicleDistance = 9999   # find this to check if the pedestrian is before the vehicle 

    # first check if there are leading vehicles
    for agent in trajectories.getNonEgoAgents(ti):
        if agent.isLeadingDirectly(ego) and u.objectTypeNames[agent.type] == 'Vehicle':
            distance = h.dist(agent.x - ego.x, agent.y - ego.y)
            angle = h.arctan(agent.x - ego.x, agent.y - ego.y) - ego.yaw
            minVehicleDistance = min(minVehicleDistance, distance * h.cos(angle))
    
    # then check pedestrians that are between the lead vehicle (possibly nonexistent) and ego
    for agent in trajectories.getNonEgoAgents(ti):
        if agent.isInFrontOf(ego):
            distance = h.dist(agent.x - ego.x, agent.y - ego.y)
            angle = h.arctan(agent.x - ego.x, agent.y - ego.y) - ego.yaw
            f_distance = min(minVehicleDistance, distance * h.cos(angle))
            if f_distance < minVehicleDistance and u.objectTypeNames[agent.type] in ['Pedestrian', 'Bicycle']:
                return u.thirdClasses['2.1.5 PedestrianCrossing']
    
    if minVehicleDistance < 9999:       # we found a leading car and there are no relevant pedestrians
        return u.thirdClasses['2.1.4 LeadVehicleStppoed']

    return '9.9.9 Invalid'


'''Below are a few repetitive functions to leave room for individual optimization (turnleft, turnright, and gostraight)'''


def ClassifyGoStraightFrame(ti, trajectories: Trajectories):

    # '2.4.1 NoVehiclesAhead'           !haslead
    # '2.4.2 WithLeadVehicle'           haslead
    # '2.4.3 VehiclesCrossing'          hascross

    # Set initial state of high-level variables (prevent unassigned variables)
    hasCross = False
    hasLead = False

    ego = trajectories.getEgo(ti)

    for egoAnchor in trajectories.getTrajectory('ego', ti+1, ti+FORWARD_EGO_ANCHOR_COUNT):

        if hasCross or hasLead:
            break

        for agent in trajectories.getNonEgoAgents(ti):

            # check for crossing in the past and the future
            for agentAnchor in trajectories.getTrajectory(agent.id, ti-BACKWARD_CROSSING_ANCHOR_COUNT, ti+FORWARD_CROSSING_ANCHOR_COUNT):
                
                if agentAnchor.matchAnchorAsCrossing(egoAnchor):
                    hasCross = True
                    break
            
            if hasCross:
                break
            
            # check for leading now
            if agent.matchAnchorAsLeading(egoAnchor):
                hasLead = True
                break

    # Return a label based on the high-level variables
    if hasCross:
        return u.thirdClasses['2.4.3 VehiclesCrossing']
    if hasLead:
        return u.thirdClasses['2.4.2 WithLeadVehicle']
    else:
        return u.thirdClasses['2.4.1 NoVehiclesAhead']


def ClassifyTurnLeftFrame(ti, trajectories: Trajectories):
    # '2.5.1 NoVehiclesAhead'           !haslead
    # '2.5.2 WithLeadVehicle'           haslead
    # '2.5.3 VehiclesCrossing'          hascross

    # Set initial state of high-level variables (prevent unassigned variables)
    hasCross = False
    hasLead = False

    ego = trajectories.getEgo(ti)

    for egoAnchor in trajectories.getTrajectory('ego', ti+1, ti+FORWARD_EGO_ANCHOR_COUNT):

        if hasCross or hasLead:
            break

        for agent in trajectories.getNonEgoAgents(ti):

            # check for crossing in the past and the future
            for agentAnchor in trajectories.getTrajectory(agent.id, ti-BACKWARD_CROSSING_ANCHOR_COUNT, ti+FORWARD_CROSSING_ANCHOR_COUNT):
                
                if agentAnchor.matchAnchorAsCrossing(egoAnchor):
                    hasCross = True
                    break
            
            if hasCross:
                break
            
            # check for leading now
            if agent.matchAnchorAsLeading(egoAnchor):
                hasLead = True
                break

    # Return a label based on the high-level variables
    if hasCross:
        return u.thirdClasses['2.5.3 VehiclesCrossing']
    if hasLead:
        return u.thirdClasses['2.5.2 WithLeadVehicle']
    else:
        return u.thirdClasses['2.5.1 NoVehiclesAhead']


def ClassifyTurnRightFrame(ti, trajectories: Trajectories):
    # '2.6.1 NoVehiclesAhead'           !haslead
    # '2.6.2 WithLeadVehicle'           haslead
    # '2.6.3 VehiclesCrossing'          hascross


    # Set initial state of high-level variables (prevent unassigned variables)
    hasCross = False
    hasLead = False

    ego = trajectories.getEgo(ti)

    for egoAnchor in trajectories.getTrajectory('ego', ti+1, ti+FORWARD_EGO_ANCHOR_COUNT):

        if hasCross or hasLead:
            break

        for agent in trajectories.getNonEgoAgents(ti):

            # check for crossing in the past and the future
            for agentAnchor in trajectories.getTrajectory(agent.id, ti-BACKWARD_CROSSING_ANCHOR_COUNT, ti+FORWARD_CROSSING_ANCHOR_COUNT):
                
                if agentAnchor.matchAnchorAsCrossing(egoAnchor):
                    hasCross = True
                    break
            
            if hasCross:
                break
            
            # check for leading now
            if agent.matchAnchorAsLeading(egoAnchor):
                hasLead = True
                break

    # Return a label based on the high-level variables
    if hasCross:
        return u.thirdClasses['2.6.3 VehiclesCrossing']
    if hasLead:
        return u.thirdClasses['2.6.2 WithLeadVehicle']
    else:
        return u.thirdClasses['2.6.1 NoVehiclesAhead']
    

def ClassifyUTurnFrame(ti, trajectory: Trajectories):
    return u.thirdClasses['2.7.1 NoVehiclesAhead']
