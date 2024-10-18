import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

TEST_SIZE = 0.1
RANDOM_STATE = 42
FRAMES_PER_VID = 40

# Put your csvs directly in these directories
startDir = Path('.')
dataPath:Path = startDir / Path('dataset/fulldata')
labelPath:Path = startDir / Path('dataset/labels')

dataFnameSet:set = set(map(lambda x: x.name, dataPath.glob('*.csv')))
labelFnameSet:set = set(map(lambda x: x.name, labelPath.glob('*.csv')))

fileNames:list = list(set.intersection(dataFnameSet, labelFnameSet))


def GetTrueLabels(fileName):
    series = pd.read_csv(labelPath / fileName)['third_class']
    index = series.map(thirdClasses)
    nonanindex = index.fillna(-1)
    intindex = nonanindex.astype(int)
    return list(intindex)


def GetRawDataFromOrigin(includeInvalid=False, toIndex=True):
    allDf = []

    # processing one data file and one label file (two files)
    for fileName in tqdm(fileNames):
        data = pd.read_csv(dataPath / fileName)
        data = data.drop(columns=['CITY_NAME'])
        if toIndex:
            data['OBJECT_TYPE'] = data['OBJECT_TYPE'].map(objectTypes)

        label = pd.read_csv(labelPath / fileName)
        label['TIMESTAMP'] = data.TIMESTAMP.unique()
        if toIndex:
            label['first_class'] = label['first_class'].map(firstClasses)
            label['second_class'] = label['second_class'].map(secondClasses)
            label['third_class'] = label['third_class'].map(thirdClasses)

        df = data.merge(label, on='TIMESTAMP')
        if includeInvalid:
            df = df.fillna(-1)  # if any nan occurs, toIndex is true
        else:
            df = df.dropna()

        df['fileName'] = fileName

        if toIndex:
            df['first_class'] = df['first_class'].astype('int')
            df['second_class'] = df['second_class'].astype('int')
            df['third_class'] = df['third_class'].astype('int')
            df['OBJECT_TYPE'] = df['OBJECT_TYPE'].astype('int')
            
        allDf.append(df)

    return pd.concat(allDf)


def WriteRawDataToCache(csvName='out/raw.csv', includeInvalid=False):
    rawData = GetRawDataFromOrigin(includeInvalid=includeInvalid)
    fullDir = startDir / csvName
    fullDir.parent.mkdir(parents=True, exist_ok=True)
    rawData.to_csv(fullDir, index=False)


def GetCachedRawData(csvName='out/raw.csv'):
    return pd.read_csv(startDir / csvName)


def GetCachedRawVideos(csvName='out/raw.csv'):
    rawData = GetCachedRawData(startDir / csvName)

    videos = {}
    for (fileName, _), agents in rawData.groupby(['fileName','TIMESTAMP']):
        if fileName not in videos:
            videos[fileName] = []
        
        videos[fileName].append(agents)
    
    return videos


def ProcessVideos(videos: dict):
    processed = {}

    # def valuable(my, agent):
        # return agent['type'] > -1   # return True
        # return vor(
        #     vand(
        #         -3 <= agent['x'],
        #         agent['x'] <= 3,
        #         -10 <= agent['theta'],
        #         agent['theta'] <= 10,
        #     ),
        #     vand(
        #         -90 <= agent['theta'],
        #         agent['theta'] <= 90,
        #         agent['r'] <= 15,
        #     ),
        #     vand(
        #         agent['yaw']-agent['theta'] >= 20,
        #         agent['r'] <= 30,
        #         -90 <= agent['theta'],
        #         agent['theta'] <= 90,
        #         agent['y'] >= 0,
        #     ),
        #     agent['type'] == 0
        # )

    for fileName, video in videos.items():
        for frame in video:
            if fileName not in processed:
                processed[fileName] = []

            processedFrame = pd.DataFrame()
            
            my = frame.iloc[0].copy(); assert my['TRACK_ID'] == 'ego'
            my_yaw = -angle(my['YAW'])

            processedFrame['x'], processedFrame['y'] = frame['X']-my['X'], frame['Y']-my['Y']
            processedFrame['yaw'], processedFrame['dyaw'], processedFrame['ddyaw'] = -angle(frame['YAW']-my_yaw), -angle(frame['DYAW']), -angle(frame['DDYAW'])
            processedFrame['x'], processedFrame['y'] = spin(processedFrame['x'], processedFrame['y'], -my_yaw)
            processedFrame['vx'], processedFrame['vy'] = spin(frame['V_X'], frame['V_Y'], -my_yaw)
            processedFrame['ax'], processedFrame['ay'] = spin(frame['A_X'], frame['A_Y'], -my_yaw)
            processedFrame['r'], processedFrame['theta'] = rtheta(processedFrame['x'], processedFrame['y'])
            processedFrame['vr'], processedFrame['vtheta'] = rtheta(processedFrame['vx'], processedFrame['vy'])
            processedFrame['ar'], processedFrame['atheta'] = rtheta(processedFrame['ax'], processedFrame['ay'])
            processedFrame['type'] = frame['OBJECT_TYPE']
            # processedFrame['lead_tendency'] = calc_lead_tendency(processedFrame)
            processedFrame['first_class'] = frame['first_class']
            processedFrame['second_class'] = frame['second_class']
            processedFrame['third_class'] = frame['third_class']

            # processedFrame = processedFrame[valuable(my, processedFrame)]

            processed[fileName].append(processedFrame)

    return processed


def SplitVideos(videos: dict):
    fileNames = list(videos.keys())
    trainFiles, testFiles = train_test_split(fileNames, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainVideos = {k:v for k,v in videos.items() if k in trainFiles}
    testVideos = {k:v for k,v in videos.items() if k in testFiles}
    return trainVideos, testVideos


def GetXFromVideosForRNN(videos: dict):
    shouldDrop = ['third_class', 'TIMESTAMP', 'TRACK_ID', 'fileName']
    data = []
    for video in videos.values():
        for frame in video:
            data.append(frame.reset_index().drop(columns=shouldDrop, errors='ignore').to_numpy())
    return data


def GetYFromVideosForRNN(videos: dict):
    labels = []
    for video in videos.values():
        for frame in video:
            labels.append(frame.iloc[0]['third_class'])
    return np.array(labels, dtype='int')


def GetXYFromVideosForRNN(videos: dict):
    return GetXFromVideosForRNN(videos), GetYFromVideosForRNN(videos)


def GetXYFromVideosForTree(videos: dict):
    data = []
    agentCount = 2
    for video in videos.values():
        for frame in video:
            frame['absx'] = abs(frame['x'])
            agentsSortedByX = frame.sort_values(by='absx', ascending=False)[:agentCount]
            agentsSortedByX = agentsSortedByX.drop(columns=['first_class', 'second_class', 'third_class'])

            # leadings = agents.sort_values(by='lead_tendency', ascending=False)[:N_AGENTS]
            # crossings = agents.sort_values(by='cross_tendency', ascending=False)[:N_AGENTS]
            # alignings = agents.sort_values(by='align_tendency', ascending=False)[:N_AGENTS]
            # following_turn = agents.sort_values(by='follow_turn_tendency', ascending=False)[:N_AGENTS]

            # self_row = self.rename(lambda x: f'{x}_self')

            # leading_row= pd.concat([e.rename(lambda x: f'{x}_leading_{i}') for i, e in enumerate(leadings.iloc)])
            # crossing_row = pd.concat([e.rename(lambda x: f'{x}_crossing_{i}') for i, e in enumerate(crossings.iloc)])
            # aligning_row = pd.concat([e.rename(lambda x: f'{x}_aligning_{i}') for i, e in enumerate(alignings.iloc)])
            # following_turn_row = pd.concat([e.rename(lambda x: f'{x}_following_turn_{i}') for i, e in enumerate(following_turn.iloc)])
            # label_row = li.rename({'first_class': 'first_class', 'second_class': 'second_class', 'third_class': 'label', 'filename': 'filename'})


            xFeatures = pd.concat([e.rename(lambda x: f'{x}_dist_{i}') for i, e in enumerate(agentsSortedByX.iloc)])
            xFeatures['first_class'] = frame.iloc[0]['first_class']
            xFeatures['second_class'] = frame.iloc[0]['second_class']
            xFeatures['third_class'] = frame.iloc[0]['third_class']

            data.append(xFeatures)
    df = pd.DataFrame(data)
    df = df[~df.isna().any(axis=1)]
    df = df.reset_index(drop=True)
    df = df.reindex(sorted(df.columns, key=lambda x: str(reversed(x))), axis=1)
    return df.drop(columns=['third_class']), df['third_class']


def PrintConfusionMatrix(mat, normalized=True):
    mat = mat.astype(float) / mat.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(15,10))
    sns.heatmap(pd.DataFrame(mat), annot=True, fmt='.1f', cmap='Blues', xticklabels=list(thirdClasses.keys()), yticklabels=list(thirdClasses.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


objectTypes = {
    'AV': 0,
    'Pedestrian': 1,
    'Vehicle' :2,
    'Bicycle' :3,
    'Barrier' :4,
}
firstClasses = {
    '1.GoStraight': 0,
    '2. Crossing': 1,
}
secondClasses = {
    '1.1 InLane': 0,
    '2.1 StopAndWait': 1,
    '2.4 GoStraight': 2,
    '2.5 TurnLeft': 3,
    '2.6 TurnRight': 4,
    '2.7 UTurn': 5,
}
thirdClasses = {
    '1.1.1 LeadVehicleConstant': 0,
    '1.1.2 LeadVehicleCutOut': 1,
    '1.1.3 VehicleCutInAhead': 2,
    '1.1.4 LeadVehicleDecelerating': 3,
    '1.1.5 LeadVehicleStppoed': 4,
    '1.1.6 LeadVehicleAccelerating': 5,
    '1.1.7  LeadVehicleWrongDriveway': 6,
    '2.1.4 LeadVehicleStppoed': 7,
    '2.1.5 PedestrianCrossing': 8,
    '2.4.1 NoVehiclesAhead': 9,
    '2.4.2 WithLeadVehicle': 10,
    '2.4.3 VehiclesCrossing': 11,
    '2.5.1 NoVehiclesAhead': 12,
    '2.5.2 WithLeadVehicle': 13,
    '2.5.3 VehiclesCrossing': 14,
    '2.6.1 NoVehiclesAhead': 15,
    '2.6.2 WithLeadVehicle': 16,
    '2.6.3 VehiclesCrossing': 17,
    '2.7.1 NoVehiclesAhead': 18,
}
objectTypeNames = {
    0: 'AV',
    1: 'Pedestrian',
    2: 'Vehicle',
    3: 'Bicycle',
    4: 'Barrier',
}
firstClassNames = {
    0: '1.GoStraight',
    1: '2. Crossing',
}
secondClassNames = {
    0: '1.1 InLane',
    1: '2.1 StopAndWait',
    2: '2.4 GoStraight',
    3: '2.5 TurnLeft',
    4: '2.6 TurnRight',
    5: '2.7 UTurn',
}
thirdClassNames = {
    0: '1.1.1 LeadVehicleConstant',
    1: '1.1.2 LeadVehicleCutOut',
    2: '1.1.3 VehicleCutInAhead',
    3: '1.1.4 LeadVehicleDecelerating',
    4: '1.1.5 LeadVehicleStppoed',
    5: '1.1.6 LeadVehicleAccelerating',
    6: '1.1.7  LeadVehicleWrongDriveway',
    7: '2.1.4 LeadVehicleStppoed',
    8: '2.1.5 PedestrianCrossing',
    9: '2.4.1 NoVehiclesAhead',
    10: '2.4.2 WithLeadVehicle',
    11: '2.4.3 VehiclesCrossing',
    12: '2.5.1 NoVehiclesAhead',
    13: '2.5.2 WithLeadVehicle',
    14: '2.5.3 VehiclesCrossing',
    15: '2.6.1 NoVehiclesAhead',
    16: '2.6.2 WithLeadVehicle',
    17: '2.6.3 VehiclesCrossing',
    18: '2.7.1 NoVehiclesAhead',
}

#---
def N(mu, sigma):
    return lambda x:np.exp(-(x-mu)**2/(2*sigma*sigma))

def angle(theta):
    return (theta+180) % (360) - 180

def deg(t):
    return t/np.pi*180

def rad(t):
    return t/180*np.pi

def rtheta(x, y):
    return np.sqrt(x**2 + y**2), deg(np.arctan2(x,y))       # right is positive, left is negative

def xy(r, theta):
    theta_rad = rad(theta)
    return r * np.sin(theta_rad), r * np.cos(theta_rad)

def spin(x, y, dtheta):
    r, t = rtheta(x, y)
    x, y = xy(r, angle(t+dtheta))
    return x, y

def vand(*conds):
    return np.all(conds, axis=0)

def vor(*conds):
    return np.any(conds, axis=0)
#---

# sigma_align = 0.08
# sigma_cross = 0.5
# sigma_close = 20

sigma_align = 0.08
sigma_cross = 0.5
sigma_close = 25


def how_aligned(theta1, theta2, sigma=sigma_align):
    '''Measures how close two angles are with N(0, sigma)'''
    dist = theta1 - theta2
    sigma_sq = sigma*sigma
    norm = lambda x:np.exp(-x**2/(2*sigma_sq))/np.sqrt(2*np.pi*sigma_sq)
    return norm(dist) / norm(0)
    
    # '''Gives a [-1,1] measure of how aligned two angles are'''
    # return np.cos(add(theta1,-theta2))**11

def how_crossed(theta1, theta2, sigma=sigma_cross):
    '''Measures how perpendicular two angles are with N(0, sigma)'''
    dist = np.pi/2 - abs(theta1-theta2)
    sigma_sq = sigma*sigma
    norm = lambda x:np.exp(-x**2/(2*sigma_sq))/np.sqrt(2*np.pi*sigma_sq)
    return norm(dist) / norm(0)
    
    # '''Gives a [0,1] mesaure of how perpendicular two angles are'''
    # return abs(np.sin(add(theta1,-theta2)))


def how_close(dist, sigma=sigma_close):
    '''Gives a measure of how close a distance is with N(0, sigma)'''
    sigma_sq = sigma*sigma
    norm = lambda x:np.exp(-x**2/(2*sigma_sq))/np.sqrt(2*np.pi*sigma_sq)
    return norm(dist) / norm(0)


def calc_lead_tendency(agent, self):
    distance = how_close(agent['r'])
    yaw_alignment = how_aligned(0, agent['yaw'])
    chasing_now = how_aligned(self['vtheta'], agent['theta'])
    ahead = how_aligned(0, agent['theta'])
    chasing = yaw_alignment + chasing_now + ahead

    # sign = np.all([sign_of(chasing, ahead), agent['type'] == 'Vehicle'], axis=0)
    is_vehicle = agent['type'] == 'Vehicle'
    return distance * chasing * is_vehicle


def calc_cross_tendency(agent, self):
    distance = how_close(agent['r'])
    yaw_cross = how_crossed(0, agent['yaw'])
    velocity_cross = how_crossed(self['vtheta'], agent['vtheta'])
    crossing = how_crossed(0, agent['vtheta'])
    x_velocity_degree = 1 - how_close(agent['vx'])
    cross = velocity_cross * yaw_cross * crossing * x_velocity_degree
    # ahead = how_aligned(0, agent['theta'])

    return distance * cross


def calc_align_tendency(agent, self):
    distance = how_close(agent['r'])
    align_yaw = how_aligned(0, agent['yaw'])
    align_velocity = how_aligned(self['vtheta'], agent['vtheta'])
    
    return distance * align_yaw * align_velocity
