import numpy as np

def transformRawYaw(yaw):
    yaw = yaw + 90                      # normalize
    yaw = to180scale(yaw)               # to [-180, 180)
    return yaw / 180 * np.pi            # convert to radians


def to180scale(yaw):
    return (yaw + 180) % 360 - 180


def rad(yaw):
    return yaw / 180 * np.pi


def deg(yaw):
    return yaw / np.pi * 180


def arctan(x, y):
    return to180scale(deg(np.arctan2(y, x)))


def dist(dx, dy):
    return np.sqrt(dx**2 + dy**2)


def cos(yaw):
    return np.cos(rad(yaw))


def sin(yaw):
    return np.sin(rad(yaw))