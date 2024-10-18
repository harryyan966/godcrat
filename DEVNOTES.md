Maybe train a RNN model with each agent containing information about the current frame and the next two frames?

## 2024-10-13

changed a bit colors or the plotting function, previously overlooked issue with vehicle type and caused a bit of confusion. Now cyan signifies all non-vehicle agents.

changed the inFrontOf function: previous function covers too much range, current function is based on the simple parabola (y=x**2), which makes more sense to me

confused about 0.csv, the visualization does not seem to match the video: the vehicle that's actually cutting out seems to be missing, I don't know if this is a distance issue (i.e. the vehicle is too far away and is not recorded in the csv table dataset)

also encountered an issue when trying to getAgentByCode (it just gets the agent if you type 3 instead of scene-00000x-3). despite importlib.reload(g), the particular function getAgentByCode does not seem to load while other functions like plotting are updating fine.

**SOLVED**: you are defining a class instance whose class is defined by g before the reload. so if you are using the old g to construct the object, the object's method content will not update even if you reload g afterwards.

**SOLUTION**: reload the g when defining the top level variable (in this case vid)

Recap: the objective is to classify frames. To do this, the current stategy is to get high level variables and build a decision tree with those variables by hand (manually). The decision tree is now ready (i suppose (cause the tree is very simple)), but the variables are very high level so they have to be calculated with care. The high level variables we are using here are the properties of certain types of vehicle (or statements that they do not exist), more on that later.

There are five types of vehicles: the leading vehicle, the crossing vehicle, the cutout vehicle, the cutin vehicle, and the useless vehicle.

The leading vehicle: if the current car is going to be about the same place in the future as another vehicle, the "another vehicle" is going to be the leading vehicle (because we are tracing its steps). Note the leading vehicle is the vehicle in the position we get to first, so the less timestamp delta, the better.

There is a special case for leading vehicle: if the current vehicle is stopping the the video ends, it is not possible to capture the current vehicle's movement in the future and miss obvious leading vehicles. Therefore, if a vehicle is near and directly in front of the current vehicle, it is considered to be leading the current vehicle as well.

The crossing vehicle: if the future current vehicle and the future "vehicle X" meets ("vehicle X" is just some other vehicle), OR the future current vehicle and the past "vehicle X" meets, "vehicle X" is said to be a "crossing vehicle". 

The cutin vehicle: if the future current vehicle and the future "vehicle X" meets AND it reasonably heads about the same direction of the current vehicle, "vehicle X" is said to be a "cutin vehicle".

To classify those vehicles, one critical problem is finding out whether two vehicle states "meet". The current solution is to get
1. the distance component in the facing direction (forward direction) (FD: forward distance)
2. the distance component in the crossing direction (perpendicular direction or door to door direction) (CD: cross distance)
3. the difference in the direction of velocity vectors (VDD: velocity direction difference)

## 2024-10-16

a leading vehicle and the future current vehicle could have
1. a larger FD
2. a smaller CD
3. a smaller VDD

a crossing vehicle and the future current vehicle could have
1. a mid-range FD
2. a larger CD
3. a VDD nearing +-90 degrees

a cutin vehicle ...
1. a larger FD
2. a mid-range CD
3. a somewhat larger VDD
4. velocity goes inwards

a cutout vehicle ...
1. a larger FD
2. a mid-range CD
3. a somwhat larger VDD
4. velocity goes outwards

with these vehicles classified, these labels will be somewhat adequately classified.

## 2024-10-17

Implement vehicle classification within agent class (yet to propagate that api change to dependent functions (functions that are using agent functions))
