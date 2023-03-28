import math
def dist (dict_waypoint,waypoint_start, waypoint_end):
    distsq=pow((dict_waypoint[waypoint_start][0]-dict_waypoint[waypoint_end][0]),2)+pow((dict_waypoint[waypoint_start][1]-dict_waypoint[waypoint_end][1]),2)
    dist=math.sqrt(distsq)
    #print(dist)
    return dist

