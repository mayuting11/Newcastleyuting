import csv
import json
import sys
import codecs
import pandas as pd
import matplotlib.pyplot as plt
from carlar_routing_model import dist
from carlar_routing_model import get_key
def trans(path):
    #transform json into dataframe
    jsonData = codecs.open(path + '.json', 'r', 'utf-8')
    for line in jsonData:
        dic = json.loads(line)
        list_segment=[]
        list_road=[]
        list_section=[]
        list_waypoint_start_X=[]
        list_waypoint_start_Y = []
        list_waypoint_start_Z=[]
        list_waypoint_end_X=[]
        list_waypoint_end_Y=[]
        list_waypoint_end_Z=[]
        list_waypoint=[]
        new_list_waypoint=[]
        for key1 in dic:
            list_road.append(key1)
            for key2 in dic[key1]:
                list_section.append(key2)
                for key3 in dic[key1][key2]:
                    list_segment.append(key3)
                    #list_waypoint.extend(dic[key1][key2][key3])
                    list_waypoint_start_X.append(dic[key1][key2][key3][0][0])
                    list_waypoint_start_Y.append(dic[key1][key2][key3][0][1])
                    list_waypoint_start_Z.append(dic[key1][key2][key3][0][2])
                    list_waypoint_end_X.append(dic[key1][key2][key3][1][0])
                    list_waypoint_end_Y.append(dic[key1][key2][key3][1][1])
                    list_waypoint_end_Z.append(dic[key1][key2][key3][1][2])
        dataframe = pd.DataFrame({'ID_Segment':list_segment,
                                  'Waypoint_Start_X':list_waypoint_start_X,'Waypoint_Start_Y':list_waypoint_start_Y,
                                  'Waypoint_Start_Z':list_waypoint_start_Z,'Waypoint_End_X':list_waypoint_end_X,
                                  'Waypoint_End_Y':list_waypoint_end_Y,'Waypoint_End_Z':list_waypoint_end_Z})
    jsonData.close()


    #transform dataframe into csv
    for i in range(len(dataframe)):
        list_waypoint.append([dataframe['Waypoint_Start_X'][i],dataframe['Waypoint_Start_Y'][i],dataframe['Waypoint_Start_Z'][i]])
        list_waypoint.append([dataframe['Waypoint_End_X'][i],dataframe['Waypoint_End_Y'][i],dataframe['Waypoint_End_Z'][i]])

    for i in list_waypoint:
        if i not in new_list_waypoint:
            new_list_waypoint.append(i)
    dict_waypoint = {}
    for i in range(len(new_list_waypoint)):
        dict_waypoint[i] = new_list_waypoint[i]
    index_waypoint_start=[]
    index_waypoint_end=[]
    for i in range(len(dataframe)):
        index_waypoint_start.append(get_key.get_key(dict_waypoint,[dataframe['Waypoint_Start_X'][i],dataframe['Waypoint_Start_Y'][i],dataframe['Waypoint_Start_Z'][i]])[0])
        index_waypoint_end.append(get_key.get_key(dict_waypoint,[dataframe['Waypoint_End_X'][i],dataframe['Waypoint_End_Y'][i],dataframe['Waypoint_End_Z'][i]])[0])
    #print(index_waypoint_start,index_waypoint_end)
    dataframe_true = pd.DataFrame({'ID_Segment':list_segment,
                                   'ID_Waypoint_Start':index_waypoint_start,
                                  'Waypoint_Start_X':list_waypoint_start_X,
                                   'Waypoint_Start_Y':list_waypoint_start_Y,
                                  'Waypoint_Start_Z':list_waypoint_start_Z,
                                   'ID_Waypoint_End':index_waypoint_end,
                                   'Waypoint_End_X':list_waypoint_end_X,
                                  'Waypoint_End_Y':list_waypoint_end_Y,
                                   'Waypoint_End_Z':list_waypoint_end_Z})
    dataframe_true.to_csv("road.csv", index=False, sep=',')


    #find out the adjacency matrix
    datas=[]
    df=pd.read_csv('/home/w3sunrui/yuting/carla_routing/road.csv')
    #print(dict_waypoint)
    for i in range(len(df)):
        #print(df['ID_Waypoint_Start'][i])
        datas.append([df['ID_Waypoint_Start'][i],df['ID_Waypoint_End'][i],dist.dist(dict_waypoint,df['ID_Waypoint_Start'][i],df['ID_Waypoint_End'][i])])
    #print(datas)
    inf = 1e7
    n=len(new_list_waypoint)
    matrix = [[(lambda x: 0 if x[0] == x[1] else inf)([i, j]) for j in range(n)] for i in range(n)]
    for u, v, c in datas:
        matrix[u][v] = c

    #print(matrix)
    #print(new_list_waypoint)
    for i in new_list_waypoint:
        plt.plot(i[0],i[1],marker='o',markersize=5,color='black')

    plt.plot([dataframe_true['Waypoint_Start_X'],dataframe_true['Waypoint_End_X']],[dataframe_true['Waypoint_Start_Y'],dataframe_true['Waypoint_End_Y']],color='black')

    return matrix, new_list_waypoint, dict_waypoint


    #print(dict_waypoint)
    #print(len(list_waypoint),len(new_list_waypoint))


