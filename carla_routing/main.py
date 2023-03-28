import random

from carlar_routing_model import Dijkstra
from carlar_routing_model import trans
import matplotlib.pyplot as plt
from carlar_routing_model import write_list_to_json
from carlar_routing_model import path_plot
from carlar_routing_model import sorce_end_plot

def main(G,node_index_start,node_index_end,C):
    distance, path_optimal = Dijkstra.Dijkstra(G, node_index_start)
    path_optimal_coordinate = []
    for j in path_optimal[node_index_end]:
        path_optimal_coordinate.append(C[j])
    return path_optimal_coordinate

if __name__ == '__main__':
    path = '/home/w3sunrui/yuting/carla_routing/road'  # 获取path参数
    json_file_save_path='/home/w3sunrui/yuting/carla_routing'
    json_file_name='path_vehicle'
    G,L,C = trans.trans(path)
    node_num=len(L)
    node_index_start=random.randint(0,node_num)
    node_index_end=random.randint(0,node_num)
    sorce_end_plot.sorce_end_plot(C[node_index_start][0],C[node_index_start][1])
    sorce_end_plot.sorce_end_plot(C[node_index_end][0],C[node_index_end][1])
    #print('path_optimal:')
    #print(path_optimal)
    path_optimal_coordinate=main(G,node_index_start,node_index_end,C)
    print('path_optimal_coordinate:')
    print(path_optimal_coordinate)
    #print('distance:')
    #print(distance)
    write_list_to_json.write_list_to_json(path_optimal_coordinate, json_file_name, json_file_save_path)
    for i in range(len(path_optimal_coordinate)-1):
        path_plot.path_plot(path_optimal_coordinate[i][0],path_optimal_coordinate[i+1][0],path_optimal_coordinate[i][1],path_optimal_coordinate[i+1][1])
    plt.show()

