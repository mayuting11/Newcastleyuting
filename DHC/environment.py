import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import configs

plt.ion() #Transform the display mode of matplotlib into interactive mode. Please add plt.ioff() before plt.show()
from matplotlib import colors

action_list = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int)

color_map = np.array([[255, 255, 255],  # white
                      [190, 190, 190],  # gray
                      [0, 191, 255],  # blue
                      [255, 165, 0],  # orange
                      [0, 250, 154]])  # green


def map_partition(map):
    '''
    partitioning map into independent partitions
    '''

    empty_pos = np.argwhere(map == 0).astype(np.int).tolist() #First return the indexes of the elements that equal to 0; Then modify the type of each data in the array into integer;
    #Finally transform the array into list

    empty_pos = [tuple(pos) for pos in empty_pos] #transform each element in the list into tuple

    if not empty_pos:
        raise RuntimeError('no empty position')

    partition_list = list()#create an empty list
    while empty_pos:

        start_pos = empty_pos.pop()#remove the last element from the list and assign the value of the last element to start_pos

        open_list = list()
        open_list.append(start_pos)#append the start_pos to the end of the list open_list
        close_list = list()

        while open_list:
            x, y = open_list.pop(0) #remove the first element of the list open_list and assign the value of the first element to x and y
            up = x - 1, y #determine the relative position of the agent's up
            if up[0] >= 0 and map[up] == 0 and up in empty_pos:
                empty_pos.remove(up)
                open_list.append(up)

            down = x + 1, y #determine the relative position of the agent's down
            if down[0] < map.shape[0] and map[down] == 0 and down in empty_pos: #compare down[0] with the dimension of the array map
                empty_pos.remove(down)
                open_list.append(down)

            left = x, y - 1 #determine the relative position of the agent's left
            if left[1] >= 0 and map[left] == 0 and left in empty_pos:
                empty_pos.remove(left)
                open_list.append(left)

            right = x, y + 1 #determine the relative position of the agent's right
            if right[1] < map.shape[1] and map[right] == 0 and right in empty_pos:
                empty_pos.remove(right)
                open_list.append(right)

            close_list.append((x, y))#append the relative position of the agent into the list close_list

        partition_list.append(close_list) #partition_list is a 2-dimension list that stores the open(free) positions for each agent[[(),(),...,()],[(),(),...,()],...,[(),(),...,()]]

    return partition_list


class Environment:
    def __init__(self, num_agents: int = configs.init_env_settings[0], map_length: int = configs.init_env_settings[1],
                 obs_radius: int = configs.obs_radius, reward_fn: dict = configs.reward_fn, fix_density=None,
                 curriculum=False, init_env_settings_set=configs.init_env_settings): #define a magic method

        #set the number of agents and the size of the map
        self.curriculum = curriculum
        if curriculum:
            self.env_set = [init_env_settings_set]
            self.num_agents = init_env_settings_set[0]
            self.map_size = (init_env_settings_set[1], init_env_settings_set[1])
        else:
            self.num_agents = num_agents
            self.map_size = (map_length, map_length)

        # set as same as in PRIMAL
        if fix_density is None:#random
            self.fix_density = False
            self.obstacle_density = np.random.triangular(0, 0.33, 0.5) #return the random value from the triangular distribution.
            # The lower bound, upper bound, and the peak value are respectively 0,0.5,0.33
        else:#assign
            self.fix_density = True
            self.obstacle_density = fix_density

        self.map = np.random.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
            np.int)#First return a numpy.ndarray (2-dimension) that has the same size as the map (..*..) and consists of 0 and 1 and the p[0] denotes the probability of returning 0,
        # p[1] denotes the probability of returning 1

        partition_list = map_partition(self.map)#call the function map_partition.
        # return partition_list which is a 2-dimension list that stores the open(free) positions for each agent[[(),(),...,()],[(),(),...,()],...,[(),(),...,()]]
        partition_list = [partition for partition in partition_list if len(partition) >= 2]#delete the list denoting that an agent has no feasible move at the current position,
        # len(partition)==1 denotes that the up, down, left, and right of the agent are obstacles

        while len(partition_list) == 0:#if there is no any feasible move for all agents then recreate a map
            self.map = np.random.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
                np.int)
            partition_list = map_partition(self.map)
            partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)#set the positions of agents
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int)#set the goal positions of agents

        pos_num = sum([len(partition) for partition in partition_list]) #calculate the sum of the number of all feasible move for all agents

        # loop to assign agent original position and goal position for each agent
        for i in range(self.num_agents):
            '''
            Assign agent original position for each agent
            '''
            pos_idx = random.randint(0, pos_num - 1) #return an integer in the range [0, pos_num-1]
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)#equal to pos_idx=pos_idx-len(partition)
                    partition_idx += 1#equal to  partition_idx=partition_idx+1
                else:
                    break# if you use the nested loop, break will stop the deepest loop and start to execute the code in the next line

            pos = random.choice(partition_list[partition_idx])#randomly select a tuple from partition_list[partition_idx]
            partition_list[partition_idx].remove(pos) #remove the first match item of pos from the list partition_list[partition_idx]
            self.agents_pos[i] = np.asarray(pos, dtype=np.int)#convert a tuple into an array

            '''
            Assign goal position for each agent
            '''
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int)

            partition_list = [partition for partition in partition_list if len(partition) >= 2]
            pos_num = sum([len(partition) for partition in partition_list])

        self.obs_radius = obs_radius

        self.reward_fn = reward_fn
        self.get_heuri_map()
        self.steps = 0
        '''
        last_actions: Save the observation for each potential action for each agent  
        '''
        self.last_actions = np.zeros((self.num_agents, 5, 2 * obs_radius + 1, 2 * obs_radius + 1), dtype=np.bool)#return a array consists of bool value (False) with
        # a given shape and data type, the FOV of an agent is (2 * obs_radius + 1)*(2 * obs_radius + 1)

    def update_env_settings_set(self, new_env_settings_set):#update the environment settings set
        self.env_set = new_env_settings_set

    def reset(self, num_agents=None, map_length=None):
        '''

      reset the environment (similar to the def __init__())
        '''

        if self.curriculum:
            rand = random.choice(self.env_set)#randomly choose an environment set
            self.num_agents = rand[0]
            self.map_size = (rand[1], rand[1])

        elif num_agents is not None and map_length is not None:
            self.num_agents = num_agents
            self.map_size = (map_length, map_length)

        if not self.fix_density:
            self.obstacle_density = np.random.triangular(0, 0.33, 0.5)

        self.map = np.random.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
            np.float32)

        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]

        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
                np.float32)
            partition_list = map_partition(self.map)
            partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int)

        pos_num = sum([len(partition) for partition in partition_list])

        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num - 1)#randomly return a number in the closed interval [0,pos_num-1]
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int)#convert a tuple into an array

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int)

            partition_list = [partition for partition in partition_list if len(partition) >= 2]#In order to make sure there is no overlap
            # among the selected positions of starting and goal points
            pos_num = sum([len(partition) for partition in partition_list])

        self.steps = 0
        self.get_heuri_map()
        '''
        last_actions: Save the observation for each potential action for each agent  
        '''
        self.last_actions = np.zeros((self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=np.bool)#for each agent

        return self.observe()

    def load(self, map: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray):#load the map, the agent positions, and the goal positions

        self.map = np.copy(map)#create a deep copy of an array
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)

        self.num_agents = agents_pos.shape[0]
        self.map_size = (self.map.shape[0], self.map.shape[1])

        self.steps = 0

        self.imgs = []

        self.get_heuri_map()

        self.last_actions = np.zeros((self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=np.bool)

    def get_heuri_map(self):
        '''
        calculate the distance from each grid to the goal point
        '''
        dist_map = np.ones((self.num_agents, *self.map_size), dtype=np.int32) #First return an array consists of 1
        for i in range(self.num_agents):
            open_list = list()#create an empty list
            x, y = tuple(self.goals_pos[i])
            open_list.append((x, y))
            dist_map[i, x, y] = 0#the distance between the goal point of the agent i and the goal point of the agent i is 0

            while open_list:
                x, y = open_list.pop(0)
                dist = dist_map[i, x, y]

                up = x - 1, y
                if up[0] >= 0 and self.map[up] == 0 and dist_map[i, x - 1, y] > dist + 1:
                    dist_map[i, x - 1, y] = dist + 1
                    if up not in open_list:
                        open_list.append(up)

                down = x + 1, y
                if down[0] < self.map_size[0] and self.map[down] == 0 and dist_map[i, x + 1, y] > dist + 1:
                    dist_map[i, x + 1, y] = dist + 1
                    if down not in open_list:
                        open_list.append(down)

                left = x, y - 1
                if left[1] >= 0 and self.map[left] == 0 and dist_map[i, x, y - 1] > dist + 1:
                    dist_map[i, x, y - 1] = dist + 1
                    if left not in open_list:
                        open_list.append(left)

                right = x, y + 1
                if right[1] < self.map_size[1] and self.map[right] == 0 and dist_map[i, x, y + 1] > dist + 1:
                    dist_map[i, x, y + 1] = dist + 1
                    if right not in open_list:
                        open_list.append(right)

        self.heuri_map = np.zeros((self.num_agents, 4, *self.map_size), dtype=np.bool)

        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                if self.map[x, y] == 0:
                    for i in range(self.num_agents):

                        if x > 0 and dist_map[i, x - 1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x - 1, y] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 0, x, y] = 1

                        if x < self.map_size[0] - 1 and dist_map[i, x + 1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x + 1, y] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 1, x, y] = 1

                        if y > 0 and dist_map[i, x, y - 1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y - 1] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 2, x, y] = 1

                        if y < self.map_size[1] - 1 and dist_map[i, x, y + 1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y + 1] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 3, x, y] = 1

        self.heuri_map = np.pad(self.heuri_map, (
        (0, 0), (0, 0), (self.obs_radius, self.obs_radius), (self.obs_radius, self.obs_radius) )) #Fill operation: fill the third and fourth dimension of the array heuri_map with o

    def step(self, actions: List[int]):#the action set for all agents is a list consisting of 0,1,2,3,4 indicating stay,up,down,left,right.
        '''
        actions:
            list of indices
                0 stay
                1 up
                2 down
                3 left
                4 right
        '''
        assert len(actions) == self.num_agents, 'only {} actions as input while {} agents in environment'.format(
            len(actions), self.num_agents)#if the expression after assert is false then produce an error
        assert all([5 > action_idx >= 0 for action_idx in actions]), 'action index out of range'#decide whether all elements in actions are True. if so return true, otherwise
        # return false

        checking_list = [i for i in range(self.num_agents)]

        rewards = []
        next_pos = np.copy(self.agents_pos)#create a deep copy of the array the positions of agents

        # remove unmoving(静止的) agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 0:
                # unmoving
                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):#check where the two array self.agents_pos[agent_id] and self.goals_pos[agent_id]
                    # have the same shape and elements
                    rewards.append(self.reward_fn['stay_on_goal'])#stay on goal=0
                else:
                    rewards.append(self.reward_fn['stay_off_goal'])#saty off goal=-0.075

                checking_list.remove(agent_id)
            else:
                # move
                next_pos[agent_id] += action_list[actions[agent_id]]
                rewards.append(self.reward_fn['move'])#move=-0.075

        # first round check, these two conflicts have the highest priority(one is that the agent is out of range)
        for agent_id in checking_list.copy():
            if np.any(next_pos[agent_id] < 0) or np.any(next_pos[agent_id] >= self.map_size[0]):#do Or operation for all elements in next_pos[agent_id] if <0 then set it as True
                # agent out of map range
                rewards[agent_id] = self.reward_fn['collision']#collision=-0.5
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide (碰撞，相撞) obstacle
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping (交换) conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                target_agent_id = np.where(np.all(next_pos[agent_id] == self.agents_pos, axis=1))[0]#target_agend_id denotes the id of the agent that will collide with
                # the current agent if it chooses the specified action

                if target_agent_id:#target_agend_id is not an empty list
                    target_agent_id = target_agent_id.item()# take out and return the element value of a tensor consisting of a single element
                    assert target_agent_id != agent_id, 'logic bug' #if the expression is false, raise an assertion error:logic bug

                    if np.array_equal(next_pos[target_agent_id], self.agents_pos[agent_id]):#collision is as follows: agent⇄agent
                        assert target_agent_id in checking_list, 'target_agent_id should be in checking list'

                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn['collision']

                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn['collision']

                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)

                        no_conflict = False
                        break

        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                collide_agent_id = np.where(np.all(next_pos == next_pos[agent_id], axis=1))[0].tolist()

                if len(collide_agent_id) > 1:
                    # collide agent

                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)

                    if all_in_checking:

                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):#pack the corresponding elements in the iterable objects as a tuple,
                            # and then return a list consisting of these tuples
                            pos.append(id)
                        collide_agent_pos.sort(key=lambda x: x[0] * self.map_size[0] + x[1])#the input of the lambda function is x
                        # and the output of the lambda function is x[0] * self.map_size[0] + x[1], from small to large
                        # (only one agent can move as planned and other agents return to the last state)

                        collide_agent_id.remove(collide_agent_pos[0][2])

                        #checking_list.remove(collide_agent_pos[0][2])

                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn['collision']

                    for id in collide_agent_id:
                        checking_list.remove(id)

                    no_conflict = False
                    break

        # self.history.append(np.copy(next_pos))
        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [self.reward_fn['finish'] for _ in range(self.num_agents)]#finish=3
        else:
            done = False

        info = {'step': self.steps - 1}

        # make sure no overlapping agents
        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:#remove the repeated elements，and return a new list(things like that)
            #in which the elements are from large to small
            print(self.steps)
            print(self.map)
            print(self.agents_pos)
            raise RuntimeError('unique')

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=np.bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1

        return self.observe(), rewards, done, info

    def observe(self):
        '''
        return observation and position for each agent
        obs: shape (num_agents, 11, 2*obs_radius+1, 2*obs_radius+1)
            layer 1: agent map
            layer 2: obstacle map
            layer 3-6: heuristic map
            layer 7-11: one-hot(一位有效编码) representation of agent's last action

        pos: used for calculating communication task
        '''
        obs  = np.zeros((self.num_agents, 6, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1), dtype=np.bool)

        # 0 represents obstacle to match 0 padding in CNN
        obstacle_map = np.pad(self.map, self.obs_radius, 'constant', constant_values=0) #pad the array map with 0

        agent_map = np.zeros((self.map_size), dtype=np.bool)
        agent_map[self.agents_pos[:, 0], self.agents_pos[:, 1]] = 1#get the first and second element of all lines(the first dimension)
        agent_map = np.pad(agent_map, self.obs_radius, 'constant', constant_values=0)

        for i, agent_pos in enumerate(self.agents_pos):#combine an iterable object as a sequence which contain the data and its index
            x, y = agent_pos
            obs[i, 0] = agent_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 0, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 2:] = self.heuri_map[i, :, x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]

        # obs = np.concatenate((obs, self.last_actions), axis=1)

        return obs, np.copy(self.agents_pos)

    def render(self):
        if not hasattr(self, 'fig'):#check whether an object contains a specified attribute
            self.fig = plt.figure()

        map = np.copy(self.map)
        for agent_id in range(self.num_agents):
            if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                map[tuple(self.agents_pos[agent_id])] = 4
            else:
                map[tuple(self.agents_pos[agent_id])] = 2
                map[tuple(self.goals_pos[agent_id])] = 3

        map = map.astype(np.uint8)#modify the datatype as np.unit8
        # plt.xlabel('step: {}'.format(self.steps))

        # add text in plot
        self.imgs.append([])
        if hasattr(self, 'texts'):
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                self.texts[i].set_position((agent_y, agent_x))
                self.texts[i].set_text(i)
        else:
            self.texts = []
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                self.texts.append(text)

        plt.imshow(color_map[map], animated=True)

        plt.show()
        # plt.ion()
        plt.pause(0.5)

    def close(self, save=False):
        plt.close()
        del self.fig
