import random
import numpy as np
import matplotlib.pyplot as plt
rand = random.choice([(1,10)])
a=None
map = np.random.choice(2, (3,3), p=[1 - 0.33, 0.33]).astype(
            np.float32)
agents_pos = np.asarray((1,2), dtype=int)
last_actions = np.zeros((3, 1, 2, 2),dtype=bool)
dist_map = np.ones((3, *(2,1)), dtype=np.int32)* 2147483647
goals_pos = np.empty((3, 2), dtype=int)
x,y=tuple(goals_pos[0])
up=x,y
a=list()
a.append(up)
A = np.arange(95,99).reshape(2,2)
B=np.pad(A,((3,2),(2,3)))
a=np.asarray([1,2,3,4,5,6])
b1 = np.arange(6).reshape(6)     # # 创建一个2*3*6矩阵`

list1=[2,1,34]
list2=sorted(list1)
c=[[1,2],[2,3]]
d=[1,2]
for pos, id in zip(c,
                   d):  # pack the corresponding elements in the iterable objects as a tuple,
    # and then return a list consisting of these tuples
    pos.append(id)
a=[(1,2),(1,2),(3,4)]
np.unique(a,axis=0)


array_0 = np.zeros([1,1])
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
agent_map = np.empty((4,3))
obs  = np.zeros((1, 2, 2 * 1 + 1, 2 * 1 + 1))
b=np.asarray([2.3,4.7])
map = b.astype(np.uint8)
c=[(0,0),(1,2)]
d=[(3,4),(5,6)]
zipped=zip(c,d)
text = plt.text(3, 4, 0, color='black', ha='center', va='center')
a=[]
a.append(text)


