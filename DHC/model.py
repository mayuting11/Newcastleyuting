import torch
import torch.nn as nn#neural network which is a module that can construct the neural network
import torch.nn.functional as F # most layers in the neural network have a corresponding function in "functional"
from torch.nn.utils.rnn import pack_padded_sequence
from torch.cuda.amp import autocast# automatically mix the accuracy
import configs


class ResBlock(nn.Module):#inherit from nn.Module(create a convolution nural network with two layers)(will be called three times in the paper)
    def __init__(self, channel):
        super().__init__()

        self.block1 = nn.Conv2d(channel, channel, 3, 1, 1)#The parameters are respectively the channels of input, the channels of output,the kernel size(3*3),stride,padding,dilation,
        #groups,bias,padding_mode
        self.block2 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x): #forward propagation function
        identity = x

        x = self.block1(x)
        x = F.relu(x)

        x = self.block2(x)

        x += identity#x=x+identity

        x = F.relu(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        '''
        in order to construct the fully connected layer
        '''
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        #the parameters are respectively the size of each input sample, the size of each output sample, bias(if set to False, the layer will not learn an additive bias)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)

    def forward(self, input, attn_mask):
        # input: [batch_size x num_agents x input_dim]
        batch_size, num_agents, input_dim = input.size()
        assert input_dim == self.input_dim#when the expression is false, then induce abnormity

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # q_s: [batch_size x num_heads x num_agents x output_dim]#reshape and transposition
        #-1 denotes the automatic determination of the size of the dimension
        k_s = self.W_K(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # k_s: [batch_size x num_heads x num_agents x output_dim]
        v_s = self.W_V(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # v_s: [batch_size x num_heads x num_agents x output_dim]

        if attn_mask.dim() == 2: #return the number of dimensions of self tensor
            attn_mask = attn_mask.unsqueeze(0)#Add a dimension of size one to the specified location
        assert attn_mask.size(0) == batch_size, 'mask dim {} while batch size {}'.format(attn_mask.size(0), batch_size)

        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.num_heads,1)  # attn_mask : [batch_size x num_heads x num_agents x num_agents]
        assert attn_mask.size() == (batch_size, self.num_heads, num_agents, num_agents)

        # context: [batch_size x num_heads x num_agents x output_dim]
        with autocast(enabled=False):
            scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) / (
                        self.output_dim ** 0.5)  # scores : [batch_size x n_heads x num_agents x num_agents]#The multiplication of matrixes of multiple dimensions
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one. keep the initial value where mask is 0.#-1e9=-1*10^9
            attn = F.softmax(scores, dim=-1)#dim means the Softmax operation should be done in the dimension -1  是在哪个维度进行Softmax操作
            #dim = -1 denotes line(1)，dim = -2 (0)denotes column
        context = torch.matmul(attn, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_agents,self.num_heads * self.output_dim)  # context: [batch_size x len_q x n_heads * d_v]
        # After the dimension changes through the operation transpose or permute，tensor is not saved contiguously in the memory but the operation view(resize)
        # requires save of tensor in the memory is contiguous，so we need to use contiguous () to return a contiguous copy
        output = self.W_O(context)

        return output  # output: [batch_size x num_agents x output_dim]


class CommBlock(nn.Module):
    def __init__(self, input_dim, output_dim=64, num_heads=configs.num_comm_heads, num_layers=configs.num_comm_layers):
        super().__init__()#当需要继承父类构造函数中的内容，且子类需要在父类的基础上补充时，使用super().__init__()方法
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.self_attn = MultiHeadAttention(input_dim, output_dim, num_heads)

        self.update_cell = nn.GRUCell(output_dim, input_dim)#Gated Recurrent Unit

    def forward(self, latent, comm_mask):
        '''
        latent shape: batch_size x num_agents x latent_dim
        '''
        num_agents = latent.size(1)

        # agent indices(指数) of agent that use communication
        update_mask = comm_mask.sum(dim=-1) > 1#dimension=-1 demotes line (dimension=1)
        # True==1 and False==0
        comm_idx = update_mask.nonzero(as_tuple=True)#return the index of non-zero elements in a tensor
        #函数返回元素为 1D 张量的元组，每一个 1D 张量对应输入张量的一个维度，而 1D 张量中的每个元素值表示输入张量中的非零元素在该维度上的索引

        # no agent use communication, return
        if len(comm_idx[0]) == 0:
            return latent

        if len(comm_idx) > 1:
            update_mask = update_mask.unsqueeze(2)#Add a dimension of size one to the specified location 2

        attn_mask = comm_mask == False#negation of the comm_mask

        for _ in range(self.num_layers):
            info = self.self_attn(latent, attn_mask=attn_mask)
            # latent = attn_layer(latent, attn_mask=attn_mask)
            if len(comm_idx) == 1:
                batch_idx = torch.zeros(len(comm_idx[0]), dtype=torch.long)#return a tensor whose shape is len(comm_idx[0]) and data type is torch.long（64 byte integer），
                # and all values of elements are 0
                latent[batch_idx, comm_idx[0]] = self.update_cell(info[batch_idx, comm_idx[0]],
                                                                  latent[batch_idx, comm_idx[0]])#the output after multi-head attention+latent
            else:
                update_info = self.update_cell(info.view(-1, self.output_dim), latent.view(-1, self.input_dim)).view(
                    configs.batch_size, num_agents, self.input_dim)
                # update_mask = update_mask.unsqueeze(2)
                latent = torch.where(update_mask, update_info, latent)#根据条件update_mask，返回从update_info,latent中选择元素所组成的张量。
                # 如果满足条件，则返回update_info中元素。若不满足，返回latent中元素
                # latent[comm_idx] = self.update_cell(info[comm_idx], latent[comm_idx])
                # latent = self.update_cell(info, latent)

        return latent


class Network(nn.Module):
    def __init__(self, input_shape=configs.obs_shape, cnn_channel=configs.cnn_channel, hidden_dim=configs.hidden_dim,
                 max_comm_agents=configs.max_comm_agents):#obs_shape = (6, 2*obs_radius+1, 2*obs_radius+1)

        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = 16 * 7 * 7
        self.hidden_dim = hidden_dim
        self.max_comm_agents = max_comm_agents

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(self.input_shape[0], cnn_channel, 3, 1),
            nn.ReLU(True),#equal to (inplace=True)
            #A sequential container.
            # Modules will be added to it in the order they are passed in the constructor. Alternatively, an ordered dict of modules can also be passed in
            #self.input_shape[0] denotes the number of input channels, cnn_channel denotes the number of output channels, kernel_size is 3, stride is 1,

            ResBlock(cnn_channel),

            ResBlock(cnn_channel),

            ResBlock(cnn_channel),

            nn.Conv2d(cnn_channel, 16, 1, 1),
            nn.ReLU(True),

            nn.Flatten(),#torch.nn.Flatten()默认从第二维开始平坦化
        )

        self.recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim)#latent_dim= 16 * 7 * 7(output_dim),hidden_dim = 256(input_dim)

        self.comm = CommBlock(hidden_dim)#input_dim=hidden_dim,output_dim=64

        # dueling q structure
        self.adv = nn.Linear(hidden_dim, 5)#advantages
        #construct the fully connected layer，the parameters are respectively in_features and out_features
        self.state = nn.Linear(hidden_dim, 1)#state-value

        self.hidden = None

        for _, m in self.named_modules():#返回网络中所有模块的迭代器，同时产生模块的名称（_）以及模块本身（m）
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):#判断一个对象是否是一个已知的类型，如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False
                nn.init.xavier_uniform_(m.weight)#xavier初始化其核心思想是使层的输出数据的方差与其输入数据的方差相等
                if m.bias is not None:#用来比较两个对象m.bias and None
                    nn.init.constant_(m.bias, 0)#使用0来填充m.bias

    @torch.no_grad()#作为一个装饰器,停止对梯度的计算和存储，从而减少对内存的消耗，不会进行反向传播。
    def step(self, obs, pos):
        num_agents = obs.size(0)#obs is equal to the o^{t}_{i} (the local observation)

        latent = self.obs_encoder(obs)#observation encoder module(latent is equal to O^{^t}_{i})

        if self.hidden is None:
            self.hidden = self.recurrent(latent)#self.recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim) latent_dim= 16 * 7 * 7(output_dim),hidden_dim = 256(input_dim)
        else:
            self.hidden = self.recurrent(latent, self.hidden) #self.hidden is equal to e^{t}_{i}

        # from num_agents x hidden_dim to 1 x num_agents x hidden_dim
        self.hidden = self.hidden.unsqueeze(0)

        # masks for communication block
        agents_pos = pos
        pos_mat = (agents_pos.unsqueeze(1) - agents_pos.unsqueeze(0)).abs()#return the absolute value of the input of abs function
        dist_mat = (pos_mat[:, :, 0] ** 2 + pos_mat[:, :, 1] ** 2).sqrt() #return the square root of the input of sqrt function
        # mask out (蒙住，遮掩)agents that out of range of FOV
        in_obs_mask = (pos_mat <= configs.obs_radius).all(2)#configs.obs_radius=4，
        # a.all()==0determine whether all the values of elements in the input are 0. If yes return True, otherwise return False

        # mask out agents that are far away
        _, ranking = dist_mat.topk(min(self.max_comm_agents, num_agents), dim=1, largest=False)# calculate the first k elements and the corresponding index
        # (from small to large in this case) in some dimension of a tensor
        dist_mask = torch.zeros((num_agents, num_agents), dtype=torch.bool)
        dist_mask.scatter_(1, ranking, True)#dist_mask[i][ranking[i][j]]=True

        comm_mask = torch.bitwise_and(in_obs_mask, dist_mask)#计算 in_obs_mask 和 dist_mask 的按位与

        self.hidden = self.comm(self.hidden, comm_mask)
        self.hidden = self.hidden.squeeze(0)#用于删除指定的维度

        adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)

        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)#calculate the mean value of each line,keepdim=True is to keep the dimension unchanged

        actions = torch.argmax(q_val, 1).tolist()#return the index of the maximum in the specified dimension

        return actions, q_val.numpy(), self.hidden.numpy(), comm_mask.numpy()

    def reset(self):
        self.hidden = None

    @autocast()
    def forward(self, obs, steps, hidden, comm_mask):
        # comm_mask shape: batch_size x seq_len x max_num_agents x max_num_agents
        max_steps = obs.size(1)
        num_agents = comm_mask.size(2)

        assert comm_mask.size(2) == configs.max_num_agents

        obs = obs.transpose(1, 2)

        obs = obs.contiguous().view(-1, *self.input_shape)

        latent = self.obs_encoder(obs)

        latent = latent.view(configs.batch_size * num_agents, max_steps, self.latent_dim).transpose(0, 1)

        hidden_buffer = []
        for i in range(max_steps):
            # hidden size: batch_size*num_agents x self.hidden_dim
            hidden = self.recurrent(latent[i], hidden)
            hidden = hidden.view(configs.batch_size, num_agents, self.hidden_dim)
            hidden = self.comm(hidden, comm_mask[:, i])
            # only hidden from agent 0
            hidden_buffer.append(hidden[:, 0])
            hidden = hidden.view(configs.batch_size * num_agents, self.hidden_dim)

        # hidden buffer size: batch_size x seq_len x self.hidden_dim
        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1)

        # hidden size: batch_size x self.hidden_dim
        hidden = hidden_buffer[torch.arange(configs.batch_size), steps - 1]

        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val
