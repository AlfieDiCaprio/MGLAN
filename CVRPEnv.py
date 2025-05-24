
from dataclasses import dataclass
import torch
import pickle
import numpy as np

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    #node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_colors: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    pre_step_color: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        self.depot_node_colors = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)
        self.pre_step_color=None

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)
        for key, value in loaded_dict.items():
            if isinstance(value, torch.Tensor):  # 检查值是否为张量
                loaded_dict[key] = value.to(device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        # self.saved_node_demand = loaded_dict['node_demand']
        self.saved_node_colors = loaded_dict['colors_xy']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        #global colors_xy
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy,colors_xy = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            #node_demand = self.saved_node_demand[self.saved_index:self.saved_index + batch_size]
            colors_xy = self.saved_node_colors[self.saved_index:self.saved_index + batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                #node_demand = node_demand.repeat(8, 1)
                colors_xy= colors_xy.repeat(8,1,1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        #修改了
        #depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        #self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        depot_color = torch.ones(size=(self.batch_size, 1,4))
        self.depot_node_colors=torch.cat((depot_color, colors_xy), dim=1)


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        #self.reset_state.node_demand = node_demand
        self.reset_state.node_colors = colors_xy

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)
        self.pre_step_color=torch.ones((self.batch_size, self.pomo_size, 4))
        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.pre_step_color=self.pre_step_color
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)



        color_list=self.depot_node_colors[:, None, :,:].expand(self.batch_size, self.pomo_size,-1, self.depot_node_colors.size(-1))
        gathering_index2=selected[:, :, None,None].expand(-1, -1, -1, 4)
        output = color_list.gather(dim=2, index=gathering_index2)
        selected_colors = output.squeeze(2)
        self.pre_step_color=selected_colors.long() & self.pre_step_color.long()
        self.pre_step_color[self.at_the_depot] = torch.tensor([1, 1, 1, 1], dtype=torch.long,device=self.pre_step_color.device)
        pre_colors=self.pre_step_color.unsqueeze(2).long() & color_list.long()
        is_all_zero = (pre_colors == 0).all(dim=3)
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0
        # mask_depot = self.visited_ninf_flag == float('-inf')
        # mask_end=mask_depot.sum(dim=2)==self.problem_size
        # self.visited_ninf_flag[:, :, 0][mask_end] = 0
        # self.visited_ninf_flag[mask_depot] = 0

        # shape: (batch, pomo, problem+1)
        # self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot
        self.ninf_mask = self.visited_ninf_flag.clone()
        self.ninf_mask[is_all_zero] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished
        depot_finished = (self.ninf_mask == float('-inf')).all(dim=2)
        self.ninf_mask[:, :, 0][self.finished] = 0
        self.ninf_mask[:, :, 0][depot_finished] = 0
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.pre_step_color = self.pre_step_color
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

