import random
from typing import List

from pulp import LpStatusInfeasible

import device
import model_util
import pulp as pl



class Algo:
    def __init__(self, RSUs: List[device.RSU], task_list):
        self.RSUs = RSUs
        self.rsu_num = len(RSUs)
        self.task_list = task_list
        self.allocate_task_for_rsu()

    def get_all_task_num(self):
        task_num = 0
        for rsu_idx in range(self.rsu_num):
            task_num = len(self.RSUs[rsu_idx].task_list) + task_num
        return task_num

    def MA(self, task_list, min_gap=0.1):
        rsu_to_rsu_structure = [[[[0 for _ in range(len(model_util.Sub_Model_Structure_Size))]]
                                 for _ in range(self.rsu_num)] for _ in range(self.rsu_num+1)] #一维是被传输的rsu，二维是输出rsu
        rsu_task_queue = self.generate_rsu_request_queue(None)
        T_max = self.cal_objective_value(rsu_to_rsu_structure, rsu_task_queue, is_Initial=True)
        T = T_max
        print("T_max:", T_max)
        T_min = 0
        obj = T_max
        while T_max - T_min >= min_gap:
            throughput, objective_value = self.ma()
            if throughput == self.get_all_task_num():
                T_max = T_max - (T_max - T_min) / 2
                T = T_max
                if obj > objective_value:
                    obj = objective_value
            else:
                T_min = T_max
                T_max = T_max + (T_max - T_min) / 2
                T = T_max
        return obj

    def generate_rsu_request_queue(self, task_remove):
        rsu_request_queue = [[] for _ in range(self.rsu_num)]
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                if task == task_remove:
                    continue
                else:
                    rsu_request_queue[rsu_idx].append(task)
        return rsu_request_queue

    def generate_new_position_request(self, task):
        rsu_request_queue = self.generate_rsu_request_queue(task)

        pass

    def generate_new_position_sub_task(self, sub_task):
        pass

    def move_task(task, new_position):
        pass

    def ma(self):
        changed = True
        while changed:
            changed = False
            for rsu_idx in range(self.rsu_num): # 先遍历请求
                for task in self.RSUs[rsu_idx].task_list:
                    new_position = self.generate_new_position_request(task)
                    if self.move_task(task, new_position):
                        changed = True

            for rsu_idx in range(self.rsu_num): # 遍历子任务
                for task in self.RSUs[rsu_idx].task_list:
                    for sub_task in task:
                        new_position = self.generate_new_position_sub_task(sub_task)
                        if self.move_task(sub_task, new_position):
                            changed = True






    def get_all_task_num_all(self):
        task_num = 0
        for i in range(len(self.task_list)):
            task_num = len(self.task_list[i]) + task_num
        return task_num

    def cal_objective_value(self, rsu_to_rsu_model_structure_list, rsu_task_queue, is_Initial=False, is_Shared=True):
        obj = []
        for rsu_idx in range(self.rsu_num):
            obj.append(self.cal_single_rsu_obj(rsu_to_rsu_model_structure_list[rsu_idx], rsu_task_queue[rsu_idx], rsu_idx, is_Shared, is_Initial)
        return max(obj)

    def cal_single_rsu_obj(self, rsu_download_model, rsu_task_queue, rsu_idx, is_Shared=True, is_Initial=False): #还没有判断存储空间
        device_id = self.RSUs[rsu_idx].device_idx
        task_exec_time = 0
        download_time = 0
        trans_time = 0
        for task in rsu_task_queue:
            sub_task_exec_time_list = []
            for sub_task in task:
                if is_Initial:  #一开始默认从cloud部署
                    trans_time = 0
                    not_added_model_structure = self.RSUs[rsu_idx].has_model_structure(sub_task["model_structure"])
                    if len(not_added_model_structure) != 0:
                        if is_Shared:
                            task_model_size = model_util.get_model_sturctures_size(not_added_model_structure)
                        else:
                            task_model_size = model_util.get_model_sturctures_size(sub_task["model_structure"])
                        download_time += task_model_size / self.RSUs[rsu_idx].download_rate
                    else:
                        download_time = download_time + 0
                else:
                    generated_rsu_id = sub_task["rsu_idx"]
                    if generated_rsu_id != rsu_idx:
                        trans_time += model_util.get_model(sub_task["model_idx"]).single_task_size / \
                                      self.RSUs[generated_rsu_id].rsu_rate
                sub_task_exec_time = self.RSUs[rsu_idx].latency_list[device_id][sub_task["model_idx"]][sub_task["sub_model_idx"]]
                sub_task_exec_time_list.append(sub_task_exec_time)
            if is_Shared:
                task_exec_time += max(sub_task_exec_time_list)
            else:
                task_exec_time += sum(sub_task_exec_time)
        if not is_Initial: #不是初始状况则按照卸载方案计算下载模型时间
            for rsu_idx_other in range(self.rsu_num+1):
                if rsu_idx_other == rsu_idx:
                    continue
                for model_structure_idx in len(model_util.Sub_Model_Structure_Size):
                    if rsu_download_model[rsu_idx_other][model_structure_idx] == 1:
                        download_time += model_util.Sub_Model_Structure_Size[model_structure_idx] / \
                                         (self.RSUs[rsu_idx_other].rsu_rate if rsu_idx_other != self.rsu_num
                                          else self.RSUs[rsu_idx].download_rate)
        singl_obj_value = task_exec_time + download_time




    def allocate_model_for_rsu_later(self, rsu_model_list):
        pass

    def allocate_task_for_rsu_later(self, rsu_job_list):
        for rsu_idx in range(len(rsu_job_list)):
            self.RSUs[rsu_idx].task_list = rsu_job_list[rsu_idx]

    def clear_rsu_task_list(self):
        for rsu_idx in range(self.rsu_num):
            self.RSUs[rsu_idx].task_list = []

    def allocate_task_for_rsu(self):
        for task in self.task_list:
            rsu_id = task[0]["rsu_id"]
            self.RSUs[rsu_id].task_list.append(task)

    def cal_max_response_time(self, rsu_task_queue):
        T_max = 0
        total_time_list = []  # 记录每个RSU的执行时间
        for rsu_idx in range(self.rsu_num):
            flag = 0
            device_id = self.RSUs[rsu_idx].device_idx
            for task in self.RSUs[rsu_idx].task_list:
                flag = 1
                exec_time_sub_task_list = []
                task_exec_time_list = []
                for sub_task in task:
                    seq_num = sub_task["seq_num"]
                    model_idx = sub_task["model_idx"]
                    sub_model_idx = sub_task["sub_model_idx"]
                    exec_time_sub_task = 0
                    rsu_cached_models = self.RSUs[rsu_idx].get_cached_model()
                    sub_task_model = sub_task["model_structure"]
                    if len(self.RSUs[rsu_idx].has_model_structure(sub_task_model)) == 0:
                        exec_time = self.RSUs[rsu_idx].latency_list[model_idx][sub_model_idx][device_id][seq_num]
                        download_time = 0
                    else:
                        if is_shared:
                            not_added_model = self.RSUs[rsu_idx].has_model_structure(sub_task_model)
                            task_model_size = model_util.get_model_sturctures_size(not_added_model)
                        else:
                            task_model_size = model_util.get_model_sturctures_size(sub_task["model_structure"])
                        if is_Initial:
                            download_time = task_model_size / self.RSUs[rsu_idx].download_rate

                        exec_time = self.RSUs[rsu_idx].latency_list[model_idx][sub_model_idx][device_id][seq_num]
                    exec_time_sub_task = download_time + exec_time
                    exec_time_sub_task_list.append(exec_time_sub_task)
                if is_shared:
                    task_exec_time_list.append(max(exec_time_sub_task_list))
                else:
                    task_exec_time_list.append(sum(exec_time_sub_task_list))
            if flag:
                total_time_list.append(sum(task_exec_time_list))
        print("total_time_list:", total_time_list)
        T_max = max(total_time_list)
        return T_max
