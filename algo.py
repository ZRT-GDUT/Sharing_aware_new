import random
import sys
from typing import List

from matplotlib import pyplot as plt
from pulp import LpStatusInfeasible
from tqdm import tqdm

import DQN
import device
import model_util
import pulp as pl
import numpy as np


class Algo:
    def __init__(self, RSUs: List[device.RSU], task_list, sub_task_list, model_download_time_list):
        self.RSUs = RSUs
        self.rsu_num = len(RSUs)
        self.task_list = task_list
        self.sub_task_list = sub_task_list
        self.cloudidx = self.rsu_num
        self.model_download_time_list = model_download_time_list

    def MA(self, min_gap=0.1):
        for rsu_idx in range(self.rsu_num):
            self.RSUs[rsu_idx].sub_task_list = []
            self.RSUs[rsu_idx].task_list = []
            self.RSUs[rsu_idx].model_structure_list = self.RSUs[rsu_idx].initial_model_structure_list.copy()
        self.allocate_task_for_rsu()
        rsu_to_rsu_structure = {}  # xx-xx:xx-xx-[...]
        rsu_to_rsu_model_structure_sub_task = {}
        is_init = True
        # for job_id in len(task_list):
        #     for sub_task in task_list[job_id]:
        #         rsu_to_rsu_structure[{} - {}.format(job_id, sub_task["sub_model_idx"])] = []
        T_max, rsu_to_rsu_model_structure_list = self.cal_objective_value(rsu_to_rsu_structure, is_Initial=True)
        T = T_max
        print("T_max:", T_max)
        T_min = 0
        obj = T_max
        while T_max - T_min >= min_gap:
            throughput, objective_value, rsu_to_rsu_model_structure_sub_task = \
                self.ma(rsu_to_rsu_model_structure_sub_task, rsu_to_rsu_model_structure_list, T_max, is_init)
            if throughput == self.get_total_sub_num():
                T_max = T_max - (T_max - T_min) / 2
                T = T_max
                if obj > objective_value:
                    obj = objective_value
            else:
                T_min = T_max
                T_max = T_max + (T_max - T_min) / 2
                T = T_max
            is_init = False
        return -obj

    def generate_rsu_request_queue(self):
        rsu_request_queue = [[] for _ in range(self.rsu_num)]
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                rsu_request_queue[rsu_idx].append(task)
        return rsu_request_queue

    def generate_rsu_request_queue(self):
        rsu_request_queue = [[] for _ in range(self.rsu_num)]
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                rsu_request_queue[rsu_idx].append(task)
        return rsu_request_queue

    def generate_new_position_request(self, task, rsu_to_rsu_model_structure_list, T_max, is_Shared=True):
        obj_value = self.cal_objective_value(rsu_to_rsu_model_structure_list, is_Request=True)
        task_copy = task.copy()
        rsu_idx_task_initial = task_copy[0]['position']
        rsu_idx_task = task_copy[0]['position']
        job_id = task_copy[0]["job_id"]
        task_model_structure_list = set()
        latency_requiredment = task_copy[0]["latency"]
        for sub_task in task:
            for model_structure_idx in sub_task["model_structure"]:
                task_model_structure_list.add(model_structure_idx)
        for rsu_idx in range(self.rsu_num):  # 遍历task在每个rsu上部署的情况
            if rsu_idx == rsu_idx_task_initial:
                continue
            not_added_model_structure = self.RSUs[rsu_idx].has_model_structure(task_model_structure_list)
            not_added_model_structure_initial = self.RSUs[rsu_idx].has_model_structure_initial(
                task_model_structure_list)
            generated_id = task[0]["rsu_id"]
            model_idx = task[0]["model_idx"]
            model = model_util.get_model(model_idx)
            task_size = model.single_task_size
            offload_time = 0
            download_time = 0
            exec_time_list = []
            device_id = self.RSUs[rsu_idx].device_idx
            if generated_id != rsu_idx:
                offload_time = task_size / self.RSUs[generated_id].rsu_rate
            for model_structure_idx in not_added_model_structure_initial:  # 通过greeedy方式获取应从哪些rsu下载model
                if self.model_download_time_list.get(model_structure_idx) is not None:
                    if self.RSUs[rsu_idx].download_rate > self.RSUs[
                        self.model_download_time_list[model_structure_idx]].rsu_rate:
                        download_time += model_util.Sub_Model_Structure_Size[model_structure_idx] / \
                                         self.RSUs[rsu_idx].download_rate
                    else:
                        download_time += model_util.Sub_Model_Structure_Size[model_structure_idx] / \
                                         self.RSUs[self.model_download_time_list[model_structure_idx]].rsu_rate
            for sub_task_exec in task:
                sub_model_idx = sub_task_exec["sub_model_idx"]
                seq_num = sub_task_exec["seq_num"]
                exec_time_ = self.RSUs[rsu_idx].latency_list[model_idx][sub_model_idx][device_id][seq_num]
                exec_time_list.append(exec_time_)
            exec_time = max(exec_time_list)
            if exec_time + download_time + offload_time <= latency_requiredment:
                if self.RSUs[rsu_idx].satisfy_add_task_constraint(task):
                    self.RSUs[rsu_idx].add_task(task)
                    if self.RSUs[rsu_idx].satisfy_add_model_structure_constraint(not_added_model_structure):
                        self.RSUs[rsu_idx_task].remove_task(task)
                    else:
                        self.RSUs[rsu_idx].remove_task(task)
                        continue
                else:
                    continue
            else:
                continue
            if len(not_added_model_structure_initial) != 0:
                download_model_rsu_info_list = []
                download_model_rsu_list = {}
                download_model_rsu_info_list_before = []
                for model_structure_idx in not_added_model_structure_initial:  # 通过greeedy方式获取应从哪些rsu下载model
                    if self.model_download_time_list.get(model_structure_idx) is not None:
                        if self.RSUs[rsu_idx].download_rate > self.RSUs[
                            self.model_download_time_list[model_structure_idx]].rsu_rate:
                            if download_model_rsu_list.get(self.cloudidx) is None:
                                download_model_rsu_list[self.cloudidx] = set()
                            download_model_rsu_list[self.cloudidx].add(model_structure_idx)
                        else:
                            if download_model_rsu_list.get(self.model_download_time_list[model_structure_idx]) is None:
                                download_model_rsu_list[self.model_download_time_list[model_structure_idx]] = set()
                            download_model_rsu_list[self.model_download_time_list[model_structure_idx]].add(
                                model_structure_idx)
                    else:
                        if download_model_rsu_list.get(self.cloudidx) is None:
                            download_model_rsu_list[self.cloudidx] = set()
                        download_model_rsu_list[self.cloudidx].add(model_structure_idx)
                for download_rsu_idxs in download_model_rsu_list.keys():
                    download_model_rsu_info = self.get_download_model_rsu(download_rsu_idxs, rsu_idx, list(
                        download_model_rsu_list[download_rsu_idxs]))
                    download_model_rsu_info_list.append(download_model_rsu_info)
                download_model_rsu_info_list_before = rsu_to_rsu_model_structure_list[job_id]
                rsu_to_rsu_model_structure_list[job_id] = download_model_rsu_info_list
            else:
                download_model_rsu_info_list_before = rsu_to_rsu_model_structure_list[job_id]
                download_model_rsu_info = self.get_download_model_rsu(rsu_idx, rsu_idx, [])
                rsu_to_rsu_model_structure_list[job_id] = []
                rsu_to_rsu_model_structure_list[job_id].append(download_model_rsu_info)
            obj_value_new = self.cal_objective_value(rsu_to_rsu_model_structure_list, is_Initial=False, is_Request=True)
            if obj_value_new < obj_value and obj_value_new < T_max:
                self.RSUs[rsu_idx].add_model_structure(not_added_model_structure)
                removed_model_list = set()
                for removed_model_off in download_model_rsu_info_list_before:
                    _, _, removed_models = self.get_download_model_rsu_info(removed_model_off)
                    for removed_model_idx in removed_models:
                        if removed_model_idx not in self.RSUs[rsu_idx_task].initial_model_structure_list:
                            removed_model_list.add(removed_model_idx)
                for task_ in self.RSUs[rsu_idx_task].task_list:
                    for model_off in rsu_to_rsu_model_structure_list[task_[0]["job_id"]]:
                        _, _, download_model_ = self.get_download_model_rsu_info(model_off)
                        set_download_model_ = set(download_model_)
                        inter_model = set_download_model_.intersection(removed_model_list)
                        if len(inter_model) != 0:
                            for removed_model_idxs in inter_model:
                                removed_model_list.remove(removed_model_idxs)
                self.RSUs[rsu_idx_task].remove_model_structure(removed_model_list)  # ??
                obj_value = obj_value_new
                for sub_task_ in task:
                    sub_task_["position"] = rsu_idx
                rsu_idx_task = rsu_idx
            else:
                self.RSUs[rsu_idx].remove_task(task)
                self.RSUs[rsu_idx_task].add_task(task)
                rsu_to_rsu_model_structure_list[job_id] = download_model_rsu_info_list_before
        return rsu_to_rsu_model_structure_list

    def generate_new_position_sub_task(self, task, rsu_to_rsu_model_structure_list, T_max, is_Shared=True):
        obj_value = self.cal_objective_value(rsu_to_rsu_model_structure_list, is_Request=False)
        rsu_idx_task_initial = task["position"]
        rsu_idx_task = task["position"]
        job_id = task["job_id"]
        sub_task_id = task["sub_model_idx"]
        sub_task_key = "{}-{}".format(job_id, sub_task_id)
        latency_requirement = task["latency"]
        for rsu_idx in range(self.rsu_num):  # 遍历task在每个rsu上部署的情况
            if rsu_idx == rsu_idx_task_initial:
                continue
            not_added_model_structure = self.RSUs[rsu_idx].has_model_structure(task["model_structure"])
            not_added_model_structure_initial = self.RSUs[rsu_idx].has_model_structure_initial(task["model_structure"])
            generated_id = task["rsu_id"]
            model_idx = task["model_idx"]
            model = model_util.get_model(model_idx)
            task_size = model.single_task_size
            offload_time = 0
            download_time = 0
            device_id = self.RSUs[rsu_idx].device_idx
            sub_model_idx = task["sub_model_idx"]
            seq_num = task["seq_num"]
            exec_time = self.RSUs[rsu_idx].latency_list[model_idx][sub_model_idx][device_id][seq_num]
            if generated_id != rsu_idx:
                offload_time = task_size / self.RSUs[generated_id].rsu_rate
            for model_structure_idx in not_added_model_structure_initial:  # 通过greeedy方式获取应从哪些rsu下载model
                if self.model_download_time_list.get(model_structure_idx) is not None:
                    if self.RSUs[rsu_idx].download_rate > self.RSUs[
                        self.model_download_time_list[model_structure_idx]].rsu_rate:
                        download_time += model_util.Sub_Model_Structure_Size[model_structure_idx] / \
                                         self.RSUs[rsu_idx].download_rate
                    else:
                        download_time += model_util.Sub_Model_Structure_Size[model_structure_idx] / \
                                         self.RSUs[self.model_download_time_list[model_structure_idx]].rsu_rate
            if exec_time + download_time + offload_time <= latency_requirement:
                if self.RSUs[rsu_idx].satisfy_add_task_constraint(task, is_Request=False):
                    self.RSUs[rsu_idx].sub_task_list.append(task)
                    if self.RSUs[rsu_idx].satisfy_add_model_structure_constraint(not_added_model_structure,
                                                                                 is_Request=False):
                        self.RSUs[rsu_idx_task].sub_task_list.remove(task)
                    else:
                        self.RSUs[rsu_idx].sub_task_list.remove(task)
                        continue
                else:
                    continue
            else:
                continue
            if len(not_added_model_structure_initial) != 0:
                download_model_rsu_info_list = []
                download_model_rsu_list = {}
                download_model_rsu_info_list_before = []
                for rsu_idxs in range(self.rsu_num):
                    download_model_rsu_list[rsu_idxs] = set()
                for model_structure_idx in not_added_model_structure_initial:  # 通过greeedy方式获取应从哪些rsu下载model
                    if self.model_download_time_list.get(model_structure_idx) is not None:
                        if self.RSUs[rsu_idx].download_rate > self.RSUs[
                            self.model_download_time_list[model_structure_idx]].rsu_rate:
                            if download_model_rsu_list.get(self.cloudidx) is None:
                                download_model_rsu_list[self.cloudidx] = set()
                            download_model_rsu_list[self.cloudidx].add(model_structure_idx)
                        else:
                            if download_model_rsu_list.get(self.model_download_time_list[model_structure_idx]) is None:
                                download_model_rsu_list[self.model_download_time_list[model_structure_idx]] = set()
                            download_model_rsu_list[self.model_download_time_list[model_structure_idx]].add(
                                model_structure_idx)
                    else:
                        if download_model_rsu_list.get(self.cloudidx) is None:
                            download_model_rsu_list[self.cloudidx] = set()
                        download_model_rsu_list[self.cloudidx].add(model_structure_idx)
                for download_rsu_idxs in download_model_rsu_list.keys():
                    download_model_rsu_info = self.get_download_model_rsu(download_rsu_idxs, rsu_idx, list(
                        download_model_rsu_list[download_rsu_idxs]))
                    download_model_rsu_info_list.append(download_model_rsu_info)
                download_model_rsu_info_list_before = rsu_to_rsu_model_structure_list[sub_task_key]
                rsu_to_rsu_model_structure_list[sub_task_key] = download_model_rsu_info_list
            else:
                download_model_rsu_info_list_before = rsu_to_rsu_model_structure_list[sub_task_key]
                download_model_rsu_info = self.get_download_model_rsu(rsu_idx, rsu_idx, [])
                rsu_to_rsu_model_structure_list[sub_task_key] = []
                rsu_to_rsu_model_structure_list[sub_task_key].append(download_model_rsu_info)
            obj_value_new = self.cal_objective_value(rsu_to_rsu_model_structure_list, is_Initial=False,
                                                     is_Request=False)
            if obj_value_new < obj_value and obj_value_new < T_max:
                self.RSUs[rsu_idx].add_model_structure(task["model_structure"])
                removed_model_list = set()
                for removed_model_off in download_model_rsu_info_list_before:
                    _, _, removed_models = self.get_download_model_rsu_info(removed_model_off)
                    for removed_model_idx in removed_models:
                        if removed_model_idx not in self.RSUs[rsu_idx_task].initial_model_structure_list:
                            removed_model_list.add(removed_model_idx)
                for task_ in self.RSUs[rsu_idx_task].sub_task_list:
                    sub_task_key_ = "{}-{}".format(task_["job_id"], task_["sub_model_idx"])
                    for model_off in rsu_to_rsu_model_structure_list[sub_task_key_]:
                        _, _, download_model_ = self.get_download_model_rsu_info(model_off)
                        set_download_model_ = set(download_model_)
                        inter_model = set_download_model_.intersection(removed_model_list)
                        if len(inter_model) != 0:
                            for removed_model_idxs in inter_model:
                                removed_model_list.remove(removed_model_idxs)
                self.RSUs[rsu_idx_task].remove_model_structure(removed_model_list)
                obj_value = obj_value_new
                task['position'] = rsu_idx
                rsu_idx_task = rsu_idx
            else:
                self.RSUs[rsu_idx].sub_task_list.remove(task)
                self.RSUs[rsu_idx_task].sub_task_list.append(task)
                rsu_to_rsu_model_structure_list[sub_task_key] = download_model_rsu_info_list_before
        return rsu_to_rsu_model_structure_list

    def get_download_model_rsu(self, download_rsu_idx, rsu_idx, model_structure_list):
        return "{}-{}-{}".format(download_rsu_idx, rsu_idx, model_structure_list)

    def get_download_model_rsu_info(self, download_model_rsu_info):
        info = download_model_rsu_info.split("-")
        string = info[2]
        my_list = list(eval(string))
        return int(info[0]), int(info[1]), my_list

    def ma(self, rsu_to_rsu_structure_sub_task, rsu_to_rsu_model_structure_list, T_max, is_init=True):
        changed_sub_task = True
        changed_request = True
        rsu_to_rsu_model_structure_sub_task = {}
        if is_init:
            while changed_request:
                changed_request = False
                for task in self.task_list:
                    old_position = task[0]["position"]
                    rsu_to_rsu_model_structure_list = \
                        self.generate_new_position_request(task, rsu_to_rsu_model_structure_list, T_max)
                    if old_position != task[0]['position']:
                        changed_request = True
            rsu_to_rsu_structure_sub_task = self.trans_request_to_sub_task(rsu_to_rsu_model_structure_list)
            self.allocate_sub_task_for_rsu()
        while changed_sub_task:
            changed_sub_task = False
            for task_ in self.task_list:  # 遍历子任务
                for sub_task in task_:
                    old_position_sub = sub_task['position']
                    rsu_to_rsu_model_structure_list_sub_task = self.generate_new_position_sub_task(
                        sub_task, rsu_to_rsu_structure_sub_task, T_max)
                    if sub_task['position'] != old_position_sub:
                        changed_sub_task = True
        obj = self.cal_objective_value(rsu_to_rsu_model_structure_list_sub_task)
        throughput = self.get_total_sub_num()
        return throughput, obj, rsu_to_rsu_structure_sub_task

    def trans_request_to_sub_task(self, rsu_to_rsu_model_structure_list):
        rsu_to_rsu_model_structure_list_sub_task = {}
        for job_id_ in range(len(self.task_list)):
            for sub_task_ in range(len(self.task_list[job_id_])):
                rsu_to_rsu_model_structure_list_sub_task["{}-{}".format(job_id_, sub_task_)] = []
        for job_id in rsu_to_rsu_model_structure_list.keys():
            for sub_task in self.task_list[job_id]:
                sub_task_id = sub_task["sub_model_idx"]
                key = "{}-{}".format(job_id, sub_task_id)
                for download_info in rsu_to_rsu_model_structure_list[job_id]:
                    download_rsu_idx, task_rsu_idx, download_models = self.get_download_model_rsu_info(download_info)
                    inter_model = set(sub_task["model_structure"]).intersection(set(download_models))
                    if len(inter_model) != 0:
                        download_info_ = self.get_download_model_rsu(download_rsu_idx, task_rsu_idx, inter_model)
                        rsu_to_rsu_model_structure_list_sub_task[key].append(download_info_)
        for download_info_key in rsu_to_rsu_model_structure_list_sub_task.keys():
            if len(rsu_to_rsu_model_structure_list_sub_task[download_info_key]) == 0:
                download_info_keys = download_info_key.split("-")
                job_idx = int(download_info_keys[0])
                sub_task_idx = int(download_info_keys[1])
                task_none = self.task_list[job_idx][sub_task_idx]
                download_info_none = self.get_download_model_rsu(task_none['position'], task_none['position'], [])
                rsu_to_rsu_model_structure_list_sub_task[download_info_key].append(download_info_none)
        return rsu_to_rsu_model_structure_list_sub_task

    def get_all_task_num_all(self):
        task_num = 0
        for i in range(len(self.task_list)):
            task_num = len(self.task_list[i]) + task_num
        return task_num

    def get_total_sub_num(self):
        return len(self.sub_task_list)

    def cal_objective_value(self, rsu_to_rsu_model_structure_list, is_Initial=False, is_Shared=True,
                            is_Request=False, is_dqn=False):
        obj = []
        for rsu_idx in range(self.rsu_num):
            if is_Initial:
                obj_single, rsu_to_rsu_model_structure_list = self.cal_single_rsu_obj_initial(
                    rsu_to_rsu_model_structure_list, rsu_idx, is_Shared)
                obj.append(obj_single)
            else:
                obj.append(
                    self.cal_single_rsu_obj(rsu_to_rsu_model_structure_list, rsu_idx, is_Shared, is_Request))
        if is_Initial:
            if is_dqn:
                return max(obj), rsu_to_rsu_model_structure_list, obj
            else:
                return max(obj), rsu_to_rsu_model_structure_list
        else:
            if is_dqn:
                return max(obj), obj
            else:
                return max(obj)

    def cal_single_rsu_obj_initial(self, rsu_download_model, rsu_idx, is_Shared=True):  # 还没有判断存储空间
        device_id = self.RSUs[rsu_idx].device_idx
        task_exec_time = 0
        download_time = 0
        already_download = set()
        for task in self.RSUs[rsu_idx].task_list:
            sub_task_exec_time_list = []
            task_model_structure_list = set()
            for sub_task in task:
                for model_structure_idx in sub_task["model_structure"]:
                    task_model_structure_list.add(model_structure_idx)
                sub_task_exec_time = self.RSUs[rsu_idx].latency_list[sub_task["model_idx"]][
                    sub_task["sub_model_idx"]][device_id][sub_task["seq_num"]]
                sub_task_exec_time_list.append(sub_task_exec_time)
            if is_Shared:
                task_exec_time += max(sub_task_exec_time_list)
            else:
                task_exec_time += sum(sub_task_exec_time_list)
            not_added_model_structure = self.RSUs[rsu_idx].has_model_structure_initial(task_model_structure_list)
            if len(not_added_model_structure) != 0:
                task_model_size = 0
                for model_structure_idxs in not_added_model_structure:
                    if model_structure_idxs not in already_download:
                        task_model_size += model_util.Sub_Model_Structure_Size[model_structure_idxs]
                        already_download.add(model_structure_idxs)
                download_model_rsu = self.get_download_model_rsu(self.cloudidx, task[0]["rsu_id"],
                                                                 not_added_model_structure)
                # if is_Shared:
                #     task_model_size = model_util.get_model_sturctures_size(not_added_model_structure)
                #     download_model_rsu = self.get_download_model_rsu(self.cloudidx, task[0]["rsu_id"],
                #                                                      not_added_model_structure)
                # else:
                #     task_model_size = model_util.get_model_sturctures_size(task_model_structure_list)
                #     download_model_rsu = self.get_download_model_rsu(self.cloudidx, task[0]["rsu_id"],
                #                                                      task_model_structure_list)
                if model_util.get_model_sturctures_size(not_added_model_structure) + \
                        model_util.get_model_sturctures_size(self.RSUs[rsu_idx].initial_model_structure_list) + \
                        self.RSUs[rsu_idx].get_total_task_size(is_Request=True) < self.RSUs[rsu_idx].storage_capacity:
                    download_time += task_model_size / self.RSUs[rsu_idx].download_rate
                else:
                    download_time += 999999
                self.RSUs[rsu_idx].add_model_structure(not_added_model_structure)
            else:
                download_time += 0
                download_model_rsu = self.get_download_model_rsu(rsu_idx, rsu_idx, [])
            if rsu_download_model.get(task[0]["job_id"]) is None:
                rsu_download_model[task[0]["job_id"]] = []
            rsu_download_model[task[0]["job_id"]].append(download_model_rsu)
        singl_obj_value = task_exec_time + download_time
        return singl_obj_value, rsu_download_model

    def cal_single_rsu_obj(self, rsu_download_model, rsu_idx, is_Shared=True, is_Request=False):  # 还没有判断存储空间
        device_id = self.RSUs[rsu_idx].device_idx
        task_exec_time_list = {0: [], 1: [], 2: []}
        download_time_list = {}
        download_time = 0
        trans_time = 0
        already_trans = {0: [], 1: [], 2: []}
        already_download = {model_structure_idx: set() for model_structure_idx in
                            range(len(model_util.Sub_Model_Structure_Size))}
        already_download_no_shared = self.RSUs[rsu_idx].initial_model_structure_list.copy()
        if is_Request:
            for job_id in rsu_download_model.keys():
                generated_id = self.task_list[job_id][0]["rsu_id"]
                model_idx = self.task_list[job_id][0]["model_idx"]
                value = rsu_download_model[job_id][0]
                trans_rsu_idx, task_rsu_idx, _ = self.get_download_model_rsu_info(value)
                if task_rsu_idx == rsu_idx:
                    if task_rsu_idx == generated_id or trans_rsu_idx in already_trans[model_idx]:
                        trans_time += 0
                    else:
                        already_trans[model_idx].append(trans_rsu_idx)
                        sub_task_size = model_util.get_model(model_idx).single_task_size
                        trans_time_current = sub_task_size / self.RSUs[generated_id].rsu_rate
                        trans_time += trans_time_current
                    for sub_task in self.task_list[job_id]:  # 每种类型的task在shared情况下计算最大计算时间，反之则计算总和
                        sub_task_exectime = \
                            self.RSUs[rsu_idx].latency_list[model_idx][sub_task["sub_model_idx"]][device_id][
                                sub_task["seq_num"]]
                        task_exec_time_list[model_idx].append(sub_task_exectime)
                    for download_info in rsu_download_model[job_id]:
                        download_rsu_idx, task_rsu_idx, download_models = self.get_download_model_rsu_info(
                            download_info)
                        if len(download_models) == 0:
                            download_time_current = 0
                        else:
                            download_model_size = 0
                            for model_structure_idx in download_models:
                                if len(already_download[model_structure_idx]) != 0:
                                    continue
                                else:
                                    already_download[model_structure_idx].add(download_rsu_idx)
                                    download_model_size += model_util.Sub_Model_Structure_Size[model_structure_idx]
                            download_time_current = download_model_size / (self.RSUs[download_rsu_idx].rsu_rate if
                                                                           download_rsu_idx != self.cloudidx else
                                                                           self.RSUs[task_rsu_idx].download_rate)
                        if download_time_list.get(download_rsu_idx) is None:
                            download_time_list[download_rsu_idx] = []
                        download_time_list[download_rsu_idx].append(download_time_current)
        else:
            for sub_task_id in rsu_download_model.keys():
                sub_key = sub_task_id.split("-")
                job_id = int(sub_key[0])
                sub_task = int(sub_key[1])
                generated_id = self.task_list[job_id][sub_task]["rsu_id"]
                model_idx = self.task_list[job_id][sub_task]["model_idx"]
                value = rsu_download_model[sub_task_id][0]
                trans_rsu_idx, task_rsu_idx, _ = self.get_download_model_rsu_info(value)
                if task_rsu_idx == rsu_idx:
                    if task_rsu_idx == generated_id or trans_rsu_idx in already_trans[model_idx]:
                        trans_time += 0
                    else:
                        already_trans[model_idx].append(trans_rsu_idx)
                        sub_task_size = model_util.get_model(model_idx).single_task_size
                        trans_time_current = sub_task_size / self.RSUs[generated_id].rsu_rate
                        trans_time += trans_time_current
                    sub_task_exectime = self.RSUs[rsu_idx].latency_list[model_idx][sub_task][device_id][
                        self.task_list[job_id][sub_task]["seq_num"]]
                    task_exec_time_list[model_idx].append(sub_task_exectime)
                    for download_info in rsu_download_model[sub_task_id]:
                        download_rsu_idx, task_rsu_idx, download_models = self.get_download_model_rsu_info(
                            download_info)
                        if len(download_models) == 0:
                            download_time_current = 0
                        else:
                            download_model_size = 0
                            if is_Shared:
                                for model_structure_idx in download_models:
                                    if len(already_download[model_structure_idx]) != 0:
                                        continue
                                    else:
                                        already_download[model_structure_idx].add(download_rsu_idx)
                                        download_model_size += model_util.Sub_Model_Structure_Size[model_structure_idx]
                                download_time_current = download_model_size / (self.RSUs[download_rsu_idx].rsu_rate if
                                                                               download_rsu_idx != self.cloudidx else
                                                                               self.RSUs[task_rsu_idx].download_rate)
                            else:
                                flag = True
                                set_download_models = set(download_models)
                                if set_download_models.issubset(already_download_no_shared):
                                    download_time_current = 0
                                else:
                                    for x in set_download_models:
                                        already_download_no_shared.add(x)
                                    download_model_size = model_util.get_model_sturctures_size(download_models)
                                    download_time_current = download_model_size / (
                                        self.RSUs[download_rsu_idx].rsu_rate if
                                        download_rsu_idx != self.cloudidx else
                                        self.RSUs[task_rsu_idx].download_rate)
                        if download_time_list.get(download_rsu_idx) is None:
                            download_time_list[download_rsu_idx] = []
                        download_time_list[download_rsu_idx].append(download_time_current)
        task_exec_time = 0
        for model_idxs in task_exec_time_list.keys():
            if len(task_exec_time_list[model_idxs]) == 0:
                continue
            if is_Shared:
                task_exec_time += max(task_exec_time_list[model_idxs])
            else:
                task_exec_time += sum(task_exec_time_list[model_idxs])
        for download_rsu_idxs in download_time_list.keys():
            download_time += sum(download_time_list[download_rsu_idxs])
        obj_value = trans_time + download_time + task_exec_time
        return obj_value

    def allocate_sub_task_for_rsu(self):
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                for sub_task in task:
                    self.RSUs[rsu_idx].sub_task_list.append(sub_task)

    def allocate_task_for_rsu(self):
        for task in self.task_list:
            rsu_id = task[0]["rsu_id"]
            self.RSUs[rsu_id].task_list.append(task)

    # ------------------------------------------------------------------------------
    #                DQN algorithm
    # ------------------------------------------------------------------------------
    def dqn(self, num_epoch=500):

        def employ_action(action_value, rsu_model_queue):
            action_value = int(action_value)
            src_rsu_id = int(action_value / (self.rsu_num * len(model_util.Sub_Model_Structure_Size)))
            des_rsu_id = int(action_value % self.rsu_num)
            model_id = int(action_value % len(model_util.Sub_Model_Structure_Size))
            des_rsu_model_size = 0
            for model_structure_idx in range(len(model_util.Sub_Model_Structure_Size)):
                if rsu_model_queue[des_rsu_id][model_structure_idx] == 1:
                    des_rsu_model_size += model_util.Sub_Model_Structure_Size[model_structure_idx]
            if src_rsu_id == des_rsu_id:
                download_time = -10000
                return download_time, rsu_model_queue
            if src_rsu_id != self.rsu_num:
                if model_id not in self.RSUs[src_rsu_id].initial_model_structure_list or \
                        model_id in self.RSUs[des_rsu_id].initial_model_structure_list:
                    download_time = -10000
                    return download_time, rsu_model_queue
                else:
                    if des_rsu_model_size + model_util.Sub_Model_Structure_Size[model_id] > \
                            self.RSUs[des_rsu_id].storage_capacity:
                        download_time = -10000
                    else:
                        rsu_model_queue[des_rsu_id][model_id] = 1
                        download_time = -(
                                model_util.Sub_Model_Structure_Size[model_id] / self.RSUs[src_rsu_id].rsu_rate)
                    return download_time, rsu_model_queue
            else:
                if model_id in self.RSUs[des_rsu_id].initial_model_structure_list:
                    download_time = -10000
                else:
                    if self.RSUs[des_rsu_id].get_total_model_size() + model_util.Sub_Model_Structure_Size[model_id] > \
                            self.RSUs[des_rsu_id].storage_capacity:
                        download_time = -10000
                    else:
                        rsu_model_queue[des_rsu_id][model_id] = 1
                        download_time = -(
                                model_util.Sub_Model_Structure_Size[model_id] / self.RSUs[des_rsu_id].download_rate)
                return download_time, rsu_model_queue

        def get_observation(rsu_model_queue) -> list:
            obs = [element for sublist in rsu_model_queue for element in sublist]
            return obs

        def get_observation_task(rsu_to_rsu_model_structure_list) -> list:
            _, observations = self.cal_objective_value(rsu_to_rsu_model_structure_list, is_dqn=True)
            return observations

        def employ_action_task(action_value, rsu_to_rsu_model_structure_list_sub, DRL_model):
            # 更新策略
            # 0: 完成修改
            # 1: 不满足约束
            # 2: 不需要修改
            rsu_model_queue = self.generate_rsu_model_queue()
            # 根据model迁移情况判断每个rsu的model存储情况
            for key in rsu_to_rsu_model_structure_list_sub.keys():
                for model_policy_ in rsu_to_rsu_model_structure_list_sub[key]:
                    src_rsu_, des_rsu_, model_list_ = self.get_download_model_rsu_info(model_policy_)
                    for model_idx in model_list_:
                        rsu_model_queue[des_rsu_][model_idx] = 1
            model_env = get_observation(rsu_model_queue)
            task_num = len(self.sub_task_list)
            action_value = int(action_value)
            rsu_id = int(action_value / task_num)
            task_id = action_value % task_num
            task = self.sub_task_list[task_id]
            request_id = task['job_id']
            sub_task_id = task["sub_model_idx"]
            sub_task_key = "{}-{}".format(request_id, sub_task_id)
            model_policy = rsu_to_rsu_model_structure_list_sub[sub_task_key][0]
            src_rsu, des_rsu, _ = self.get_download_model_rsu_info(model_policy)
            if des_rsu == rsu_id:
                return 2, rsu_to_rsu_model_structure_list_sub
            change_after = []
            change_before = rsu_to_rsu_model_structure_list_sub[sub_task_key]
            model_policy_rsu_list = {i: [] for i in range(self.rsu_num + 1)}
            for model_policy_ in rsu_to_rsu_model_structure_list_sub[sub_task_key]:
                _, _, model_list = self.get_download_model_rsu_info(model_policy_)
                for model_structure_idx in model_list:
                    if rsu_model_queue[rsu_id][model_structure_idx] == 1:
                        continue
                    des_rsu_list = [i for i in range(rsu_id, 480, 5)]
                    model_list_key = [i for i in range(model_structure_idx, 480, 16)]
                    intersection = set(des_rsu_list).intersection(set(model_list_key))
                    intersection_list = list(intersection)
                    model_src_rsu = DRL_model.choose_action(model_env, finished=True,
                                                            intersection_list=intersection_list,
                                                            rsu_num=self.rsu_num)
                    model_policy_rsu_list[model_src_rsu].append(model_structure_idx)
            for model_policy_key in model_policy_rsu_list.keys():
                m_list = model_policy_rsu_list[model_policy_key]
                change_after_info = self.get_download_model_rsu(model_policy_key, rsu_id, m_list)
                change_after.append(change_after_info)
            rsu_to_rsu_model_structure_list_sub[sub_task_key] = change_after
            for rsu_idx in range(self.rsu_num):
                self.RSUs[rsu_idx].model_structure_list = self.RSUs[rsu_idx].initial_model_structure_list.copy()
                self.RSUs[rsu_idx].sub_task_list = []
            if self.is_satisfied_constraint(rsu_to_rsu_model_structure_list_sub, rsu_id, sub_task_key):
                return 0, rsu_to_rsu_model_structure_list_sub
            # rsu_to_rsu_model_structure_list_sub[sub_task_key] = change_before
            return 1, rsu_to_rsu_model_structure_list_sub

        num_state = (self.rsu_num + 1) * len(model_util.Sub_Model_Structure_Size)
        num_action = (self.rsu_num + 1) * len(model_util.Sub_Model_Structure_Size) * self.rsu_num
        DRL_model = DQN.DQN(num_state, num_action)
        train_base = 3.0
        train_bais = 30.0
        LOSS_model = []
        for epoch in tqdm(range(350), desc="dqn"):
            rsu_model_queue = self.generate_rsu_model_queue()
            observation = get_observation(rsu_model_queue)
            for _ in range(400):
                action_value = DRL_model.choose_action(observation)
                # if action_value == num_state - 1:
                #     # print("DRL think this state is the optimal, thus break..")
                #     DRL.store_transition(observation, action_value, 0, observation)
                #     break
                # employ action .....
                reward, rsu_model_queue = employ_action(action_value, rsu_model_queue)
                observation_ = get_observation(rsu_model_queue)
                DRL_model.store_transition(observation, action_value, reward, observation_)
                observation = observation_
            # print("objective_value: {}".format(best_optimal))
            if epoch >= train_bais and epoch % train_base == 0:
                # print("DRL is learning......")
                loss = DRL_model.learn()
                LOSS_model.append(float(loss))
        with open("loss.txt", "w+") as f:
            f.write("loss: {}\n".format(LOSS_model))
        # ------------------------------------------------------------------------------
        #                任务部署模型
        # ------------------------------------------------------------------------------
        task_model_state = self.rsu_num
        task_model_action = self.get_total_sub_num() * self.rsu_num
        task_model = DQN.DQN(task_model_state, task_model_action)
        REWARDS = []
        LOSS = []
        OPT_RESULT = []
        best_optimal = -10000
        for epoch in tqdm(range(500), desc="dqn_task"):
            rsu_to_rsu_structure = {}
            for rsu_idx in range(self.rsu_num):
                self.RSUs[rsu_idx].clear_added_model()
                self.RSUs[rsu_idx].task_list = []
                self.RSUs[rsu_idx].sub_task_list = []
            self.allocate_sub_task_initial()
            self.allocate_task_for_rsu()
            _, rsu_to_rsu_model_structure_list, observation = self.cal_objective_value(rsu_to_rsu_structure,
                                                                                       is_Initial=True, is_dqn=True)
            if -max(observation) > best_optimal:
                best_optimal = -max(observation)
            rsu_to_rsu_model_structure_list_sub = self.trans_request_to_sub_task(rsu_to_rsu_model_structure_list)
            total_reward = 0
            for _ in range(500):
                action_value = task_model.choose_action(observation)
                flag, rsu_to_rsu_model_structure_list_sub = employ_action_task(action_value,
                                                                               rsu_to_rsu_model_structure_list_sub,
                                                                               DRL_model)
                if flag == 2:
                    continue
                observation_ = get_observation_task(rsu_to_rsu_model_structure_list_sub)
                if flag == 1:
                    reward = -100000
                    task_model.store_transition(observation, action_value, reward, observation_)
                    observation = observation_
                    continue
                reward = -max(observation_)
                task_model.store_transition(observation, action_value, reward, observation_)
                if -max(observation_) > best_optimal:
                    best_optimal = -max(observation_)
                observation = observation_
            REWARDS.append(total_reward)
            OPT_RESULT.append(best_optimal)
            # print("objective_value: {}".format(best_optimal))
            if epoch >= train_bais and epoch % train_base == 0:
                # print("DRL is learning......")
                loss = task_model.learn()
                LOSS.append(float(loss))
            if epoch % 50 == 0:
                # print("\nepoch: {}, objective_value: {}".format(epoch, best_optimal))
                pass
        # plt.plot(LOSS)
        # plt.title("loss curve......")
        # plt.show()
        # plt.plot(OPT_RESULT)
        # plt.title("best_optimal")
        # plt.ylabel("objective, minimal is better.")
        # plt.show()
        with open("loss.txt", "w+") as f:
            f.write("reward: {}\n".format(REWARDS))
            f.write("loss: {}\n".format(LOSS))
        return best_optimal

    def is_satisfied_constraint(self, rsu_to_rsu_model_structure_list_sub, rsu_id, sub_task_key):
        for task_id in rsu_to_rsu_model_structure_list_sub.keys():
            if task_id == sub_task_key:
                continue
            policy_list = rsu_to_rsu_model_structure_list_sub[task_id]
            _, des_rsu, _ = self.get_download_model_rsu_info(policy_list[0])
            if des_rsu == rsu_id:
                sub_key = task_id.split("-")
                job_id = int(sub_key[0])
                sub_task_id = int(sub_key[1])
                sub_task = self.task_list[job_id][sub_task_id]
                self.RSUs[rsu_id].sub_task_list.append(sub_task)
                for policy in policy_list:
                    _, _, model_added_list = self.get_download_model_rsu_info(policy)
                    self.RSUs[rsu_id].add_model_structure(model_added_list)
                self.RSUs[rsu_id].sub_task_list.append(sub_task)
        policy_cons = rsu_to_rsu_model_structure_list_sub[sub_task_key]
        sub_key_ = sub_task_key.split("-")
        job_id_ = int(sub_key_[0])
        sub_task_id_ = int(sub_key_[1])
        sub_task_ = self.task_list[job_id_][sub_task_id_]
        model_idx_ = sub_task_["model_idx"]
        sub_model_idx_ = sub_task_["sub_model_idx"]
        generated_id_ = sub_task_["rsu_id"]
        seq_num = sub_task_["seq_num"]
        model_ = model_util.get_model(model_idx_)
        task_size = model_.single_task_size
        device_id = self.RSUs[rsu_id].device_idx
        exec_time = self.RSUs[rsu_id].latency_list[model_idx_][sub_model_idx_][device_id][seq_num]
        latency_requirement = sub_task_["latency"]
        if rsu_id == generated_id_:
            offload_time = 0
        else:
            offload_time = task_size / self.RSUs[generated_id_].rsu_rate
        download_time = 0
        for policy_ in policy_cons:
            src_rsu_, _, model_list_ = self.get_download_model_rsu_info(policy_)
            model_list_added = []
            if self.RSUs[rsu_id].satisfy_add_task_constraint(sub_task_, is_Request=False):
                pass
            else:
                # print("task不满足")
                return False
            for model_structure_idx in model_list_:
                if model_structure_idx not in self.RSUs[rsu_id].initial_model_structure_list:
                    model_list_added.append(model_structure_idx)
            if self.RSUs[rsu_id].satisfy_add_model_structure_constraint(model_list_added, is_Request=False):
                pass
            else:
                # print("model不满足")
                return False
            if len(model_list_) == 0:
                download_time += 0
            else:
                model_size = model_util.get_model_sturctures_size(model_list_added)
                download_time += model_size / (self.RSUs[src_rsu_].rsu_rate if src_rsu_ != self.rsu_num else
                                               self.RSUs[rsu_id].download_rate)
        if download_time + offload_time + exec_time > latency_requirement:
            # time = download_time + offload_time + exec_time
            print("latency不满足")
            return False
        return True

    def generate_rsu_model_queue(self):
        rsu_model_queue = [[0 for _ in range(len(model_util.Sub_Model_Structure_Size))] for _ in
                           range(self.rsu_num + 1)]
        for rsu_idx in range(self.rsu_num):
            for model_structure_idx in self.RSUs[rsu_idx].initial_model_structure_list:
                rsu_model_queue[rsu_idx][model_structure_idx] = 1
        for model_structure_idxs in range(len(model_util.Sub_Model_Structure_Size)):
            rsu_model_queue[self.rsu_num][model_structure_idxs] = 1
        return rsu_model_queue

    def allocate_sub_task_initial(self):
        for rsu_idx in range(self.rsu_num):
            self.RSUs[rsu_idx].sub_task_list = []
        for sub_task in self.sub_task_list:
            rsu_id = sub_task["rsu_id"]
            self.RSUs[rsu_id].sub_task_list.append(sub_task)

    # ------------------------------------------------------------------------------
    #                DQN_no_shared algorithm
    # ------------------------------------------------------------------------------
    def dqn_(self):

        def employ_action(action_value, rsu_model_queue):
            action_value = int(action_value)
            src_rsu_id = int(action_value / (self.rsu_num * len(model_util.Sub_Model_Structure_Size)))
            des_rsu_id = int(action_value % self.rsu_num)
            model_id = int(action_value % len(model_util.Sub_Model_Structure_Size))
            des_rsu_model_size = 0
            for model_structure_idx in range(len(model_util.Sub_Model_Structure_Size)):
                if rsu_model_queue[des_rsu_id][model_structure_idx] == 1:
                    des_rsu_model_size += model_util.Sub_Model_Structure_Size[model_structure_idx]
            if src_rsu_id == des_rsu_id:
                download_time = -10000
                return download_time, rsu_model_queue
            if src_rsu_id != self.rsu_num:
                if model_id not in self.RSUs[src_rsu_id].initial_model_structure_list or \
                        model_id in self.RSUs[des_rsu_id].initial_model_structure_list:
                    download_time = -10000
                    return download_time, rsu_model_queue
                else:
                    if des_rsu_model_size + model_util.Sub_Model_Structure_Size[model_id] > \
                            self.RSUs[des_rsu_id].storage_capacity:
                        download_time = -10000
                    else:
                        rsu_model_queue[des_rsu_id][model_id] = 1
                        download_time = -(
                                model_util.Sub_Model_Structure_Size[model_id] / self.RSUs[src_rsu_id].rsu_rate)
                    return download_time, rsu_model_queue
            else:
                if model_id in self.RSUs[des_rsu_id].initial_model_structure_list:
                    download_time = -10000
                else:
                    if self.RSUs[des_rsu_id].get_total_model_size() + model_util.Sub_Model_Structure_Size[model_id] > \
                            self.RSUs[des_rsu_id].storage_capacity:
                        download_time = -10000
                    else:
                        rsu_model_queue[des_rsu_id][model_id] = 1
                        download_time = -(
                                model_util.Sub_Model_Structure_Size[model_id] / self.RSUs[des_rsu_id].download_rate)
                return download_time, rsu_model_queue

        def get_observation(rsu_model_queue) -> list:
            obs = [element for sublist in rsu_model_queue for element in sublist]
            return obs

        def get_observation_task(rsu_to_rsu_model_structure_list) -> list:
            _, observations = self.cal_objective_value(rsu_to_rsu_model_structure_list, is_dqn=True)
            return observations

        def employ_action_task(action_value, rsu_to_rsu_model_structure_list_sub, DRL_model):
            # 更新策略
            # 0: 完成修改
            # 1: 不满足约束
            # 2: 不需要修改
            rsu_model_queue = self.generate_rsu_model_queue()
            # 根据model迁移情况判断每个rsu的model存储情况
            for key in rsu_to_rsu_model_structure_list_sub.keys():
                for model_policy_ in rsu_to_rsu_model_structure_list_sub[key]:
                    src_rsu_, des_rsu_, model_list_ = self.get_download_model_rsu_info(model_policy_)
                    for model_idx in model_list_:
                        rsu_model_queue[des_rsu_][model_idx] = 1
            model_env = get_observation(rsu_model_queue)
            task_num = len(self.sub_task_list)
            action_value = int(action_value)
            rsu_id = int(action_value / task_num)
            task_id = action_value % task_num
            task = self.sub_task_list[task_id]
            request_id = task['job_id']
            sub_task_id = task["sub_model_idx"]
            sub_task_key = "{}-{}".format(request_id, sub_task_id)
            model_policy = rsu_to_rsu_model_structure_list_sub[sub_task_key][0]
            src_rsu, des_rsu, _ = self.get_download_model_rsu_info(model_policy)
            if des_rsu == rsu_id:
                return 2, rsu_to_rsu_model_structure_list_sub
            change_after = []
            rsu_to_rsu_model_structure_list_sub[sub_task_key] = change_after
            for rsu_idx in range(self.rsu_num):
                self.RSUs[rsu_idx].model_structure_list = self.RSUs[rsu_idx].initial_model_structure_list.copy()
                self.RSUs[rsu_idx].sub_task_list = []
            if self.is_satisfied_constraint(rsu_to_rsu_model_structure_list_sub, rsu_id, sub_task_key):
                return 0, rsu_to_rsu_model_structure_list_sub
            # rsu_to_rsu_model_structure_list_sub[sub_task_key] = change_before
            return 1, rsu_to_rsu_model_structure_list_sub
        # ------------------------------------------------------------------------------
        #                任务部署模型
        # ------------------------------------------------------------------------------
        task_model_state = (self.rsu_num + 1) * len(model_util.Sub_Model_Structure_Size) * (len(self.sub_task_list)) * \
                    self.rsu_num
        task_model_action = (self.rsu_num + 1) * len(model_util.Sub_Model_Structure_Size) * self.rsu_num * self.rsu_num * \
                     (len(self.sub_task_list))
        task_model = DQN.DQN(task_model_state, task_model_action)
        REWARDS = []
        LOSS = []
        OPT_RESULT = []
        best_optimal = -10000
        train_base = 3.0
        train_bais = 30.0
        for epoch in tqdm(range(500), desc="dqn_task"):
            rsu_to_rsu_structure = {}
            for rsu_idx in range(self.rsu_num):
                self.RSUs[rsu_idx].clear_added_model()
                self.RSUs[rsu_idx].task_list = []
                self.RSUs[rsu_idx].sub_task_list = []
            self.allocate_sub_task_initial()
            self.allocate_task_for_rsu()
            _, rsu_to_rsu_model_structure_list, observation = self.cal_objective_value(rsu_to_rsu_structure,
                                                                                       is_Initial=True, is_dqn=True)
            if -max(observation) > best_optimal:
                best_optimal = -max(observation)
            rsu_to_rsu_model_structure_list_sub = self.trans_request_to_sub_task(rsu_to_rsu_model_structure_list)
            for _ in range(500):
                action_value = task_model.choose_action(observation)
                flag, rsu_to_rsu_model_structure_list_sub = employ_action_task(action_value,
                                                                               rsu_to_rsu_model_structure_list_sub)
                if flag == 2:
                    continue
                observation_ = get_observation_task(rsu_to_rsu_model_structure_list_sub)
                if flag == 1:
                    reward = -100000
                    task_model.store_transition(observation, action_value, reward, observation_)
                    observation = observation_
                    continue
                reward = -max(observation_)
                task_model.store_transition(observation, action_value, reward, observation_)
                if -max(observation_) > best_optimal:
                    best_optimal = -max(observation_)
                observation = observation_
            OPT_RESULT.append(best_optimal)
            # print("objective_value: {}".format(best_optimal))
            if epoch >= train_bais and epoch % train_base == 0:
                # print("DRL is learning......")
                loss = task_model.learn()
                LOSS.append(float(loss))
            if epoch % 50 == 0:
                # print("\nepoch: {}, objective_value: {}".format(epoch, best_optimal))
                pass
        # plt.plot(LOSS)
        # plt.title("loss curve......")
        # plt.show()
        # plt.plot(OPT_RESULT)
        # plt.title("best_optimal")
        # plt.ylabel("objective, minimal is better.")
        # plt.show()
        with open("loss.txt", "w+") as f:
            f.write("reward: {}\n".format(REWARDS))
            f.write("loss: {}\n".format(LOSS))
        return best_optimal

    # ------------------------------------------------------------------------------
    #                Coalition algorithm
    # ------------------------------------------------------------------------------

    def preference_coalition(self, shared=False):
        for rsu_idx in range(self.rsu_num):
            self.RSUs[rsu_idx].sub_task_list = []
            self.RSUs[rsu_idx].task_list = []
            self.RSUs[rsu_idx].model_structure_list = self.RSUs[rsu_idx].initial_model_structure_list.copy()
        self.allocate_task_for_rsu()
        rsu_to_rsu_structure = {}  # xx-xx:xx-xx-[...]
        _, rsu_to_rsu_model_structure_list = self.cal_objective_value(rsu_to_rsu_structure, is_Initial=True)
        rsu_to_rsu_model_structure_list = self.trans_request_to_sub_task(rsu_to_rsu_model_structure_list)
        self.allocate_sub_task_for_rsu()
        utility = self.cal_objective_value(rsu_to_rsu_model_structure_list, is_Shared=shared, is_Request=False)
        for task_key in rsu_to_rsu_model_structure_list.keys():
            info = task_key.split("-")
            job_id = int(info[0])
            sub_model_idx = int(info[1])
            task = self.task_list[job_id][sub_model_idx]
            latency_requirement = task["latency"]
            _, rsu_idx_task, _ = self.get_download_model_rsu_info(rsu_to_rsu_model_structure_list[task_key][0])
            for rsu_idx in range(self.rsu_num):
                if rsu_idx_task == rsu_idx:
                    continue
                not_added_model_structure = self.RSUs[rsu_idx].has_model_structure(task["model_structure"])
                not_added_model_structure_initial = self.RSUs[rsu_idx].has_model_structure_initial(
                    task["model_structure"])
                generated_id = task["rsu_id"]
                model_idx = task["model_idx"]
                model = model_util.get_model(model_idx)
                task_size = model.single_task_size
                offload_time = 0
                download_time = 0
                device_id = self.RSUs[rsu_idx].device_idx
                sub_model_idx = task["sub_model_idx"]
                seq_num = task["seq_num"]
                exec_time = self.RSUs[rsu_idx].latency_list[model_idx][sub_model_idx][device_id][seq_num]
                if generated_id != rsu_idx:
                    offload_time = task_size / self.RSUs[generated_id].rsu_rate
                for model_structure_idx in not_added_model_structure_initial:  # 通过greeedy方式获取应从哪些rsu下载model
                    if self.model_download_time_list.get(model_structure_idx) is not None:
                        if self.RSUs[rsu_idx].download_rate > self.RSUs[
                            self.model_download_time_list[model_structure_idx]].rsu_rate:
                            download_time += model_util.Sub_Model_Structure_Size[model_structure_idx] / \
                                             self.RSUs[rsu_idx].download_rate
                        else:
                            download_time += model_util.Sub_Model_Structure_Size[model_structure_idx] / \
                                             self.RSUs[self.model_download_time_list[model_structure_idx]].rsu_rate
                if exec_time + download_time + offload_time <= latency_requirement:
                    if self.RSUs[rsu_idx].satisfy_add_task_constraint(task, is_Request=False):
                        self.RSUs[rsu_idx].sub_task_list.append(task)
                        if self.RSUs[rsu_idx].satisfy_add_model_structure_constraint(not_added_model_structure,
                                                                                     is_Request=False):
                            self.RSUs[rsu_idx_task].sub_task_list.remove(task)
                        else:
                            self.RSUs[rsu_idx].sub_task_list.remove(task)
                            continue
                    else:
                        continue
                else:
                    continue
                if len(not_added_model_structure_initial) != 0:
                    download_model_rsu_info_list = []
                    download_model_rsu_list = {}
                    download_model_rsu_info_list_before = []
                    for rsu_idxs in range(self.rsu_num):
                        download_model_rsu_list[rsu_idxs] = set()
                    for model_structure_idx in not_added_model_structure_initial:  # 通过greeedy方式获取应从哪些rsu下载model
                        if self.model_download_time_list.get(model_structure_idx) is not None:
                            if self.RSUs[rsu_idx].download_rate > self.RSUs[
                                self.model_download_time_list[model_structure_idx]].rsu_rate:
                                if download_model_rsu_list.get(self.cloudidx) is None:
                                    download_model_rsu_list[self.cloudidx] = set()
                                download_model_rsu_list[self.cloudidx].add(model_structure_idx)
                            else:
                                if download_model_rsu_list.get(
                                        self.model_download_time_list[model_structure_idx]) is None:
                                    download_model_rsu_list[
                                        self.model_download_time_list[model_structure_idx]] = set()
                                download_model_rsu_list[self.model_download_time_list[model_structure_idx]].add(
                                    model_structure_idx)
                        else:
                            if download_model_rsu_list.get(self.cloudidx) is None:
                                download_model_rsu_list[self.cloudidx] = set()
                            download_model_rsu_list[self.cloudidx].add(model_structure_idx)
                    for download_rsu_idxs in download_model_rsu_list.keys():
                        download_model_rsu_info = self.get_download_model_rsu(download_rsu_idxs, rsu_idx, list(
                            download_model_rsu_list[download_rsu_idxs]))
                        download_model_rsu_info_list.append(download_model_rsu_info)
                    download_model_rsu_info_list_before = rsu_to_rsu_model_structure_list[task_key]
                    rsu_to_rsu_model_structure_list[task_key] = download_model_rsu_info_list
                else:
                    download_model_rsu_info_list_before = rsu_to_rsu_model_structure_list[task_key]
                    download_model_rsu_info = self.get_download_model_rsu(rsu_idx, rsu_idx, [])
                    rsu_to_rsu_model_structure_list[task_key] = []
                    rsu_to_rsu_model_structure_list[task_key].append(download_model_rsu_info)
                tmp_utility = self.cal_objective_value(rsu_to_rsu_model_structure_list, is_Initial=False,
                                                       is_Request=False, is_Shared=shared)
                if tmp_utility < utility:
                    utility = tmp_utility
                    self.RSUs[rsu_idx].add_model_structure(task["model_structure"])
                    removed_model_list = set()
                    for removed_model_off in download_model_rsu_info_list_before:
                        _, _, removed_models = self.get_download_model_rsu_info(removed_model_off)
                        for removed_model_idx in removed_models:
                            if removed_model_idx not in self.RSUs[rsu_idx_task].initial_model_structure_list:
                                removed_model_list.add(removed_model_idx)
                    for task_ in self.RSUs[rsu_idx_task].sub_task_list:
                        sub_task_key_ = "{}-{}".format(task_["job_id"], task_["sub_model_idx"])
                        for model_off in rsu_to_rsu_model_structure_list[sub_task_key_]:
                            _, _, download_model_ = self.get_download_model_rsu_info(model_off)
                            set_download_model_ = set(download_model_)
                            inter_model = set_download_model_.intersection(removed_model_list)
                            if len(inter_model) != 0:
                                for removed_model_idxs in inter_model:
                                    removed_model_list.remove(removed_model_idxs)
                    self.RSUs[rsu_idx_task].remove_model_structure(removed_model_list)
                    rsu_idx_task = rsu_idx
                else:
                    self.RSUs[rsu_idx].sub_task_list.remove(task)
                    self.RSUs[rsu_idx_task].sub_task_list.append(task)
                    rsu_to_rsu_model_structure_list[task_key] = download_model_rsu_info_list_before
        return -utility
