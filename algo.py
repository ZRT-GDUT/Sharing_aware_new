import random
from typing import List

from pulp import LpStatusInfeasible

import device
import model_util
import pulp as pl

class Algo:
    def __init__(self, RSUs: List[device.RSU], task_list, model_download_time_list):
        self.RSUs = RSUs
        self.rsu_num = len(RSUs)
        self.task_list = task_list
        self.allocate_task_for_rsu()
        self.cloudidx = self.rsu_num
        self.model_download_time_list = model_download_time_list

    def get_all_task_num(self):
        task_num = 0
        for rsu_idx in range(self.rsu_num):
            task_num = len(self.RSUs[rsu_idx].task_list) + task_num
        return task_num

    def MA(self, task_list, min_gap=0.1):
        rsu_to_rsu_structure = {}  # xx-xx:xx-xx-[...]
        # for job_id in len(task_list):
        #     for sub_task in task_list[job_id]:
        #         rsu_to_rsu_structure[{} - {}.format(job_id, sub_task["sub_model_idx"])] = []
        rsu_task_queue = self.generate_rsu_request_queue()
        T_max, rsu_to_rsu_model_structure_list = self.cal_objective_value(rsu_to_rsu_structur e, rsu_task_queue,
                                                                          is_Initial=True)
        T = T_max
        print("T_max:", T_max)
        T_min = 0
        obj = T_max
        while T_max - T_min >= min_gap:
            throughput, objective_value = self.ma(rsu_to_rsu_structure, rsu_task_queue, T_max)
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

    def generate_rsu_request_queue(self):
        rsu_request_queue = [[] for _ in range(self.rsu_num)]
        for rsu_idx in range(self.rsu_num):
            for task in self.RSUs[rsu_idx].task_list:
                rsu_request_queue[rsu_idx].append(task)
        return rsu_request_queue

    def generate_new_position_request(self, task, rsu_id, rsu_to_rsu_model_structure_list, rsu_task_queue, T_max,
                                      is_Shared=True):
        obj_value = self.cal_objective_value(rsu_to_rsu_model_structure_list, rsu_task_queue, is_Request=True)
        rsu_task_queue[rsu_id].remove(task)
        rsu_idx_task = rsu_id
        job_id = task[0]["job_id"]
        task_model_structure_list = set()
        for sub_task in task:
            for model_structure_idx in sub_task["model_structure"]:
                task_model_structure_list.add(model_structure_idx)
        for rsu_idx in range(self.rsu_num):  # 遍历task在每个rsu上部署的情况
            not_added_model_structure = self.RSUs[rsu_idx].has_model_structure(task_model_structure_list)
            if self.RSUs[rsu_idx].satisfy_add_task_constraint(task) and \
                    self.RSUs[rsu_idx].satisfy_add_model_structure_constraint(not_added_model_structure):
                rsu_task_queue[rsu_idx].append(task)
            else:
                continue
            if len(not_added_model_structure) != 0:
                download_model_rsu_info_list = []
                download_model_rsu_list = {}
                download_model_rsu_info_list_before = []
                for rsu_idx in range(self.rsu_num):
                    download_model_rsu_list[rsu_idx] = set()
                for model_structure_idx in not_added_model_structure:  # 通过greeedy方式获取应从哪些rsu下载model
                    if self.RSUs[rsu_idx].download_rate > self.RSUs[self.model_download_time_list[model_structure_idx]].rsu_rate:
                        download_model_rsu_list[self.cloudidx].add(model_structure_idx)
                    else:
                        download_model_rsu_list[self.model_download_time_list[model_structure_idx]].add(
                            model_structure_idx)
                for download_rsu_idxs in download_model_rsu_list.keys():
                    download_model_rsu_info = self.get_download_model_rsu(download_rsu_idxs, rsu_idx, list(
                        download_model_rsu_list[download_rsu_idxs]))
                    download_model_rsu_info_list.append(download_model_rsu_info)
                download_model_rsu_info_list_before = rsu_to_rsu_model_structure_list[job_id]
                rsu_to_rsu_model_structure_list[job_id] = download_model_rsu_info_list
            obj_value_new = self.cal_objective_value(rsu_to_rsu_model_structure_list, rsu_task_queue,
                                                     is_Initial=False, is_Request=True)
            if obj_value_new < obj_value and obj_value_new < T_max:
                self.RSUs[rsu_idx].add_task(task)
                self.RSUs[rsu_idx_task].remove_task(task)
                added_models = self.RSUs[rsu_idx_task].has_model_structure(task_model_structure_list)
                self.RSUs[rsu_idx].model_structure_list.add(added_models)
                removed_model_list = list(task_model_structure_list)
                for task_ in self.RSUs[rsu_idx_task].task_list:
                    for model_off in rsu_to_rsu_model_structure_list[task_]:

                obj_value = obj_value_new
                rsu_idx_task = rsu_idx
            else:
                rsu_task_queue[rsu_idx].remove(task)
                rsu_to_rsu_model_structure_list[job_id] = download_model_rsu_info_list_before
        if rsu_id != rsu_idx_task:
            added_models = self.RSUs[rsu_idx_task].has_model_structure(task_model_structure_list)
            self.RSUs[rsu_idx_task].model_structure_list.add(added_models)
            self.RSUs[rsu_idx_task].add_task(task)
            self.RSUs[rsu_id].remove_task(task)
            for task_ in self.RSUs[rsu_idx_task].task_list:


        return rsu_idx_task, rsu_to_rsu_model_structure_list, rsu_task_queue

    def generate_new_position_sub_task(self, task, rsu_id, rsu_to_rsu_model_structure_list, rsu_task_queue, T_max,
                                       is_Shared=True):
        obj_value = self.cal_objective_value(rsu_to_rsu_model_structure_list, rsu_task_queue, is_Request=False)
        rsu_idx_task = rsu_id
        job_id = task["job_id"]
        sub_task_id = task["sub_task_id"]
        for rsu_idx in range(self.rsu_num):  # 遍历task在每个rsu上部署的情况
            self.RSUs[rsu_idx].task_list = rsu_task_queue[rsu_idx]
            task_model_structure_list = set()
            not_added_model_structure = self.RSUs[rsu_idx].has_model_structure(task["model_structure"])
            if self.RSUs[rsu_idx].satisfy_add_task_constraint(task) and \
                    self.RSUs[rsu_idx].satisfy_add_model_structure_constraint(not_added_model_structure):
                rsu_task_queue[rsu_idx].append(task)
            else:
                continue
            if len(not_added_model_structure) != 0:
                download_model_rsu_info_list = []
                download_model_rsu_list = {}
                download_model_rsu_info_list_before = []
                for rsu_idxs in range(self.rsu_num):
                    download_model_rsu_list[rsu_idxs] = set()
                for model_structure_idx in not_added_model_structure:  # 通过greeedy方式获取应从哪些rsu下载model
                    if self.RSUs[rsu_idx].download_rate > self.RSUs[
                        self.model_download_time_list[model_structure_idx]].rsu_rate:
                        download_model_rsu_list[self.cloudidx].add(model_structure_idx)
                    else:
                        download_model_rsu_list[self.model_download_time_list[model_structure_idx]].add(
                            model_structure_idx)
                for download_rsu_idxs in download_model_rsu_list.keys():
                    download_model_rsu_info = self.get_download_model_rsu(download_rsu_idxs, rsu_idx, list(
                        download_model_rsu_list[download_rsu_idxs]))
                    download_model_rsu_info_list.append(download_model_rsu_info)
                download_model_rsu_info_list_before = rsu_to_rsu_model_structure_list[{}-{}.format(job_id, sub_task_id)]
                rsu_to_rsu_model_structure_list[{} - {}.format(job_id, sub_task_id)] = download_model_rsu_info_list
            obj_value_new = self.cal_objective_value(rsu_to_rsu_model_structure_list, rsu_task_queue,
                                                     is_Initial=False, is_Request=False)
            if obj_value_new < obj_value and obj_value_new < T_max:
                obj_value = obj_value_new
                rsu_idx_task = rsu_idx
            else:
                rsu_task_queue[rsu_idx].remove(task)
                rsu_to_rsu_model_structure_list[{}-{}.format(job_id, sub_task_id)] = download_model_rsu_info_list_before
        if rsu_id != rsu_idx_task:
            added_models = self.RSUs[rsu_idx_task].has_model_structure(task["model_structure"])
            for info in rsu_to_rsu_model_structure_list.keys():
                _, info_rsu, _ = self.ge
            self.RSUs[rsu_idx_task].model_structure_list.add(added_models)
            self.RSUs[rsu_idx_task].add_task(task)
            self.RSUs[rsu_id].remove_task(task)
        return rsu_idx_task, rsu_to_rsu_model_structure_list, rsu_task_queue

    def get_download_model_rsu(self, download_rsu_idx, rsu_idx, model_structure_list):
        return "{}-{}-{}".format(download_rsu_idx, rsu_idx, model_structure_list)

    def get_download_model_rsu_info(self, download_model_rsu_info):
        info = download_model_rsu_info.split("-")
        return int(info[0]), int(info[1]), int(info[2])

    def ma(self, rsu_to_rsu_model_structure_list, rsu_task_queue, T_max):
        changed_sub_task, changed_request = True
        while changed_request:
            changed_request = False
            for rsu_idx in range(self.rsu_num):  # 先遍历请求
                task_list = rsu_task_queue[rsu_idx]
                for task in task_list:
                    rsu_task_queue[rsu_idx].remove(task)
                    new_position, rsu_to_rsu_model_structure_list, rsu_task_queue = \
                        self.generate_new_position_request(task, rsu_idx, rsu_to_rsu_model_structure_list,
                                                           rsu_task_queue, T_max)
                    if new_position != rsu_idx:
                        changed_request = True
                    self.allocate_task_for_rsu_later(rsu_task_queue)
        rsu_to_rsu_model_structure_list_sub_task = self.trans_request_to_sub_task(rsu_to_rsu_model_structure_list)
        while changed_sub_task:
            changed_sub_task = False
            for rsu_idxs in range(self.rsu_num):  # 遍历子任务
                task_lists = rsu_task_queue[rsu_idxs]
                for task in task_lists:
                    for sub_task in task:
                        new_position, rsu_to_rsu_model_structure_list, rsu_task_queue = \
                            self.generate_new_position_sub_task(sub_task, rsu_idxs,
                                                                rsu_to_rsu_model_structure_list_sub_task,
                                                                rsu_task_queue, T_max)
                        if new_position != rsu_idxs:
                            changed_sub_task = True

    def trans_request_to_sub_task(self, rsu_to_rsu_model_structure_list):
        rsu_to_rsu_model_structure_list_sub_task = {}
        for job_id in rsu_to_rsu_model_structure_list.keys():
            for sub_task in self.task_list[job_id]:
                sub_task_id = sub_task["sub_model_idx"]
                rsu_to_rsu_model_structure_list_sub_task[{} - {}.format(job_id, sub_task_id)] = \
                    rsu_to_rsu_model_structure_list[job_id]
        return rsu_to_rsu_model_structure_list_sub_task

    def get_all_task_num_all(self):
        task_num = 0
        for i in range(len(self.task_list)):
            task_num = len(self.task_list[i]) + task_num
        return task_num

    def cal_objective_value(self, rsu_to_rsu_model_structure_list, rsu_task_queue, is_Initial=False, is_Shared=True,
                            is_Request=False):
        obj = []
        for rsu_idx in range(self.rsu_num):
            if is_Initial:
                obj_single, rsu_to_rsu_model_structure_list = self.cal_single_rsu_obj_initial(
                    rsu_to_rsu_model_structure_list, rsu_task_queue[rsu_idx], rsu_idx, is_Shared)
                obj.append(obj_single)
            else:
                obj.append(
                    self.cal_single_rsu_obj(rsu_to_rsu_model_structure_list, rsu_idx, is_Shared, is_Request))
        if is_Initial:
            return max(obj), rsu_to_rsu_model_structure_list
        else:
            return max(obj)

    def cal_single_rsu_obj_initial(self, rsu_download_model, rsu_task_queue, rsu_idx, is_Shared=True):  # 还没有判断存储空间
        device_id = self.RSUs[rsu_idx].device_idx
        task_exec_time = 0
        download_time = 0
        for task in rsu_task_queue:
            sub_task_exec_time_list = []
            for sub_task in task:
                task_model_structure_list = set()
                for model_structure_idx in sub_task["model_structure"]:
                    task_model_structure_list.add(model_structure_idx)
                sub_task_exec_time = self.RSUs[rsu_idx].latency_list[device_id][sub_task["model_idx"]][
                    sub_task["sub_model_idx"]][sub_task["seq_num"]]
                sub_task_exec_time_list.append(sub_task_exec_time)
                if is_Shared:
                    task_exec_time += max(sub_task_exec_time_list)
                else:
                    task_exec_time += sum(sub_task_exec_time)
            not_added_model_structure = self.RSUs[rsu_idx].has_model_structure(task_model_structure_list)
            if len(not_added_model_structure) != 0:
                if is_Shared:
                    task_model_size = model_util.get_model_sturctures_size(not_added_model_structure)
                else:
                    task_model_size = model_util.get_model_sturctures_size(task_model_structure_list)
                if self.RSUs[rsu_idx].satisfy_add_model_structure_constraint(not_added_model_structure):
                    download_time += task_model_size / self.RSUs[rsu_idx].download_rate
                    self.RSUs[rsu_idx].add_model_structure(not_added_model_structure)
                else:
                    download_time += 999999
            else:
                download_time += 0
            if is_Shared:
                download_model_rsu = self.get_download_model_rsu(self.cloudidx, task[0]["rsu_id"],
                                                                 not_added_model_structure)
            else:
                download_model_rsu = self.get_download_model_rsu(self.cloudidx, task[0]["rsu_id"],
                                                                 task_model_structure_list)
            if len(rsu_download_model[task[0]["job_id"]]) == 0
                rsu_download_model[task[0]["job_id"]] = []
            rsu_download_model[task[0]["job_id"]].append(download_model_rsu)
        singl_obj_value = task_exec_time + download_time
        return singl_obj_value, rsu_download_model

    def cal_single_rsu_obj(self, rsu_download_model, rsu_idx, is_Shared=True, is_Request=False):  # 还没有判断存储空间
        device_id = self.RSUs[rsu_idx].device_idx
        task_exec_time_list = {0: [], 1: [], 2: []}
        download_time_list = []
        download_time = 0
        trans_time = 0
        if is_Request:
            for job_id in rsu_download_model.keys():
                generated_id = self.task_list[job_id][0]["rsu_id"]
                model_idx = self.task_list[job_id][0]["model_idx"]
                value = rsu_download_model[job_id][0]
                _, task_rsu_idx, _ = self.get_download_model_rsu_info(value)
                if task_rsu_idx == rsu_idx:
                    if task_rsu_idx == generated_id:
                        trans_time += 0
                    else:
                        sub_task_size = model_util.get_model(model_idx).single_task_size
                        trans_time_current = sub_task_size / self.RSUs[generated_id].rsu_rate
                        trans_time += trans_time_current
                    for sub_task in self.task_list[job_id]:
                        sub_task_exectime = self.RSUs[rsu_idx].latency_list[device_id][model_idx][sub_task][
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
                                download_model_size += model_util.Sub_Model_Structure_Size[model_structure_idx]
                            download_time_current = download_model_size / (self.RSUs[download_rsu_idx].rsu_rate if
                                                                           download_rsu_idx != self.cloudidx else
                                                                           self.RSUs[
                                                                               task_rsu_idx].download_rate)
                        download_time_list.append(download_time_current)
                    download_time += max(download_time_list)
        else:
            for sub_task_id in rsu_download_model.keys():
                job_id, sub_task = sub_task_id.split("-")
                generated_id = self.task_list[job_id][sub_task]["rsu_id"]
                model_idx = self.task_list[job_id][sub_task]["model_idx"]
                value = rsu_download_model[sub_task_id][0]
                _, task_rsu_idx, _ = self.get_download_model_rsu_info(value)
                if task_rsu_idx == rsu_idx:
                    if task_rsu_idx == generated_id:
                        trans_time += 0
                    else:
                        sub_task_size = model_util.get_model(model_idx).single_task_size
                        trans_time_current = sub_task_size / self.RSUs[generated_id].rsu_rate
                        trans_time += trans_time_current
                    sub_task_exectime = self.RSUs[rsu_idx].latency_list[device_id][model_idx][sub_task]
                    task_exec_time_list[model_idx].append(sub_task_exectime)
                    for download_info in rsu_download_model[sub_task_id]:
                        download_rsu_idx, task_rsu_idx, download_models = self.get_download_model_rsu_info(
                            download_info)
                        if len(download_models) == 0:
                            download_time_current = 0
                        else:
                            download_model_size = 0
                            for model_structure_idx in download_models:
                                download_model_size += model_util.Sub_Model_Structure_Size[model_structure_idx]
                            download_time_current = download_model_size / (self.RSUs[download_rsu_idx].rsu_rate if
                                                                           download_rsu_idx != self.cloudidx else
                                                                           self.RSUs[
                                                                               task_rsu_idx].download_rate)
                        download_time_list.append(download_time_current)
                    download_time += max(download_time_list)
        task_exec_time = 0
        for model_idxs in task_exec_time_list.keys():
            if is_Shared:
                task_exec_time += max(task_exec_time_list[model_idxs])
            else:
                task_exec_time += sum(task_exec_time_list[model_idxs])
        obj_value = trans_time + download_time + task_exec_time
        return obj_value

    def allocate_model_for_rsu_later(self, rsu_model_list):
        pass

    def allocate_task_for_rsu_later(self, rsu_task_queue):
        for rsu_idx in range(len(rsu_task_queue)):
            self.RSUs[rsu_idx].task_list = []
            self.RSUs[rsu_idx].task_list = rsu_task_queue[rsu_idx]

    def clear_rsu_task_list(self):
        for rsu_idx in range(self.rsu_num):
            self.RSUs[rsu_idx].task_list = []

    def allocate_task_for_rsu(self):
        for task in self.task_list:
            rsu_id = task[0]["rsu_id"]
            self.RSUs[rsu_id].task_list.append(task)

            #
