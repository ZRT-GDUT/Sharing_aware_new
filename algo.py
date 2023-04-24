from typing import List

import device
import model_util
import pulp as pl


class Algo:
    def __init__(self, RSUs: List[device.RSU]):
        self.RSUs = RSUs
        self.rsu_num = len(RSUs)

    def get_all_task_num(self):
        task_num = 0
        for rsu_idx in range(self.rsu_num):
            task_num = len(self.RSUs[rsu_idx].task_list) + task_num
        return task_num

    def iarr(self, task_list, min_gap=0.1):
        self.allocate_task_for_rsu(task_list)
        T_max = self.cal_max_response_time()
        T = T_max
        print("T_max:", T_max)
        T_min = 0
        obj = T_max
        while T_max - T_min <= min_gap:
            throughput, objective_value = self.arr(T_max)
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

    def find_no_decide_variable(self):
        for rsu_idx in self.rsu_num:
            models = self.RSUs[rsu_idx].get_cached_model()
            if len(models) == 0:
                continue
            else:
                rand_rsu_idx = rsu_idx
                rand_sub_model_idx = set()
                rand_model_structure_idx = set()
                for model in models:
                    model_idx, sub_model_idx = model_util.get_model_info(model)
                    model_object = model_util.get_model(model_idx)
                    rand_model_idx = model_idx
                    rand_sub_model_idx.add(sub_model_idx)
                    for model_structure_idx in range(model_object.require_sub_model_all[sub_model_idx]):
                        rand_model_structure_idx.add(model_structure_idx)
        return rand_rsu_idx, rand_model_idx, rand_sub_model_idx, rand_model_structure_idx

    def lin_pro(self):
        rand_rsu_id, rand_model_idx, rand_sub_model_idx, rand_model_structure_idx = self.find_no_decide_variable()
        max_system_throughput = pl.LpProblem("max_system_throughput", sense=pl.LpMaximize)  # 定义最大化吞吐率问题
        x_i_e = {(i, m, s): pl.LpVariable('x_i_e_{0}_{1}_{2}'.format(i, m, s), lowBound=0, upBound=1,
                                          cat=pl.LpContinuous)
                 for i in range(self.rsu_num)
                 for m in range(len(model_util.Model_name))
                 for s in range(model_util.Sub_model_num[m])}
        for sub_model_idx in rand_sub_model_idx:
            x_i_e[(rand_rsu_id, rand_model_idx, sub_model_idx)].lowBound = 1
            x_i_e[(rand_rsu_id, rand_model_idx, sub_model_idx)].upBound = 1

        x_i_l = {(i, l): pl.LpVariable('x_i_l_{0}_{1}'.format(i, l),
                                       lowBound=0, upBound=1, cat=pl.LpContinuous)
                 for i in range(self.rsu_num + 1)
                 for l in range(len(model_util.Sub_Model_Structure_Size))}
        for model_structure_idx in rand_model_structure_idx:
            x_i_l[rand_rsu_id, model_structure_idx].lowBound = 1
            x_i_l[rand_rsu_id, model_structure_idx].upBound = 1

        x_i_i_l = {(i, j, l): pl.LpVariable('x_i_i_l_{0}_{1}_{2}'.format(i, j, l), lowBound=0, upBound=1,
                                            cat=pl.LpContinuous)
                   for i in range(self.rsu_num + 1)
                   for j in range(self.rsu_num)
                   for l in range(len(model_util.Sub_Model_Structure_Size))}
        for other_rsu_idx in range(self.rsu_num + 1):
            for model_structure_idx in rand_model_structure_idx:
                x_i_i_l[other_rsu_idx, rand_rsu_id, model_structure_idx].lowBound = 0
                x_i_i_l[other_rsu_idx, rand_rsu_id, model_structure_idx].upBound = 0

        y_i_jk = {(i, j): pl.LpVariable('y_i_jk_{0}_{1}'.format(i, j), lowBound=0, upBound=1, cat=pl.LpContinuous)
                  for i in range(self.rsu_num)
                  for j in range(self.get_all_task_num())}
        z_i_jk_l = {(i, j, t, l): pl.LpVariable('z_{0}_{1}_{2}_{3}'.format(i, j, t, l), lowBound=0, upBound=1,
                                                cat=pl.LpContinuous)
                    for i in range(self.rsu_num + 1)
                    for j in range(self.rsu_num)
                    for t in range(self.get_all_task_num())
                    for l in range(len(model_util.Sub_Model_Structure_Size))}
        max_system_throughput += (pl.lpSum((y_i_jk[rsu_idx_lp, job_id_lp] for rsu_idx_lp in range(self.rsu_num))
                                           for job_id_lp in range(self.get_all_task_num())))  # 目标函数
        for rsu_idx_lp in range(self.rsu_num + 1):
            for other_rsu_idx_lp in range(self.rsu_num):
                for job_id_lp in range(self.get_all_task_num()):
                    for model_structure_idx_lp in range(len(model_util.Sub_Model_Structure_Size)):
                        max_system_throughput += (
                                z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, model_structure_idx_lp] <= y_i_jk[
                            other_rsu_idx_lp, job_id_lp])
                        max_system_throughput += (
                                z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, model_structure_idx_lp] <=
                                x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp])
                        max_system_throughput += (
                                z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, model_structure_idx_lp] >=
                                y_i_jk[other_rsu_idx_lp, job_id_lp] + x_i_i_l[
                                    rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] - 1)

    def arr(self, T_max):
        self.lin_pro()
        throughput = 0
        objective_value = 0
        return throughput, objective_value

    def allocate_task_for_rsu(self, task_list):
        for task in task_list:
            rsu_id = task["rsu_id"]
            self.RSUs[rsu_id].task_list.append(task)

    def get_task_model_list(self, model_idx, sub_models):
        task_models = set()
        for sub_model in sub_models:
            model_name = model_util.get_model_name(model_idx, sub_model)
            task_models.add(model_name)
        return task_models

    def cal_max_response_time(self):
        T_max = 0
        total_time_list = []  # 记录每个RSU的执行时间
        for rsu_idx in range(self.rsu_num):
            total_time = 0
            if self.RSUs[rsu_idx].has_gpu:
                device_id = self.RSUs[rsu_idx].gpu_idx
            else:
                device_id = self.RSUs[rsu_idx].cpu_idx
            for task in self.RSUs[rsu_idx].task_list:
                exec_time = 0
                model_idx = task["model_idx"]
                sub_models = task["sub_model"]
                rsu_cached_models = self.RSUs[rsu_idx].get_cached_model()
                task_models = self.get_task_model_list(model_idx, sub_models)
                intersection_models = rsu_cached_models & task_models
                if len(intersection_models) > 0:
                    task_models = task_models - intersection_models
                    task_models_size = self.RSUs[rsu_idx].get_caching_models_size(task_models)
                    download_time = task_models_size / self.RSUs[rsu_idx].download_rate
                    for sub_model in sub_models:
                        model = model_util.get_model(model_idx)
                        exec_time += model.cal_execution_delay(seq_num=self.RSUs[rsu_idx].seq_num[model_idx][sub_model],
                                                               sub_model_idx=sub_model, device_id=device_id)
                        self.RSUs[rsu_idx].seq_num[model_idx][sub_model] += 1
                else:
                    task_models_size = self.RSUs[rsu_idx].get_caching_models_size(task_models)
                    download_time = task_models_size / self.RSUs[rsu_idx].download_rate
                    for sub_model in sub_models:
                        model = model_util.get_model(model_idx)
                        exec_time += model.cal_execution_delay(seq_num=self.RSUs[rsu_idx].seq_num[model_idx][sub_model],
                                                               sub_model_idx=sub_model, device_id=device_id)
                        self.RSUs[rsu_idx].seq_num[model_idx][sub_model] += 1
                total_time = download_time + exec_time + total_time
            total_time_list.append(total_time)
            self.RSUs[rsu_idx].seq_num = [[0 for _ in range(model_util.Sub_model_num[i])] for i in
                                          range(len(model_util.Model_name))]
        T_max = max(total_time_list)
        return T_max
