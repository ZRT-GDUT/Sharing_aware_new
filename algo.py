import random
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
        while T_max - T_min >= min_gap:
            throughput, objective_value = self.arr(T_max, task_list)
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

    def find_decide_variable(self):
        model_list_all = {}
        model_structure_list = set()
        for rsu_idx in range(self.rsu_num):
            models = self.RSUs[rsu_idx].get_cached_model()
            if len(models) == 0:
                continue
            else:
                for model_name in models:
                    model_idx, sub_model_idx = model_util.get_model_info(model_name)
                    model = model_util.get_model(model_idx)
                    for structure_idx in model.require_sub_model_all[sub_model_idx]:
                        model_structure_list.add(structure_idx)
                    if model_list_all.get(model_idx, 0) == 0:
                        model_list_all[model_idx] = {sub_model_idx}
                    else:
                        model_list_all[model_idx].add(sub_model_idx)
        return model_list_all, model_structure_list

    def generate_jobid_model_idx(self, task_list: List[dict]):
        model_idx_jobid_list = []
        for task in task_list:
            model_idx_jobid_list.append(task[0]["model_idx"])
        return model_idx_jobid_list

    def get_all_task_num_all(self, task_list):
        task_num = 0
        for i in range(len(task_list)):
            task_num = len(task_list[i]) + task_num
        return task_num

    def get_model_list_merging(self, model_list, model_list_init):
        for rand_model_idx in model_list_init.keys():
            if model_list.get(rand_model_idx, 0) == 0:
                model_list[rand_model_idx] = model_list_init[rand_model_idx]
            else:
                for rand_sub_model in model_list_init[rand_model_idx]:
                    model_list[rand_model_idx].add(rand_sub_model)
        return model_list

    def get_rsu_model_list(self):
        rsu_model_list = {}
        rsu_structure_list = {}
        for rsu_idx in range(self.rsu_num):
            rsu_model_list[rsu_idx] = self.RSUs[rsu_idx].get_cached_model()
            for model_name in self.RSUs[rsu_idx].get_cached_model():
                model_idx, sub_model_idx = model_util.get_model_info(model_name)
                model = model_util.get_model(model_idx)
                for structure_idx in model.require_sub_model_all[sub_model_idx]:
                    if rsu_structure_list.get(rsu_idx, 0) == 0:
                        rsu_structure_list[rsu_idx] = {structure_idx}
                    else:
                        rsu_structure_list[rsu_idx].add(structure_idx)
        return rsu_model_list, rsu_structure_list

    def lin_var(self, model_list, model_structure_list, rsu_model_list, rsu_structure_list,
                task_list, model_list_keys):
        x_i_e = {(i, m, s): pl.LpVariable('x_i_e_{0}_{1}_{2}'.format(i, m, s), lowBound=0, upBound=1,
                                          cat='Continuous')
                 for i in range(self.rsu_num)
                 for m in model_list_keys
                 for s in model_list[m]}
        for rsu_idx in rsu_model_list.keys():
            for rsu_model in rsu_model_list[rsu_idx]:
                model_idx, sub_model_idx = model_util.get_model_info(rsu_model)
                x_i_e[rsu_idx, model_idx, sub_model_idx].lowBound = 1
                x_i_e[rsu_idx, model_idx, sub_model_idx].upBound = 1

        x_i_l = {(i, l): pl.LpVariable('x_i_l_{0}_{1}'.format(i, l),
                                       lowBound=0, upBound=1, cat='Continuous')
                 for i in range(self.rsu_num + 1)
                 for l in model_structure_list}
        for model_structure_idx in model_structure_list:  # 默认云上部署了所有的model
            x_i_l[self.rsu_num, model_structure_idx].lowBound = 1
            x_i_l[self.rsu_num, model_structure_idx].upBound = 1
        for rsu_idx in rsu_structure_list.keys():
            for model_structure_idx in rsu_structure_list[rsu_idx]:
                x_i_l[rsu_idx, model_structure_idx].lowBound = 1
                x_i_l[rsu_idx, model_structure_idx].upBound = 1

        x_i_i_l = {(i, j, l): pl.LpVariable('x_i_i_l_{0}_{1}_{2}'.format(i, j, l), lowBound=0, upBound=1,
                                            cat='Continuous')
                   for i in range(self.rsu_num + 1)
                   for j in range(self.rsu_num)
                   for l in model_structure_list}
        for rsu_idx in rsu_structure_list.keys():
            for model_structure_idx in rsu_structure_list[rsu_idx]:
                for other_rsu_idx in range(self.rsu_num + 1):
                    if other_rsu_idx != rsu_idx:
                        x_i_i_l[other_rsu_idx, rsu_idx, model_structure_idx].lowBound = 0
                        x_i_i_l[other_rsu_idx, rsu_idx, model_structure_idx].upBound = 0
                    else:
                        x_i_i_l[other_rsu_idx, rsu_idx, model_structure_idx].lowBound = 1
                        x_i_i_l[other_rsu_idx, rsu_idx, model_structure_idx].upBound = 1

        y_i_jk = {
            (i, j, m): pl.LpVariable('y_i_jk_{0}_{1}_{2}'.format(i, j, m), lowBound=0, upBound=1, cat='Continuous')
            for i in range(self.rsu_num)
            for j in range(self.get_all_task_num())
            for m in range(len(task_list[j]))}

        z_i_jk_l = {(i, j, t, m, l): pl.LpVariable('z_{0}_{1}_{2}_{3}_{4}'.format(i, j, t, m, l), lowBound=0, upBound=1,
                                                   cat='Continuous')
                    for i in range(self.rsu_num + 1)
                    for j in range(self.rsu_num)
                    for t in range(self.get_all_task_num())
                    for m in range(len(task_list[t]))
                    for l in model_structure_list}
        return x_i_e, x_i_l, x_i_i_l, y_i_jk, z_i_jk_l

    def lin_pro(self, model_list, model_structure_list, task_list, model_idx_jobid_list, T_max, x_structure_model,
                x_task_structure):
        model_list_init, model_structure_list_init = self.find_decide_variable()
        model_structure_list = model_structure_list | model_structure_list_init
        model_list = self.get_model_list_merging(model_list, model_list_init)
        model_list_keys = list(model_list.keys())
        model_list_keys.sort()
        print(model_list, model_structure_list)
        rsu_model_list, rsu_structure_list = self.get_rsu_model_list()
        ######
        # LP #
        ######
        max_system_throughput = pl.LpProblem("max_system_throughput", sense=pl.LpMaximize)  # 定义最大化吞吐率问题
        x_i_e, x_i_l, x_i_i_l, y_i_jk, z_i_jk_l = self.lin_var(model_list, model_structure_list,
                                                               rsu_model_list, rsu_structure_list, task_list,
                                                               model_list_keys)
        max_system_throughput += (pl.lpSum(((y_i_jk[rsu_idx_lp, job_id_lp, sub_task]
                                             for sub_task in range(len(task_list[job_id_lp])))
                                            for job_id_lp in range(self.get_all_task_num()))
                                           for rsu_idx_lp in range(self.rsu_num)))  # 目标函数
        for rsu_idx_lp in range(self.rsu_num + 1):
            for other_rsu_idx_lp in range(self.rsu_num):
                for job_id_lp in range(self.get_all_task_num()):
                    for sub_task in range(len(task_list[job_id_lp])):
                        for model_structure_idx_lp in model_structure_list:
                            max_system_throughput += (
                                    z_i_jk_l[
                                        rsu_idx_lp, other_rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp] <=
                                    y_i_jk[other_rsu_idx_lp, job_id_lp, sub_task])
                            max_system_throughput += (
                                    z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp]
                                    <= x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp])
                            max_system_throughput += (
                                    z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp]
                                    >= (y_i_jk[other_rsu_idx_lp, job_id_lp, sub_task] + x_i_i_l[
                                rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] - 1))

        for job_id_lp in range(self.get_all_task_num()):
            for sub_task in range(len(task_list[job_id_lp])):
                max_system_throughput += (pl.lpSum(y_i_jk[rsu_idx_lp, job_id_lp, sub_task] *
                                                   self.RSUs[rsu_idx_lp].latency_list
                                                   [task_list[job_id_lp][sub_task]["model_idx"]]
                                                   [task_list[job_id_lp][sub_task]["sub_model_idx"]]
                                                   [self.RSUs[rsu_idx_lp].device_idx]
                                                   [task_list[job_id_lp][sub_task]["seq_num"]]
                                                   for rsu_idx_lp in range(self.rsu_num)) +
                                          pl.lpSum(
                                              ((y_i_jk[other_rsu_idx_lp, job_id_lp, sub_task] * model_util.get_model(
                                                  model_idx_jobid_list[job_id_lp]).single_task_size / self.RSUs[
                                                    rsu_idx_lp].rsu_rate)
                                               if (task_list[job_id_lp][sub_task][
                                                       "rsu_id"] == rsu_idx_lp and rsu_idx_lp != other_rsu_idx_lp) else 0
                                               for rsu_idx_lp in range(self.rsu_num))
                                              for other_rsu_idx_lp in range(self.rsu_num))
                                          + pl.lpSum((((((z_i_jk_l[
                                                              rsu_idx_lp, other_rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp] *
                                                          x_task_structure[job_id_lp][sub_task][model_structure_idx_lp]
                                                          * model_util.Sub_Model_Structure_Size[model_structure_idx_lp])
                                                         / (self.RSUs[
                                                                rsu_idx_lp].rsu_rate if rsu_idx_lp != self.rsu_num else
                                                            self.RSUs[other_rsu_idx_lp].download_rate))
                                                        if other_rsu_idx_lp != rsu_idx_lp else 0)
                                                       for model_structure_idx_lp in model_structure_list)
                                                      for other_rsu_idx_lp in range(self.rsu_num))
                                                     for rsu_idx_lp in range(self.rsu_num + 1)) <= task_list[job_id_lp]
                                          [sub_task]["latency"])  # Constraint(32)

        for rsu_idx_lp in range(self.rsu_num):
            for other_rsu_idx_lp in range(self.rsu_num):
                max_system_throughput += (pl.lpSum(
                    (((model_util.get_model(model_idx_jobid_list[job_id_lp]).single_task_size * y_i_jk[
                        other_rsu_idx_lp, job_id_lp, sub_task] / self.RSUs[rsu_idx_lp].rsu_rate)
                      if (rsu_idx_lp != other_rsu_idx_lp and task_list[job_id_lp][sub_task][
                        "rsu_id"] == rsu_idx_lp) else 0)
                     for sub_task in range(len(task_list[job_id_lp])))
                    for job_id_lp in range(self.get_all_task_num()))
                                          + pl.lpSum((((x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] *
                                                        model_util.Sub_Model_Structure_Size[model_structure_idx_lp]) /
                                                       (self.RSUs[
                                                            rsu_idx_lp].rsu_rate)) if rsu_idx_lp != other_rsu_idx_lp else 0)
                                                     for model_structure_idx_lp in
                                                     model_structure_list) <= T_max)  # Constraint(34)
        for rsu_idx_lp in range(self.rsu_num):
            max_system_throughput += (pl.lpSum(
                ((x_i_i_l[self.rsu_num, rsu_idx_lp, model_structure_idx_lp] * model_util.Sub_Model_Structure_Size[
                    model_structure_idx_lp])
                 / self.RSUs[rsu_idx_lp].download_rate) for model_structure_idx_lp in
                model_structure_list) <= T_max)  # Constraint(35)

        for rsu_idx_lp in range(self.rsu_num):
            max_system_throughput += (pl.lpSum(
                ((y_i_jk[rsu_idx_lp, job_id_lp, sub_task] *
                  self.RSUs[rsu_idx_lp].latency_list[task_list[job_id_lp][sub_task]["model_idx"]][
                      task_list[job_id_lp][sub_task]
                      ["sub_model_idx"]][self.RSUs[rsu_idx_lp].device_idx][task_list[job_id_lp][sub_task]["seq_num"]])
                 for sub_task in range(len(task_list[job_id_lp])))
                for job_id_lp in range(self.get_all_task_num())) <= T_max)  # Constraint(36)

        for job_id_lp in range(self.get_all_task_num()):
            for sub_task in range(len(task_list[job_id_lp])):
                max_system_throughput += (pl.lpSum(
                    y_i_jk[rsu_idx_lp, job_id_lp, sub_task] for rsu_idx_lp in
                    range(self.rsu_num)) <= 1)  # Constraint(37)

        for rsu_idx_lp in range(self.rsu_num):
            max_system_throughput += (pl.lpSum((x_i_l[rsu_idx_lp, model_structure_idx_lp] *
                                                 model_util.Sub_Model_Structure_Size[model_structure_idx_lp])
                                                for model_structure_idx_lp in model_structure_list)
                                      + pl.lpSum(((y_i_jk[rsu_idx_lp, job_id_lp, sub_task] * model_util.get_model(
                        model_idx_jobid_list[job_id_lp]).single_task_size)
                                                  for sub_task in range(len(task_list[job_id_lp])))
                                                 for job_id_lp in range(self.get_all_task_num())) <= self.RSUs[
                                          rsu_idx_lp].storage_capacity)  # Constraint(14)

        for rsu_idx_lp in range(self.rsu_num + 1):
            for other_rsu_idx_lp in range(self.rsu_num):
                for model_structure_idx_lp in model_structure_list:
                    max_system_throughput += (x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] <= x_i_l
                    [rsu_idx_lp, model_structure_idx_lp])  # Constraint(16)

        for rsu_idx_lp in range(self.rsu_num):
            for model_idx_lp in model_list_keys:
                for sub_model_idx_lp in model_list[model_idx_lp]:
                    for model_structure_idx_lp in model_structure_list:
                        max_system_throughput += ((x_structure_model[model_idx_lp][sub_model_idx_lp][
                                                       model_structure_idx_lp] * x_i_e[
                                                       rsu_idx_lp, model_idx_lp, sub_model_idx_lp]) <=
                                                  pl.lpSum(x_i_i_l[other_rsu_idx_lp, rsu_idx_lp, model_structure_idx_lp]
                                                           for other_rsu_idx_lp in
                                                           range(self.rsu_num + 1)))  # Constraint(17)

        for rsu_idx_lp in range(self.rsu_num):
            for model_structure_idx_lp in model_structure_list:
                max_system_throughput += (pl.lpSum(
                    x_i_i_l[other_rsu_idx_lp, rsu_idx_lp, model_structure_idx_lp] for other_rsu_idx_lp in
                    range(self.rsu_num + 1)
                ) <= 1)  # Constraint(18)

        for rsu_idx_lp in range(self.rsu_num):
            for job_id_lp in range(self.get_all_task_num()):
                for sub_task in range(len(task_list[job_id_lp])):
                    for model_structure_idx_lp in model_structure_list:
                        max_system_throughput += (
                                y_i_jk[rsu_idx_lp, job_id_lp, sub_task] * x_task_structure[job_id_lp][sub_task][
                            model_structure_idx_lp]
                                <= x_i_l[rsu_idx_lp, model_structure_idx_lp])  # Constraint(19)

        for rsu_idx_lp in range(self.rsu_num):
            for job_id_lp in range(self.get_all_task_num()):
                for sub_task in range(len(task_list[job_id_lp])):
                    max_system_throughput += (y_i_jk[rsu_idx_lp, job_id_lp, sub_task] <=
                                              x_i_e[rsu_idx_lp, task_list[job_id_lp][sub_task]["model_idx"],
                                              task_list[job_id_lp][sub_task]["sub_model_idx"]])  # Extra Constraint

        print("sub_task_num:", self.get_all_task_num_all(task_list))

        status = max_system_throughput.solve()
        print(pl.LpStatus[status])
        for v in y_i_jk.values():
            if v.varValue != 0 and v.varValue != None:
                print(v.name, "=", v.varValue)
        # for v in x_i_i_l.values():
        #     if v.varValue != 0 and v.varValue != None:
        #         print(v.name, "=", v.varValue)
        for v in x_i_e.values():
            if v.varValue != 0 and v.varValue != None:
                print(v.name, "=", v.varValue)
        for v in x_i_l.values():
            if v.varValue != 0 and v.varValue != None:
                print(v.name, "=", v.varValue)
        # for v in z_i_jk_l.values():
        #     if v.varValue != 0 and v.varValue != None:
        #         print(v.name, "=", v.varValue)
        print('objective =', pl.value(max_system_throughput.objective))
        rsu_model_list_rr, rsu_to_rsu_model_structure_list_rr, rsu_model_structure_list_rr, rsu_job_list_rr \
            = self.RR(x_i_l, x_i_i_l, x_i_e, y_i_jk, model_list_keys, model_list, model_structure_list, task_list)
        print("rsu_model_list_rr", rsu_model_list_rr[0])
        print("rsu_to_rsu_model_structure_list_rr", rsu_to_rsu_model_structure_list_rr[:][0])
        print("rsu_model_structure_list_rr", rsu_model_structure_list_rr[0])
        print("rsu_job_list_rr", rsu_job_list_rr[0])
        print(self.RSUs[0].get_cached_model())
        throughput = 0
        for rsu_idx in range(self.rsu_num):
            throughput += len(rsu_job_list_rr[rsu_idx])
        print(throughput)
        t = self.calculate_objective_value(record_task_dict, is_shared=True)
        object_value = sum(t)
        return throughput, object_value, []

    def RR(self, x_i_l, x_i_i_l, x_i_e, y_i_jk, model_list_keys, model_list, model_structure_list, task_list):
        task_list_new = task_list
        rsu_model_list_rr = {}
        rsu_model_structure_list_rr = {}
        rsu_to_rsu_model_structure_list_rr = [[[0 for _ in range(len(model_util.Sub_Model_Structure_Size))]
                                               for _ in range(self.rsu_num)] for rsu_idx in range(self.rsu_num + 1)]
        rsu_job_list_rr = [[] for rsu_idx in range(self.rsu_num)]
        rsu_model_list = [[[0 for _ in range(model_util.Sub_model_num[model_idx])]
                           for model_idx in range(len(model_util.Model_name))] for rsu_idx in range(self.rsu_num)]
        rsu_model_structure_list = [[0 for _ in range(len(model_util.Sub_Model_Structure_Size))]
                                    for rsu_idx in range(self.rsu_num + 1)]

        for rsu_idx in range(self.rsu_num):
            for model_idx in model_list_keys:
                for sub_model_idx in model_list[model_idx]:
                    x_i_e_value = x_i_e[(rsu_idx, model_idx, sub_model_idx)].value()
                    if random.uniform(0, 1) <= x_i_e_value:
                        rsu_model_list[rsu_idx][model_idx][sub_model_idx] = 1
                        model_name = model_util.get_model_name(model_idx, sub_model_idx)
                        if rsu_model_list_rr.get(rsu_idx, 0) == 0:
                            rsu_model_list_rr[rsu_idx] = {model_name}
                        else:
                            rsu_model_list_rr[rsu_idx].add(model_name)
                    else:
                        rsu_model_list[rsu_idx][model_idx][sub_model_idx] == 0

        for rsu_idx in range(self.rsu_num + 1):
            for model_structure_idx in model_structure_list:
                x_i_l_value = x_i_l[(rsu_idx, model_structure_idx)].value()
                if random.uniform(0, 1) <= x_i_l_value:
                    rsu_model_structure_list[rsu_idx][model_structure_idx] = 1
                    if rsu_model_structure_list_rr.get(rsu_idx, 0) == 0:
                        rsu_model_structure_list_rr[rsu_idx] = {model_structure_idx}
                    else:
                        rsu_model_structure_list_rr[rsu_idx].add(model_structure_idx)
                else:
                    rsu_model_structure_list[rsu_idx][model_structure_idx] = 0

        for rsu_idx in range(self.rsu_num + 1):
            for other_rsu_idx in range(self.rsu_num):
                for model_structure_idx in model_structure_list:
                    x_i_i_l_value = x_i_i_l[(rsu_idx, other_rsu_idx, model_structure_idx)].value()
                    if rsu_model_structure_list[rsu_idx][model_structure_idx] == 1:
                        if random.uniform(0, 1) <= (x_i_i_l_value / x_i_l[(rsu_idx, model_structure_idx)].value()):
                            rsu_to_rsu_model_structure_list_rr[rsu_idx][other_rsu_idx][model_structure_idx] = 1

        for rsu_idx in range(self.rsu_num):
            for job_id in range(len(task_list_new)):
                for sub_task in range(len(task_list_new[job_id])):
                    if task_list_new[job_id][sub_task] is not None:
                        model_idx = task_list_new[job_id][sub_task]["model_idx"]
                        sub_model_idx = task_list_new[job_id][sub_task]["sub_model_idx"]
                        y_i_jk_value = y_i_jk[(rsu_idx, job_id, sub_task)].value()
                        if rsu_model_list[rsu_idx][model_idx][sub_model_idx] == 1:
                            if random.uniform(0, 1) <= (
                                    y_i_jk_value / x_i_e[(rsu_idx, model_idx, sub_model_idx)].value()):
                                rsu_job_list_rr[rsu_idx].append(task_list_new[job_id][sub_task])
                                task_list_new[job_id][sub_task] = None

        return rsu_model_list_rr, rsu_to_rsu_model_structure_list_rr, rsu_model_structure_list_rr, rsu_job_list_rr

    def get_variable_range(self, task_list):
        model_list = {}
        model_structure_list = set()
        for task in task_list:
            for sub_task in task:
                model_idx = sub_task["model_idx"]
                sub_model_idx = sub_task["sub_model_idx"]
                for model_structure_idx in sub_task["model_structure"]:
                    model_structure_list.add(model_structure_idx)
                if model_list.get(model_idx, 0) == 0:
                    model_list[model_idx] = {sub_model_idx}
                else:
                    model_list[model_idx].add(sub_model_idx)
        return model_list, model_structure_list

    def generate_task_to_structure(self, task_list):
        x_task_structure = [[[0 for _ in range(len(model_util.Sub_Model_Structure_Size))]
                             for _ in range(len(task_list[i]))] for i in range(self.get_all_task_num())]
        for task in task_list:
            for sub_task in task:
                task_job_id = sub_task["job_id"]
                task_structure = sub_task["model_structure"]
                sub_task_id = sub_task["sub_model_idx"]
                for task_structure_idx in task_structure:
                    x_task_structure[task_job_id][sub_task_id][task_structure_idx] = 1
        return x_task_structure

    def generate_structure_to_model(self):
        x_structure_model = [
            [[0 for _ in range(len(model_util.Sub_Model_Structure_Size))] for _ in range(model_util.Sub_model_num[i])]
            for i in range(len(model_util.Model_name))]  # 用来定义每个model需要哪些structure
        for model_idx in range(len(model_util.Model_name)):
            for sub_model_idx in range(model_util.Sub_model_num[model_idx]):
                model_structure = model_util.get_model(model_idx).require_sub_model_all[sub_model_idx]
                for model_structure_idx in model_structure:
                    x_structure_model[model_idx][sub_model_idx][model_structure_idx] = 1
        return x_structure_model

    def arr(self, T_max, task_list):
        model_idx_jobid_list = self.generate_jobid_model_idx(task_list)
        x_task_structure = self.generate_task_to_structure(task_list)
        x_structure_model = self.generate_structure_to_model()
        model_list, model_structure_list = self.get_variable_range(task_list)
        self.lin_pro(model_list, model_structure_list, task_list, model_idx_jobid_list, T_max, x_structure_model,
                     x_task_structure)
        throughput = 0
        objective_value = 0
        return throughput, objective_value

    def allocate_task_for_rsu(self, task_list):
        for task in task_list:
            rsu_id = task[0]["rsu_id"]
            self.RSUs[rsu_id].task_list.append(task)

    def get_task_model_list(self, model_idx, sub_models):
        task_models = set()
        for sub_model in sub_models:
            model_name = model_util.get_model_name(model_idx, sub_model)
            task_models.add(model_name)
        return task_models

    def cal_max_response_time(self, is_shared=True):
        T_max = 0
        total_time_list = []  # 记录每个RSU的执行时间
        for rsu_idx in range(self.rsu_num):
            flag = 0
            if self.RSUs[rsu_idx].has_gpu:
                device_id = self.RSUs[rsu_idx].gpu_idx
            else:
                device_id = self.RSUs[rsu_idx].cpu_idx
            for task in self.RSUs[rsu_idx].task_list:
                flag = 1
                exec_time_sub_task_list = []
                task_exec_time_list = []
                for sub_task in task:
                    seq_num = sub_task["seq_num"]
                    exec_time_sub_task = 0
                    model_idx = sub_task["model_idx"]
                    sub_model_idx = sub_task["sub_model_idx"]
                    rsu_cached_models = self.RSUs[rsu_idx].get_cached_model()
                    sub_task_model = model_util.get_model_name(model_idx, sub_model_idx)
                    if sub_task_model in rsu_cached_models:
                        exec_time = self.RSUs[rsu_idx].latency_list[model_idx][sub_model_idx][device_id][seq_num]
                        download_time = 0
                    else:
                        task_model_size = self.RSUs[rsu_idx].get_caching_model_size(sub_task_model)
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
