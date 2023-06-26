import random
from typing import List

from pulp import LpStatusInfeasible

import device
import model_util
import pulp as pl


def get_variable_range(task_list):
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


def get_task_model_list(model_idx, sub_models):
    task_models = set()
    for sub_model in sub_models:
        model_name = model_util.get_model_name(model_idx, sub_model)
        task_models.add(model_name)
    return task_models


def generate_structure_to_model():
    x_structure_model = [
        [[0 for _ in range(len(model_util.Sub_Model_Structure_Size))] for _ in range(model_util.Sub_model_num[i])]
        for i in range(len(model_util.Model_name))]  # 用来定义每个model需要哪些structure
    for model_idx in range(len(model_util.Model_name)):
        for sub_model_idx in range(model_util.Sub_model_num[model_idx]):
            model_structure = model_util.get_model(model_idx).require_sub_model_all[sub_model_idx]
            for model_structure_idx in model_structure:
                x_structure_model[model_idx][sub_model_idx][model_structure_idx] = 1
    return x_structure_model


class Algo:
    def __init__(self, RSUs: List[device.RSU], task_list):
        self.RSUs = RSUs
        self.rsu_num = len(RSUs)
        self.task_list = task_list
        self.allocate_task_for_rsu()
        self.model_idx_jobid_list = self.generate_jobid_model_idx()
        self.x_task_structure = self.generate_task_to_structure()
        self.x_structure_model = generate_structure_to_model()
        self.rsu_model_list, self.rsu_structure_list = self.get_rsu_model_list()
        self.model_list, self.model_structure_list, self.model_list_keys = self.find_decide_variable_new()

    def find_decide_variable_new(self):
        model_list_init, model_structure_list_init = self.find_decide_variable()
        model_list_task, model_structure_list_task = get_variable_range(self.task_list)
        model_structure_list = model_structure_list_task | model_structure_list_init
        model_list = self.get_model_list_merging(model_list_task, model_list_init)
        model_list_keys = list(model_list.keys())
        model_list_keys.sort()
        return model_list, model_structure_list, model_list_keys

    def get_all_task_num(self):
        task_num = 0
        for rsu_idx in range(self.rsu_num):
            task_num = len(self.RSUs[rsu_idx].task_list) + task_num
        return task_num

    def iarr(self, task_list, min_gap=0.1):
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

    def generate_jobid_model_idx(self):
        model_idx_jobid_list = []
        for task in self.task_list:
            model_idx_jobid_list.append(task[0]["model_idx"])
        return model_idx_jobid_list

    def get_all_task_num_all(self):
        task_num = 0
        for i in range(len(self.task_list)):
            task_num = len(self.task_list[i]) + task_num
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
        rsu_structure_list = {i: set() for i in range(self.rsu_num)}
        for rsu_idx in range(self.rsu_num):
            rsu_model_list[rsu_idx] = self.RSUs[rsu_idx].get_cached_model()
            for model_name in self.RSUs[rsu_idx].get_cached_model():
                model_idx, sub_model_idx = model_util.get_model_info(model_name)
                model = model_util.get_model(model_idx)
                for structure_idx in model.require_sub_model_all[sub_model_idx]:
                    rsu_structure_list[rsu_idx].add(structure_idx)
        return rsu_model_list, rsu_structure_list

    def lin_var(self):
        x_i_l = {(i, l): pl.LpVariable('x_i_l_{0}_{1}'.format(i, l),
                                       lowBound=0, upBound=1, cat='Continuous')
                 for i in range(self.rsu_num + 1)
                 for l in self.model_structure_list}
        for model_structure_idx in self.model_structure_list:  # 默认云上部署了所有的model
            x_i_l[self.rsu_num, model_structure_idx].setInitialValue(1)
            x_i_l[self.rsu_num, model_structure_idx].fixValue()
        for rsu_idx in self.rsu_structure_list.keys():
            for model_structure_idx in self.rsu_structure_list[rsu_idx]:  # 将已经部署的structure设为1
                x_i_l[rsu_idx, model_structure_idx].setInitialValue(1)
                x_i_l[rsu_idx, model_structure_idx].fixValue()

        x_i_i_l = {(i, j, l): pl.LpVariable('x_i_i_l_{0}_{1}_{2}'.format(i, j, l), lowBound=0, upBound=1,
                                            cat='Continuous')
                   for i in range(self.rsu_num + 1)
                   for j in range(self.rsu_num)
                   for l in self.model_structure_list}
        for rsu_idx in range(self.rsu_num):  # 设定其它rsu无法为已经部署structure的rsu传输相同的structure
            for model_structure_idx in self.model_structure_list:
                if model_structure_idx not in self.rsu_structure_list[rsu_idx]:
                    x_i_i_l[rsu_idx, rsu_idx, model_structure_idx].setInitialValue(0)
                    x_i_i_l[rsu_idx, rsu_idx, model_structure_idx].fixValue()
        for rsu_idx in self.rsu_structure_list.keys():
            for model_structure_idx in self.rsu_structure_list[rsu_idx]:
                for other_rsu_idx in range(self.rsu_num + 1):
                    if rsu_idx == other_rsu_idx:
                        x_i_i_l[other_rsu_idx, rsu_idx, model_structure_idx].setInitialValue(1)
                        x_i_i_l[other_rsu_idx, rsu_idx, model_structure_idx].fixValue()
                    else:
                        x_i_i_l[other_rsu_idx, rsu_idx, model_structure_idx].setInitialValue(0)
                        x_i_i_l[other_rsu_idx, rsu_idx, model_structure_idx].fixValue()

        y_i_jk = {
            (i, j, m): pl.LpVariable('y_i_jk_{0}_{1}_{2}'.format(i, j, m), lowBound=0, upBound=1, cat='Continuous')
            for i in range(self.rsu_num)
            for j in range(self.get_all_task_num())
            for m in range(len(self.task_list[j]))}

        z_i_jk_l = {(i, j, t, m, l): pl.LpVariable('z_{0}_{1}_{2}_{3}_{4}'.format(i, j, t, m, l), lowBound=0, upBound=1,
                                                   cat='Continuous')
                    for i in range(self.rsu_num + 1)
                    for j in range(self.rsu_num)
                    for t in range(self.get_all_task_num())
                    for m in range(len(self.task_list[t]))
                    for l in self.model_structure_list}
        return x_i_l, x_i_i_l, y_i_jk, z_i_jk_l

    def lin_con(self, T_max, task_list, x_i_l, x_i_i_l, y_i_jk, z_i_jk_l, max_system_throughput, flag):
        for rsu_idx_lp in range(self.rsu_num + 1):
            for other_rsu_idx_lp in range(self.rsu_num):
                for job_id_lp in range(self.get_all_task_num()):
                    for sub_task in range(len(self.task_list[job_id_lp])):
                        for model_structure_idx_lp in self.model_structure_list:
                            constraint1 = pl.LpConstraint(
                                e=z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp] -
                                  y_i_jk[other_rsu_idx_lp, job_id_lp, sub_task],
                                sense=pl.LpConstraintLE,
                                rhs=0
                            )
                            constraint2 = pl.LpConstraint(
                                e=z_i_jk_l[rsu_idx_lp, other_rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp] -
                                  x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp],
                                sense=pl.LpConstraintLE,
                                rhs=0
                            )
                            constraint3 = pl.LpConstraint(
                                e=y_i_jk[other_rsu_idx_lp, job_id_lp, sub_task] + x_i_i_l[
                                    rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] - z_i_jk_l[
                                      rsu_idx_lp, other_rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp],
                                sense=pl.LpConstraintLE,
                                rhs=1
                            )
                            max_system_throughput += constraint1
                            max_system_throughput += constraint2
                            max_system_throughput += constraint3

        for job_id_lp in range(self.get_all_task_num()):
            for sub_task in range(len(self.task_list[job_id_lp])):
                constraint1 = pl.LpConstraint(
                    e=pl.lpSum(y_i_jk[rsu_idx_lp, job_id_lp, sub_task] *
                               self.RSUs[rsu_idx_lp].latency_list
                               [self.task_list[job_id_lp][sub_task]["model_idx"]]
                               [self.task_list[job_id_lp][sub_task]["sub_model_idx"]]
                               [self.RSUs[rsu_idx_lp].device_idx]
                               [self.task_list[job_id_lp][sub_task]["seq_num"]]
                               for rsu_idx_lp in range(self.rsu_num)) +
                      pl.lpSum(
                          (((y_i_jk[other_rsu_idx_lp, job_id_lp, sub_task] * model_util.get_model(
                              self.model_idx_jobid_list[job_id_lp]).single_task_size) / (self.RSUs[
                                                                                             self.task_list[
                                                                                                 job_id_lp][
                                                                                                 sub_task][
                                                                                                 "rsu_id"]].rsu_rate * len(
                              self.task_list[job_id_lp])))
                           if rsu_idx_lp != other_rsu_idx_lp else 0
                           for rsu_idx_lp in range(self.rsu_num))
                          for other_rsu_idx_lp in range(self.rsu_num))
                      + pl.lpSum(((((z_i_jk_l[
                                         rsu_idx_lp, other_rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp] *
                                     self.x_task_structure[job_id_lp][sub_task][
                                         model_structure_idx_lp]
                                     * model_util.Sub_Model_Structure_Size[model_structure_idx_lp])
                                    / (self.RSUs[
                                           rsu_idx_lp].rsu_rate if rsu_idx_lp != self.rsu_num else
                                       self.RSUs[other_rsu_idx_lp].download_rate))
                                   for model_structure_idx_lp in self.model_structure_list)
                                  for other_rsu_idx_lp in range(self.rsu_num))
                                 for rsu_idx_lp in range(self.rsu_num + 1)),
                    sense=pl.LpConstraintLE,
                    rhs=self.task_list[job_id_lp][sub_task]["latency"]
                )
                max_system_throughput += constraint1
                # Constraint(32)

        for rsu_idx_lp in range(self.rsu_num):
            for other_rsu_idx_lp in range(self.rsu_num):
                if rsu_idx_lp == other_rsu_idx_lp:
                    continue
                constraint1 = pl.LpConstraint(
                    e=pl.lpSum(
                        ((model_util.get_model(self.model_idx_jobid_list[job_id_lp]).single_task_size * y_i_jk[
                            other_rsu_idx_lp, job_id_lp, sub_task] / (
                                  self.RSUs[self.task_list[job_id_lp][sub_task]["rsu_id"]].rsu_rate * len(
                              self.task_list[job_id_lp]))) if other_rsu_idx_lp != self.task_list[job_id_lp][sub_task][
                            "rsu_id"] else 0
                         for sub_task in range(len(self.task_list[job_id_lp])))
                        for job_id_lp in range(self.get_all_task_num())),
                    # + pl.lpSum(((x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] *
                    #              model_util.Sub_Model_Structure_Size[model_structure_idx_lp]) /
                    #             self.RSUs[
                    #                 rsu_idx_lp].rsu_rate)
                    #            for model_structure_idx_lp in
                    #            self.model_structure_list),
                    sense=pl.LpConstraintLE,
                    rhs=T_max
                )
                max_system_throughput += constraint1  # Constraint(35)

        for rsu_idx_lp in range(self.rsu_num):
            constraint1 = pl.LpConstraint(
                e=pl.lpSum(
                    ((x_i_i_l[self.rsu_num, rsu_idx_lp, model_structure_idx_lp] * model_util.Sub_Model_Structure_Size[
                        model_structure_idx_lp])
                     / self.RSUs[rsu_idx_lp].download_rate) for model_structure_idx_lp in
                    self.model_structure_list),
                sense=pl.LpConstraintLE,
                rhs=T_max
            )
            max_system_throughput += constraint1  # Constraint(36)

        for rsu_idx_lp in range(self.rsu_num):
            constraint1 = pl.LpConstraint(
                e=pl.lpSum(
                    ((y_i_jk[rsu_idx_lp, job_id_lp, sub_task] *
                      self.RSUs[rsu_idx_lp].latency_list[self.task_list[job_id_lp][sub_task]["model_idx"]][
                          self.task_list[job_id_lp][sub_task]
                          ["sub_model_idx"]][self.RSUs[rsu_idx_lp].device_idx][
                          self.task_list[job_id_lp][sub_task]["seq_num"]] / len(self.task_list[job_id_lp]))
                     for sub_task in range(len(self.task_list[job_id_lp])))
                    for job_id_lp in range(self.get_all_task_num())) +
                  pl.lpSum(
                      ((model_util.get_model(self.model_idx_jobid_list[job_id_lp]).single_task_size * y_i_jk[
                          rsu_idx_lp, job_id_lp, sub_task] / (
                                self.RSUs[self.task_list[job_id_lp][sub_task]["rsu_id"]].rsu_rate * len(
                            self.task_list[job_id_lp]))) if rsu_idx_lp != self.task_list[job_id_lp][sub_task][
                          "rsu_id"] else 0
                       for sub_task in range(len(self.task_list[job_id_lp])))
                      for job_id_lp in range(self.get_all_task_num())),
                sense=pl.LpConstraintLE,
                rhs=T_max
            )
            max_system_throughput += constraint1  # Constraint(37)

        for job_id_lp in range(self.get_all_task_num()):
            for sub_task in range(len(self.task_list[job_id_lp])):
                constraint1 = pl.LpConstraint(
                    e=pl.lpSum(
                        y_i_jk[rsu_idx_lp, job_id_lp, sub_task] for rsu_idx_lp in
                        range(self.rsu_num)),
                    sense=pl.LpConstraintLE,
                    rhs=1
                )
                max_system_throughput += constraint1  # Constraint(38)

        for rsu_idx_lp in range(self.rsu_num):
            constraint1 = pl.LpConstraint(
                e=pl.lpSum((x_i_l[rsu_idx_lp, model_structure_idx_lp] *
                            model_util.Sub_Model_Structure_Size[model_structure_idx_lp])
                           for model_structure_idx_lp in self.model_structure_list)
                  + pl.lpSum(((y_i_jk[rsu_idx_lp, job_id_lp, sub_task] * model_util.get_model(
                    self.model_idx_jobid_list[job_id_lp]).single_task_size / len(
                    self.task_list[job_id_lp][sub_task]))
                              for sub_task in range(len(self.task_list[job_id_lp])))
                             for job_id_lp in range(self.get_all_task_num())),
                sense=pl.LpConstraintLE,
                rhs=self.RSUs[rsu_idx_lp].storage_capacity
            )
            max_system_throughput += constraint1  # Constraint(15)

        for rsu_idx_lp in range(self.rsu_num + 1):
            for other_rsu_idx_lp in range(self.rsu_num):
                for model_structure_idx_lp in self.model_structure_list:
                    constraint1 = pl.LpConstraint(
                        e=x_i_i_l[rsu_idx_lp, other_rsu_idx_lp, model_structure_idx_lp] - x_i_l[
                            rsu_idx_lp, model_structure_idx_lp],
                        sense=pl.LpConstraintLE,
                        rhs=0
                    )
                    max_system_throughput += constraint1  # Constraint(16)

        for rsu_idx_lp in range(self.rsu_num):
            for model_structure_idx_lp in self.model_structure_list:
                constraint1 = pl.LpConstraint(
                    e=pl.lpSum(x_i_i_l[other_rsu_idx_lp, rsu_idx_lp, model_structure_idx_lp] for other_rsu_idx_lp in
                               range(self.rsu_num + 1)),
                    sense=pl.LpConstraintLE,
                    rhs=1
                )
                max_system_throughput += constraint1  # Constraint(18)

        for rsu_idx_lp in range(self.rsu_num):
            for job_id_lp in range(self.get_all_task_num()):
                for sub_task in range(len(self.task_list[job_id_lp])):
                    for model_structure_idx_lp in self.model_structure_list:
                        max_system_throughput += (
                                y_i_jk[rsu_idx_lp, job_id_lp, sub_task] * self.x_task_structure[job_id_lp][sub_task][
                            model_structure_idx_lp]
                                <= x_i_l[rsu_idx_lp, model_structure_idx_lp])  # Constraint(19)

        for rsu_idx_lp in range(self.rsu_num):
            for model_structure_idx_lp in self.model_structure_list:
                constraint1 = pl.LpConstraint(
                    e=x_i_l[rsu_idx_lp, model_structure_idx_lp] - pl.lpSum(x_i_i_l[other_rsu_idx_lp, rsu_idx_lp,
                                                                                   model_structure_idx_lp] for
                                                                           other_rsu_idx_lp in range(self.rsu_num + 1)),
                    sense=pl.LpConstraintLE,
                    rhs=0
                )
                max_system_throughput += constraint1

        status = max_system_throughput.solve()
        print(pl.LpStatus[status])
        for v in y_i_jk.values():
            if v.varValue != 0 and v.varValue != None:
                print(v.name, "=", v.varValue)
        for v in x_i_i_l.values():
            if v.varValue != 0 and v.varValue != None:
                print(v.name, "=", v.varValue)
        for v in x_i_l.values():
            if v.varValue != 0 and v.varValue != None:
                print(v.name, "=", v.varValue)
        print('objective =', pl.value(max_system_throughput.objective))
        throughput = pl.value(max_system_throughput.objective)
        if flag:
            unsatisfied_constraints = [c for c in max_system_throughput.constraints.values() if c.slack < 0]
            half = len(unsatisfied_constraints) // 2
            for c in range(half):
                print(
                    f"Unsatisfied constraint: {unsatisfied_constraints[c]}, slack: {unsatisfied_constraints[c].slack},values: {[v.value() for v in unsatisfied_constraints[c].keys()]}")
            rsu_model_list_rr, rsu_to_rsu_model_structure_list_rr, rsu_model_structure_list_rr, rsu_job_list_rr, \
            rsu_model_list, rsu_model_structure_list, rsu_job_list = self.RR(x_i_l, x_i_i_l, y_i_jk,
                                                                             task_list)
            return rsu_model_list_rr, rsu_to_rsu_model_structure_list_rr, rsu_model_structure_list_rr, rsu_job_list_rr, \
                   rsu_model_list, rsu_model_structure_list, rsu_job_list, throughput
        else:
            # 假设 prob 为 PuLP 中的问题对象
            unsatisfied_constraints = [c for c in max_system_throughput.constraints.values() if c.slack < 0]
            half = len(unsatisfied_constraints) // 2
            for c in range(half):
                print(
                    f"Unsatisfied constraint: {unsatisfied_constraints[c]}, slack: {unsatisfied_constraints[c].slack},values: {[v.value() for v in unsatisfied_constraints[c].keys()]}")
            return pl.LpStatus[status], pl.value(max_system_throughput.objective)

    def lin_pro(self, T_max, task_list):
        ######
        # LP #
        ######
        status = "Infeasible"
        # while status == "Infeasible":
        max_system_throughput = pl.LpProblem("max_system_throughput", sense=pl.LpMaximize)  # 定义最大化吞吐率问题
        x_i_l, x_i_i_l, y_i_jk, z_i_jk_l = self.lin_var()
        max_system_throughput += (pl.lpSum(((y_i_jk[rsu_idx_lp, job_id_lp, sub_task]
                                             for sub_task in range(len(self.task_list[job_id_lp])))
                                            for job_id_lp in range(self.get_all_task_num()))
                                           for rsu_idx_lp in range(self.rsu_num)))  # 目标函数
        rsu_model_list_rr, rsu_to_rsu_model_structure_list_rr, rsu_model_structure_list_rr, rsu_job_list_rr, \
        rsu_model_list, rsu_model_structure_list, rsu_job_list, throughput = \
            self.lin_con(T_max, task_list, x_i_l, x_i_i_l, y_i_jk, z_i_jk_l, max_system_throughput, 1)

        for rsu_idx in range(self.rsu_num):
            for job_id in range(len(self.task_list)):
                for sub_task in range(len(self.task_list[job_id])):
                    y_i_jk[rsu_idx, job_id, sub_task].setInitialValue(rsu_job_list[rsu_idx][job_id][sub_task])
                    y_i_jk[rsu_idx, job_id, sub_task].fixValue()

        for rsu_idx in range(self.rsu_num + 1):
            for model_structure_idx in self.model_structure_list:
                x_i_l[rsu_idx, model_structure_idx].setInitialValue(rsu_model_structure_list[rsu_idx][
                                                                        model_structure_idx])
                x_i_l[rsu_idx, model_structure_idx].fixValue()

        for rsu_idx in range(self.rsu_num + 1):
            for other_rsu_idx in range(self.rsu_num):
                for model_structure_idx in self.model_structure_list:
                    x_i_i_l[rsu_idx, other_rsu_idx, model_structure_idx].setInitialValue(
                        rsu_to_rsu_model_structure_list_rr[rsu_idx][other_rsu_idx][model_structure_idx])
                    x_i_i_l[rsu_idx, other_rsu_idx, model_structure_idx].fixValue()

        for other_rsu_idx_lp in range(self.rsu_num + 1):
            for rsu_idx_lp in range(self.rsu_num):
                for job_id_lp in range(self.get_all_task_num()):
                    for sub_task in range(len(self.task_list[job_id_lp])):
                        for model_structure_idx_lp in self.model_structure_list:
                            if rsu_to_rsu_model_structure_list_rr[other_rsu_idx_lp][rsu_idx_lp][
                                model_structure_idx_lp] == 1 and rsu_job_list[rsu_idx_lp][job_id_lp][sub_task] == 1:
                                z_i_jk_l[
                                    other_rsu_idx_lp, rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp].setInitialValue(
                                    1)
                                z_i_jk_l[
                                    other_rsu_idx_lp, rsu_idx_lp, job_id_lp, sub_task, model_structure_idx_lp].fixValue()
        status, obj = self.lin_con(T_max, task_list, x_i_l, x_i_i_l, y_i_jk, z_i_jk_l, max_system_throughput, 0)
        print(self.get_all_task_num_all())
        return rsu_model_list_rr, rsu_job_list_rr, rsu_model_structure_list_rr, rsu_to_rsu_model_structure_list_rr, obj1

    def RR(self, x_i_l, x_i_i_l, y_i_jk, task_list):
        task_list_new = [[] for _ in range(self.get_all_task_num())]
        rsu_model_list = [[[0 for _ in range(model_util.Sub_model_num[i])] for i in range(len(model_util.Model_name))]
                          for _ in
                          range(self.rsu_num)]
        rsu_model_list_rr = {i: set() for i in range(self.rsu_num)}
        rsu_model_structure_list_rr = {i: set() for i in range(self.rsu_num + 1)}
        rsu_to_rsu_model_structure_list_rr = [[[0 for _ in range(len(model_util.Sub_Model_Structure_Size))]
                                               for _ in range(self.rsu_num)] for rsu_idx in range(self.rsu_num + 1)]
        for rsu_idx in self.rsu_structure_list.keys():
            for model_structure_idx in self.rsu_structure_list[rsu_idx]:
                rsu_to_rsu_model_structure_list_rr[rsu_idx][rsu_idx][model_structure_idx] = 1
        rsu_job_list_rr = [[] for rsu_idx in range(self.rsu_num)]
        rsu_job_list = [[[0 for _ in range(len(self.task_list[i]))] for i in range(len(self.task_list))]
                        for rsu_idx in range(self.rsu_num)]
        rsu_model_structure_list = [[0 for _ in range(len(model_util.Sub_Model_Structure_Size))]
                                    for _ in range(self.rsu_num + 1)]
        x_i_l_value_sorted = {}
        x_i_i_l_value_sorted = {}
        y_i_jk_value_sorted = {}
        for rsu_idx in range(self.rsu_num + 1):
            for model_structure_idx in self.model_structure_list:
                value = x_i_l[rsu_idx, model_structure_idx].value()
                key = "{}-{}".format(rsu_idx, model_structure_idx)
                x_i_l_value_sorted[key] = value

        for rsu_idx in range(self.rsu_num + 1):
            for other_rsu_idx in range(self.rsu_num):
                for model_structure_idx in self.model_structure_list:
                    value = x_i_i_l[rsu_idx, other_rsu_idx, model_structure_idx].value()
                    key = "{}-{}-{}".format(rsu_idx, other_rsu_idx, model_structure_idx)
                    x_i_i_l_value_sorted[key] = value

        for rsu_idx in range(self.rsu_num):
            for job_id in range(len(self.task_list)):
                for sub_task in range(len(self.task_list[job_id])):
                    value = y_i_jk[rsu_idx, job_id, sub_task].value()
                    key = "{}-{}-{}".format(rsu_idx, job_id, sub_task)
                    y_i_jk_value_sorted[key] = value

        y_i_jk_value_sorted = {k: v for k, v in sorted(y_i_jk_value_sorted.items(), key=lambda x: x[1], reverse=True)}
        x_i_l_value_sorted = {k: v for k, v in sorted(x_i_l_value_sorted.items(), key=lambda x: x[1], reverse=True)}
        x_i_i_l_value_sorted = {k: v for k, v in sorted(x_i_i_l_value_sorted.items(), key=lambda x: x[1], reverse=True)}

        for key in x_i_l_value_sorted.keys():
            value = x_i_l_value_sorted[key]
            if random.uniform(0, 1) <= value:
                info = key.split("-")
                rsu_idx = int(info[0])
                model_structure_idx = int(info[1])
                rsu_model_structure_list[rsu_idx][model_structure_idx] = 1
                if rsu_idx != self.rsu_num:
                    if model_structure_idx not in self.RSUs[rsu_idx].cached_model_structure_list:
                        rsu_model_structure_list_rr[rsu_idx].add(model_structure_idx)
            else:
                rsu_model_structure_list[rsu_idx][model_structure_idx] = 0

        for model_structure_idx in self.model_structure_list:
            rsu_model_structure_list[self.rsu_num][model_structure_idx] = 1

        for key in y_i_jk_value_sorted.keys():
            value = y_i_jk_value_sorted[key]
            info = key.split("-")
            rsu_idx = int(info[0])
            job_id = int(info[1])
            sub_task = int(info[2])
            if self.task_list[job_id][sub_task] not in task_list_new[job_id]:
                model_idx = self.task_list[job_id][sub_task]["model_idx"]
                sub_model_idx = self.task_list[job_id][sub_task]["sub_model_idx"]
                y_i_jk_value = y_i_jk[(rsu_idx, job_id, sub_task)].value()
                flag = 1
                for model_structure_idx in self.task_list[job_id][sub_task]["model_structure"]:
                    if rsu_model_structure_list[rsu_idx][model_structure_idx] == 0:
                        flag = 0
                if flag:
                    if random.uniform(0, 1) <= value:
                        rsu_job_list[rsu_idx][job_id][sub_task] = 1
                        rsu_job_list_rr[rsu_idx].append(self.task_list[job_id][sub_task])
                        task_list_new[job_id].append(self.task_list[job_id][sub_task])

        for key in x_i_i_l_value_sorted.keys():
            value = x_i_i_l_value_sorted[key]
            info = key.split("-")
            rsu_idx = int(info[0])
            other_rsu_idx = int(info[1])
            model_structure_idx = int(info[2])
            if rsu_model_structure_list[rsu_idx][model_structure_idx] == 1:
                key_x_i_l = "{}-{}".format(rsu_idx, model_structure_idx)
                if random.uniform(0, 1) <= (value / x_i_l_value_sorted[key_x_i_l]):
                    flag = 1
                    for rsu_idx_new in range(self.rsu_num + 1):  # 排除出现多个rsu给突然同一个rsu传输相同model structure
                        if rsu_to_rsu_model_structure_list_rr[rsu_idx_new][other_rsu_idx][
                            model_structure_idx] == 1:
                            flag = 0
                    if flag:
                        rsu_to_rsu_model_structure_list_rr[rsu_idx][other_rsu_idx][model_structure_idx] = 1
                        rsu_model_structure_list_rr[other_rsu_idx].add(model_structure_idx)
                        rsu_model_structure_list[other_rsu_idx][model_structure_idx] = 1

        rsu_model_structure_list_all = rsu_model_structure_list_rr
        for rsu_idx in range(self.rsu_num):
            for model_structure_idx in self.rsu_structure_list[rsu_idx]:
                rsu_model_structure_list_all[rsu_idx].add(model_structure_idx)

        for rsu_idx in range(self.rsu_num):
            for model_structure_idx in rsu_model_structure_list_rr[rsu_idx]:
                flag = 1
                for other_rsu_idx in range(self.rsu_num + 1):
                    if rsu_to_rsu_model_structure_list_rr[other_rsu_idx][rsu_idx][model_structure_idx] == 1:
                        flag = 0
                if flag:
                    download_time = 999999
                    for other_rsu_idx in range(self.rsu_num + 1):
                        if other_rsu_idx == rsu_idx:
                            continue
                        if model_structure_idx in rsu_model_structure_list_all[other_rsu_idx]:
                            download_time_temp = model_util.Sub_Model_Structure_Size[model_structure_idx] / (
                                self.RSUs[other_rsu_idx].rsu_rate
                                if other_rsu_idx != self.rsu_num else self.RSUs[rsu_idx].download_rate)
                            if download_time_temp <= download_time:
                                download_time = download_time_temp
                                offloading_rsu = other_rsu_idx

                    rsu_to_rsu_model_structure_list_rr[offloading_rsu][rsu_idx][rsu_idx] = 1

        return rsu_model_list_rr, rsu_to_rsu_model_structure_list_rr, rsu_model_structure_list_rr, rsu_job_list_rr, \
               rsu_model_list, rsu_model_structure_list, rsu_job_list

    def generate_task_to_structure(self):
        x_task_structure = [[[0 for _ in range(len(model_util.Sub_Model_Structure_Size))]
                             for _ in range(len(self.task_list[i]))] for i in range(self.get_all_task_num())]
        for task in self.task_list:
            for sub_task in task:
                task_job_id = sub_task["job_id"]
                task_structure = sub_task["model_structure"]
                sub_task_id = sub_task["sub_model_idx"]
                for task_structure_idx in task_structure:
                    x_task_structure[task_job_id][sub_task_id][task_structure_idx] = 1
        return x_task_structure

    def arr(self, T_max, task_list):
        rsu_model_list_rr, rsu_job_list_rr, rsu_model_structure_list_rr, rsu_to_rsu_model_structure_list_rr, \
        throughput = self.lin_pro(T_max, task_list)
        # self.clear_rsu_task_list()
        # self.allocate_task_for_rsu(rsu_job_list_rr)
        # self.allocate_model_for_rsu_later(rsu_model_list_rr)
        objective_value = self.cal_objective_value(rsu_to_rsu_model_structure_list_rr, rsu_model_list_rr,
                                                   rsu_job_list_rr, rsu_model_structure_list_rr)
        return throughput, objective_value

    def cal_objective_value(self, rsu_to_rsu_model_structure_list, rsu_model_list, rsu_job_list,
                            rsu_model_structure_list):
        obj = []
        for rsu_idx in range(self.rsu_num):
            obj.append(self.cal_single_rsu_obj(rsu_to_rsu_model_structure_list, rsu_model_list[rsu_idx],
                                               rsu_job_list[rsu_idx], rsu_model_structure_list[rsu_idx], rsu_idx))
        return max(obj)

    def cal_single_rsu_obj(self, rsu_to_rsu_model_structure_list, rsu_model_list, rsu_job_list,
                           rsu_model_structure_list, rsu_idx, is_Shared=True):
        def arrange_task():
            job_id_list_ = set()
            for task_ in rsu_job_list:
                job_id_list_.add(task_["job_id"])
            rsu_request_list_ = {i: [] for i in job_id_list_}
            for task_ in rsu_job_list:
                job_id_ = task_["job_id"]
                rsu_request_list_[job_id_].append(task_)
            return rsu_request_list_, job_id_list_

        rsu_caching_model_list = set()
        request_exec_time_list = []
        rsu_request_list, job_id_list = arrange_task()
        download_time = 0
        transmission_time = 0
        for job_id in job_id_list:
            task_exec_time_list = []
            request_exec_time = 0
            for task in rsu_request_list[job_id]:
                generated_id = task["rsu_id"]
                task_exec_time = self.RSUs[rsu_idx].latency_list[task["model_idx"]][task["sub_model_idx"]][
                    self.RSUs[rsu_idx].device_idx][task["seq_num"]]
                task_exec_time_list.append(task_exec_time)
                if generated_id != rsu_idx:
                    transmission_time += (model_util.get_model(task["model_idx"]).single_task_size / len(
                        self.task_list[job_id])) / self.RSUs[generated_id].rsu_rate
            if is_Shared:
                request_exec_time = max(task_exec_time_list)
            else:
                request_exec_time = sum(task_exec_time_list)
            request_exec_time_list.append(request_exec_time)
        for other_rsu_idx in range(self.rsu_num + 1):
            for model_structure_idx in rsu_caching_model_list:
                if rsu_to_rsu_model_structure_list[other_rsu_idx][rsu_idx][model_structure_idx] == 1:
                    if other_rsu_idx == self.rsu_num:
                        download_time += model_util.Sub_Model_Structure_Size[model_structure_idx] / self.RSUs[
                            rsu_idx].download_rate
                    else:
                        download_time += model_util.Sub_Model_Structure_Size[model_structure_idx] / self.RSUs[
                            other_rsu_idx].rsu_rate
        objective_value = download_time + sum(request_exec_time_list) + transmission_time
        return objective_value

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

    def cal_max_response_time(self, is_shared=True):
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
                    exec_time_sub_task = 0
                    model_idx = sub_task["model_idx"]
                    sub_model_idx = sub_task["sub_model_idx"]
                    rsu_cached_models = self.RSUs[rsu_idx].get_cached_model()
                    sub_task_model = model_util.get_model_name(model_idx, sub_model_idx)
                    if sub_task_model in rsu_cached_models:
                        exec_time = self.RSUs[rsu_idx].latency_list[model_idx][sub_model_idx][device_id][seq_num]
                        download_time = 0
                    else:
                        if is_shared:
                            task_model_size = self.RSUs[rsu_idx].get_caching_model_structure_size(
                                sub_task["model_structure"])
                        else:
                            task_model_size = model_util.get_model_sturctures_size(sub_task["model_structure"])
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
