from typing import List

import device
import model_util


class Algo:
    def __init__(self, RSUs: List[device.RSU]):
        self.RSUs = RSUs
        self.rsu_num = len(RSUs)

    def iarr(self, task_list):
        self.allocate_task_for_rsu(task_list)
        T_max = self.cal_max_response_time()

    def allocate_task_for_rsu(self, task_list):
        for task in range(task_list):
            rsu_id = task["rsu_id"]
            self.RSUs[rsu_id].task_list.append(task)

    def get_task_model_list(self, model_idx, sub_models):
        task_models = set()
        for sub_model in sub_models:
            model_name = model_util.get_model_name(model_name, sub_model)
            task_models.add(model_name)
        return task_models

    def cal_max_response_time(self):
        T_max = 0
        total_time_list = []  # 记录每个RSU的执行时间
        for rsu_idx in self.rsu_num:
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
                total_time = download_time + exec_time
            total_time_list.append(total_time)
            self.RSUs[rsu_idx].seq_num = [[0 for _ in range(model_util.Sub_model_num[i])] for i in
                                          range(len(model_util.Model_name))]
        T_max = max(total_time_list)
        return T_max
