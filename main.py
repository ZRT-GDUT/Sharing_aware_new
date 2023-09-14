# This is a sample Python script.
import random

import device
import model_util
from algo import Algo
from data import google_data_util


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_date():
    import datetime
    with open("performance.txt", "a+") as f:
        f.write("-" * 100)
        f.write("\n")
        f.write("{}\n".format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S  %A')))
    with open("tmp.data.txt", "a+") as f:
        f.write("-" * 100)
        f.write("\n")
        f.write("-" * 100)
        f.write("\n")
        f.write("{}\n".format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S  %A')))


def generate_rsu(rsu_num, device_ration, download_rate, rsu_rate, max_storage):
    return [device.RSU(device_ration, max_storage, download_rate, rsu_rate, rsu_num) for i in range(rsu_num)]


def init_model_deploy(model_ration, rsu_num, RSUs):
    model_list_all = ["0-0", "0-1", "0-2", "0-3", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "2-3"]
    model_list_all_selected = []
    rsu_model_list = [[] for _ in range(rsu_num)]
    for model_num in range(model_ration):
        flag = 0
        while flag == 0:
            rand_rsu_id = random.randint(0, rsu_num - 1)
            rand_model_name_idx = random.randint(0, len(model_list_all) - 1)
            rand_model_name = model_list_all[rand_model_name_idx]
            rand_model_idx, rand_sub_model_idx = model_util.get_model_info(rand_model_name)
            rand_model = model_util.get_model(rand_model_idx)
            rand_model_structure_list = rand_model.require_sub_model_all[rand_sub_model_idx]
            rand_model_structure_size = 0
            task_model_structure_size = 0
            task_model_list = set()
            for task in RSUs[rand_rsu_id].task_list:
                for sub_task in task:
                    for model_structure_idx in sub_task["model_structure"]:
                        if model_structure_idx not in rand_model_structure_list:
                            task_model_list.add(model_structure_idx)
            task_model_structure_size = model_util.get_model_sturctures_size(task_model_list)
            rand_model_structure_size = model_util.get_model_sturctures_size(rand_model_structure_list)
            if rand_model_name not in rsu_model_list[rand_rsu_id] and rand_model_name not in model_list_all_selected \
                and task_model_structure_size + rand_model_structure_size + RSUs[rand_rsu_id].get_total_model_size() + \
                    RSUs[rand_rsu_id].get_total_task_size() < RSUs[rand_rsu_id].storage_capacity:
                model_list_all_selected.append(rand_model_name)
                rsu_model_list[rand_rsu_id].append(rand_model_name)
                flag = 1
                RSUs[rand_rsu_id].add_model_structure(rand_model_structure_list)
    for rsu_idx in range(rsu_num):
        RSUs[rsu_idx].initial_model_structure_list = RSUs[rsu_idx].model_structure_list.copy()


def run_algo(device_ration=0.5, download_rate=120, rsu_rate=100, rsu_num=20, max_storage=1200, model_ration=6):
    result = []
    RSUs = generate_rsu(rsu_num, device_ration, download_rate, rsu_rate, max_storage)
    task_list = google_data_util.process_task(rsu_num)
    for task in task_list:
        rsu_id = task[0]["rsu_id"]
        RSUs[rsu_id].add_task(task)
    init_model_deploy(model_ration, rsu_num, RSUs)
    model_download_time_list = {}
    for model_structure_idx in range(len(model_util.Sub_Model_Structure_Size)):
        model_download_time = 99999
        model_download_rsu = -1
        for rsu_idx in range(rsu_num):
            if model_structure_idx in RSUs[rsu_idx].model_structure_list:
                model_download_time_current = model_util.Sub_Model_Structure_Size[model_structure_idx] / \
                                              RSUs[rsu_idx].rsu_rate
                if model_download_time_current < model_download_time:
                    model_download_time = model_download_time_current
                    model_download_rsu = rsu_idx
        if model_download_rsu >= 0:
            model_download_time_list[model_structure_idx] = model_download_rsu
    Algo_new = Algo(RSUs, task_list, model_download_time_list)
    objective_value = Algo_new.MA(task_list)
    return objective_value


def rsu_num_change():
    result = run_algo(rsu_num=5)
    print("目标值:", result)
    # res = []
    # for rsu_num in range(5, 31, 5):
    #     res.append(run_algo(rsu_num=rsu_num))
    # print(res)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rsu_num_change()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
