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
    model_list_all = model_util.model_list_all
    for model_num in range(model_ration):
        flag = 0
        while flag == 0:
            rand_rsu_id = random.randint(0, rsu_num - 1)
            rand_model_name_idx = random.randint(0, len(model_list_all) - 1)
            rand_model_name = model_list_all[rand_model_name_idx]
            rand_rsu_model_list = RSUs[rand_rsu_id].get_cached_model()
            rand_model_idx, rand_sub_model_idx = model_util.get_model_info(rand_model_name)
            if rand_model_name not in rand_rsu_model_list:
                RSUs[rand_rsu_id].add_model(rand_model_idx, rand_sub_model_idx)
                caching_size, cached_model_structure_list = RSUs[rand_rsu_id].get_rsu_cached_model_size(is_share=True)
                if caching_size <= RSUs[rand_rsu_id].storage_capacity:
                    rand_model_id, rand_sub_model_id = model_util.get_model_info(rand_model_name)
                    RSUs[rand_rsu_id].add_model(rand_model_id, rand_sub_model_id)
                    for model_structure_idx in cached_model_structure_list:
                        RSUs[rand_rsu_id].cached_model_structure_list.add(model_structure_idx)
                    model_list_all.remove(rand_model_name)
                    flag = 1
                else:
                    RSUs[rand_rsu_id].remove_model(rand_model_idx, rand_sub_model_idx)


def run_algo(device_ration=0.5, download_rate=120, rsu_rate=100, rsu_num=20, max_storage=1200, model_ration=6):
    result = []
    RSUs = generate_rsu(rsu_num, device_ration, download_rate, rsu_rate, max_storage)
    init_model_deploy(model_ration, rsu_num, RSUs)
    task_list = google_data_util.process_task(rsu_num)
    Algo_new = Algo(RSUs, task_list)
    objective_value = Algo_new.iarr(task_list)
    return objective_value


def rsu_num_change():
    res = []
    for rsu_num in range(5, 31, 5):
        res.append(run_algo(rsu_num=rsu_num))
    print(res)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rsu_num_change()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
