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
    return [device.RSU(device_ration, max_storage, download_rate, rsu_rate) for i in range(rsu_num)]


def run_algo(device_ration=0.5, download_rate=120, rsu_rate=100, rsu_num=20, max_storage=200):
    RSUs = generate_rsu(rsu_num, device_ration, download_rate, rsu_rate, max_storage)
    flag = 0
    while flag == 0:
        init_model_deploy = random.uniform(0, 1)
        rand_rsu_id = random.randint(0, rsu_num - 1)
        rand_model_index = random.randint(0, len(model_util.Model_name) - 1)
        rand_sub_model_index = random.randint(0, model_util.Sub_model_num[rand_model_index] - 1)
        model = model_util.get_model(rand_model_index)
        size = 0
        if init_model_deploy <= 1 / 2:
            # 部署大模型
            for sub_model_idx in range(model_util.Sub_model_num[rand_model_index]):
                for model_structure_idx in model.require_sub_model_all[sub_model_idx]:
                    size = size + model_util.Sub_Model_Structure_Size[model_structure_idx]
            if RSUs[rand_rsu_id].storage_capacity > size:
                flag = 1
                for sub_model_idx in range(model_util.Sub_model_num[rand_model_index]):
                    RSUs[rand_rsu_id].add_model(rand_model_index, sub_model_idx)  # 默认部署在cpu
        else:
            # 部署小模型
            for model_structure_idx in model.require_sub_model_all[rand_sub_model_index]:
                size = size + model_util.Sub_Model_Structure_Size[model_structure_idx]
            if RSUs[rand_rsu_id].storage_capacity > size:
                RSUs[rand_rsu_id].add_model(rand_model_index, rand_sub_model_index)
                flag = 1

    Algo_new = Algo(RSUs)
    task_list = google_data_util.process_task(rsu_num)
    print(RSUs[0].latency_list[1][0][2][5])
    Algo_new.iarr(task_list)


def rsu_num_change():
    for rsu_num in range(10, 31, 5):
        run_algo(rsu_num=rsu_num)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rsu_num_change()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
