# This is a sample Python script.
import random
import device
import model_util
from algo import Algo
from data import google_data_util
import matplotlib.pyplot as plt

random.seed(1023)

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

second_loop_num = 10
time_slot_list = [5, 7, 2, 4, 4, 2, 4, 2, 6, 7]
base_seed = 10086


def generate_rsu(rsu_num, device_ration, download_rate, rsu_rate, max_storage):
    return [device.RSU(device_ration, max_storage, download_rate, rsu_rate, rsu_num) for i in range(rsu_num)]


def record_file(x, res, description):
    with open("performance.txt", "a+") as f:
        f.write("{}\n".format(description))
        f.write("{}\n".format(x))
        f.write("{}\n".format(res))
        f.write("\n\n")


def tmp_record(data):
    with open("tmp.data.txt", "a+") as f:
        f.write("{}\n".format(data))


def out_line():
    import datetime
    with open("performance.txt", "a+") as f:
        f.write("-" * 100)
        f.write("\n")
        f.write("{}\n".format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S  %A')))
    # with open("tmp.data.txt", "a+") as f:
    #     f.write("-" * 100)
    #     f.write("\n")
    #     f.write("-" * 100)
    #     f.write("\n")
    #     f.write("{}\n".format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S  %A')))


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
            task_model_list = set()
            for task in RSUs[rand_rsu_id].task_list:
                for sub_task in task:
                    for model_structure_idx in sub_task["model_structure"]:
                        if model_structure_idx not in rand_model_structure_list and model_structure_idx not in RSUs[rand_rsu_id].model_structure_list:
                            task_model_list.add(model_structure_idx)
            task_model_structure_size = model_util.get_model_sturctures_size(task_model_list)
            rand_model_structure_size = model_util.get_model_sturctures_size(rand_model_structure_list)
            if rand_model_name not in rsu_model_list[rand_rsu_id] and rand_model_name not in model_list_all_selected \
                    and task_model_structure_size + rand_model_structure_size + RSUs[
                rand_rsu_id].get_total_model_size() + \
                    RSUs[rand_rsu_id].get_total_task_size() < RSUs[rand_rsu_id].storage_capacity:
                model_list_all_selected.append(rand_model_name)
                rsu_model_list[rand_rsu_id].append(rand_model_name)
                flag = 1
                RSUs[rand_rsu_id].add_model_structure(rand_model_structure_list)
    for rsu_idx in range(rsu_num):
        RSUs[rsu_idx].initial_model_structure_list = RSUs[rsu_idx].model_structure_list.copy()


def run_algo(device_ration=0.5, download_rate=550, rsu_rate=120, rsu_num=10, max_storage=1700, model_ration=9,
             latency_requiredment=3, seed=666, task_num=7):
    random.seed(seed)
    res = []
    RSUs = generate_rsu(rsu_num, device_ration, download_rate, rsu_rate, max_storage)
    task_list, sub_task_list = google_data_util.process_task(rsu_num, max_latency=latency_requiredment, filename=task_num)
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
    Algo_new = Algo(RSUs, task_list, sub_task_list, model_download_time_list)
    res.append(Algo_new.preference_coalition())
    # res.append(Algo_new.dqn())
    res.append(Algo_new.MA())
    # res.append(Algo_new.dqn())
    tmp_record(res)
    return res


def rsu_num_change():
    results = []
    x_list = []
    MA_res = []
    DQN_res = []
    Pre_coa = []
    for rsu_num in range(10, 31, 5):
        x_list.append(rsu_num)
        tmp_record("\nrsu_num_change, rsu_num: {}".format(rsu_num))
        res = []
        for seed in range(base_seed, base_seed + second_loop_num, 1):
            tmp = run_algo(rsu_num=rsu_num, seed=seed)
            if len(res) == 0:
                res = [0 for _ in tmp]
            for i in range(len(res)):
                res[i] += tmp[i]
        for i in range(len(res)):
            res[i] = res[i] / second_loop_num
        results.append(res)
        Pre_coa.append(res[0])
        DQN_res.append(res[1])
        MA_res.append(res[2])
    record_file(x_list, results, "rsu_num  ")
    plt.plot(x_list, MA_res, color='red', label='MA')
    # plt.plot(x_list, DQN_res, color='blue', label='DQN')
    plt.plot(x_list, Pre_coa, color='yellow', label='COALITION')
    plt.xlabel('rsu_num')
    plt.ylabel('Utility')
    plt.legend()
    plt.show()
    plt.savefig("pic/{}.png".format("rsu_num"))
    plt.clf()

def model_num_change():
    results = []
    x_list = []
    MA_res = []
    DQN_res = []
    DQN_res_ = []
    Pre_coa = []
    for model_num in range(1, 12, 2):
        x_list.append(model_num)
        tmp_record("\nModel_num_change, Model_num: {}".format(model_num))
        res = []
        for seed in range(base_seed, base_seed + second_loop_num, 1):
            tmp = run_algo(model_ration=model_num, seed=seed)
            if len(res) == 0:
                res = [0 for _ in tmp]
            for i in range(len(res)):
                res[i] += tmp[i]
        for i in range(len(res)):
            res[i] = res[i] / second_loop_num
        results.append(res)
        Pre_coa.append(res[0])
        DQN_res.append(res[1])
        MA_res.append(res[2])
        # DQN_res_.append(res[3])
    record_file(x_list, results, "Model_num  ")
    plt.plot(x_list, MA_res, color='red', label='MA')
    plt.plot(x_list, DQN_res, color='blue', label='DQN')
    plt.plot(x_list, Pre_coa, color='yellow', label='COALITION')
    plt.xlabel('Model_num')
    plt.ylabel('Utility')
    plt.legend()
    plt.show()
    plt.savefig("pic/{}.png".format("Model_num"))
    plt.clf()
    plt.show()
def download_change():
    results = []
    x_list = []
    MA_res = []
    DQN_res = []
    Pre_coa = []
    for download_rate in range(450, 551, 25):
        x_list.append(download_rate)
        tmp_record("\ndownload_change, download_rate: {}".format(download_rate))
        res = []
        for seed in range(base_seed, base_seed + second_loop_num, 1):
            tmp = run_algo(download_rate=download_rate, seed=seed)
            if len(res) == 0:
                res = [0 for _ in tmp]
            for i in range(len(res)):
                res[i] += tmp[i]
        for i in range(len(res)):
            res[i] = res[i] / second_loop_num
        results.append(res)
        Pre_coa.append(res[0])
        DQN_res.append(res[1])
        MA_res.append(res[2])
    record_file(x_list, results, "Download_rate  ")
    plt.plot(x_list, MA_res, color='red', label='MA')
    # plt.plot(x_list, DQN_res, color='blue', label='DQN')
    plt.plot(x_list, Pre_coa, color='yellow', label='COALITION')
    plt.xlabel('Download_rate')
    plt.ylabel('Utility')
    plt.legend()
    plt.show()
    plt.savefig("pic/{}.png".format("Download_rate"))
    plt.clf()


def latency_requirement():
    results = []
    x_list = []
    MA_res = []
    DQN_res = []
    Pre_coa = []
    for latency in range(10):
        max_latency = 0.15 + 0.05 * latency
        x_list.append(max_latency)
        tmp_record("\nlatency_requirement_change, latency_requirement: {}".format(max_latency))
        res = []
        for seed in range(base_seed, base_seed + second_loop_num, 1):
            tmp = run_algo(latency_requiredment=max_latency, seed=seed)
            if len(res) == 0:
                res = [0 for _ in tmp]
            for i in range(len(res)):
                res[i] += tmp[i]
        for i in range(len(res)):
            res[i] = res[i] / second_loop_num
        results.append(res)
        Pre_coa.append(res[0])
        DQN_res.append(res[1])
        MA_res.append(res[2])
    record_file(x_list, results, "latency_requirement  ")
    plt.plot(x_list, MA_res, color='red', label='MA')
    plt.plot(x_list, DQN_res, color='blue', label='DQN')
    plt.plot(x_list, Pre_coa, color='yellow', label='COALITION')
    plt.xlabel('Latency_requiredment')
    plt.ylabel('Utility')
    plt.legend()
    plt.show()
    plt.savefig("pic/{}.png".format("Latency_requiredment"))

def rsu_rate_change():
    results = []
    x_list = []
    MA_res = []
    DQN_res = []
    Pre_coa = []
    for rsu_rate in range(80, 121, 10):
        x_list.append(rsu_rate)
        tmp_record("\nrsu_rate_change, rsu_rate: {}".format(rsu_rate))
        res = []
        for seed in range(base_seed, base_seed + second_loop_num, 1):
            tmp = run_algo(rsu_rate=rsu_rate, seed=seed)
            if len(res) == 0:
                res = [0 for _ in tmp]
            for i in range(len(res)):
                res[i] += tmp[i]
        for i in range(len(res)):
            res[i] = res[i] / second_loop_num
        Pre_coa.append(res[0])
        DQN_res.append(res[1])
        MA_res.append(res[2])
    record_file(x_list, results, "rsu_rate  ")
    plt.plot(x_list, MA_res, color='red', label='MA')
    plt.plot(x_list, DQN_res, color='blue', label='DQN')
    plt.plot(x_list, Pre_coa, color='yellow', label='COALITION')
    plt.xlabel('rsu_rate')
    plt.ylabel('Utility')
    plt.legend()
    plt.show()
    plt.savefig("pic/{}.png".format("rsu_rate"))
    plt.clf()

def storage_change():
    results = []
    x_list = []
    MA_res = []
    DQN_res = []
    Pre_coa = []
    for max_storage in range(10):
        max_storage = 300 + 50 * max_storage
        x_list.append(max_storage)
        tmp_record("\nmax_storage_change, max_storage: {}".format(max_storage))
        res = []
        for seed in range(base_seed, base_seed + second_loop_num, 1):
            tmp = run_algo(max_storage=max_storage, seed=seed)
            if len(res) == 0:
                res = [0 for _ in tmp]
            for i in range(len(res)):
                res[i] += tmp[i]
        for i in range(len(res)):
            res[i] = res[i] / second_loop_num
        results.append(res)
        Pre_coa.append(res[0])
        DQN_res.append(res[1])
        MA_res.append(res[2])
    record_file(x_list, results, "max_storage  ")
    plt.plot(x_list, MA_res, color='red', label='MA')
    plt.plot(x_list, DQN_res, color='blue', label='DQN')
    plt.plot(x_list, Pre_coa, color='yellow', label='COALITION')
    plt.xlabel('max_storage')
    plt.ylabel('Utility')
    plt.legend()
    plt.show()
    plt.savefig("pic/{}.png".format("max_storage"))
    plt.clf()

def time_slot_change():
    results = []
    x_list = []
    MA_res = []
    DQN_res = []
    Pre_coa = []
    time_slot = 610
    for task_num in time_slot_list:
        x_list.append(time_slot)
        tmp_record("\ntime_slot_change, time_slot: {}".format(time_slot))
        res = []
        for seed in range(base_seed, base_seed + second_loop_num, 1):
            tmp = run_algo(task_num=task_num, seed=seed)
            if len(res) == 0:
                res = [0 for _ in tmp]
            for i in range(len(res)):
                res[i] += tmp[i]
        for i in range(len(res)):
            res[i] = res[i] / second_loop_num
        time_slot += 10
        results.append(res)
        Pre_coa.append(res[0])
        DQN_res.append(res[1])
        MA_res.append(res[2])
    record_file(x_list, results, "time_slot  ")
    plt.plot(x_list, MA_res, color='red', label='MA')
    # plt.plot(x_list, DQN_res, color='blue', label='DQN')
    plt.plot(x_list, Pre_coa, color='yellow', label='COALITION')
    plt.xlabel('time_slot')
    plt.ylabel('Utility')
    plt.legend()
    plt.show()
    plt.savefig("pic/{}.png".format("time_slot"))
    plt.clf()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    out_line()
    # model_num_change()
    # latency_requirement()
    storage_change()
    # rsu_rate_change()
    # download_change()
    # rsu_num_change()
    # time_slot_change()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
