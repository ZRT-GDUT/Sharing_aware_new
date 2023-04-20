# This is a sample Python script.
import device
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

def generate_rsu(rsu_num, device_ration, download_rate, rsu_rate, rsu_num, max_storage):
    return [device.RSU(device_ration, max_storage, download_rate, rsu_rate) for i in range(rsu_num)]

def run_algo(device_ration=0.5, download_rate=120, rsu_rate=100, rsu_num=20, max_storage=200):
    RSUs = generate_rsu(rsu_num, device_ration, download_rate, rsu_rate, rsu_num, max_storage)
    Algo_new = Algo(RSUs)
    task_list = google_data_util.process_task(rsu_num)
    Algo_new.iarr(task_list)



def rsu_num_change():
    for rsu_num in range(10, 31, 5):
        run_algo(rsu_num=rsu_num)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_date()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
