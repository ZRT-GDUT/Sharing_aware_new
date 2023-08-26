"""
定义论文中的所有设备
目前文章只使用RSU。
"""
import random
from typing import List

import model_util


class RSU:
    def __init__(self, device_ration=0.5, max_storage=1200, download_rate=None, rsu_rate=None, rsu_num=10):
        # transmission rate
        self.seq_num = [[1 for _ in range(model_util.Sub_model_num[i])] for i in range(len(model_util.Model_name))]
        self.cached_model_structure_list = set()
        if download_rate is None:
            self.download_rate = random.uniform(450, 550) / rsu_num  # Mbps
        else:
            self.download_rate = (download_rate / rsu_num)
        if rsu_rate is None:
            self.rsu_rate = random.uniform(80, 120)  # Mbps
        else:
            self.rsu_rate = rsu_rate
        # computation
        self.gpu_idx = -1  # -1, no-gpu, 0, 1: gpu_type_idx
        if random.uniform(0, 1) < device_ration:  #
            self.gpu_idx = self.get_device_id(random.randint(0, 1), is_gpu=True)
            self.device_idx = self.gpu_idx
            self.has_gpu = True
            self.has_cpu = False
        else:
            self.cpu_idx = self.get_device_id(random.randint(0, 1), is_gpu=False)
            self.device_idx = self.cpu_idx
            self.has_gpu = False
            self.has_cpu = True
        self.trans_cpu_gpu = 16 * 1024  # Gbps
        # storage
        self.storage_capacity = random.uniform(300, max_storage)  # to do ...
        # task
        self.task_list = []
        self.model_structure_list = set()
        self.__caching_model_list = set()  # data: get_model_name(model_idx, sub_model_idx)
        self.task_size = 0
        self.latency_list = [
            # model 1
            [
                #  每一个元素中有四项分别代表着使用哪一个cpu或gpu
                [  # 0
                    [0.1973605, 0.1982527, 0.1977889, 0.2009442, 0.2027571, 0.204893, 0.1968997, 0.2066418, 0.1978363,
                     0.1971634],
                    [0.2299988, 0.2331354, 0.2455341, 0.2345285, 0.2314658, 0.2365028, 0.2595511, 0.228322, 0.236145,
                     0.2300552],
                    [0.1645303, 0.0192916, 0.0140529, 0.0140292, 0.0140198, 0.0140286, 0.0141624, 0.0140435, 0.0140469,
                     0.0140267],
                    [0.1645303, 0.0192916, 0.0140529, 0.0140292, 0.0140198, 0.0140286, 0.0141624, 0.0140435, 0.0140469,
                     0.0140267]
                ],
                [  # 1
                    [0.20755, 0.1955544, 0.191715, 0.2003362, 0.1924398, 0.1957741, 0.1987331, 0.1997689, 0.2000075,
                     0.1947023],
                    [0.2192278, 0.2236783, 0.239304, 0.2314798, 0.2267759, 0.2299196, 0.2502516, 0.2345959, 0.2361438,
                     0.2236564],
                    [0.197561, 0.0219852, 0.0196325, 0.0196459, 0.0190636, 0.018966, 0.0189843, 0.0188254, 0.018835,
                     0.0187074],
                    [0.1770275, 0.0171917, 0.0130068, 0.0132736, 0.0132129, 0.0132522, 0.0131701, 0.0133126, 0.0130964,
                     0.012643]
                ],
                [  # 2
                    [0.2110046, 0.206003, 0.2037366, 0.2111136, 0.2041781, 0.207264, 0.2106943, 0.2120338, 0.2119275,
                     0.205587],
                    [0.22227, 0.2344547, 0.2515713, 0.2433699, 0.2384606, 0.2423721, 0.2629923, 0.246728, 0.2486427,
                     0.2352069],
                    [0.1973163, 0.0200291, 0.0155259, 0.0154012, 0.0153429, 0.0153581, 0.0154425, 0.0152935, 0.0152867,
                     0.0149271],
                    [0.1767534, 0.0174833, 0.0132263, 0.013547, 0.0134845, 0.0135205, 0.0134498, 0.0135853, 0.0133631,
                     0.0128731]
                ],
                [  # 3
                    [0.39929, 0.3936066, 0.4082943, 0.417456, 0.3996705, 0.4069022, 0.4039221, 0.4075282, 0.4075422,
                     0.4049913],
                    [0.4576806, 0.482701, 0.4764581, 0.4748913, 0.4733252, 0.4858324, 0.4780169, 0.478016, 0.4717729,
                     0.4905155],
                    [0.1855786, 0.0416556, 0.039362, 0.041933, 0.041193, 0.0405013, 0.0407579, 0.0397856, 0.0399682,
                     0.0396941],
                    [0.1545, 0.0244677, 0.0191376, 0.0187591, 0.0188785, 0.018762, 0.0188769, 0.0188578, 0.0188102,
                     0.0187884]
                ]
            ],
            # model 2
            [
                [
                    [1.345184, 1.3951448, 1.4360982, 1.4330647, 1.4047035, 1.4245633, 1.4154638, 1.4279191, 1.4171983,
                     1.4255445],
                    [1.8276929, 1.9026842, 1.9229911, 1.908936, 1.8933103, 1.8902004, 1.9151701, 1.9089415, 1.9089346,
                     1.9151921],
                    [0.2519119, 0.061573, 0.0494595, 0.0479253, 0.0477197, 0.0479401, 0.048034, 0.0473943, 0.0468735,
                     0.0468108],
                    [0.1915035, 0.0522405, 0.0440092, 0.0438338, 0.0436341, 0.0434896, 0.043353, 0.0433929, 0.0431076,
                     0.0469716]
                ],
                [
                    [1.4169746, 1.4624698, 1.4799298, 1.4960632, 1.4808722, 1.4937343, 1.4921776, 1.4886177, 1.5030005,
                     1.492997],
                    [1.8417474, 1.9136175, 1.968292, 1.9511149, 1.9386106, 1.9417362, 1.9370469, 1.9511076, 1.9495515,
                     1.9401691],
                    [0.2264391, 0.0684185, 0.0541684, 0.0519295, 0.0516059, 0.0510713, 0.0556128, 0.0530082, 0.0519835,
                     0.0517727],
                    [0.1860892, 0.0539773, 0.0441473, 0.0439235, 0.045913, 0.0453405, 0.0453362, 0.0456352, 0.0453773,
                     0.0441715]
                ],
                [
                    [1.799822, 1.8847428, 1.9282357, 1.9231366, 1.9174413, 1.9508356, 1.9413683, 1.9318613, 1.9550918,
                     1.956064],
                    [2.5442903, 2.6396756, 2.7175733, 2.678594, 2.6534723, 2.6803657, 2.6933662, 2.6867981, 2.70779,
                     2.7169834],
                    [0.2387834, 0.086218, 0.0657115, 0.0656308, 0.0653607, 0.0670349, 0.0665059, 0.0666279, 0.0665136,
                     0.0665525],
                    [0.2099016, 0.070507, 0.0568322, 0.0567594, 0.0569891, 0.0568056, 0.0564236, 0.0565652, 0.0561634,
                     0.056785]
                ]
            ],
            # model 3
            [
                [  # 2
                    [0.135990858, ] * 10,
                    [0.201506972, ] * 10,
                    [0.194757056, 0.045601606, 0.043881512, 0.042294717, 0.041257095, 0.040289712, 0.041384459,
                     0.04009223,
                     0.042858768, 0.039830589],
                    [0.181886792, 0.044860005, 0.0440382, 0.042821765, 0.041881132, 0.039645624, 0.042106056,
                     0.041171479,
                     0.03931706, 0.04142592]
                ],
                [  # 3
                    [0.13459971, ] * 10,
                    [0.204626083, ] * 10,
                    [0.189062619, 0.048365259, 0.043605471, 0.04241128, 0.041698623, 0.041868114, 0.041815138,
                     0.040962243,
                     0.041610646, 0.039768434],
                    [0.182476449, 0.044616342, 0.043532491, 0.043518043, 0.040133238, 0.040185833, 0.041477394,
                     0.040636206,
                     0.039343548, 0.041982532]
                ],
                [  # 4
                    [0.152732325, ] * 10,
                    [0.201511741, ] * 10,
                    [0.204985499, 0.042389655, 0.040116858, 0.038677812, 0.039412808, 0.037670755, 0.038818765,
                     0.038382888,
                     0.038452697, 0.03672843],
                    [0.195576763, 0.040828538, 0.03947804, 0.039878082, 0.038290191, 0.038246274, 0.038030124,
                     0.037859988,
                     0.036315417, 0.037552762]
                ],
                [  # 5
                    [0.137135577, ] * 10,
                    [0.1890131, ] * 10,
                    [0.19721477, 0.040685987, 0.039271116, 0.037487793, 0.036696982, 0.036797166, 0.035172009,
                     0.035724592,
                     0.035730457, 0.034275293],
                    [0.18819325, 0.041623545, 0.038406134, 0.037783098, 0.036522651, 0.036203599, 0.036601329,
                     0.03701334,
                     0.034984875, 0.035346556]
                ]
            ]
        ]

    def get_total_task_size(self):
        task_size_total = 0
        for task in self.task_list:
            model_idx = task[0]["model_idx"]
            task_model = model_util.get_model(model_idx)
            task_size = task_model.single_task_size
            task_size_total += task_size
        return task_size_total

    def add_task(self, task):
        if self.satisfy_add_task_constraint(task):
            self.task_list.add(task)
            return True
        else:
            return False

    def remove_task(self, task):
        if task in self.task_list:
            self.task_list.remove(task)
        else:
            print("task不在队列中")

    def add_model_structure(self, model_structure):
        if self.satisfy_add_model_structure_constraint(model_structure):
            self.model_structure_list.add(model_structure)
            return True
        else:
            return False

    def remove_model_structure(self, model_structure):
        if model_structure in self.model_structure_list:
            self.model_structure_list.remove(model_structure)
        else:
            print(model_structure, "不在缓存列表中")

    def get_total_model_size(self):
        model_total_size = 0
        for model_structure_idx in self.model_structure_list:
            model_total_size += model_util.Sub_Model_Structure_Size[model_structure_idx]
        return model_total_size

    def satisfy_add_model_structure_constraint(self, model_structure_list):
        model_total_size = self.get_total_model_size()
        task_total_size = self.get_total_task_size()
        add_model_size = model_util.get_model_sturctures_size(model_structure_list)
        if task_total_size + add_model_size + model_total_size > self.storage_capacity:
            return False
        else:
            return True

    def satisfy_add_task_constraint(self, task):
        model_idx = task["model_idx"]
        task_model = model_util.get_model(model_idx)
        task_size = task_model.single_task_size
        if task_size + self.get_total_model_size() + self.get_total_task_size() <= self.storage_capacity:
            self.task_list.add(task)
            return True
        else:
            return False

    def has_model_structure(self, model_structure_list):
        not_added_model = set()
        for model_structure_idx in model_structure_list:
            if model_structure_idx not in self.model_structure_list:
                not_added_model.add(model_structure_idx)
        return not_added_model

    def get_device_id(self, device_id, is_gpu=False, gpu_num=2):
        if is_gpu:
            return device_id
        else:
            return device_id + gpu_num
