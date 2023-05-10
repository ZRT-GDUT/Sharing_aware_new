"""
定义论文中的所有设备
目前文章只使用RSU。
"""
import random
from typing import List

import model_util


class RSU:
    def __init__(self, device_ration=0.5, max_storage=1200, download_rate=None, rsu_rate=None):
        # transmission rate
        self.seq_num = [[1 for _ in range(model_util.Sub_model_num[i])] for i in range(len(model_util.Model_name))]
        if download_rate is None:
            self.download_rate = random.uniform(450, 550)  # Mbps
        else:
            self.download_rate = download_rate
        if rsu_rate is None:
            self.rsu_rate = random.uniform(80, 120)  # Mbps
        else:
            self.rsu_rate = rsu_rate
        # computation
        self.gpu_idx = -1  # -1, no-gpu, 0, 1: gpu_type_idx
        if random.uniform(0, 1) < device_ration:  #
            self.gpu_idx = get_device_id(random.randint(0, 1), is_gpu=True)
            self.device_idx = self.gpu_idx
            self.has_gpu = True
            self.has_cpu = False
        else:
            self.cpu_idx = get_device_id(random.randint(0, 1), is_gpu=False)
            self.device_idx = self.cpu_idx
            self.has_gpu = False
            self.has_cpu = True
        self.trans_cpu_gpu = 16 * 1024  # Gbps
        # storage
        self.storage_capacity = random.uniform(300, max_storage)  # to do ...
        # task
        self.task_list = []
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

    def add_task(self, task_size, exec_latency):  # 计算add一个task之后的queue_latency，以及rsu存储的task size
        # self.queue_latency += exec_latency
        self.task_size += task_size

    def satisfy_caching_constraint(self, task_size=0):
        return self.storage_capacity > self.cal_caching_size() + task_size

    def get_rsu_cached_model_size(self, is_share=True):  # 获取已经缓存的模型大小
        model_size = 0
        model_layers = set()
        for model_name in self.__caching_model_list:
            model_idx, sub_model_idx = model_util.get_model_info(model_name)
            model = model_util.get_model(model_idx)
            sub_model_layers = set(model.require_sub_model_all[sub_model_idx])
            if is_share:
                model_layers.add(sub_model_layers)
            else:
                for sub_model_layer in sub_model_layers:
                    model_size += model.sub_model_size[sub_model_layer]
        if is_share:
            for sub_model_layer in model_layers:
                model_size += model_util.Sub_Model_Structure_Size[sub_model_layer]
        return model_size

    def add_model(self, model_idx, sub_model_idx):  # 为rsu添加模型
        model_name = model_util.get_model_name(model_idx, sub_model_idx)
        self.__caching_model_list.add(model_name)

    def remove_model(self, model_idx, sub_model_idx):  # 移除模型
        model_name = model_util.get_model_name(model_idx, sub_model_idx)
        self.__caching_model_list.remove(model_name)

    def get_surplus_size(self):  # 获得rsu的剩余存储空间
        return self.storage_capacity - self.get_rsu_cached_model_size() - self.task_size

    def get_cached_model(self) -> set:
        return self.__caching_model_list

    def get_caching_model_size(self, cache_model):  # 计算需要缓存的模型大小
        models = {}
        model_idx, sub_model_idx = model_util.get_model_info(cache_model)
        models[model_idx] = {sub_model_idx}
        model_size = 0
        for model_idx in models.keys():
            model_idxs = models[model_idx]  # 每个model_idx对应的sub_model
            model = model_util.get_model(model_idx)
            model_size += model.require_model_size(model_idxs, is_share=True)
        return model_size

    def cal_extra_caching_size(self, model_idx, sub_models: List[int]):
        """
        calculate the cache size when model_idx[sub_models] are added.
        :param model_idx:
        :param sub_models:
        :param is_gpu:
        :return:
        """
        pre_model_size = self.get_rsu_cached_model_size()  # 获得已缓存模型的size
        models = self.get_cached_model()  # 获取已缓存模型
        for sub_model_idx in sub_models:
            model_name = model_util.get_model_name(model_idx, sub_model_idx)
            models.add(model_name)
        after_model_size = self.get_caching_models_size(models)
        return after_model_size - pre_model_size

    def add_all_sub_model(self, model_idx, sub_models: List[int], is_gpu=False) -> List[int]:
        """
        :param model_idx:
        :param sub_models:
        :param is_gpu:
        :return: the new added sub_model_idx
        """
        add_success_models = []
        for sub_model_idx in sub_models:
            if self.add_model(model_idx, sub_model_idx, is_gpu=is_gpu):
                add_success_models.append(sub_model_idx)
        return add_success_models

    def add_model(self, model_idx, sub_model_idx):
        """
        :param model_idx:
        :param sub_model_idx:
        :param is_gpu:
        :return: true-> add a new model, false-> model has been added.
        """
        model_name = model_util.get_model_name(model_idx, sub_model_idx)
        if self.has_gpu:
            size = len(self.__caching_model_list)
            self.__caching_model_list.add(model_name)  # 没有理解，意思是gpu有了，cpu有一样的model就要删除吗
            # if self.has_model(model_idx, sub_model_idx):
            #     self.remove_model(model_idx, sub_model_idx)
            return len(self.__caching_model_list) - size != 0
        else:
            size = len(self.__caching_model_list)  # 为什么这里是获得gpu的model数量
            self.__caching_model_list.add(model_name)
            # if self.has_model(model_idx, sub_model_idx):
            #     self.remove_model(model_idx, sub_model_idx)
            return len(self.__caching_model_list) - size != 0

    def has_model(self, model_idx, sub_model_idx):
        model_name = model_util.get_model_name(model_idx, sub_model_idx)
        if self.has_gpu:
            return model_name in self.__caching_model_list_gpu
        else:
            return model_name in self.__caching_model_list

    def remove_add_models(self, model_idx: int, sub_models: List[int], is_gpu=False):
        """
        remove model_idx-[sub_models] in RSU
        :param model_idx:
        :param sub_models:
        :return:
        """
        for sub_model_idx in sub_models:
            self.remove_model(model_idx, sub_model_idx, is_gpu=is_gpu)

    def has_all_model(self, sub_models) -> bool:
        """
        :return the result whether RSU caches all sub_models....
        :param sub_models:
        :return:
        """
        return len(sub_models - self.__caching_model_list - self.__caching_model_list_gpu) == 0

    def can_executed(self, model_idx, sub_models):
        """
        return the result whether the task can be executed by the RSU
        :param model_idx:
        :param sub_models:
        :return:
        """
        s = set()
        for sub_model_idx in sub_models:
            s.add(model_util.get_model_name(model_idx, sub_model_idx))
        return self.has_all_model(s)

    def get_add_models(self):
        return self.get_cached_model(is_gpu=True).union(self.get_cached_model(is_gpu=False))

    def get_model_idx_series(self, model_idx, is_gpu=False) -> set:
        """
        get the model_idx series...
        if model_idx = 1, and the cached_model is '1-1', '0-2', '0-3', '1-3',
        then the result is set(1, 3).
        :param model_idx: model_series_idx
        :param is_gpu:  in cpu or gpu
        :return:
        """
        sub_model_idxs = set()
        if is_gpu:
            cached_model = self.__caching_model_list_gpu
        else:
            cached_model = self.__caching_model_list
        for model_info in cached_model:
            _model_idx, sub_model_idx = model_util.get_model_info(model_info)
            if model_idx == _model_idx:
                sub_model_idxs.add(sub_model_idx)
        return sub_model_idxs

    def clear_cached_model(self):
        self.__caching_model_list.clear()

    def rebuild_model_list(self, model_list):
        self.__caching_model_list = model_list

def get_device_id(device_id, is_gpu=False, gpu_num=2):
    if is_gpu:
        return device_id
    else:
        return device_id + gpu_num
