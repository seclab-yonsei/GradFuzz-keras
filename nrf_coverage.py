import numpy as np
from collections import OrderedDict
from colorama import Fore, Back, Style
    

class NRFCoverage:

    def scale(self, layer_outputs, rmax=1, rmin=0):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_outputs: the layer's output tensor
        :param rmax: the upper bound of scale
        :param rmin: the lower bound of scale
        :return:
        '''
        divider = (layer_outputs.max() - layer_outputs.min())
        if divider == 0:
            return np.zeros(shape=layer_outputs.shape)
        X_std = (layer_outputs - layer_outputs.min()) / divider
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled


    def vectorCoverage(self, coverages, labels):
        return self.coverageFunction2(coverages, labels)


    def bitCoverage(self, coverages, labels):
        batch_num = len(coverages[0])
        ptr = np.tile(np.zeros(self.total_size, dtype=np.uint8), (batch_num,1))
        for idx in range(len(coverages)-1, 0, -1):
            if idx in self.layer_not_to_compute:
                del coverages[idx]
        self.coverageFunction2(coverages, ptr)
        return ptr


    def gv_coverage_function(self, gradient_batches, labels):
        gradients = []
        for gradient in gradient_batches:
            tempgradient = []
            for layer in gradient:
                layer = np.ravel(layer)
                tempgradient.extend(layer)
            flattenGradient = np.array(tempgradient).flatten()
            gradients.append(flattenGradient)
        return gradients


    def gv_partial_coverage_function(self, gradient_batches, labels):
        gradients = []
        for index in range(len(gradient_batches)):
            tempgradient = []
            for layer in gradient_batches[index]:
                layer = np.transpose(layer)[labels[index]]
                layer = np.ravel(layer)
                tempgradient.extend(layer)
            flattenGradient = np.array(tempgradient).flatten()
            gradients.append(flattenGradient)
        return gradients


    def av_coverage_function(self, activation_dict, labels):
        '''
            레이어 전체를 볼때 배치별로 묶이지 않으니깐 코드 다시 고쳐야됨
            어차피 사이즈 커서 모든 레이어를 못보긴 함
        '''

        batchsize = len(activation_dict[0])
        coverage_list = [[] for i in range(batchsize)]

        for i in range(len(activation_dict)) :
            for batchnum in range(len(activation_dict[i])):
                coverage_list[batchnum].extend(activation_dict[i][batchnum].flatten())

        return np.array(coverage_list)


    def kmnc_update_coverage(self, outputs, ptr):

        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]

            for seed_id, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(layer_output.shape[-1]):
                    profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]

                    lower_bound = profiling_data_list[3]
                    upper_bound = profiling_data_list[4]

                    unit_range = (upper_bound - lower_bound) / self.k

                    output = np.mean(layer_output[..., neuron_idx])

                    # the special case, that a neuron output profiling is a fixed value
                    # TODO: current solution see whether test data cover the specific value
                    # if it covers the value, then it covers the entire range by setting to all 1s
                    if unit_range == 0:
                        continue
                    # we ignore output cases, where output goes out of profiled ranges,
                    # this could be the surprised/exceptional case, and we leave it to
                    # neuron boundary coverage criteria
                    if output > upper_bound or output < lower_bound:
                        continue

                    subrange_index = int((output - lower_bound) / unit_range)

                    if subrange_index == self.k:
                        subrange_index -= 1

                    # print "subranges: ", subrange_index

                    id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + subrange_index
                    num = ptr[seed_id][id]
                    assert(num==0)
                    if num < 255:
                        num += 1
                        ptr[seed_id][id] = num
                        

    def bknc_update_coverage(self, outputs, ptr, rev=False):

        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]

            for seed_id, layer_output in enumerate(layer_outputs):
                # print(layer_output.shape)

                layer_output_dict = {}
                for neuron_idx in range(layer_output.shape[-1]):
                    output = np.mean(layer_output[..., neuron_idx])

                    layer_output_dict[neuron_idx] = output

                # sort the dict entry order by values
                sorted_index_output_dict = OrderedDict(
                    sorted(layer_output_dict.items(), key=lambda x: x[1], reverse=rev))

                # for list if the top_k > current layer neuron number,
                # the whole list would be used, not out of bound
                top_k_node_index_list = list(sorted_index_output_dict.keys())[:self.k]

                for top_sec, top_idx in enumerate(top_k_node_index_list):
                    id = self.start + self.layer_start_index[idx] + top_idx * self.bytearray_len + top_sec
                    num = ptr[seed_id][id]
                    if num < 255:
                        num += 1
                        ptr[seed_id][id] = num


    def tknc_update_coverage(self, outputs, ptr, rev=True):

        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]

            for seed_id, layer_output in enumerate(layer_outputs):
                # print(layer_output.shape)

                layer_output_dict = {}
                for neuron_idx in range(layer_output.shape[-1]):
                    output = np.mean(layer_output[..., neuron_idx])

                    layer_output_dict[neuron_idx] = output

                # sort the dict entry order by values
                sorted_index_output_dict = OrderedDict(
                    sorted(layer_output_dict.items(), key=lambda x: x[1], reverse=rev))

                # for list if the top_k > current layer neuron number,
                # the whole list would be used, not out of bound
                top_k_node_index_list = list(sorted_index_output_dict.keys())[:self.k]

                for top_sec, top_idx in enumerate(top_k_node_index_list):
                    id = self.start + self.layer_start_index[idx] + top_idx * self.bytearray_len + top_sec
                    num = ptr[seed_id][id]
                    if num < 255:
                        num += 1
                        ptr[seed_id][id] = num


    def nbc_update_coverage(self, outputs, ptr):

        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]

            for seed_id, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(layer_output.shape[-1]):

                    output = np.mean(layer_output[..., neuron_idx])

                    profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]
                    
                    lower_bound = profiling_data_list[3]
                    upper_bound = profiling_data_list[4]

                    # this version uses k multi_section as unit range, instead of sigma
                    # TODO: need to handle special case, std=0
                    # TODO: this might be moved to args later
                    k_multisection = self.k
                    unit_range = (upper_bound - lower_bound) / k_multisection
                    if unit_range == 0:
                        unit_range = 0.05

                    # the hypo active case, the store targets from low to -infi
                    if output < lower_bound:
                        # float here
                        target_idx = (lower_bound - output) / unit_range

                        if target_idx > (self.k - 1):
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + self.k - 1


                        else:
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + int(target_idx)


                        num = ptr[seed_id][id]
                        if num < 255:
                            num += 1
                            ptr[seed_id][id] = num
                        continue

                    # the hyperactive case
                    if output > upper_bound:
                        target_idx = (output - upper_bound) / unit_range

                        if target_idx > (self.k - 1):
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + self.k - 1
                        else:
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + int(target_idx)
                        num = ptr[seed_id][id]
                        if num < 255:
                            num += 1
                            ptr[seed_id][id] = num
                        continue


    def snac_update_coverage(self, outputs, ptr):

        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]

            for seed_id, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(layer_output.shape[-1]):
                    output = np.mean(layer_output[..., neuron_idx])

                    profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]
                    
                    lower_bound = profiling_data_list[3]
                    upper_bound = profiling_data_list[4]

                    # this version uses k multi_section as unit range, instead of sigma
                    # TODO: need to handle special case, std=0
                    # TODO: this might be moved to args later
                    # this supposes that the unit range of boundary range is the same as k multi-1000
                    k_multisection = self.k
                    unit_range = (upper_bound - lower_bound) / k_multisection
                    if unit_range == 0:
                        unit_range = 0.05

                    # the hyperactive case
                    if output > upper_bound:
                        target_idx = (output - upper_bound) / unit_range

                        if target_idx > (self.k - 1):
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + self.k - 1
                        else:
                            id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + int(
                                target_idx)

                        num = ptr[seed_id][id]
                        if num < 255:
                            num += 1
                            ptr[seed_id][id] = num

                        continue
                    

    def nc_update_coverage(self, outputs, ptr):
        '''
                Given the input, update the neuron covered in the model by this input.
                    This includes mark the neurons covered by this input as "covered"
                :param input_data: the input image
                :return: the neurons that can be covered by the input
                '''
                
        for idx, layer_name in enumerate(self.layer_to_compute):
            layer_outputs = outputs[idx]
            for seed_id, layer_output in enumerate(layer_outputs):
                scaled = self.scale(layer_output) # scale하면 값이 달성이 안될텐데..?
                for neuron_idx in range(scaled.shape[-1]):
                    if np.mean(scaled[..., neuron_idx]) > self.k:
                        id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + 0
                        ptr[seed_id][id] = 1


    def deephunter_init(self, criteria, scope='all', k = 10, profiling_dict={}, exclude_layer=['input', 'flatten']):
        
        if scope == 'all':
            self.layers = self.model.layers
        elif scope == 'logit':
            self.layers = [self.model.layers[-1]]
        else:
            print('Please enter a right scope range.', scope)
            exit(0)

        if criteria == 'nbc':
            self.k = 10 # k + 1
            self.bytearray_len = self.k * 2
        elif criteria == 'snac':
            self.k = 10 # k + 1
            self.bytearray_len = self.k
        elif criteria == 'nc':
            self.k = 0.5 # k
            self.bytearray_len = 1
        elif criteria == 'kmnc':
            self.k = 100 # k
            self.bytearray_len = self.k
        elif criteria == 'tknc':
            self.k = 3
            self.bytearray_len = self.k
        else:
            self.k = k
            self.bytearray_len = self.k

        self.criteria = criteria
        self.profiling_dict = profiling_dict


        self.layer_not_to_compute = []
        self.layer_to_compute = []
        self.layer_neuron_num = []
        self.layer_start_index = []
        self.start = 0

        

        num = 0
        for layer in self.layers:
            if all(ex not in layer.name for ex in exclude_layer):
                self.layer_start_index.append(num)
                self.layer_to_compute.append(layer.name)#str(layer).split('"')[1].split('/')[0])#layer.name)
                self.layer_neuron_num.append(layer.output.shape[-1])
                num += int(layer.output.shape[-1] * self.bytearray_len)
            else:
                self.layer_not_to_compute.append(self.model.layers.index(layer))

        self.total_size = num

        self.cov_dict = OrderedDict()


    def __init__(self, model, metric, scope, profile):

        self.model = model
        self.total_size = 0
        
        # 벡터 거리 기반 커버리지 측정 기법들
        if 'gradfuzz' in metric and 'partial' in metric:
            self.coverageFunction = self.vectorCoverage
            self.coverageFunction2 = self.gv_partial_coverage_function
        elif 'gradfuzz' in metric:
            self.coverageFunction = self.vectorCoverage
            self.coverageFunction2 = self.gv_coverage_function
        elif 'tensorfuzz' in metric:
            self.coverageFunction = self.vectorCoverage
            self.coverageFunction2 = self.av_coverage_function
        # bitwise 연산 기반 커버리지 측정 기법들
        elif metric == 'nc':
            self.coverageFunction = self.bitCoverage
            self.coverageFunction2 = self.nc_update_coverage
            self.deephunter_init(metric, scope=scope, profiling_dict=profile)
        elif metric == 'kmnc':
            self.coverageFunction = self.bitCoverage
            self.coverageFunction2 = self.kmnc_update_coverage
            self.deephunter_init(metric, scope=scope, profiling_dict=profile)
        elif metric == 'bknc':
            self.coverageFunction = self.bitCoverage
            self.coverageFunction2 = self.bknc_update_coverage
            self.deephunter_init(metric, scope=scope, profiling_dict=profile)
        elif metric == 'tknc':
            self.coverageFunction = self.bitCoverage
            self.coverageFunction2 = self.tknc_update_coverage
            self.deephunter_init(metric, scope=scope, profiling_dict=profile)
        elif metric == 'nbc':
            self.coverageFunction = self.bitCoverage
            self.coverageFunction2 = self.nbc_update_coverage
            self.deephunter_init(metric, scope=scope, profiling_dict=profile)
        elif metric == 'snac':
            self.coverageFunction = self.bitCoverage
            self.coverageFunction2 = self.snac_update_coverage
            self.deephunter_init(metric, scope=scope, profiling_dict=profile)
        else:
            self.coverageFunction = self.vectorCoverage
            self.coverageFunction2 = self.gv_coverage_function
            

    def checkCoverage(self, coverages, labels=None):
        return self.coverageFunction(coverages, labels)
