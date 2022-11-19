import numpy as np
import tensorflow as tf
import copy
#import pyflann

_BUFFER_SIZE = 50

# 하나의 샘플을 담는 곳
class NRFElement(object):
    def __init__(self, data, label, prediction, metadata, coverage, loss, parent, filterVector=None, sampleNo=0, generation=0, tid=[-1], ref=None, clss=None, l0_ref=None, linf_ref=None):
        self.data = data
        self.label = label
        self.prediction = prediction
        self.metadata = metadata
        self.coverage = coverage
        self.loss = loss
        self.parent = parent
        self.found = 0
        self.filterVector = filterVector
        self.sampleNo = sampleNo
        self.generation = generation
        self.tid = tid

        # DeepHunter variable
        self.ref = ref
        self.l0_ref = l0_ref
        self.linf_ref = linf_ref
        self.clss = clss

        self.fuzzed_time = 0

    def setDeepHunterVariable(self, ref, clss, l0_ref, linf_ref):
        self.ref = ref
        self.clss = clss
        self.l0_ref = l0_ref
        self.linf_ref = linf_ref

    def oldest_ancestor(self):
        
        # 뮤테이션된 횟수 
        # parent 값을 순차적으로 찾는다.

        if self.parent is None: 
            return self, 0
        else:
            ancestor, generation = self.parent.oldest_ancestor()
            return ancestor, generation + 1
        
        

class NRFCorpus(object):
    def __init__(self, metric, elements=[]):
        self.elements = elements # 역대 추가됐던 element들
        self.candidates = copy.deepcopy(elements) # 현재 샘플링 가능한 element들
        self.current_id = 0
        if 'gradfuzz' in metric or 'tensorfuzz' in metric or 'random' in metric:
            self.coverages = dict()
        try:
            self.weights = [1 / len(candidates)] * len(candidates)
        except:
            self.weights = list()
