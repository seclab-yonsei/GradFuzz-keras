import random
from random import randint
import numpy as np

### Sample Functions ###
def randomSample(corpus):
    '''  Description : 임의로 하나 뽑기 (확률 기반)
    '''
    
    index = np.random.choice(range(len(corpus.candidates)), p=corpus.weights)

    return corpus.candidates[index]

def tensorfuzzSample(corpus):
    """Grabs new input from corpus according to sample_function."""
    # choice = self.sample_function(self)
    reservoir = list(range(len(corpus.candidates)))[-5:] + [random.choice(list(range(len(corpus.candidates))))]
    choice = random.choice(reservoir)
    return corpus.candidates[choice]
    # return random.choice(self.queue)

def prob_next(corpus):
    """Grabs new input from corpus according to sample_function."""
    while True:
        if corpus.current_id == len(corpus.candidates):
            corpus.current_id = 0

        cur_seed = corpus.candidates[corpus.current_id]
        if randint(0,100) < cur_seed.probability * 100:
            # Based on the probability, we decide whether to select the current seed.
            cur_seed.fuzzed_time += 1
            corpus.current_id += 1
            return cur_seed
        else:
            corpus.current_id += 1


def recentSample(corpus):
    ''' Description : 최신꺼 우선
    '''

    return corpus.candidates[len(corpus.elements)-1]

class NRFSampler:

    def __init__(self, sampletype):
        if sampletype == 'prob':
            self.sampleFunction = prob_next
        elif sampletype == 'tensorfuzz':
            self.sampleFunction = tensorfuzzSample
        elif sampletype == 'uniform':
            self.sampleFunction = randomSample
        else:
            self.sampleFunction = recentSample

    def doWeight(self, corpus):
        ''' Description : 가중치 업데이트 함수 돌리는 껍데기 함수
        '''
        self.weightFunction(corpus)

    def doSample(self, corpus):
        ''' Description : 샘플 함수 돌리는 껍데기 함수
        '''
        choice = self.sampleFunction(corpus)
        #index = self.sampleFunction(corpus)
        #choice = corpus.candidates[index]

        ## 뽑은거는 목록에서 제거하기
        #corpus.elements = np.delete(corpus.elements, index, axis=0)
        #corpus.weights = np.delete(corpus.weights, index, axis = 0)

        ## 가중치 조정 (하나 빠졌으니깐 다시 1로 맞추기)
        #scale = np.sum(corpus.weights)
        #corpus.weights /= scale

        return choice
