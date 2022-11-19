import numpy as np
import pyflann
import tensorflow as tf


class NRFUpdater(object):

    ### Weight Functions ###
    def uniformWeight(self, corpus):
        ''' Description : 완전 랜덤 가중치
        '''
        newWeight = 1 / (len(corpus.candidates))
        corpus.weights = np.tile([newWeight] , len(corpus.candidates))


    def decayWeight(self, corpus):
        ''' Description : 새로 추가된 배치 확률 50, 이전 샘플 확률 50
            배치 추가되면 
        '''
        
        # 새로운 weight list
        corpus.weights /= 2
        newWeight = 0.5 / len(corpus.newBatch)
        newWeights = [newWeight] * len(corpus.newBatch)
        corpus.weights = np.append(corpus.weights, newWeights)

    
    def setProb(self, corpus):
        seed = corpus.elements[-1]
        seed.probability = self.REG_INIT_PROB

    # 측정된 커버리지가 max_coverage 보다 높을 경우 측정된 커버리지를 max 커버리지로 변경하고 corpus에 추가 시킨다.
    def deepXplore(self, corpus, element):
        if self.numLabels == 1:
            label = 0
        else:
            label = element.label

        if element.coverage > self.max_coverage[label]:
            self.max_coverage[label] = element.coverage
            corpus.elements.append(element)
            corpus.candidates.append(element)
            self.weightFunction(corpus)

    def has_new_bits(self, seed):

        if self.numLabels == 1:
            label = 0
        else:
            label = seed.label

        temp = np.invert(seed.coverage, dtype = np.uint8)
        cur = np.bitwise_and(self.virgin_bits[label], temp)
        has_new = not np.array_equal(cur, self.virgin_bits[label])
        if has_new:
            # If the coverage is increased, we will update the coverage
            self.virgin_bits[label] = cur
        return has_new

    def bitUpdate(self, corpus, element, initial):

        def addToCorpus(corpus, element):
            ''' Description : Add an element to corpus
                                      The corpus size is limited to 1,000
            '''
            corpus.elements.append(element)
            corpus.candidates.append(element)
            #if len(corpus.elements) > 1000:
            #    corpus.elements = corpus.elements[len(corpus.elements)-1000:]
            #    return True
            return False, None

        full = False
        update = False

        if self.has_new_bits(element):
            update = True
            full = addToCorpus(corpus, element)
            self.weightFunction(corpus) # 가중치 조정
        
        return update, None

    def label_build_index_and_flush_buffer(self, corpus, label):
        """Builds the nearest neighbor index and flushes buffer of examples.

        This method first empties the buffer of examples that have not yet
        been added to the nearest neighbor index.
        Then it rebuilds that index using the contents of the whole corpus.
        Args:
            corpus_object: InputCorpus object.
        """
        self.corpus_buffer[label][:] = [] # 버퍼 비우기
        self.lookup_array[label] = np.array( # corpus에 있는 커버리지로 lookup_array 구성
            [element.coverage for element in corpus.elements]
        )

        self.flann[label].build_index(self.lookup_array[label], algorithm=self.algorithm) # ??????????????????
        # tf.logging.info("Flushing buffer and building index.")


    def label_update_function(self, corpus_object, element, initial):
        """Checks if coverage is new and updates corpus if so.

        The updater maintains both a corpus_buffer and a lookup_array.
        When the corpus_buffer reaches a certain size, we empty it out
        and rebuild the nearest neighbor index.
        Whenever we check for neighbors, we get exact neighbors from the
        buffer and approximate neighbors from the index.
        This stops us from building the index too frequently.
        FLANN supports incremental additions to the index, but they require
        periodic rebalancing anyway, and so far this method seems to be
        working OK.
        Args:
            corpus_object: InputCorpus object.
            element: CorpusElement object to maybe be added to the corpus.
        """

        def addToCorpus(corpus, element):
            ''' Description : Add an element to corpus
                                      The corpus size is limited to 1,000
            '''
            corpus.elements.append(element)
            corpus.candidates.append(element)
            #if len(corpus.elements) > 1000:
            #    corpus.elements = corpus.elements[len(corpus.elements)-1000:]
            #    return True
            return False

        update = False
        nearest_distance = np.inf
        full = False
        if self.numLabels == 1:
            label = 0
        else:
            label = element.label

        if not len(corpus_object.elements):
            #update = True
            full = addToCorpus(corpus_object, element)
            #corpus_object.elements.append(element)
            self.weightFunction(corpus_object) # 가중치 조정
            self.label_build_index_and_flush_buffer(corpus_object, label)
            update = True
        else:
            # Approximate nearest neighbor 찾기
            approx_indices, approx_distances = self.flann[label].nn_index(
                np.array([element.coverage]), 1, algorithm=self.algorithm
            )

            # 최근에 추가된 _BUFFER_SIZE 만큼의 corpus element와의 거리 측정
            exact_distances = [
                np.sum(np.square(element.coverage - buffer_elt))
                for buffer_elt in self.corpus_buffer[label]
            ]
            nearest_distance = min(exact_distances + approx_distances.tolist()) # ANN, 최근 꺼와의 거리중 가장 가까운거

            if nearest_distance > self.threshold[label]: # 거리가 threshold를 넘으면?
                if self.includeLoss and not initial: # when including loss
                    nearest_index = np.argmin(exact_distances + approx_distances.tolist())
                    if nearest_index < len(exact_distances):
                        nearest_neighbor = corpus_object.elements[-len(exact_distances)+nearest_index]
                    else:
                        nearest_neighbor = corpus_object.elements[approx_indices[0]]

                    if element.loss >= element.parent.loss:#element.loss > nearest_neighbor.loss: # loss must be larger than the previous one
                        update = True
                        full = addToCorpus(corpus_object, element) # 해당 샘플을 corpus에 추가
                        #corpus_object.elements.append(element) # 해당 샘플을 corpus에 추가
                        self.weightFunction(corpus_object) # 가중치 조정
                        self.corpus_buffer[label].append(element.coverage) # 버퍼 (가장 최근 꺼를 저장하는)에 추가
                    else:
                        update = False
                else: # when not including loss
                    update = True
                    full = addToCorpus(corpus_object, element) # 해당 샘플을 corpus에 추가
                    #corpus_object.elements.append(element) # 해당 샘플을 corpus에 추가
                    self.weightFunction(corpus_object) # 가중치 조정
                    self.corpus_buffer[label].append(element.coverage) # 버퍼 (가장 최근 꺼를 저장하는)에 추가
                #tf.logging.info(
                #    "corpus_size %s mutations_processed %s",
                #    len(corpus_object.corpus),
                #    corpus_object.mutations_processed,
                #)
                #tf.logging.info(
                #    "coverage: %s, metadata: %s",
                #    element.coverage,
                #    element.metadata,
                #)
            else: # 거리가 threshold를 넘지 못하면?
                update = False

            if len(self.corpus_buffer[label]) >= self._BUFFER_SIZE or full: # 버퍼가 꽉차면 or Corpus 꽉차면
                self.label_build_index_and_flush_buffer(corpus_object, label) # 버퍼를 비우고 lookup_array 업데이트

            # threshold 조정
            if self.flexThreshold:
                if update:
                    self.miss[label] = 0
                    self.hit[label] += 1
                else:
                    self.miss[label] += 1
                    self.hit[label] = 0

                if self.miss[label] >= self.misscombo:
                    self.threshold[label] /= 10 # threshold 더 작게
                    if self.threshold[label] < self.mindist[label]:
                        self.mindist[label] = self.threshold[label]
                    self.miss[label] = 0
                    self.hit[label] = 0
            
                if self.hit[label] >= self.hitcombo:
                    self.threshold[label] *= 10 # threshold 더 크게
                    if self.threshold[label] > self.maxdist[label]:
                        self.maxdist[label] = self.threshold[label]
                    self.miss[label] = 0
                    self.hit[label] = 0

                if self.threshold[label] < 1e-32:
                    self.threshold[label] = 1

        return update, nearest_distance


    def __init__(self, metric, numLabels, includeLoss, flexThreshold, misscombo, sampletype, cov_num=0, threshold=0.0001, algorithm='kdtree', weightIndex=0, _BUFFER_SIZE=100):
        # 큰 이미지에서 특성들을 매칭할 때 성능을 위해 최적화된 라이브러리 모음

        self.metric = metric
        self.numLabels = numLabels
        self.includeLoss = includeLoss
        self.flexThreshold = flexThreshold

        #if metric == 'deepxplore':
        #    self.updateFunction = deepXplore
        #    self.max_coverage = dict()
        #    for i in range(self.numLabels):
        #        self.max_coverage[i] = 0

        if metric in ['nc', 'kmnc', 'bknc', 'tknc', 'nbc', 'snac']:
            self.updateFunction = self.bitUpdate
            self.virgin_bits = dict()

            for i in range(self.numLabels):
                self.virgin_bits[i] = np.full(cov_num, 0xFF, dtype=np.uint8)
        
        elif 'gradfuzz' in metric or 'tensorfuzz' in metric or 'random' in metric:
            self.algorithm = algorithm
            self._BUFFER_SIZE = _BUFFER_SIZE
            if misscombo:
                self.misscombo = misscombo
            else:
                self.misscombo = 32
            self.hitcombo = 2
            
            self.mindist = [np.inf for label in range(self.numLabels)]
            self.maxdist = [-np.inf for label in range(self.numLabels)]

            self.updateFunction = self.label_update_function
            self.flann = dict()
            for i in range(self.numLabels):
                self.flann[i] = pyflann.FLANN()
            self.threshold = dict()
            for i in range(self.numLabels):
                self.threshold[i] = threshold
            self.corpus_buffer = dict()
            for i in range(self.numLabels):
                self.corpus_buffer[i] = []
            self.lookup_array = dict()
            for i in range(self.numLabels):
                self.lookup_array[i] = []
            
            if self.flexThreshold:
                self.miss = dict()
                for i in range(self.numLabels):
                    self.miss[i] = 0
                self.hit = dict()
                for i in range(self.numLabels):
                    self.hit[i] = 0

        else:
            print('Please enter a right metric name.', metric)
            exit(0)

        # REG_MIN and REG_GAMMA are the p_min and gamma in Equation 3
        self.REG_GAMMA = 5
        self.REG_MIN = 0.3
        self.REG_INIT_PROB = 0.8

        if sampletype == 'prob':
            self.weightFunction = self.setProb
        else:
            self.weightFunctions = [self.uniformWeight, self.decayWeight]
            self.weightFunction = self.weightFunctions[weightIndex]

    def update(self, corpus, element, initial):
        update = self.updateFunction(corpus, element, initial)
        return update
