import numpy as np
import csv
import sys
import copy
import pickle
from datetime import datetime
# from matplotlib import pyplot as plt

sys.setrecursionlimit(10000)


from nrf_corpus import NRFCorpus, NRFElement
from nrf_sampling import NRFSampler
from nrf_mutator import NRFMutator
from nrf_coverage import NRFCoverage
from nrf_metadata import NRFMetadata
from nrf_update import NRFUpdater
from nrf_fetcher import NRFFetcher

from nrf_mutator import L2, L0

import tensorflow as tf

metrics_para = {
    'kmnc': 1000,
    'bknc': 10,
    'tknc': 10,
    'nbc': 10,
    'newnc': 10,
    'nc': 0.75,
    'snac': 10
}
bitMetrics = ['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc']

class NRFFuzzer:

    def __init__(self, execution, target, metric, mode, scope, includeLoss, flexThreshold, mutation, model, preprocess, profile, filterVector, batchsize=1, numLabels=1, sampletype='uniform', clss=1, misscombo=None):
        
        # 함수, 클래스 인스턴스들 정의
        def createCorpus(numLabels):
            corpus = []
            for i in range(numLabels):
                # avoiding corpus elements sharing their elements (I don't know why this happens)
                newcorpus = NRFCorpus(metric=metric)
                newcorpus.elements = dict()
                newcorpus.elements = []
                corpus.append(newcorpus)
            return corpus

        if mode == 'labeled':
            self.corpus = createCorpus(numLabels)
            #self.corpus2 = createCorpus(numLabels)
            self.numLabels = numLabels
        else: # shared coverage mode
            self.corpus = createCorpus(1)
            #self.corpus2 = createCorpus(1)
            self.numLabels = 1

        #if 'tensorfuzz' in metric:
        #    self.sampletype = 'tensorfuzz'
        #else:
        #    self.sampletype = sampletype
        self.sampletype = sampletype
        self.sampler = NRFSampler(sampletype)
        self.mutation = mutation
        self.mutator = NRFMutator(batchsize=batchsize, mutation_function=self.mutation)
        self.Fetcher = NRFFetcher(model, preprocess, metric, scope, batchsize)
        #self.Fetcher2 = NRFFetcher(model, 'tensorfuzz', scope, batchsize)
        self.fetcher = self.Fetcher.fetchFunction
        #self.fetcher2 = self.Fetcher2.fetchFunction
        self.coverageChecker = NRFCoverage(model, metric, scope, profile)
        #self.coverageChecker2 = NRFCoverage(model, 'tensorfuzz', scope, profile)
        self.metadataChecker = NRFMetadata()
        self.updater = NRFUpdater(metric=metric, numLabels=self.numLabels, includeLoss=includeLoss, flexThreshold=flexThreshold, misscombo=misscombo, sampletype=self.sampletype, cov_num=self.coverageChecker.total_size)
        #self.updater2 = NRFUpdater(metric='tensorfuzz', numLabels=self.numLabels, includeLoss=includeLoss, flexThreshold=flexThreshold, misscombo=misscombo, cov_num=self.coverageChecker.total_size)

        print('Function and class inialization complete.')
        
        self.execution = execution
        self.filterVector = filterVector
        self.target = target
        self.batchsize = batchsize
        
        self.metric = metric
        self.mode = mode
        self.scope = scope
        self.includeLoss = includeLoss
        self.flexThreshold = flexThreshold
        self.maxCrashCount = 256
        self.clss = clss

        class recoder():
            def __init__(self):
                # The number of each data
                self.noIteration = list()
                self.periodTime = list()
                self.noCrash = list()
                self.noElement = list()
                self.noCrashParents = list()
                self.crashParentsCount = 0
                self.noErrorCategories = list()
                
        self.timeRecoder = recoder()
        self.iterRecoder = recoder()
        self.recoders = [self.timeRecoder, self.iterRecoder]

        self.crashDict = dict()
        self.crash = []
        for label in range(numLabels):
            self.crash.append(0)
        self.crashParents = list()
        self.errorCategories = dict()
        self.diffElements = list()

        self.parents = list()

        self.crashFiles = list()
        for cls in range(numLabels):
            self.crashFiles.append(open('result/%s/%d/%s_%s_%s_%s_%s_%s_%s.crash%d' %
                        (self.metric, self.execution, self.target, self.metric, self.scope, self.mode, self.batchsize, self.includeLoss, self.flexThreshold, cls), 'wb'))
        self.parentFile = open('result/%s/%d/%s_%s_%s_%s_%s_%s_%s.parent' % (self.metric, self.execution, self.target, self.metric, self.scope, self.mode, self.batchsize, self.includeLoss, self.flexThreshold), 'wb')


		
    def objective_function(self, Label, Predict):
        if Label != Predict:
            return True
        else:
            return False

    def deepHunterQueue(self, element, crash=False):
        for bitMetric in bitMetrics:

            # Predict the mutant and obtain the outputs
            # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
            coverage_batches,  metadata_batches = self.dry_run_fetch[bitMetric]((0,[element.data],0,0,0))
            # Based on the output, compute the coverage information
            coverage_list = self.coverage_function[bitMetric](coverage_batches)
            # Create a new seed
            input = Seed(0, coverage_list[0], None, None, None, None)
            if crash:
                self.crashQueue[bitMetric].has_new_bits(input)
            else:
                self.queue[bitMetric].has_new_bits(input)
		
    def foundCrash(self, label, element):
        keyTuple = (np.mean(element.data), np.std(element.data), element.label)
        if keyTuple not in self.crashDict.keys():
            self.crashDict[keyTuple] = element.data
            
            self.crash[element.label] += 1#.append(newElement)
            element.parent.found += 1

            if element.parent not in self.crashParents:
                self.crashParents.append(element.parent)
                for recoder in self.recoders:
                    recoder.crashParentsCount += 1
            if element.prediction not in self.errorCategories[element.sampleNo]:
                self.errorCategories[element.sampleNo].append(element.prediction)

            if element.parent not in self.parents:
                self.parents.append(element.parent)
                crashParent = copy.deepcopy(element.parent)
                crashParent.parent = None
                pickle.dump(crashParent, self.parentFile)
                del crashParent
            #element = [element.data, element.label, element.prediction]#element.parent = None
            pickle.dump(element, self.crashFiles[label])

    def recordCurrentState_time(self):
        self.timeRecoder.noIteration.append(self.iteration)
        self.timeRecoder.noCrash.append(np.sum(self.crash))
        self.timeRecoder.noElement.append(np.sum([len(corpus.elements) for corpus in self.corpus]))
        self.timeRecoder.noCrashParents.append(self.timeRecoder.crashParentsCount)
        self.timeRecoder.crashParentsCount = 0
        self.timeRecoder.noErrorCategories.append(np.sum([len(self.errorCategories[sample]) for sample in self.errorCategories.keys()]))
        #for bitMetric in bitMetrics:
        #    coverage = round(float(self.coverage_handler[bitMetric].total_size - np.count_nonzero(self.queue[bitMetric].virgin_bits == 0xFF)) * 100 / self.coverage_handler[bitMetric].total_size, 2)
        #    crashCoverage = round(float(self.coverage_handler[bitMetric].total_size - np.count_nonzero(self.crashQueue[bitMetric].virgin_bits == 0xFF)) * 100 / self.coverage_handler[bitMetric].total_size, 2)
        #    print('DeepHunter coverage %s : queue = %f, crash queue = %f' % (bitMetric, coverage, crashCoverage))

    def recordCurrentState_iteration(self, time):
        self.iterRecoder.noIteration.append(self.iteration)
        self.iterRecoder.periodTime.append(time)
        self.iterRecoder.noCrash.append(np.sum(self.crash))
        self.iterRecoder.noElement.append(np.sum([len(corpus.elements) for corpus in self.corpus]))
        self.iterRecoder.noCrashParents.append(self.iterRecoder.crashParentsCount)
        self.iterRecoder.crashParentsCount = 0
        self.iterRecoder.noErrorCategories.append(np.sum([len(self.errorCategories[sample]) for sample in self.errorCategories.keys()]))
        #for bitMetric in bitMetrics:
        #    coverage = round(float(self.coverage_handler[bitMetric].total_size - np.count_nonzero(self.queue[bitMetric].virgin_bits == 0xFF)) * 100 / self.coverage_handler[bitMetric].total_size, 2)
        #    crashCoverage = round(float(self.coverage_handler[bitMetric].total_size - np.count_nonzero(self.crashQueue[bitMetric].virgin_bits == 0xFF)) * 100 / self.coverage_handler[bitMetric].total_size, 2)
        #    print('DeepHunter coverage %s : queue = %f, crash queue = %f' % (bitMetric, coverage, crashCoverage))

    def saveResult(self, initialTime, fuzzingTime, iterations=None):
        # Save iteration information
        with open('result/%s/%d/timeRecode.csv' %
                  (self.metric, self.execution), 'w') as wfile:
            writer = csv.writer(wfile)
            
            writer.writerow(['No. Iteration'] + self.timeRecoder.noIteration)
            writer.writerow(['No. Crash'] + self.timeRecoder.noCrash)
            writer.writerow(['No. Elements'] + self.timeRecoder.noElement)
            writer.writerow(['No. Unique CrashParents'] + self.timeRecoder.noCrashParents)
            writer.writerow(['No. Error Category'] + self.timeRecoder.noErrorCategories)

        with open('result/%s/%d/iterationRecode.csv' %
                  (self.metric, self.execution), 'w') as wfile:
            writer = csv.writer(wfile)
            
            writer.writerow(['No. Iteration'] + self.iterRecoder.noIteration)
            writer.writerow(['No. Period Time'] + self.iterRecoder.periodTime)
            writer.writerow(['No. Crash'] + self.iterRecoder.noCrash)
            writer.writerow(['No. Elements'] + self.iterRecoder.noElement)
            writer.writerow(['No. Unique CrashParents'] + self.iterRecoder.noCrashParents)
            writer.writerow(['No. Error Category'] + self.iterRecoder.noErrorCategories)

        for i in range(len(self.corpus)):
            # Save corpus information
            with open('result/%s/%d/%s_%s_%s_%s_%s_%s_%s.corpus%d' %
                        (self.metric, self.execution, self.target, self.metric, self.scope, self.mode, self.batchsize, self.includeLoss, self.flexThreshold, i), 'wb') as wfile:
                for element in self.corpus[i].elements:
                    del element.coverage
                    element.coverage = None
                    del element.filterVector
                    element.filterVector = None
                    pickle.dump(element, wfile)
        #for i in range(len(self.crash)):
        #    # Save crash information
        #    with open('D:/Development/experiment/GradFuzz/result/background/%d/%s_%s_%s_%s_%sX%s_%s_%s.crash%d' %
        #                (self.execution, self.target, self.metric, self.scope, self.mode, self.batchsize, iterations, self.includeLoss, self.flexThreshold, i), 'wb') as wfile:
        #        for crash in self.crash[i]:
        #            del crash.coverage
        #            crash.coverage = None
        #            del crash.filterVector
        #            crash.filterVector = None
        #            pickle.dump(crash, wfile)

        # Save analysis time
        with open('result/%s/%d/%s_%s_%s_%s_%s_%s_%s.time' %
                    (self.metric, self.execution, self.target, self.metric, self.scope, self.mode, self.batchsize, self.includeLoss, self.flexThreshold), 'w') as wfile:
            wfile.write('***Initial Time***\n')
            wfile.write('%s' % initialTime)
            wfile.write('\n')
            wfile.write('***Fuzzing Time***\n')
            wfile.write('%s' % fuzzingTime)

        #for bitMetric in bitMetrics:
        #    # Save DeepHunter coverage information
        #    with open('result/%d/%s_%s_%s_%s_%s_%s_%s.%s.corpus' %
        #                (self.execution, self.target, self.metric, self.scope, self.mode, self.batchsize, self.includeLoss, self.flexThreshold, bitMetric), 'wb') as wfile:
        #        pickle.dump(self.queue[bitMetric], wfile)
        #    with open('result/%d/%s_%s_%s_%s_%s_%s_%s.%s.crash' %
        #                (self.execution, self.target, self.metric, self.scope, self.mode, self.batchsize, self.includeLoss, self.flexThreshold, bitMetric), 'wb') as wfile:
        #        pickle.dump(self.crashQueue[bitMetric], wfile)

                
        

    def plotCrash(self, element):
    
        from matplotlib import pyplot as plt

        def plotimg(img):
            convImg = (np.reshape(img, (28, 28)) * 255).astype(np.uint8)
            plt.figure(figsize=(3, 3))
            plt.imshow(convImg, interpolation='nearest', cmap='gray')
            plt.show()

        plotimg(element.data + 0.5)

    # def plotCrashes(self):
    
    #     from matplotlib import pyplot as plt

    #     def plotimg(img):
    #         convImg = (np.reshape(img, (28, 28)) * 255).astype(np.uint8)
    #         plt.figure(figsize=(3, 3))
    #         plt.imshow(convImg, interpolation='nearest', cmap='gray')
    #         plt.show()

    #     for element in self.crash:

    # Insert initial seeds into corpus without testing
    def initialSeed(self, initialData, initialLabels):
        testCount = 0
        #cleanCount = 0
        #datas = list([list(), list()])
        batchsize = self.batchsize
        while batchsize * testCount < len(initialData):
            batchImage = initialData[batchsize * testCount : batchsize * (testCount + 1)]
            batchLabel = initialLabels[batchsize * testCount : batchsize * (testCount + 1)]
            
            coverages, metadata_, predict_batches, loss = self.fetcher(batchImage, batchLabel)#self.fetcher.doFetch(inputBatch)

            # 커버리지
            coverages = self.coverageChecker.checkCoverage(coverages, batchLabel)

            # 업데이트
            for i in range(len(batchImage)):
                # 새로운 엘리먼트
                newElement = NRFElement(batchImage[i], batchLabel[i], predict_batches[i], metadata_[i], coverages[i], loss[i], None, self.filterVector, (batchsize * testCount) + i, 0)
                self.errorCategories[(batchsize * testCount) + i] = list()
                if self.mutation == 'deephunter':
                    newElement.setDeepHunterVariable(batchImage[i], self.clss, 0, 0)
                
                if self.objective_function(batchLabel[i], predict_batches[i]):
                    continue
                #cleanCount += 1
                #datas[0].append(batchImage[i])
                #datas[1].append(batchLabel[i])
                #if cleanCount == 5000:
                #    # # Activate when saving dataset
                #    with open('data/gtsrb/test_5000.data', 'wb') as wfile:
                #        pickle.dump(datas, wfile)
                #    print((batchsize * testCount) + i)
                #    exit()

                if self.mode == 'labeled':
                    label = batchLabel[i]
                else:
                    label = 0
                corpus = self.corpus[label]
                
                corpus.elements.append(newElement)
                corpus.candidates.append(newElement)

                if 'gradfuzz' in self.metric or 'tensorfuzz' in self.metric or 'random' in self.metric:
                    self.updater.corpus_buffer[label].append(newElement.coverage)
                else:
                    self.updater.has_new_bits(newElement)
                self.updater.weightFunction(corpus)
                #self.deepHunterQueue(newElement)

            testCount += 1
            sys.stdout.write('\rMetric: %s, Execution %d, Initial %d/%d'%(self.metric, self.execution, batchsize * testCount, len(initialData)))

        for label in range(self.numLabels):
            corpus = self.corpus[label]
            if 'gradfuzz' in self.metric or 'tensorfuzz' in self.metric or 'random' in self.metric and len(corpus.elements):
                self.updater.label_build_index_and_flush_buffer(corpus, label)

    # 단일 샘플로 퍼징하기
    def singleFuzz(self, singleSample):
        self.batchFuzz(np.expend_dim(singleSample, axis=0))

    # 배치로 퍼징하기
    def batchFuzz(self, mutatedBatches, label, iteration, parent=None, initial=True):
        # 돌리고 뻬치
        # print('---Current state---')
        ref_batches = mutatedBatches[0]
        inputBatch = mutatedBatches[1]
        inputLabels = np.tile(label, (len(mutatedBatches[1])))
        cl_batches = mutatedBatches[2]
        l0_batches = mutatedBatches[3]
        linf_batches = mutatedBatches[4]
        tids = mutatedBatches[5]
        coverages, metadata_, predict_batches, loss = self.fetcher(inputBatch, inputLabels)#self.fetcher.doFetch(inputBatch)
        #coverages2, metadata_2, predict_batches2 = self.fetcher2(inputBatch, inputLabels)#self.fetcher.doFetch(inputBatch)

        if not coverages:
            return None

        # 커버리지
        coverages = self.coverageChecker.checkCoverage(coverages, inputLabels)
        #coverages2 = self.coverageChecker2.checkCoverage(coverages, inputLabels)

        # 메타데이터
        # metadata_ = self.metadataChecker.checkMetadata(metadata_)

        # 업데이트
        for i in range(len(inputBatch)):
            # 새로운 엘리먼트
            newElement = NRFElement(inputBatch[i], inputLabels[i], predict_batches[i], metadata_[i], coverages[i], loss[i], parent, self.filterVector, parent.sampleNo, parent.generation+1, parent.tid+[tids[i]])
            #newElement2 = NRFElement(inputBatch[i], inputLabels[i], predict_batches[i], metadata_[i], coverages2[i], None, parent, self.filterVector)
            if self.mutation == 'deephunter':
                newElement.setDeepHunterVariable(ref_batches[i], cl_batches[i], l0_batches[i], linf_batches[i])
                #newElement2.setDeepHunterVariable(ref_batches[i], cl_batches[i], l0_batches[i], linf_batches[i])

            # Check it is an adversarial example
            # Crash = Pass

            if self.mode == 'labeled':
                label = inputLabels[i]
            else:
                label = 0
            corpus = self.corpus[label]
            #corpus2 = self.corpus2[label]

            if self.objective_function(inputLabels[i], predict_batches[i]):
                if not initial:
                    #self.deepHunterQueue(newElement, True)
                    self.foundCrash(inputLabels[i], newElement)
                    #if 'gradfuzz' in self.metric and parent.found >= self.maxCrashCount and parent in corpus.candidates:
                    #    corpus.candidates.remove(parent)
                    #    self.updater.weightFunction(corpus)
                    #    del parent

                    #if parent in self.diffElements:
                    #    parents = list()
                    #    while True:
                    #        if parent in self.diffElements:
                    #            parents.append(parent)
                    #            self.plotCrash(parent)
                    #        parent = parent.parent
                    #        if parent == None:
                    #            break
                    #    print(parents, inputLabels[i], predict_batches[i])
    
                    #from matplotlib import pyplot as plt

                    #convImg = (np.reshape(inputBatch[i], (32, 32)) * 255).astype(np.uint8)
                    #plt.figure(figsize=(3, 3))
                    #plt.imshow(convImg, interpolation='nearest', cmap='gray')
                    #plt.show()
                    
                #print('-------------------')
                #print('Found Crash: ', [newElement, inputLabels[i], predict_batches[i]])
                #print('-------------------')

            else:
                # 커버리지 갱신? & 추가
                if self.metric is not 'random':
                    update, distance = self.updater.update(corpus, newElement, initial)
                #update2, distance2 = self.updater2.update(corpus2, newElement2, initial)
                #if not initial and update and not update2:
                #    print(update, update2, iteration, newElement, distance, distance2)
                #    self.diffElements.append(newElement)
                #if update:
                #    self.deepHunterQueue(newElement)

        # Update the probability based on the Equation 3 in the paper
        if self.sampletype == 'prob':
            if parent.probability > self.updater.REG_MIN and parent.fuzzed_time < self.updater.REG_GAMMA * (1 - self.updater.REG_MIN):
                parent.probability = self.updater.REG_INIT_PROB - float(parent.fuzzed_time) / self.updater.REG_GAMMA


    # initial seed 먼저 테스트
    def initialTest(self, initialData, initialLabels):
        testCount = 0
        batchsize = self.batchsize
        while batchsize * testCount < len(initialData):
            mutatedBatches = list()
            mutatedBatches.append(initialData[batchsize * testCount : batchsize * (testCount + 1)])
            mutatedBatches.append(initialData[batchsize * testCount : batchsize * (testCount + 1)])
            mutatedBatches.append(np.tile(self.clss, batchsize))
            mutatedBatches.append(np.tile(0, batchsize))
            mutatedBatches.append(np.tile(0, batchsize))
            self.batchFuzz(mutatedBatches, initialLabels[batchsize * testCount : batchsize * (testCount + 1)], testCount, initial=True)
            testCount += 1

    # 퍼징 시작
    def doFuzzing(self, iterations, timeout):
        startTime = datetime.now()
        if iterations and not timeout: # Given iteration number
            self.iteration = 0
            while True:
                if self.iteration >= iterations:
                    break
                for self.i in range(self.numLabels):
                    #sys.stdout.write('\rMetric: %s, Execution %d, %d/%d, Corpus elements %d, Crashes %d, Crash parents %d, min threshold %.02E, max threshold %.02E' % (self.metric, self.execution, self.iteration, iterations, np.sum([len(corpus.elements) for corpus in self.corpus]), np.sum(self.crash), len(self.crashParents), min([self.updater.threshold[threshold] for threshold in self.updater.threshold]), max([self.updater.threshold[threshold] for threshold in self.updater.threshold])))
                    #sys.stdout.write('\rMetric: %s, Execution %d, %d/%d, Corpus elements %d, Crashes %d'%(self.metric, self.execution, self.iteration, iterations, np.sum([len(corpus.elements) for corpus in self.corpus]), np.sum(self.crash)))
                    #sys.stdout.flush()

                    # 뮤테이션
                    
                    #while True:
                    # 샘플 뽑기
                    seed = self.sampler.doSample(self.corpus[self.i]) # NRFElement 형태 시드 반환됨

                    # 뮤테이션
                    mutatedBatches = self.mutator.doMutation(seed)
                        #if not np.sum(mutatedBatches[1]):
                        #    if 'gradfuzz' in self.metric:
                        #        continue
                        #        corpus = self.corpus[i]
                        #        corpus.candidates.remove(seed)
                        #        self.updater.weightFunction(corpus)
                        #else:
                        #    break

                    # 배치샘플 분석
                    self.batchFuzz(mutatedBatches, seed.label, self.iteration, seed, initial=False)
                    self.iteration += 1
                    if self.iteration % 5000 == 0:
                        endTime = datetime.now()
                        self.recordCurrentState_iteration(endTime - startTime)
                    if self.iteration == iterations:
                        endTime = datetime.now()
                        print('\nFuzzing reached max iteration, execution time = %s' % (endTime - startTime))
            print('\n')
        elif iterations and timeout:
            import signal

            class Scheduler(object):
                def __init__(self, fuzzer):
                    self._tasks = [(300, self._heartbeat), (timeout, self._stop)]
                    self._tick = 0
                    self.fuzzer = fuzzer
                    self.stopped = False

                def _heartbeat(self):
                    self.fuzzer.recordCurrentState_time()
                    #sys.stdout.write('\rMetric: %s, Execution %d, Tick %2dh %2dm %2ds, Corpus elements %d, Crashes %d, Crash parents %d, min threshold %.02E, max threshold %.02E' % (self.fuzzer.metric, self.fuzzer.execution, self._tick/3600, (self._tick%3600)/60, (self._tick%3600)%60, np.sum([len(corpus.elements) for corpus in self.fuzzer.corpus]), np.sum(self.fuzzer.crash), len(self.fuzzer.crashParents), min([self.fuzzer.updater.threshold[threshold] for threshold in self.fuzzer.updater.threshold]), max([self.fuzzer.updater.threshold[threshold] for threshold in self.fuzzer.updater.threshold])))
                    sys.stdout.write('\rMetric: %s, Execution %d, Tick %2dh %2dm %2ds, Corpus elements %d, Crashes %d, Crash parents %d, Error categories %d' % (self.fuzzer.metric, self.fuzzer.execution, self._tick/3600, (self._tick%3600)/60, (self._tick%3600)%60, np.sum([len(corpus.elements) for corpus in self.fuzzer.corpus]), np.sum(self.fuzzer.crash), len(self.fuzzer.crashParents), self.fuzzer.timeRecoder.noErrorCategories[-1]))

                def _stop(self):
                    print('\nFuzzing reached timeout, Total iteration = %d' % self.fuzzer.iteration)
                    self.fuzzer.stopped = True
                    self.stopped = True

                def _execute(self, signum, stack):
                    if self.stopped:
                        return

                    self._tick += 300
                    for period, task in self._tasks:
                        if 0 == self._tick % period:
                            task()
                    signal.alarm(300)

                def start(self):
                    signal.signal(signal.SIGALRM, self._execute)
                    signal.alarm(300)

            s = Scheduler(self)
            s.start()

            self.iteration = 0
            self.stopped = False
            while True: # Run fuzzing with given running time (without iteration numbers)
                if self.iteration >= iterations and self.stopped == True:
                    break
                for self.i in range(self.numLabels):
                    #sys.stdout.write('\r%d'% iteration)
                    sys.stdout.flush()
                    
                    #while True:
                    # 샘플 뽑기
                    seed = self.sampler.doSample(self.corpus[self.i]) # NRFElement 형태 시드 반환됨

                    # 뮤테이션
                    mutatedBatches = self.mutator.doMutation(seed)
                        #if not np.sum(mutatedBatches[1]):
                        #    if 'gradfuzz' in self.metric:
                        #        continue
                        #        corpus = self.corpus[self.i]
                        #        corpus.candidates.remove(seed)
                        #        self.updater.weightFunction(corpus)
                        #else:
                        #    break
                            

                    # 배치샘플 분석
                    self.batchFuzz(mutatedBatches, seed.label, self.iteration, seed, initial=False)
                    self.iteration += 1
                    if self.iteration % 5000 == 0:
                        endTime = datetime.now()
                        self.recordCurrentState_iteration(endTime - startTime)
                    if self.iteration == iterations:
                        endTime = datetime.now()
                        print('\nFuzzing reached max iteration, execution time = %s\n' % (endTime - startTime))
        else:
            exit()
