import numpy as np
import csv
import sys
import pickle
# from matplotlib import pyplot as plt

import fuzz_utils

from nrf_corpus import NRFCorpus, NRFElement
from nrf_sampling import NRFSampler
from nrf_mutator import NRFMutator
from nrf_fetcher import NRFFetcher
from nrf_coverage import NRFCoverage
from nrf_metadata import NRFMetadata
from nrf_update import NRFUpdater

from nrf_mutator import L2

import tensorflow as tf

class NRFFuzzer:

    def __init__(self, target, metric, mode, scope, includeLoss, flexThreshold, sess, tensor_map, profile, filterVector, batchsize=1, numLabels=1, misscombo=None):
        
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
            self.numLabels = numLabels
        else: # shared coverage mode
            self.corpus = createCorpus(1)
            self.numLabels = 1
        self.sampler = NRFSampler()
        self.mutator = NRFMutator(batchsize=batchsize, constraint=L2())
        self.fetcher = fuzz_utils.build_fetch_function(sess, scope, tensor_map) #NRFFetcher(sess, tensor_map)
        self.coverageChecker = NRFCoverage(metric, tensor_map, scope, profile)
        self.metadataChecker = NRFMetadata()
        self.updater = NRFUpdater(metric=metric, numLabels=self.numLabels, includeLoss=includeLoss, flexThreshold=flexThreshold, misscombo=misscombo, cov_num=self.coverageChecker.total_size)

        print('Function and class inialization complete.')
        
        self.filterVector = filterVector
        self.target = target
        self.batchsize = batchsize
        self.crash = []
        for label in range(numLabels):
            self.crash.append([])
        
        self.metric = metric
        self.mode = mode
        self.scope = scope
        self.includeLoss = includeLoss
        self.flexThreshold = flexThreshold
        self.maxCrashCount = 8

        # The number of each data
        self.noCrash = []
        self.noElement = []
        self.coverageTrend = []
        self.dupCoverage = 0
		
    def objective_function(self, Label, Predict):
        if Label != Predict:
            return 1
        else:
            return 0

    def recordCurrentState(self, iteration, i):
        #print('Iteration : ', iteration)
        #print('Batch No. : ', i)
        #print('Crash: ', np.sum([len(crash) for crash in self.crash]))
        #print('Corpus Elements : ', np.sum([len(corpus.elements) for corpus in self.corpus]))
        #print('Dup. Coverage : ', self.dupCoverage)
        self.noCrash.append(np.sum([len(crash) for crash in self.crash]))
        self.noElement.append(np.sum([len(corpus.elements) for corpus in self.corpus]))
        #self.coverageTrend.append(self.updater.max_coverage)

    def saveResult(self, initialTime, fuzzingTime, iterations=None):
        # Save iteration information
        with open('D:/Development/experiment/GradFuzz/result/background_gt/%s_%s_%s_%s_%sX%s_%s_%s.csv' %
                  (self.target, self.metric, self.scope, self.mode, self.batchsize, iterations, self.includeLoss, self.flexThreshold), 'w') as wfile:
            writer = csv.writer(wfile)

            writer.writerow(['No. Crash'] + self.noCrash)
            writer.writerow(['No. Elements'] + self.noElement)
            #writer.writerow(['Coverage Trend'] + self.coverageTrend)
            writer.writerow(['No. Duplicated Coverage', self.dupCoverage])

        for i in range(len(self.corpus)):
            # Save corpus information
            with open('D:/Development/experiment/GradFuzz/result/background_gt/%s_%s_%s_%s_%sX%s_%s_%s.corpus%d' %
                        (self.target, self.metric, self.scope, self.mode, self.batchsize, iterations, self.includeLoss, self.flexThreshold, i), 'wb') as wfile:
                for element in self.corpus[i].elements:
                    del element.coverage
                    element.coverage = None
                    del element.filterVector
                    element.filterVector = None
                    pickle.dump(element, wfile)
        for i in range(len(self.crash)):
            # Save crash information
            with open('D:/Development/experiment/GradFuzz/result/background_gt/%s_%s_%s_%s_%sX%s_%s_%s.crash%d' %
                        (self.target, self.metric, self.scope, self.mode, self.batchsize, iterations, self.includeLoss, self.flexThreshold, i), 'wb') as wfile:
                for crash in self.crash[i]:
                    del crash.coverage
                    crash.coverage = None
                    del crash.filterVector
                    crash.filterVector = None
                    pickle.dump(crash, wfile)

        # Save analysis time
        with open('D:/Development/experiment/GradFuzz/result/background_gt/%s_%s_%s_%s_%sX%s_%s_%s.time' %
                    (self.target, self.metric, self.scope, self.mode, self.batchsize, iterations, self.includeLoss, self.flexThreshold), 'w') as wfile:
            wfile.write('***Initial Time***\n')
            wfile.write('%s' % initialTime)
            wfile.write('\n')
            wfile.write('***Fuzzing Time***\n')
            wfile.write('%s' % fuzzingTime)



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
        batchsize = self.batchsize
        while batchsize * testCount < len(initialData):
            batchImage = initialData[batchsize * testCount : batchsize * (testCount + 1)]
            batchLabel = initialLabels[batchsize * testCount : batchsize * (testCount + 1)]

            coverages, metadata_, predict_batches, gradient_batches, losses = self.fetcher([batchImage, batchLabel])#self.fetcher.doFetch(inputBatch)

            # 커버리지
            if self.metric in ['gradfuzz', 'gradfuzz_label']:
                coverages = self.coverageChecker.checkCoverage(gradient_batches, batchLabel)
            else:
                coverages = self.coverageChecker.checkCoverage(coverages, batchLabel)

            # 업데이트
            for i in range(len(batchImage)):
                # 새로운 엘리먼트
                newElement = NRFElement(batchImage[i], batchLabel[i], predict_batches[i], metadata_[i], coverages[i], losses[i], None, self.filterVector)

                if self.mode == 'labeled':
                    label = batchLabel[i]
                else:
                    label = 0
                corpus = self.corpus[label]
                
                corpus.elements.append(newElement)
                corpus.candidates.append(newElement)

                if self.metric in ['gradfuzz', 'gradfuzz_label', 'tensorfuzz']:
                    self.updater.corpus_buffer[label].append(newElement.coverage)
                else:
                    self.updater.has_new_bits(newElement)
                self.updater.weightFunction(corpus)

            testCount += 1

        for label in range(self.numLabels):
            corpus = self.corpus[label]
            if self.metric in ['gradfuzz', 'gradfuzz_label', 'tensorfuzz'] and len(corpus.elements):
                self.updater.label_build_index_and_flush_buffer(corpus, label)

    # 단일 샘플로 퍼징하기
    def singleFuzz(self, singleSample):
        self.batchFuzz(np.expend_dim(singleSample, axis=0))

    # 배치로 퍼징하기
    def batchFuzz(self, inputBatch, inputLabels, iteration, parent=None, initial=True):
        # 돌리고 뻬치
        # print('---Current state---')
        coverages, metadata_, predict_batches, gradient_batches, losses = self.fetcher([inputBatch, inputLabels])#self.fetcher.doFetch(inputBatch)
        #print(activation_batches)

        #print(coverages)

        # 커버리지
        if self.metric in ['gradfuzz', 'gradfuzz_label']:
            coverages = self.coverageChecker.checkCoverage(gradient_batches, inputLabels)
        else:
            coverages = self.coverageChecker.checkCoverage(coverages, inputLabels)

        # 메타데이터
        metadata_ = self.metadataChecker.checkMetadata(metadata_)

        # 업데이트
        for i in range(len(inputBatch)):
            # 새로운 엘리먼트
            newElement = NRFElement(inputBatch[i], inputLabels[i], predict_batches[i], metadata_[i], coverages[i], losses[i], parent, self.filterVector)

            # Duplicated Coverage = Pass
            # if parent:
            #     if self.coverageChecker.checkDuplicate(self.corpus, parent.oldest_ancestor(), coverages[i], predict_batches[i]):
            #         self.dupCoverage += 1
            #         print('-------------------')
            #         print('Duplicated Coverage')
            #         print('-------------------')
            #         self.recordCurrentState(iteration, i)
            #         continue

            # Check it is an adversarial example
            # Crash = Pass
            if self.mode == 'labeled':
                corpus = self.corpus[inputLabels[i]]
            else:
                corpus = self.corpus[0]

            if self.objective_function(inputLabels[i], predict_batches[i]):
                if not initial:
                    self.crash[inputLabels[i]].append(newElement)
                    parent.found += 1
                    if parent.found >= 32:
                        corpus.candidates.remove(parent)
                        self.updater.weightFunction(corpus)
                        del parent
    
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
                    self.updater.update(corpus, newElement, initial)


    # initial seed 먼저 테스트
    def initialTest(self, initialData, initialLabels):
        testCount = 0
        batchsize = self.batchsize
        while batchsize * testCount < len(initialData):
            self.batchFuzz(initialData[batchsize * testCount : batchsize * (testCount + 1)], initialLabels[batchsize * testCount : batchsize * (testCount + 1)], testCount, initial=True)
            testCount += 1

    # 퍼징 시작
    def doFuzzing(self, iterations, timeout):
        if iterations: # Given iteration number
            for self.iteration in range(iterations):
                for i in range(self.numLabels):
                    sys.stdout.write('\rMetric: %s, %d/%d'%(self.metric, self.iteration, iterations))
                    sys.stdout.flush()
                    # 샘플 뽑기
                    seed = self.sampler.doSample(self.corpus[i]) # NRFElement 형태 시드 반환됨

                    # 뮤테이션
                    mutatedBatches = self.mutator.doMutation(seed)

                    # 배치샘플 분석
                    self.batchFuzz(mutatedBatches, np.tile(seed.label, (len(mutatedBatches))), self.iteration, seed, initial=False)
                    self.recordCurrentState(self.iteration, i)
            print('\n')
        else:
            import signal

            class Scheduler(object):
                def __init__(self, fuzzer):
                    self._tasks = [(1, self._heartbeat), (timeout, self._stop)]
                    self._tick = 0
                    self.fuzzer = fuzzer
                    self.stopped = False

                def _heartbeat(self):
                    sys.stdout.write('\rMetric: %s, Tick %2dh %2dm %2ds' % (self.fuzzer.metric, self._tick/3600, (self._tick%3600)/60, (self._tick%3600)%60))
                    self.fuzzer.recordCurrentState(self.fuzzer.iteration, self.fuzzer.i)

                def _stop(self):
                    print('\nStopping fuzzer, Total iteration = %d' % self.fuzzer.iteration)
                    self.fuzzer.stopped = True
                    self.stopped = True

                def _execute(self, signum, stack):
                    if self.stopped:
                        return

                    self._tick += 1
                    for period, task in self._tasks:
                        if 0 == self._tick % period:
                            task()
                    signal.alarm(1)

                def start(self):
                    signal.signal(signal.SIGALRM, self._execute)
                    signal.alarm(1)

            s = Scheduler(self)
            s.start()

            self.iteration = 0
            self.stopped = False
            while True: # Run fuzzing with given running time (without iteration numbers)
                if self.stopped == True:
                    break
                for self.i in range(self.numLabels):
                    #sys.stdout.write('\r%d'% iteration)
                    sys.stdout.flush()
                    # 샘플 뽑기
                    seed = self.sampler.doSample(self.corpus[self.i]) # NRFElement 형태 시드 반환됨

                    # 뮤테이션
                    mutatedBatches = self.mutator.doMutation(seed)

                    # 배치샘플 분석
                    self.batchFuzz(mutatedBatches, np.tile(seed.label, (len(mutatedBatches))), self.iteration, seed, initial=False)
                self.iteration += 1