import sys
import cv2
import numpy as np
import random
import time
import copy

class constraint:
    max_distance = 1

    def distance_metric(self, noise):
        pass

    def clip():
        pass

class L0(constraint):
    max_distance = 0.01

    def distance_metric(self, noise):
        return np.count_nonzero(noise)

    # 통째로 줄이는 clip 함수
    #def clip(self, noises):
    #    for noise in noises:
    #        clip_value = 1
    #        distance = self.distance_metric(noise)
    #        if distance > self.max_distance:
    #            clip_value = self.max_distance / distance

    #        # 노이즈만큼 값 자르기
    #        noise *= clip_value

    #    return noises

    # 이번에 변경된 픽셀만 줄이는 clip 함수
    def clip(self, totalnoise, noise):
        for index in range(len(totalnoise)):

            distance = self.distance_metric(totalnoise[index])
            if distance/np.shape(totalnoise[index].flatten())[0] > self.max_distance:
                # Max distance에 딱 맞게 noise 조정
                totalnoise[index] = noise[index]
                print('Mutation distance exceeded', self.distance_metric(totalnoise[index]))
                return np.zeros(1)

        return totalnoise

class L2(constraint):
    max_distance = 500

    def distance_metric(self, noise):
        return np.sum(np.square(noise)) ** 0.5

    # 통째로 줄이는 clip 함수
    #def clip(self, noises):
    #    for noise in noises:
    #        clip_value = 1
    #        distance = self.distance_metric(noise)
    #        if distance > self.max_distance:
    #            clip_value = self.max_distance / distance

    #        # 노이즈만큼 값 자르기
    #        noise *= clip_value

    #    return noises

    # 이번에 변경된 픽셀만 줄이는 clip 함수
    def clip(self, totalnoise, noise):
        for index in range(len(totalnoise)):
            distance = self.distance_metric(totalnoise[index])
            if distance > self.max_distance:
                # Max distance에 딱 맞게 noise 조정
                totalnoise[index] -= noise[index]
                original_distance = self.distance_metric(totalnoise[index])
                newnoise = (self.max_distance ** 2) - (original_distance ** 2)
                noise[index][np.nonzero(noise[index])] = newnoise
                totalnoise[index] += noise[index]
                #print(self.distance_metric(totalnoise[index]))

        return totalnoise

def basic_mutation_function(self, seed_input):
    ''' TensorFuzz�� ���ٰ� ������ �� �ְ�����
        filterVector�� �ֱ� ������ ������ �ٸ��� '''

    '''
    Input parameter : Mutation 할 원본 이미지, 필터
    Output : Mutated image batch
    '''

    # 원본 이미지와 필터 불러오기
    newshape = tuple([self.batchsize] + [1] * len(np.shape(seed_input.data)))
    original_image = np.tile([seed_input.data], newshape)

    # 조상 이미지 불러오기
    ancestor, _ = seed_input.oldest_ancestor()
    ancestor_image = np.tile([ancestor.data], newshape)

    # 필터 적용하기
    if seed_input.filterVector is None: # *****
        filterVector = np.tile([1], np.shape(original_image))
    else:
        filterVector = np.tile([seed_input.filterVector], newshape)

    # 랜덤 노이즈 생성
    #noise = np.random.rand(*np.shape(original_image)) - 0.5
    noise = np.zeros(np.shape(original_image))
    for n in noise:
        shapes = list()
        tempN = n
        for s in np.shape(n):
            index = np.random.randint(len(tempN))
            shapes.append(index)
            tempN = tempN[index]
        n[tuple(shapes)] = np.random.randint(0, 255)#np.random.random() - 0.5

    # 누적 노이즈 찾기
    totalnoise = (original_image + (noise * filterVector)) - ancestor_image

    # Constraint를 통해 노이즈 잘라주기
    if self.constraint:
        noise = self.constraint.clip(totalnoise, noise)
        if not np.sum(noise):
            return None, [], None, None, None, []

    # 원본 이미지에 노이즈 씌우기
    mutated_batch = ancestor_image + noise

    # 픽셀 최대, 최소 값 넘어가는 값 자르기
    mutated_batch = np.clip(mutated_batch, self.min, self.max)
    
    #return mutated_batch
    return None, mutated_batch, None, None, None, [9 for i in range(self.batchsize)]







# keras 1.2.2 tf:1.2.0
class Mutators():
    def image_translation(img, params):

        rows, cols, ch = img.shape
        # rows, cols = img.shape

        # M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
        M = np.float32([[1, 0, params], [0, 1, params]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst.astype(np.uint8)

    def image_scale(img, params):

        # res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
        rows, cols, ch = img.shape
        res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
        res = res.reshape((res.shape[0],res.shape[1],ch))
        y, x, z = res.shape
        if params > 1:  # need to crop
            startx = x // 2 - cols // 2
            starty = y // 2 - rows // 2
            return res[starty:starty + rows, startx:startx + cols]
        elif params < 1:  # need to pad
            sty = round((rows - y) / 2)
            stx = round((cols - x) / 2)
            return np.pad(res, [(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)], mode='constant',
                          constant_values=0)
        return res.astype(np.uint8)

    def image_shear(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        factor = params * (-1.0)
        M = np.float32([[1, factor, 0], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst.astype(np.uint8)

    def image_rotation(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_AREA)
        return dst.astype(np.uint8)

    def image_contrast(img, params):
        alpha = params
        new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
        # new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

        return new_img.astype(np.uint8)

    def image_brightness(img, params):
        beta = params
        new_img = cv2.add(img, beta)  # new_img = img*alpha + beta
        return new_img.astype(np.uint8)

    def image_blur(img, params):

        # print("blur")
        blur = []
        if params == 1:
            blur = cv2.blur(img, (3, 3))
        if params == 2:
            blur = cv2.blur(img, (4, 4))
        if params == 3:
            blur = cv2.blur(img, (5, 5))
        if params == 4:
            blur = cv2.GaussianBlur(img, (3, 3), 0)
        if params == 5:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
        if params == 6:
            blur = cv2.GaussianBlur(img, (7, 7), 0)
        if params == 7:
            blur = cv2.medianBlur(img, 3)
        if params == 8:
            blur = cv2.medianBlur(img, 5)
        # if params == 9:
        #     blur = cv2.blur(img, (6, 6))
        if params == 9:
            blur = cv2.bilateralFilter(img, 6, 50, 50)
            # blur = cv2.bilateralFilter(img, 9, 75, 75)
        return blur.astype(np.uint8)

    def image_pixel_change(img, params):
        # random change 1 - 5 pixels from 0 -255
        img_shape = img.shape
        img1d = np.ravel(img)
        arr = np.random.randint(0, len(img1d), params)
        for i in arr:
            img1d[i] = np.random.randint(0, 256)
        new_img = img1d.reshape(img_shape)
        return new_img.astype(np.uint8)

    def image_noise(img, params):
        if params == 1:  # Gaussian-distributed additive noise.
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss
            return noisy.astype(np.uint8)
        elif params == 2:  # Replaces random pixels with 0 or 1.
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(img)
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                      for i in img.shape]
            out[tuple(coords)] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper))
                      for i in img.shape]
            out[tuple(coords)] = 0
            return out
        elif params == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
            row, col, ch = img.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = img + img * gauss
            return noisy.astype(np.uint8)















    transformations = [image_translation, image_scale, image_shear, image_rotation,
                       image_contrast, image_brightness, image_blur, image_pixel_change, image_noise]

    # these parameters need to be carefullly considered in the experiment
    # to consider the feedbacks
    params = []
    params.append(list(range(-3, 3)))  # image_translation
    params.append(list(map(lambda x: x * 0.1, list(range(7, 12)))))  # image_scale
    params.append(list(map(lambda x: x * 0.1, list(range(-6, 6)))))  # image_shear
    params.append(list(range(-50, 50)))  # image_rotation
    params.append(list(map(lambda x: x * 0.1, list(range(5, 13)))))  # image_contrast
    params.append(list(range(-20, 20)))  # image_brightness
    params.append(list(range(1, 10)))  # image_blur
    params.append(list(range(1, 10)))  # image_pixel_change
    params.append(list(range(1, 3)))#4)))  # image_noise

    classA = [7, 8]  # pixel value transformation
    classB = [0, 1, 2, 3, 4, 5, 6] # Affine transformation
    @staticmethod
    def mutate_one(ref_img, img, cl, l0_ref, linf_ref, try_num=50):

        # ref_img is the reference image, img is the seed

        # cl means the current state of transformation
        # 0 means it can select both of Affine and Pixel transformations
        # 1 means it only select pixel transformation because an Affine transformation has been used before

        # l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
        # between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1}) in Equation 2 of the paper

        # tyr_num is the maximum number of trials in Algorithm 2


        x, y, z = img.shape

        # a, b is the alpha and beta in Equation 1 in the paper
        a = 0.01
        b = 0.20

        # l0: alpha * size(s), l_infinity: beta * 255 in Equation 1
        l0 = int(a * x * y * z)
        l_infinity = int(b * 255)

        ori_shape = ref_img.shape
        for ii in range(try_num):
            random.seed(time.time())
            if cl == 0:  # 0: can choose class A and B
                tid = random.sample(Mutators.classA + Mutators.classB, 1)[0]
                # Randomly select one transformation   Line-7 in Algorithm2
                transformation = Mutators.transformations[tid]
                params = Mutators.params[tid]
                # Randomly select one parameter Line 10 in Algo2
                param = random.sample(params, 1)[0]

                # Perform the transformation  Line 11 in Algo2
                img_new = transformation(copy.deepcopy(img), param)
                img_new = img_new.reshape(ori_shape)
                img_new = np.clip(img_new, 0, 255)

                if tid in Mutators.classA:
                    sub = ref_img - img_new
                    # check whether it is a valid mutation. i.e., Equation 1 and Line 12 in Algo2
                    l0_ref = l0_ref + np.sum(sub != 0)
                    linf_ref = max(linf_ref, np.max(abs(sub)))
                    if l0_ref < l0 or linf_ref < l_infinity:
                        return ref_img, img_new, 0, 1, l0_ref, linf_ref, tid
                else:  # B, C
                    # If the current transformation is an Affine trans, we will update the reference image and
                    # the transformation state of the seed.
                    ref_img = transformation(copy.deepcopy(ref_img), param)
                    ref_img = ref_img.reshape(ori_shape)
                    ref_img = np.clip(ref_img, 0, 255)
                    return ref_img, img_new, 1, 1, l0_ref, linf_ref, tid
            if cl == 1: # 0: can choose class A
                tid = random.sample(Mutators.classA, 1)[0]
                transformation = Mutators.transformations[tid]
                params = Mutators.params[tid]
                param = random.sample(params, 1)[0]
                img_new = transformation(copy.deepcopy(img), param)
                img_new = np.clip(img_new, 0, 255)
                sub = ref_img - img_new

                # To compute the value in Equation 2 in the paper.
                l0_ref = l0_ref + np.sum(sub != 0)
                linf_ref = max(linf_ref, np.max(abs(sub)))

                if  l0_ref < l0 or linf_ref < l_infinity:
                    return ref_img, img_new, 1, 1, l0_ref, linf_ref, tid
        # Otherwise the mutation is failed. Line 20 in Algo 2
        return ref_img, img, cl, 0, l0_ref, linf_ref, -1

    @staticmethod
    def mutate_without_limitation(ref_img):

        tid = random.sample(Mutators.classA + Mutators.classB, 1)[0]
        transformation = Mutators.transformations[tid]
        ori_shape = ref_img.shape
        params = Mutators.params[tid]
        param = random.sample(params, 1)[0]
        img_new = transformation(ref_img, param)
        img_new = img_new.reshape(ori_shape)
        return img_new
    @staticmethod
    #Algorithm 2
    def image_random_mutate(self, seed):

        ref_img = seed.ref#test[0]
        img = seed.data#test[1]
        cl = seed.clss
        ref_batches = []
        batches = []
        cl_batches = []
        l0_ref_batches = []
        linf_ref_batches = []
        tids = []
        for i in range(self.batchsize):
            ref_out, img_out, cl_out, changed, l0_ref, linf_ref, tid = Mutators.mutate_one(ref_img, img, cl, seed.l0_ref, seed.linf_ref)
            if changed:
                ref_batches.append(ref_out)
                batches.append(img_out)
                cl_batches.append(cl_out)
                l0_ref_batches.append(l0_ref)
                linf_ref_batches.append(linf_ref)
                tids.append(tid)

        return np.asarray(ref_batches), np.asarray(batches), cl_batches, l0_ref_batches, linf_ref_batches, tids
    

# pylint: disable=too-many-locals
def tensorfuzz_mutation(self, corpus_element):
    """Mutates image inputs with white noise.

  Args:
    corpus_element: A CorpusElement object. It's assumed in this case that
      corpus_element.data[0] is a numpy representation of an image and
      corput_element.data[1] is a label or something we *don't* want to change.
    mutations_count: Integer representing number of mutations to do in
      parallel.
    constraint: If not None, a constraint on the norm of the total mutation.

  Returns:
    A list of batches, the first of which is mutated images and the second of
    which is passed through the function unchanged (because they are image
    labels or something that we never intended to mutate).
  """
    # Here we assume the corpus.data is of the form (image, label)
    # We never mutate the label.

    image = corpus_element.data
    image_batch = np.tile(image, [self.batchsize, 1, 1, 1])

    sigma = 0.2
    noise = np.random.normal(size=image_batch.shape, scale=sigma) # 랜덤 노이즈 생성

    if self.min and self.max:
        # (image - original_image) is a single image. it gets broadcast into a batch
        # when added to 'noise'
        ancestor, _ = corpus_element.oldest_ancestor()  # 가장 오래된 조상을 찾음
        original_image = ancestor.data
        original_image_batch = np.tile(
            original_image, [self.batchsize, 1, 1, 1]
        )
        cumulative_noise = noise + (image_batch - original_image_batch) # 이전 노이즈에서 랜덤 생성된 노이즈를 축적
        # pylint: disable=invalid-unary-operand-type
        noise = np.clip(cumulative_noise, a_min=self.min, a_max=self.max) # constraint를 넘어가면 짤라냄
        mutated_image_batch = noise + original_image_batch
    else:
        mutated_image_batch = noise + image_batch

    mutated_image_batch = np.clip(
        mutated_image_batch, a_min=self.min, a_max=self.max
    ) # constraint 없으면 input parameter만큼 짤라냄

    return None, mutated_image_batch, None, None, None


class NRFMutator:

    def __init__(self, mutation_function = 'deephunter', min = 0, max = 255, batchsize = 1, constraint = L2()):
        self.min = min
        self.max = max
        self.batchsize = batchsize
        self.constraint = constraint
        if mutation_function == 'deephunter':
            self.mutation_function = Mutators.image_random_mutate
        elif mutation_function == 'tensorfuzz':
            self.mutation_function = tensorfuzz_mutation
        else:
            self.mutation_function = basic_mutation_function


    def doMutation(self, seed_input):
        mutated_batch = self.mutation_function(self, seed_input)

        return mutated_batch

