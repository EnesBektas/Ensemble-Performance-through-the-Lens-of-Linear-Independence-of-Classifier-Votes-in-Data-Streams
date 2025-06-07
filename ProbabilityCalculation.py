import numpy as np
import itertools

# rows of the matrix are the vectors, i is a integer
def find_i_linearly_independent_vectors(matrix, i):
    row, col = matrix.shape
    if i > row:
        return None

    independent_vectors = np.zeros((i, col))
    dependent_vectors = np.zeros((row - i, col))

    curRank = 0
    independent_index = 0
    dependent_index = 0
    for j in range(row):
        independent_vectors[independent_index] = matrix[j]
        if np.linalg.matrix_rank(independent_vectors, tol=1e-20) > curRank:
            curRank += 1
            independent_index += 1
        else:
            dependent_vectors[dependent_index] = matrix[j]
            dependent_index += 1

        if curRank == i or dependent_index == row - i:
            temp = matrix[j + 1:]
            break

    if independent_index < i:
        independent_vectors[independent_index:] = temp

    if dependent_index < row - i:
        dependent_vectors[dependent_index:] = temp
    

    return independent_vectors, dependent_vectors



class ProbabilityCalculation():
    def __init__(self, numOfClasses, numOfClassifiers):
        self.p_array = np.zeros(numOfClasses - 1)
        self.p_total = np.zeros(numOfClasses - 1)
        self.numOfClasses = numOfClasses
        self.numOfClassifiers = numOfClassifiers

        self.numberOfIterations = 0

    # vectors are the rows of the matrix, voteMatrix[i] is the i-th vector
    def iterateVotes(self, voteMatrix):
        self.numberOfIterations += 1
        if self.numberOfIterations % self.numOfClasses != 0:    # only calculate the probability every numOfClasses iterations for efficiency
            return

        n, col = voteMatrix.shape

        matrix = voteMatrix.copy()

        np.random.shuffle(matrix)
        max_rank = np.linalg.matrix_rank(matrix, tol=1e-20) - 1
        max_rank = min(max_rank, self.numOfClasses - 1)

        # to calculate p_i
        for i in range(1, max_rank + 1):
            # print("i = ", i)
            independent_matrix, dependent_matrix = find_i_linearly_independent_vectors(matrix, i)

            for j in range( n - i):
                temp_matrix = np.vstack((independent_matrix, dependent_matrix[j]))
                rank = np.linalg.matrix_rank(temp_matrix, tol=1e-20)
                if rank == i:
                    self.p_array[i - 1] += 1
                self.p_total[i - 1] += 1

    def calculateLinearIndependenceProbability(self):
        p = np.zeros(self.numOfClasses - 1)
        for i in range(len(self.p_array)):
            if self.p_total[i] == 0:
                p[i] = 1
            else:
                p[i] = self.p_array[i] / self.p_total[i]

        product_constant = 1
        for element in p:
            product_constant *= 1 - element

        print("product constant: ", product_constant)

        sum = 0
        for i in range(self.numOfClassifiers - self.numOfClasses + 1):
            if i == 0:
                sum += 1
                continue

            x_array = np.array(list(self.partitions(i, len(self.p_array))))
            for row in x_array:
                product = 1
                for j in range(len(row)):
                    product *= pow(p[j], row[j])
                sum += product

        print("sum: ", sum)
        probability = product_constant * sum

        return probability
    
    def calculateLinearIndependenceProbability2(self, p_array, p_total):
        p = np.zeros(self.numOfClasses - 1)
        for i in range(len(self.p_array)):
            if p_total[i] == 0:
                p[i] = 1
            else:
                p[i] = p_array[i] / p_total[i]

        product_constant = 1
        for element in p:
            product_constant *= 1 - element

        print("product constant: ", product_constant)

        sum = 0
        for i in range(self.numOfClassifiers - self.numOfClasses + 1):
            if i == 0:
                sum += 1
                continue

            x_array = np.array(list(self.partitions(i, len(p_array))))
            for row in x_array:
                product = 1
                for j in range(len(row)):
                    product *= pow(p[j], row[j])
                sum += product

            # print("sum: ", sum)

        print("sum: ", sum)
        probability = product_constant * sum

        return probability
    
    def calculateNumberOfClassifiers(self, p_arr,  p_tot , probabilityLimit=0.9999):
        for i in range(len(p_arr)):
            if p_tot[i] == 0:
                p_arr[i] = 1
            else:
                p_arr[i] = p_arr[i] / p_tot[i]
        print("in calculation, p arr: ", p_arr)
        productConst = 1
        for element in p_arr:
            productConst *= 1 - element

        if productConst == 0:
            print("It is not possible to have ideal weights")
            return np.infty

        prob = productConst
        print("product const: ", prob)
        sum = 0
        k = 0
        while prob < probabilityLimit:
            if k == 0:
                sum = 1
            else:
                x_array = np.array(list(self.partitions(k, len(p_arr))))
                #print("xarr", x_array)
                for row in x_array:
                    product = 1
                    for i in range(len(row)):
                        product *= pow(p_arr[i], row[i])
                    sum += product
                # print("Sum: ", sum)
            k += 1
            prob = productConst * sum
            # print("prob: ", prob)

        print("prob: ", prob)
        if k > 0:
            k -= 1
        return k + len(p_arr) + 1
        
    def partitions(self, n, b):
        masks = np.identity(b, dtype=int)
        for c in itertools.combinations_with_replacement(masks, n):
            yield sum(c)