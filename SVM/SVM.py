from numpy import * 

def loadDataSet(filename):
    """
    Load the dataset from the given file.

    Parameters:
    filename (str): The path to the file containing the dataset.

    Returns:
    dataMat (list): A list of lists representing the data features.
    labelMat (list): A list of labels corresponding to each data point.
    """
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    """
    Randomly selects an integer between 0 and m (exclusive) that is not equal to i.

    Parameters:
    i (int): The integer to exclude from the selection.
    m (int): The upper bound (exclusive) for the random selection.

    Returns:
    int: The randomly selected integer.

    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    """
    Clips the value of `aj` to ensure it is within the range [L, H].

    Parameters:
    aj (float): The value to be clipped.
    H (float): The upper bound of the range.
    L (float): The lower bound of the range.

    Returns:
    float: The clipped value of `aj`.
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def kernelTrans(X, A, kTup): 
    """
    Transforms the input data using a specified kernel function.

    Parameters:
    X (matrix): The feature matrix of support vectors.
    A (matrix): The row of feature data to be transformed.
    kTup (tuple): A tuple specifying the kernel type and its parameters.

    Returns:
    K (matrix): The transformed matrix.

    Raises:
    NameError: If the specified kernel type is not recognized.
    """

    m, n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin': # Linear function
        K = X * A.T
    elif kTup[0] == 'rbf': # Radial basis function
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1]**2)) # Return the generated result
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    """
    A class that represents the optimization structure for SVM.

    Attributes:
        X (ndarray): The data features.
        labelMat (ndarray): The data class labels.
        C (float): The soft margin parameter C. A larger value allows for stronger non-linear fitting.
        tol (float): The stopping threshold.
        m (int): The number of data rows.
        alphas (ndarray): The Lagrange multipliers.
        b (float): The initial bias.
        eCache (ndarray): The cache for storing errors.
        K (ndarray): The computed kernel matrix.
    """

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        """
        Initialize the optStruct object.

        Args:
            dataMatIn (ndarray): The data features.
            classLabels (ndarray): The data class labels.
            C (float): The soft margin parameter C. A larger value allows for stronger non-linear fitting.
            toler (float): The stopping threshold.
            kTup (tuple): The kernel function parameters.
        """
        self.X = dataMatIn  # Data features
        self.labelMat = classLabels # Data categories
        self.C = C # Soft margin parameter C, the larger the parameter, the stronger the non-linear fitting ability
        self.tol = toler # Stopping threshold
        self.m = shape(dataMatIn)[0] # Number of rows in the data
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 # Initially set to 0
        self.eCache = mat(zeros((self.m,2))) # Cache
        self.K = mat(zeros((self.m,self.m))) # The calculation result of the kernel function
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


def calcEk(oS, k):
    """
    Calculate the error (Ek) for a given data point (k).

    Parameters:
    oS (object): The SVM object containing the necessary data.
    k (int): The index of the data point.

    Returns:
    float: The calculated error (Ek).
    """
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    """
    Selects the second alpha (aj) for the optimization process in the SMO algorithm.
    This function will randomly select aj and return its E value.

    Parameters:
    i (int): The index of the first alpha (ai) in the alpha vector.
    oS (object): The SVM optimization object containing necessary data.
    Ei (float): The error of the first alpha (ai).

    Returns:
    int: The index of the second alpha (aj) in the alpha vector.
    float: The error of the second alpha (aj).
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  # Returns the row numbers of non-zero positions in the matrix
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): # Return aj with the largest step size
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    """
    Update the error cache for a given data point.

    Parameters:
    oS (object): The SVM object containing the data and parameters.
    k (int): The index of the data point to update.

    Returns:
    None
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    """
    Performs the inner loop of the Sequential Minimal Optimization (SMO) algorithm for training a Support Vector Machine (SVM).
    This function will firstly check whether ai meets the KKT conditions. If not, randomly select aj for optimization, update ai, aj, and b values

    Parameters:
    i (int): The index of the first alpha parameter to optimize.
    oS (object): The object containing all the necessary data and parameters for the SVM training.

    Returns:
    int: Returns 1 if the alpha parameters were updated, 0 otherwise.
    """
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        # Check whether this row of data meets the KKT conditions
        j,Ej = selectJ(i, oS, Ei) # Randomly select aj, and return its E value
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol): # Threshold for alpha change size (set by yourself)
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i) # Update data
        # The following is the process of solving b
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)): 
    """
    Implements the simplified SMO algorithm for training a support vector machine (SVM).
    This function is used to solve the `alpha` values in the SVM model.

    Args:
        dataMatIn (numpy.ndarray): The input data matrix of shape (m, n), where m is the number of samples and n is the number of features.
        classLabels (numpy.ndarray): The class labels of the input data, of shape (m, 1).
        C (float): The regularization parameter.
        toler (float): The tolerance threshold.
        maxIter (int): The maximum number of iterations.
        kTup (tuple, optional): The kernel function type and parameters. Defaults to ('lin', 0) for linear kernel.

    Returns:
        float: The bias term (b) of the SVM.
        numpy.ndarray: The Lagrange multipliers (alphas) of the SVM.

    """
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m): # Traverse all data
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)) 
                # Display the number of iterations, which row of feature data caused alpha to change, and how many times alpha has changed this time
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: # Traverse non-boundary data
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

def testRbf(data_train, data_test):
    """
    Test the performance of the SVM model using the RBF kernel.

    Args:
        data_train (str): The file path of the training data.
        data_test (str): The file path of the test data.

    Returns:
        None
    """
    dataArr, labelArr = loadDataSet(data_train)  # Read the training data
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', 1.3))  # Get b and alphas using SMO algorithm
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas)[0]  # Select the indices of non-zero alphas (support vectors)
    sVs = datMat[svInd]  # Features of the support vectors
    labelSV = labelMat[svInd]  # Class labels of the support vectors (1 or -1)
    print("There are %d Support Vectors" % shape(sVs)[0])  # Print the number of support vectors

    m, n = shape(datMat)  # Number of rows and columns in the training data
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', 1.3))  # Convert the support vectors to kernel function
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        # The prediction result for this line, note that the final separation plane is determined by the support vectors only.
        if sign(predict) != sign(labelArr[i]):  # sign function: -1 if x < 0, 0 if x == 0, 1 if x > 0
            errorCount += 1
    print("The Training Error Rate is: %2.2f %%" % (100 * float(errorCount) / m))  # Print the error rate

    dataArr_test, labelArr_test = loadDataSet(data_test)  # Read the test data
    errorCount_test = 0
    datMat_test = mat(dataArr_test)
    labelMat = mat(labelArr_test).transpose()
    m, n = shape(datMat_test)
    for i in range(m):  # Check the error rate on the test data
        kernelEval = kernelTrans(sVs, datMat_test[i, :], ('rbf', 1.3))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr_test[i]):
            errorCount_test += 1
    print("The Test Error Rate is: %2.2f %%" % (100 * float(errorCount_test) / m))

def testlin(data_train, data_test):
    """
    Test the linear SVM model using the given training and testing data.

    Parameters:
    - data_train: The file path of the training data.
    - data_test: The file path of the testing data.

    Returns: None
    """
    dataArr, labelArr = loadDataSet(data_train) # Read the training data
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('lin', 1.3)) # Get b and alphas using SMO algorithm
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas)[0]  # Select the indices of non-zero alphas (support vectors)
    sVs = datMat[svInd]  # Features of the support vectors
    labelSV = labelMat[svInd]  # Class labels of the support vectors (1 or -1)
    print("There are %d Support Vectors" % shape(sVs)[0])  # Print the number of support vectors
    
    m, n = shape(datMat)  # Number of rows and columns in the training data
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('lin', 1.3))  # Convert the support vectors to kernel function
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        # The prediction result for this line, note that the final separation plane is determined by the support vectors only.
        if sign(predict) != sign(labelArr[i]):  # sign function: -1 if x < 0, 0 if x == 0, 1 if x > 0
            errorCount += 1
    print("The Training Error Rate is: %2.2f %%" % (100 * float(errorCount) / m))  # Print the error rate
    
    dataArr_test, labelArr_test = loadDataSet(data_test)  # Read the test data
    errorCount_test = 0
    datMat_test = mat(dataArr_test)
    labelMat = mat(labelArr_test).transpose()
    m, n = shape(datMat_test)
    for i in range(m):  # Check the error rate on the test data
        kernelEval = kernelTrans(sVs, datMat_test[i, :], ('rbf', 1.3))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr_test[i]):
            errorCount_test += 1
    print("The Test Error Rate is: %2.2f %%" % (100 * float(errorCount_test) / m))

# Main function
def main():
    filename_traindata='train.txt'
    filename_testdata='test.txt'
    testRbf(filename_traindata,filename_testdata)
    #testlin(filename_traindata,filename_testdata)

if __name__=='__main__':
    main()