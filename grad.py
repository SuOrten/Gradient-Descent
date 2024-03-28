# Gradient Descent for linear regression
# yhat = wx +b [our prediction]
# implement with respect to loss = (y-yhat)**2 / N

# we will use numpy to create our data
import numpy as np

# initialize some parameters
# we need training data x and y
x = np.random.randn(10,1)# create 10 array with 1 value in each one of them
y = 2*x + np.random.rand()
# We will iteratively go and find what these parameters represent. So ideally once we go and run gradient descent w=2 and b is whatever that value is.
#Parameter
w = 0.0
b = 0.0
#Hyperparameter
learning_rate = 0.1 #How fast the algorithm learns (steps)
print(x.shape[0])

# create gradient descent function
def descend(x,y,w,b,learning_rate):
    # first initialize the partial derivative
    # calculate derivative of loss with respect to each parameters(w,b)
    dldw = 0.0
    dldb = 0.0
    # since we will calculate the average we need to know how many examples we will have in our training data
    N = x.shape[0] # shape of x
    # Loop through x and y and calculate partial derivatives and make updates to w and b
    # zip allows you to loop through both of them at the same time
    # loss = (y - yhat) ** 2 / N  and yhat=wx+b so
    # loss =  (y-(wx+b)) ** 2 this is the function we will differentiate
    for xi,yi in zip(x,y):
        dldw += 2*(yi-(w*xi+b))*-1*xi
        dldb = 2*(yi-(w*xi+b))*-1
    # make updates
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb

    return w,b

#iteratively make updates
for epoch in range(400):
    #Run gradient descent
    w,b=descend(x,y,w,b,learning_rate)
    yhat = w*x+b
    loss = np.divide(np.sum((y-yhat)**2,axis=0),x.shape[0])
    print(f'{epoch} loss is {loss} , parameters w:{w} , b:{b}')
