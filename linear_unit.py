from functools import reduce

class LinearUnit(object):

    def __init__(self, input_sum, act):

        self.weights = [0.0 for _ in  range(input_sum)]
        self.bias = 0.0
        self.act = act

    def __str__(self):
        
        return ""
    
    def cal(self, input_vec):

        return self.act(reduce(lambda a,b:a+b,map(lambda x,w:x*w, input_vec,self.weights),0.0)+self.bias)

    def train(self, input_vecs, labels, iterator, rate):
        
        for i in range(iterator):
            self.one_train(input_vecs, labels, rate)
    
    def one_train(self, input_vecs, labels, rate):

        samples = zip(input_vecs,labels)
        for (input_vec,label) in samples:
            output = self.cal(input_vec);
            self.update_weights(input_vec,label,output,rate)

    def update_weights(self, input_vec, label, output, rate):
        
        detal = label-output        
        self.weights = list(map(lambda x,w:x*rate*detal+w,input_vec, self.weights))
        self.bias = rate*detal+self.bias

f = lambda x:x

def train_data_set():

    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    
    linear_unit = LinearUnit(2,f);
    input_vecs,labels = train_data_set()
    linear_unit.train(input_vecs,labels, 10, 0.01)
    return linear_unit

if __name__ == "__main__":
    
    linear_unit = train_linear_unit()
    print(linear_unit)
    
    print ('Work 3.4 years, monthly salary = %.2f' % linear_unit.cal([3.4]))
    print ('Work 15 years, monthly salary = %.2f' % linear_unit.cal([15]))
    print ('Work 1.5 years,monthly salary = %.2f' % linear_unit.cal([1.5]))
    print ('Work 6.3 years, monthly salary = %.2f' % linear_unit.cal([6.3]))
