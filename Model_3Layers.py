import tensorflow as tf
import numpy as np
import gc
import os
import platform
from sklearn.model_selection import train_test_split
from beautifultable import BeautifulTable
import webbrowser

np.set_printoptions(linewidth=999999)
print('Optimized Tensorflow ',tf.__version__)
print('Python Version ',platform.sys.version)


n_hidden_1 = 15
n_hidden_2 = 10
n_hidden_3 = 10
n_classes = 1
layer  = 3
comb_coeffient = 0.01
damp_factor = 10
iteration = 20
loss_thresold = 1.0e-6
accuracy_stop = 0.88


working_dir_path = '/Data/Model/AI'
test_ratio = 0.4
RANDOM_SEED = 42


def working_dir(working_dir_path):
    """ Set working directory """
    os.chdir(working_dir_path)

def load_data():
    """ Load spectrum """
    temp = np.genfromtxt('Data.csv',delimiter=",")

    return temp

def training_testing_data(data_input, expected_output, testrows):
    """ Split spectrum into train and test sample"""
    splitted_data = train_test_split(data_input, expected_output, test_size = testrows,
                                     random_state=RANDOM_SEED)
    return splitted_data

def init_weight(shape,bound,name):
    """ Weight initialization """
    weight = np.random.uniform(-bound,bound,shape).astype('float32')
    
    return tf.Variable(weight,name = name)

def reinit_weight(rows,columns):
    """ Weight re-initialization """
    weight = np.random.randn(rows,columns).astype('float32')*np.sqrt(2/(rows+columns))
    return weight

def init_bias_input(shape,name):
    """ Bias initialization """
    bias = tf.fill([shape],0.01)
    return tf.Variable(bias,name = name)

def reinit_bias_input(shape):
    """ Bias re-initialization """
    bias = np.full(shape,0.01).astype('float32')
    return bias

def init_bias_output(shape,name):
    """ Bias_Output initialization """
    bias = tf.fill([shape],0.01)
    return tf.Variable(bias,name = name)

def reinit_bias_output(shape):
    """ Bias re-initialization """
    bias = np.full(shape,0.01).astype('float32')
    return bias

def init_alpha(name):
    """ Alpha initialization """
    temp = tf.constant(0.001)
    return temp

def init_beta(name):
    """ Beat initialization """
    temp = tf.constant(0.1)
    return temp

def reinit_alpha():
    """ Alpha initialization """
    temp = 0.001
    return temp

def reinit_beta():
    """ Beat initialization """
    temp = 0.1
    return temp

def regularizer(variables,layer):

    weights = variables[:layer+1]
    regularization = 0

    for weight in weights:
        regularization = regularization + tf.nn.l2_loss(weight)

    return regularization

def loss_func(labels,output,size):

    loss = size*tf.losses.mean_squared_error(labels, output)

    return loss

def objective_func(loss,regularization,alpha,beta):
    """ Define loss function"""

    objective = alpha*regularization + beta*loss

    return objective


def gra_holder(yhat,y_,variables,n_testcases):

    error = y_ - yhat

    loop_vars = [
                 tf.constant(0, tf.int32),
                 tf.TensorArray(tf.float32, size = n_testcases),
    ]

    _, holder = tf.while_loop(
                          lambda i, _: i < n_testcases,
                          lambda i, result: (i+1, result.write(i, tf.gradients(error[i],variables))),loop_vars)

    holder = holder.stack()
    print(holder.shape)

    return holder

def gra_holder_list(yhat,variables,x,y,n_testcases,n_components):

    list = []
    for variable in variables:

        j = gra_holder(yhat,y,variable,n_testcases)

        if variable.name[:2] == 'W1':
            j = tf.reshape(j,[n_testcases,n_components,n_hidden_1])
        elif variable.name[:2] == 'W2':
            j = tf.reshape(j,[n_testcases,n_hidden_1,n_hidden_2])
        elif variable.name[:2] == 'W3':
            j = tf.reshape(j,[n_testcases,n_hidden_2,n_hidden_3])
        elif variable.name[:2] == 'W4':
            j = tf.reshape(j,[n_testcases,n_hidden_3,n_classes])
        elif variable.name[:2] == 'b1':
            j = tf.reshape(j,[n_testcases,1,n_hidden_1])
        elif variable.name[:2] == 'b2':
            j = tf.reshape(j,[n_testcases,1,n_hidden_2])
        elif variable.name[:2] == 'b3':
            j = tf.reshape(j,[n_testcases,1,n_hidden_3])
        elif variable.name[:2] == 'b4':
            j = tf.reshape(j,[n_testcases,1,n_classes])

        list.append(j)

    return list

def jaccob_func(yhat,x,y,n_testcases,n_components,variables):

    paras= len(variables)

    list = gra_holder_list(yhat,variables,x,y,n_testcases,n_components)

    jaccob = []
    for count in range(paras):
        weight_holder = list[count]
        cols = weight_holder.shape[1]*weight_holder.shape[2]

        loop_vars = [
                     tf.constant(0, tf.int32),
                     tf.TensorArray(tf.float32, size = n_testcases),
                     ]

        _, output = tf.while_loop(
                             lambda i, _: i < n_testcases,
                             lambda i, result: (i+1, result.write(i,tf.reshape(weight_holder[i],[1,-1]))),loop_vars)
        output = output.stack()
        output = tf.reshape(output,[n_testcases,cols])

        jaccob.append(output)

    jaccob = tf.concat(jaccob, axis=1)

    return jaccob

def hessian_func(j):

    hessian = tf.matmul(j,j,True)
    hessian = tf.linalg.inv(hessian)

    return hessian

def hessian_adj_func(j,coeffient):
    hessian = tf.matmul(j,j,True)
    diag = tf.diag_part(hessian)
    diag = diag + coeffient
    hessian = tf.linalg.set_diag(hessian,diag)
    hessian = tf.linalg.inv(hessian)

    return hessian

def fast_hessian_adj_func(j,damp):
    hessian = j.T.dot(j)
    hessian/(damp*comb_coeffient + j.dot(j.T))
    diag = hessian.diagonal()
    diag = diag - 1
    hessian[np.diag_indices_from(hessian)] = diag
    hessian = (1/damp*comb_coeffient)*hessian

    return hessian

def delta_weight(delta,variable_size,n_components,session):


    delta_w1 = delta[:variable_size[1]]
    delta_w1 = tf.reshape(delta_w1,[n_components,n_hidden_1])

    delta_w2 = delta[variable_size[1]:variable_size[2]]
    delta_w2 = tf.reshape(delta_w2,[n_hidden_1,n_hidden_2])

    delta_w3 = delta[variable_size[2]:variable_size[3]]
    delta_w3 = tf.reshape(delta_w3,[n_hidden_2,n_hidden_3])

    delta_w4 = delta[variable_size[3]:variable_size[4]]
    delta_w4 = tf.reshape(delta_w4,[n_hidden_3,n_classes])

    delta_b1 = delta[variable_size[4]:variable_size[5]]
    delta_b1 = tf.reshape(delta_b1,[n_hidden_1,])

    delta_b2 = delta[variable_size[5]:variable_size[6]]
    delta_b2 = tf.reshape(delta_b2,[n_hidden_2,])

    delta_b3 = delta[variable_size[6]:variable_size[7]]
    delta_b3 = tf.reshape(delta_b3,[n_hidden_3,])

    delta_b4 = delta[variable_size[7]:variable_size[8]]
    delta_b4 = tf.reshape(delta_b4,[n_classes,])

    return session.run([delta_w1,delta_w2,delta_w3,delta_w4,delta_b1,delta_b2,delta_b3,delta_b4])

def update_hyperparam(h,loss,regularization,model_size,n_components):

    trace_h = tf.linalg.trace(h)
    alpha = tf.divide(model_size,(2*regularization + 2*trace_h))
    gamma = model_size - 2*alpha*trace_h
    beta = tf.divide((n_components - gamma),(2*loss))

    return alpha,beta,gamma

def updated_hyperparam(h,loss,regularization,model_size,n_components, alpha):
    
    trace_h = tf.linalg.trace(h)
    gamma = model_size - 2.*alpha*trace_h
    
    alpha = tf.divide(gamma,2*regularization)
    beta = tf.divide((n_components - gamma),(2*loss))
    
    return alpha,beta,gamma

def model_relu(x,w1,w2,w3,w4,bias_1,bias_2,bias_3,bias_4):
    """ Define Deep Learning-Based Model"""
    operation_1 = tf.matmul(x, w1)
    operation_1 = tf.add(operation_1,bias_1)
    operation_1 = tf.nn.relu(operation_1)

    operation_2 = tf.matmul(operation_1, w2)
    operation_2 = tf.add(operation_2,bias_2)
    operation_2 = tf.nn.relu(operation_2)

    operation_3 = tf.matmul(operation_2, w3)
    operation_3 = tf.add(operation_3,bias_3)
    operation_3 = tf.nn.relu(operation_3)

    operation_4 = tf.matmul(operation_3, w4)
    operation_4 = tf.add(operation_4,bias_4)

    output = tf.nn.sigmoid(operation_4)

    return output

def model_sigmoid(x,w1,w2,w3,w4,bias_1,bias_2,bias_3,bias_4):
    """ Define Deep Learning-Based Model"""
    operation_1 = tf.matmul(x, w1)
    operation_1 = tf.add(operation_1,bias_1)
    operation_1 = tf.nn.sigmoid(operation_1)

    operation_2 = tf.matmul(operation_1, w2)
    operation_2 = tf.add(operation_2,bias_2)
    operation_2 = tf.nn.sigmoid(operation_2)

    operation_3 = tf.matmul(operation_2, w3)
    operation_3 = tf.add(operation_3,bias_3)
    operation_3 = tf.nn.sigmoid(operation_3)

    operation_4 = tf.matmul(operation_3, w4)
    operation_4 = tf.add(operation_4,bias_4)

    output = tf.nn.sigmoid(operation_4)

    return output


def model_accuracy(yhat,y_test):
    """ Accuracy of the model after a minibatch"""

    yhat_class = tf.greater(yhat,0.5)
    y_test_class = tf.equal(y_test,1.0)
    corrections = tf.equal(yhat_class,y_test_class)
    accuracy = tf.reduce_mean( tf.cast(corrections, 'float') )

    return accuracy

def results(component, suspiciousness):
    table = zip(component, suspiciousness)
    table = list(table)

    t = BeautifulTable()
    t.column_headers = ["Component", "Suspiciousness"]
    for i in range(len(table)):
        t.append_row(table[i])

    return(t)

def print_spectrum(spectrum,vector_error,components):

    rows, columns = spectrum.shape
    t = BeautifulTable()

    labels = []
    for i in range(1, columns+1):
        labels.append('Comp' + str(i))
    labels = np.append(labels,"Error")
    t.column_headers = labels
    data = np.append(spectrum,vector_error,axis=1)
    '''data = data[:20,(components % 10)+10:]'''
    data = list(data)
    for i in range(len(data)):
        t.append_row(data[i])

    print('Matrix of Test Results - Spectrum')

    print('Good Work')

def output_html(a,b):
    html_str = """
        <table border=1>"""
    html_str +="<tr>"

    for item in a:
        html_str +="<th>"+item+"</th>"

    html_str +="</tr>"
    #indent tag
    html_str +="<indent>"
    html_str +="<tr>"

    for item in b:
        html_str +="<td><font color=" + "&ldquo" + "red" + "&rdquo" + ">" + "&nbsp&nbsp&nbsp" +item+ "&nbsp&nbsp&nbsp" + "</font></td>"

    html_str +="</tr>"
    html_str +="</indent>"
    #indent tag
    html_str +="</table>"

    Html_file= open("out_vector.html","w")
    Html_file.write(html_str)
    Html_file.close()


    url = 'out_vector.html'
    chrome_path = "C://Program Files (x86)/Google/Chrome/Application/chrome.exe %s"

    webbrowser.get(chrome_path).open(url)

def main():

    # Change working directory
    working_dir(working_dir_path)
    print(working_dir_path)

    # Load data
    data = load_data()
    rows, cols = data.shape
    spectrum = data[:,:cols-1]
    
    # Number of components and testcases in the tarining set
    n_components = spectrum.shape[1]
    test_rows = int(spectrum.shape[0]*test_ratio)
    print("test sample: ",test_rows)
    n_testcases = spectrum.shape[0] - test_rows

    #Size of Model
    model_size = n_components*n_hidden_1 + n_hidden_1*n_hidden_2 + n_hidden_2*n_hidden_3 + n_hidden_3*n_classes + n_hidden_1 + n_hidden_2 + n_hidden_3 + n_classes
    
    # Load vector error
    vector_result = data[:,cols-1:cols]
    vector_result = np.reshape(vector_result,[rows,1])

    loop = 0
    flag_stop = False
    while(True):
    
        # Split data into training and testing sets
        
        global X_train, X_test, Y_train, Y_test
        X_train, X_test, Y_train, Y_test = training_testing_data(spectrum, vector_result, test_rows)

        # Define the input/output
        X = tf.placeholder(tf.float32, shape=[None, n_components])
        Y = tf.placeholder(tf.float32, shape=[None, 1])

        # Init the weights
        W1 = init_weight((n_components, n_hidden_1),0.25,'W1')
        W2 = init_weight((n_hidden_1, n_hidden_2),0.5,'W2')
        W3 = init_weight((n_hidden_2, n_hidden_3),0.5,'W3')
        W4 = init_weight((n_hidden_3, n_classes),1.,'W4')

        # Init the biases
        b1 = init_bias_input(n_hidden_1,'b1')
        b2 = init_bias_input(n_hidden_2,'b2')
        b3 = init_bias_input(n_hidden_3,'b3')
        b4 = init_bias_output(n_classes,'b4')

        # Init hyper_paras
        alpha = init_alpha('alpha')
        beta = init_beta('beta')
        
        
        global variable
        variables = []
        for variable in tf.trainable_variables():
            variables.append(variable)

        # Models
        Yhat_relu = model_relu(X,W1,W2,W3,W4,b1,b2,b3,b4)
        Yhat_sigmoid = model_sigmoid(X,W1,W2,W3,W4,b1,b2,b3,b4)

        # Loss
        loss_holder = loss_func(Y,Yhat_sigmoid,n_testcases)

        # Jaccob
        j_holder = jaccob_func(Yhat_sigmoid,X,Y,n_testcases,n_components,variables)

        # Regularization
        regularization_holder = regularizer(variables,layer)

        # Objective
        objective_holder = objective_func(loss_holder,regularization_holder,alpha,beta)

        # Size of variables
        global variable_size
        variable_size = [0]
        for variable in variables:
            size = variable_size[-1]+tf.size(variable)
            variable_size.append(size)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("loop: ",loop)
        accuracy = 0
        learning_coeffient = comb_coeffient
        round = 0

        while (round < iteration):

            print("Round: ",round)
            damp = damp_factor
            
            global j, yhat

            yhat = sess.run(Yhat_sigmoid, feed_dict = {X:X_train,Y: Y_train})
            j = sess.run(j_holder,feed_dict = {X:X_train,Y: Y_train})

            error = tf.cast(Y_train - yhat,tf.float32)
            objective = sess.run(objective_holder,feed_dict = {X:X_train,Y: Y_train})
            print("objective: ",objective)

            flag_exit = False
            flag_record = False
            count = 1

            while (count < 6):
                
                 global h,delta,delta_w1,delta_w2,delta_w3,delta_w4,delta_b1,delta_b2,delta_b3,delta_b4
                 global new_w1,new_w2,new_w3,new_w4,new_b1,new_b2,new_b3,new_b4
                 
                 h = hessian_adj_func(j,learning_coeffient)
                 delta = tf.matmul(j,error,True)
                 delta = tf.matmul(h,delta)
                 delta_w1,delta_w2,delta_w3,delta_w4,delta_b1,delta_b2,delta_b3,delta_b4 = delta_weight(delta,variable_size,n_components,sess)

                 # Update weights
                 new_w1 = sess.run(W1 - delta_w1)
                 new_w2 = sess.run(W2 - delta_w2)
                 new_w3 = sess.run(W3 - delta_w3)
                 new_w4 = sess.run(W4 - delta_w4)

                 new_b1 = sess.run(b1 - delta_b1)
                 new_b2 = sess.run(b2 - delta_b2)
                 new_b3 = sess.run(b3 - delta_b3)
                 new_b4 = sess.run(b4 - delta_b4)

                 W1.load(new_w1,sess)
                 W2.load(new_w2,sess)
                 W3.load(new_w3,sess)
                 W4.load(new_w4,sess)

                 b1.load(new_b1,sess)
                 b2.load(new_b2,sess)
                 b3.load(new_b3,sess)
                 b4.load(new_b4,sess)

                 # Update objective
                 new_objective = sess.run(objective_holder,feed_dict = {X:X_train,Y: Y_train})
                 print("new_objective: ",new_objective)

                 less_objective = sess.run(tf.less(new_objective,objective))
                 if less_objective:

                     h = hessian_adj_func(j_holder,comb_coeffient)
                     para_holder = update_hyperparam(h,loss_holder,regularization_holder,model_size,n_testcases)
                     alpha,beta,gamma = sess.run(para_holder,feed_dict = {X:X_train,Y: Y_train})
                     print("%s , %s , %s"  % (alpha,beta,gamma))
                     learning_coeffient /= damp

                     yhat = sess.run(Yhat_sigmoid, feed_dict = {X:X_test,Y: Y_test})
                     accuracy = sess.run(model_accuracy(yhat,Y_test))
                     print("Accuracy: ",accuracy)
                     if(accuracy >= accuracy_stop):
                         flag_record = True
                         flag_stop = True
                         break

                     round += 1
                     flag_exit = True
                     break

                 else:
                     if count < 6:
                        learning_coeffient *= damp

                        # Restore weights
                        W1.load(new_w1 + delta_w1,sess)
                        W2.load(new_w2 + delta_w2,sess)
                        W3.load(new_w3 + delta_w3,sess)
                        W4.load(new_w4 + delta_w4,sess)

                        b1.load(new_b1 + delta_b1,sess)
                        b2.load(new_b2 + delta_b2,sess)
                        b3.load(new_b3 + delta_b3,sess)
                        b4.load(new_b4 + delta_b4,sess)

                        count += 1
                        round += 1
                     else:
                        learning_coeffient *= damp
                        round += 1
                        break
        
            if flag_record:
                break
            
            if flag_exit:
                if round >= iteration:
                    break

        if flag_stop:
            break

        loop += 1
        print("new loop")

        tf.reset_default_graph()

    sess.close

if __name__ == '__main__':

    main()
