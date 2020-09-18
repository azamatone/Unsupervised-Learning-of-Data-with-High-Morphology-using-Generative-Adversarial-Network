import os
import numpy as np
from modules.dataset import Dataset
from morphogan import MORPHOGAN

if __name__ == '__main__':

    out_dir = 'output/'   
    
    db_name     = 'galaxyzoo'
    data_source = './data/galaxyzoo/'
        
    model     = 'morphogan'
    nnet_type = 'dcgan'
    loss_type = 'log' #'log' or 'hinge'
    
    is_train = True 
    
    '''
    model parameters
    '''
    noise_dim    = 100    #latent dim
    '''
    Feture dim, set your self as in the paper:
    dcgan: 4096
    morphogan: 16384
    '''
    feature_dim  = 16384
    
    n_steps      = 100000 #number of iterations
        
    lambda_p  = 10.0
    lambda_r  = 1.0    
    lambda_w  = np.sqrt(noise_dim * 1.0/feature_dim) 
    
    #output dir
    out_dir = os.path.join(out_dir, model + '_' + nnet_type, db_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source)
    
    
    # setup gan model and train
    morphogan = MORPHOGAN(model=model, \
                              loss_type = loss_type, \
                              lambda_p=lambda_p, lambda_r=lambda_r, \
                              lambda_w=lambda_w, \
                              noise_dim = noise_dim, \
                              nnet_type = nnet_type, \
                              dataset=dataset, \
                              n_steps = n_steps, \
                              out_dir=out_dir)
    if is_train == True:
        morphogan.train()
    else:
        morphogan.generate()
