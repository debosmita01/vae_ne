import json
import csv
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from smb_helper import to_text, concat_segments, to_tnsor
from SMB_models_2.of_smb_vae_1_1 import model
from cmaes import CMA
import ray


class Encoder(nn.Module):
    def __init__(self, num_features=7):
        super(Encoder, self).__init__()
        # Encoder
        self.cnv1 = nn.Conv2d(num_features, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.cnv2 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.cnv1(x)))
        x = F.leaky_relu(self.bn2(self.cnv2(x)))
        x = x.reshape(-1, 64)
        return self.fc1(x), self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, num_features=7):
        super(Decoder, self).__init__()
        # Decoder
        self.fc3 = nn.Linear(64,64)
        self.cnv3 = nn.ConvTranspose2d(4, 8, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.cnv4 = nn.ConvTranspose2d(8, num_features, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.fc3(x))
        x = x.view(-1, 4, 4, 4)
        x = F.relu(self.bn3(self.cnv3(x)))
        x = self.cnv4(x)
        x = F.softmax(x,dim=1)
        return x
        
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.en = Encoder()
        self.de = Decoder()
        self.layers = [self.en.cnv1, self.en.bn1, self.en.cnv2, self.en.bn2, self.en.fc1, self.en.fc2, 
                       self.de.fc3, self.de.cnv3, self.de.bn3, self.de.cnv4]

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
        

    def forward(self, x):
        mu, logvar = self.en(x)
        y = self.reparameterize(mu, logvar)
        z = self.de(y)
        return z, mu, logvar

def get_weights(model):
    weights = []
    for lyr in model.layers:
        weights.append(lyr.weight.view(-1).numpy())
        weights.append(lyr.bias.view(-1).numpy())
    weights = np.hstack(weights)
    return weights

def set_weights(model, weights):
    with torch.no_grad():
        n_el = 0
        for layer in model.layers:
            l_weights = weights[n_el:n_el + layer.weight.numel()]
            n_el += layer.weight.numel()
            l_weights = l_weights.reshape(layer.weight.shape)
            layer.weight = torch.nn.Parameter(torch.Tensor(l_weights))
            layer.weight.requires_grad = False
            b_weights = weights[n_el:n_el + layer.bias.numel()]
            n_el += layer.bias.numel()
            b_weights = b_weights.reshape(layer.bias.shape)
            layer.bias = torch.nn.Parameter(torch.Tensor(b_weights))
            layer.bias.requires_grad = False
    return model

def set_nograd(model):
    for param in model.parameters():
        param.requires_grad = False
        

        
class MarioDataset(Dataset):
    def __init__(self, file, transform=None):
        data = [line for line in csv.reader(open(file))]
        self.levels = data[:320]

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, idx):
        data = self.levels[idx]
        lvl = np.array(list(data[1]))
        #int_lvl = convert_int(lvl)
        tns_lvl  = self.to_tnsor(lvl)
        return tns_lvl

    def to_tnsor(self, level ,channels=7):
        level = level.reshape(16,16)
        lvl_arr = []
        for line in level:
            row = []
            for item in line:
                oh = [0] * channels
                if item == '-': oh[0] = 1    # empty/sky
                elif item == 'X': oh[1] = 1  # floor/soild
                elif item == 'S': oh[2] = 1  # brick
                elif item == 'Q': oh[3] = 1  # collecteble
                elif item == 'E': oh[4] = 1  # enemy
                elif item == 'o': oh[5] = 1  # coin
                elif item == 'p': oh[6] = 1  # pipe
                else: print("Something went wrong.", item)
                row.append(oh)
            lvl_arr.append(row)
        tns = torch.tensor(lvl_arr).float()
        return tns.transpose(0,1).transpose(0,2)

def VAE_loss(recon_x, x, mu, logvar):
    CCE = categorical_cross_entropy(recon_x,x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return CCE + KLD
        
def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


#@ray.remote
def calc_fitness(weights, dataset):
    vae = VAE().float()
    vae = set_weights(vae, weights)
    total_loss = 0
    cnt = 0
    for in_data in dataset:
        out_data, mu, logvar = vae(in_data)
        batch_loss = VAE_loss(out_data, in_data, mu, logvar)
        total_loss += batch_loss
        #print(batch_loss, total_loss)
        cnt += 1
    loss = total_loss/cnt
        
    stats = {
                "weights": np.array2string(weights),
                "loss": loss,
                "loss-type": "logit"
            }
    return loss, stats


class CMA_ES:
    def __init__(self, pop_size):
        vae = VAE().float()
        set_nograd(vae)
        model_weights = get_weights(vae)
        len_weights = len(model_weights)
        
        #int cmaes controler  Mean vector, standard deviation (sigma)
        self._optimizer = CMA(mean=np.random.uniform(-5, 5, size=len_weights), sigma=.5, population_size=pop_size)
       


    def evolve(self, dataset, gens, path):
        best = None
        min_fit = 100
        for g in range(gens+1): 
            print("gen ", g)
            #weights = [self._optimizer.ask() for _ in range(self._optimizer.population_size)]
            vae = VAE().float()
            set_nograd(vae)
            weights = [get_weights(vae) for _ in range(self._optimizer.population_size)]
        
            #futures = [calc_fitness.remote(w, dataset) for w in weights]
            #results = ray.get(futures)
            
            fitness = []
            for w in weights:
                fit, stat = calc_fitness(w, dataset)
                fitness.append(fit)
                if fit <= min_fit:
                    min_fit = fit
                    best = stat
                    
            '''for i in range(len(results)):
                r = results[i]
                fitness.append(r)
                if r <= min_fit:
                    min_fit = r
                    best = r'''
            print("gen: {} gen_best: {} global_best:{}\n".format(g, min(fitness), min_fit))
            '''log_file = open(path + "/log.txt", "a")
            log_file.write("gen: {} gen_best: {} global_best:{}\n".format(g, min(fitness), min_fit))
            log_file.close()'''
               
            # Re-package data into format expected by cmaes library
            solutions = list(zip(weights, fitness))
      
            # Pass data back into optimizer to shift the search center
            #self._optimizer.tell(solutions)
             
        '''for i in range(len(weights)):
            with open(path  + "/" + str(i) + ".json", 'w') as f:
                f.write(json.dumps(r[1]))
            cnt += 1
        
        with open(path  + "/best.json", 'w') as f:
            f.write(json.dumps(best[1]))'''

def main():
    fitness = "logit_loss"
    pop_size = 10
    generations = 5
    
    mario_data = MarioDataset("./SMB_Compare/smb_tr_data.csv")
    dloader = DataLoader(mario_data, batch_size=64, shuffle=True, num_workers=0)
    
    save_path = "./tst" 
    '''os.makedirs(save_path)
    details_file = save_path + "/details.json"
    with open(details_file, 'w') as f:
        temp = {
            "pop_size": pop_size,
            "generations": generations,
            "fitness": fitness,
            "init": "random uniform -5,5"
            }
        f.write(json.dumps(temp))'''

    cma_es = CMA_ES(pop_size)
    cma_es.evolve(dloader, generations, save_path)

#ray.init(num_cpus=2)
if __name__ == '__main__':
    main()