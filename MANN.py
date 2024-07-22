import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import argparse
import pickle
from MANN_utils import *


class Controller(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Controller, self).__init__()
        self.lstm = nn.Linear(input_size + 1 + hidden_size, hidden_size * 4, bias = True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y, hidden, cell):
        out = self.lstm(torch.cat([x, y.unsqueeze(1), hidden], dim = 1))
        f_i_o = self.sigmoid(out[:, :3*hidden.shape[-1]])
        f, i, o = f_i_o.chunk(3, dim = -1)
        u = self.tanh(out[:, 3*hidden.shape[-1]:])
        cell = f * cell + i * u 
        hidden = o * self.tanh(cell)
        return hidden, cell


class Augment(nn.Module):
    def __init__(self, batch_size, hidden_size, memory_shape,
                 nb_read, gamma = 0.95,):
        super(Augment, self).__init__()
        self.nb_read = nb_read
        self.gamma = gamma
        self.threshold = nb_read
        self.h2m = nn.Linear(hidden_size, memory_shape[1] * nb_read, bias = True)

    def forward(self, memory, hidden, read_weight, usage_weight, alpha):
        bs = hidden.shape[0]
        alpha = F.sigmoid(alpha)
        query = self.h2m(hidden).contiguous().view(bs, self.nb_read, -1) # q_t = W_q * h_t

        # Wlu_{t-1} = 0/1
        least_usage_indices = torch.argsort(usage_weight, dim=1)[:, :self.nb_read].flatten()
        # Ww_t = sigmoid(alpha) * Wr_{t-1} + (1 - sigmoid(alpha)) * Wlu_{t-1}
        write_weight = (alpha * read_weight).contiguous().view(bs * self.nb_read, -1)
        write_weight[torch.arange(write_weight.shape[0]), least_usage_indices] = write_weight[torch.arange(write_weight.shape[0]),
                                                                                              least_usage_indices] + 1 - alpha.flatten()

        write_weight = write_weight.contiguous().view(bs, self.nb_read, -1) # (bs, nb_read, memory_shape[0])

        # M_t = M_{t-1} + Ww_t * q_t
        memory = memory +  torch.matmul(write_weight.permute(0, 2, 1), query)
        scale = torch.linalg.vector_norm(query) * torch.linalg.vector_norm(memory)

        # Wr_t \in (bs, nb_read, memory_size[0])
        read_weight = F.softmax(torch.matmul(query, memory.permute(0, 2, 1)).div(scale).reshape(bs*self.nb_read, -1),
                                dim = -1) # Wr_t
        read_weight = read_weight.reshape(bs, self.nb_read, -1) # Wr_t

        # r_t = Wr_t @ M_t
        read_vec = torch.matmul(read_weight, memory) # bs x nb_read x memory_shape[1]
        read_vec = read_vec.contiguous().view(bs, -1) # to match the fc layer: bs x (memory_shape[1] * nb_read)

        # Wlu_t = gamma * Wlu_{t-1} + Wr_t + Ww_t
        usage_weight = self.gamma * usage_weight + read_weight.sum(dim = 1) + write_weight.sum(dim = 1) # Wu_t
    
        return read_vec, memory.detach(), read_weight.detach(), usage_weight.detach()



class MANN(nn.Module):
    def __init__(self, batch_size, input_size, output_size,
                 hidden_size, memory_shape, nb_read, gamma = 0.95):
        super(MANN, self).__init__()
        self.controller = Controller(input_size, hidden_size, output_size)
        self.aug = Augment(batch_size, hidden_size, memory_shape, 
                           nb_read = nb_read, gamma = gamma,)
        
        self.batch_size = batch_size
        self.memory_shape = memory_shape
        self.nb_read = nb_read
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size # controller
        self.fc = nn.Linear(hidden_size + nb_read * memory_shape[1], output_size) # o_t = cat(h_t, r_t)
        self.alpha = nn.Parameter(torch.zeros((batch_size, nb_read, 1)))


    def forward(self, x, y, memory, read_weight, usage_weight,):
        outputs = []
        bs, seq_len, _ = x.shape
        hidden, cell = self.controller(x[:, 0, :], torch.zeros_like(y[:, 0]),
                                                torch.zeros((bs, self.hidden_size)),
                                                torch.zeros((bs, self.hidden_size)))
        alpha = self.alpha
        for i in range(seq_len):
            # Wr_{t-1}, Wu_{t-1}, M_{t-1}, h_t -> r_t, M_t, Ww_t, Wu_t
            if i:
                # h_t, c_t, x_t, y_{t-1} -> h_{t+1}, c_{t+1}
                hidden, cell = self.controller(x[:, i, :], y[:, i-1],
                                                        hidden, cell)
            read_vec, memory, read_weight, usage_weight = self.aug(memory, hidden,
                                                                   read_weight, usage_weight, alpha)
            # cat(h_t, r_t) -> o_t
            outputs.append(torch.cat([hidden, read_vec], dim = -1))
        return (F.softmax(self.fc(torch.stack(outputs, dim = 1)), -1), 
                memory.detach(), read_weight.detach(), usage_weight.detach())



if __name__ == "__main__":       
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type = int, default = 1)
    parser.add_argument("--e", type=int, default= 100_000)
    parser.add_argument("--t", type=int, default= 100_000)
    parser.add_argument("--lr", type=float, default= 1e-3)

    args = parser.parse_args()

    batch_size = 15
    input_size = (20, 20)
    hidden_size = 200
    memory_shape = (128, 40)
    nb_read = 4
    nb_classes = 5
    output_size = nb_classes
    samples_per_class = 10
    gamma = 0.95
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # root_dir = 'omniglot/python/images_background'
    root_dir = 'omniglot/python/images_evaluation'

    dataset = OmniglotDataset(root_dir=root_dir)    
    memory = 1e-6 * torch.ones((batch_size,) + memory_shape)
    memory.requires_grad = False
    memory.to(device)
    usage_weight = torch.randn((batch_size, memory_shape[0]), requires_grad = False)
    read_weight = torch.randn((batch_size, nb_read, memory_shape[0]), requires_grad = False)

    # alpha = nn.Parameter(torch.randn((batch_size, nb_read, 1)))
    model = MANN(batch_size= batch_size, output_size= nb_classes, \
        memory_shape=memory_shape, hidden_size= hidden_size, \
            input_size= np.prod(input_size), nb_read= nb_read,)
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)

    model, optimizer, episode, usage_weight, read_weight = load_checkpoint(model, optimizer, episode = args.e)
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
    usage_weight.to(device)
    read_weight.to(device)
    usage_weight.requires_grad = False
    read_weight.requires_grad = False
    criterion = nn.CrossEntropyLoss()

    n_episodes = args.t
    pbar = tqdm(range(max(1, episode+1), n_episodes+1))
    criterion = nn.CrossEntropyLoss()
    losses = []
    accs = []

    if args.train > 0:
        for episode in pbar:
            model.train()
            example_inputs, example_outputs = dataset.sample_classes(num_samples=nb_classes,
                                                                samples_per_class= samples_per_class, 
                                                                batch_size=batch_size)
            
            optimizer.zero_grad()
            outputs, memory, read_weight, usage_weight, = model(example_inputs, 
                                                               example_outputs, memory, 
                                                               read_weight, usage_weight,)
            preds = torch.max(outputs, -1)[1]
            with torch.no_grad():
                accuracy = (preds.flatten() == example_outputs.flatten()).sum()
                accuracy = accuracy / example_outputs.flatten().shape[0]
                accs.append(accuracy.item())
    
            loss = criterion(outputs.permute(0, 2, 1), example_outputs.to(torch.long))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
    
            # print(model.alpha.data[0])
            # memory wiped per episode
            memory = 1e-6 * torch.ones((batch_size,) + memory_shape)
            memory.requires_grad = False
    
            if episode % 50 == 0:
                avg_loss = np.mean(losses[-50:])
                avg_acc = np.mean(accs[-50:])
                
                model.eval()
                with torch.no_grad():
                    acc = torch.zeros((batch_size, samples_per_class))
                    idx = torch.zeros((batch_size, nb_classes))
                    outputs, memory, read_weight, usage_weight, = model(example_inputs, 
                                                                        example_outputs, memory, 
                                                                       read_weight, usage_weight,)
                    preds = torch.max(outputs, -1)[1]
                    for i in range(batch_size):
                        eval_acc(i, preds, example_outputs, acc, idx)
                        
                    avg_acc_instance = acc.mean(dim = 0)/nb_classes
                    instance_accs = ''.join([f'{str(k)} instance: {v:.5f}\t' for k, v in zip(list(range(1, batch_size + 1)), 
                                                                               avg_acc_instance)])
            
                pbar.set_description(f'per 50 episodes, Loss: {avg_loss:.5f} Acc: {avg_acc:.5f}')
                print(instance_accs)
    
                with open('train.log.p', 'a') as f:
                    log_entry = f'\nepisode:\t{episode}\tavg_loss: {avg_loss}\tavg_acc: {avg_acc}\n'
                    f.write(log_entry)
                    save_checkpoint(model, optimizer, episode, usage_weight, read_weight)
                    
                with open('eval.log.p','a') as f:
                    log_entry = f'\nepisode:{episode}:\n\n{instance_accs}\n\n'
                    f.write(log_entry)
                
    else:
        model.to(device)
        model.eval()
    
        episode = 100
        acc_total = torch.zeros((episode, batch_size, samples_per_class))
        with torch.no_grad():
            for i in tqdm(range(1, episode+1)):
                example_inputs, example_outputs = dataset.sample_classes(num_samples=nb_classes,
                                                                    samples_per_class= samples_per_class, 
                                                                    batch_size=batch_size)
                acc = torch.zeros((batch_size, samples_per_class))
                idx = torch.zeros((batch_size, nb_classes))
            
                input_, output_ = example_inputs, example_outputs
                outputs, memory, read_weight, usage_weight, = model(input_, 
                                                                   output_, memory, 
                                                                   read_weight, usage_weight,)
                preds = torch.max(outputs, -1)[1]
                for k in range(batch_size):
                    eval_acc(k, preds, output_, acc, idx)
                    
                acc_total[i-1] = acc
        torch.save(acc_total, f'acc_{episode}_e_{args.e}.pt')
                
                
                
                
            
            
            
        
        
