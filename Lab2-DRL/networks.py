import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Utility function for model checkpointing
def save_checkpoint(epoch, model, opt, dir):

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
        }, os.path.join(dir, f'checkpoint-{epoch}.pt'))

# Utility function to load a model checkpoint
def load_checkpoint(fname, model, opt=None):

    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['model_state_dict'])
    if opt:
        opt.load_state_dict(checkpoint['opt_state_dict'])
    return model

class PolicyNet(nn.Module):

    def __init__(self, env, n_hidden=1, width=128):
        super().__init__()
        hidden_layers = [nn.Linear(env.observation_space.shape[0], width), nn.ReLU()]
        hidden_layers += [nn.Linear(width, width), nn.ReLU()] * (n_hidden - 1)
        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(width, env.action_space.n)

    def forward(self, s):
        s = self.hidden(s)
        s = F.softmax(self.out(s), dim=-1)
        return s

class ValueNet(nn.Module):
    ''' Neural network for the value function approximation. '''

    def __init__(self, env, n_hidden=1, width=128):
        super().__init__()
        hidden_layers = [nn.Linear(env.observation_space.shape[0], width), nn.ReLU()]
        hidden_layers += [nn.Linear(width, width), nn.ReLU()] * (n_hidden - 1)
        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(width, 1) # 1 output for the value function

    def forward(self, s):
        s = self.hidden(s)
        s = self.out(s)
        return s
