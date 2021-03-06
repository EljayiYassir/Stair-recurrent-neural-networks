import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from utils import select_network,select_optimizer
import numpy as np
import os
import pickle
from datetime import datetime

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn, ntoken, ninp, nhid, tie_weights=False):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = rnn
        self.decoder = nn.Linear(nhid, ntoken)
        self.alpha = nn.Parameter(torch.fill_(torch.ones(SLEN), 0.98))
        print(rnn)
        # self.params = rnn.params + [self.encoder.weight,self.decoder.weight,self.decoder.bias]
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_norm(self):
        return self.rnn._norm()

    def forward(self, input,hidden):

        emb = self.encoder(input)
        hs= []
        for i in range(emb.shape[0]):
            if NVAR == 'alpha':
                hidden = self.alpha[i] * self.rnn(emb[i], hidden)
            else:
                hidden = self.rnn(emb[i], hidden)
            hs.append(hidden)
        output = torch.stack(hs)
        
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden



parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--net-type',
                    type=str, default='RNN',
                    choices=['LSTM', 'RNN', 'RNN1', 'RNN2', 'RNN3', 'EURNN', 'expRNN', 'nnRNN', 'RNN-Orth', 'RNNSN'],
                    help='options: LSTM, RNN, expRNN, nnRNN')

parser.add_argument('--nvar',
                    type=str, default='alpha',
                    choices=['none','1n', '1n2', 'expn', 'logn', 'log2n', 'sqrtlogn', 'alpha'],
                    help='options: 1n 1n2 expn logn logn2')

parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--rounds', type=int, default=1,
                    help='max rounds')
parser.add_argument('--bptt', type=int, default=150,
                    help='sequence length')
parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
parser.add_argument('--tied', action='store_true', default=False, help='For tie weights')
parser.add_argument('--random-seed', type=int, default=400,
                    help='random seed')
parser.add_argument('--batch', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--lr', type=float, default=0.0008)
parser.add_argument('--lr_orth', type=float, default=8e-5)
parser.add_argument('--rinit', type=str, default="cayley",
                    choices=['random', 'cayley', 'henaff', 'xavier'],
                    help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="kaiming",
                    choices=['xavier', 'kaiming'],
                    help='input weight matrix initialization' )
parser.add_argument('--nonlin', type=str, default='modrelu',
                    choices=['none','modrelu', 'tanh', 'relu', 'sigmoid'],
                    help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--alam', type=float, default=1, help='decay for gamma values nnRNN')
parser.add_argument('--Tdecay', type=float,
                    default=0.0001, help='weight decay on upper T')
parser.add_argument('--optimizer', type=str, default='RMSprop', choices=['RMSprop', 'Adam'])
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--betas', type=float, default=(0.0, 0.9), nargs='+')
parser.add_argument('--name', type=str,
                    default='RNN', help='name of model')
args = parser.parse_args()



# Set the random seed manually for reproducibility.
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.random_seed)

        

###############################################################################
# Load data
###############################################################################

corpus = Corpus('./data/pennchar/')

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
NET_TYPE = args.net_type
inp_size = args.emsize
hid_size = args.nhid
alam = args.alam
CUDA = args.cuda
nonlin = args.nonlin
NVAR = args.nvar
SLEN = args.bptt


###############################################################################
# Training code
###############################################################################

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = None
    correct = 0
    processed = 0
    for i in range(0, data_source.size(0) - 1, args.bptt):
        
        data, targets = get_batch(data_source, i, evaluation=True)
        if i == 0 and NET_TYPE == 'LSTM':
            model.rnn.init_states(data.shape[1])
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).item()
        correct += torch.eq(torch.argmax(output_flat,dim=1),targets).sum().item()
        processed += targets.shape[0]
        hidden = hidden.detach()

        if NET_TYPE == 'LSTM':
                model.rnn.ct = model.rnn.ct.detach()
    return total_loss / len(data_source), correct/processed


def train(optimizer, orthog_optimizer):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    hidden = None
    losses = []

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.

        if hidden is not None:
            hidden = hidden.detach()
            if NET_TYPE == 'LSTM':
                model.rnn.ct = model.rnn.ct.detach()
            elif NET_TYPE  == 'nnRNN':
                model.rnn.calc_rec()

        if i == 0 and NET_TYPE == 'LSTM':
            model.rnn.init_states(data.shape[1])
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss_act = criterion(output.view(-1, ntokens), targets)
        loss = loss_act
        if NET_TYPE  == 'nnRNN' and alam > 0:
            alpha_loss = model.rnn.alpha_loss(alam)
            loss += alpha_loss

        loss.backward()
        losses.append(loss.item())

        if orthog_optimizer:
            model.rnn.orthogonal_step(orthog_optimizer)
        optimizer.step()

        total_loss += loss_act.item()

    return np.mean(losses)
# Loop over epochs.
lr = args.lr
decay = args.Tdecay

rounds = args.rounds

rounds_test_accs = []
rounds_test_losses = []
rounds_avg_epoch = []
rounds_test_bpc = []
for rnd in range(rounds):
    rnn = select_network(args, inp_size)
    model = RNNModel(rnn, ntokens, inp_size, hid_size, args.tied)
    if args.cuda:
        model.cuda()
    print('Language Task')
    print(NET_TYPE)
    print(args)

    criterion = nn.CrossEntropyLoss()
    optimizer, orthog_optimizer = select_optimizer(model, args)
    scheduler = optim.lr_scheduler.StepLR(optimizer,1,gamma=0.5)
    if orthog_optimizer:
        orthog_scheduler = optim.lr_scheduler.StepLR(orthog_optimizer,1,gamma=0.5)
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        exp_time = "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
        if rounds == 1:
            SAVEDIR = os.path.join('./saves',
                                   'PTB',
                                   NET_TYPE,
                                   args.name,
                                   str(args.random_seed),
                                   exp_time)

            if not os.path.exists(SAVEDIR):
                os.makedirs(SAVEDIR)

        else:
            SAVEDIR = os.path.join('./saves',
                                   'PTB',
                                   NET_TYPE,
                                   args.name+"_round_"+str(rnd+1),
                                   str(args.random_seed),
                                   exp_time)

            if not os.path.exists(SAVEDIR):
                os.makedirs(SAVEDIR)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    with open(SAVEDIR + 'hparams.txt','w') as fp:
         for key,val in args.__dict__.items():
            fp.write(('{}: {}'.format(key,val)))
    tr_losses = []
    v_losses = []
    v_accs = []
    norm = []
    alphas = []
    time_per_epochs = []
    best_val_loss = None
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        loss = train(optimizer, orthog_optimizer)
        elapsed = time.time() - epoch_start_time
        time_per_epochs.append(elapsed)
        tr_losses.append(loss)
        val_loss, val_acc = evaluate(val_data)
        v_losses.append(val_loss)
        v_accs.append(val_acc)
        norm.append(model.get_norm())
        alphas.append(model.alpha.clone().detach().cpu().numpy())
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f} | valid accuracy {:5.2f} '.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss),val_loss / math.log(2), val_acc))
        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(SAVEDIR + args.save, 'wb') as f:
                torch.save(model,f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            scheduler.step()
            if orthog_optimizer:
                orthog_scheduler.step()

    with open(SAVEDIR + '{}_Train_Losses'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(tr_losses,fp)

    with open(SAVEDIR + '{}_Train_NORM'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(norm,fp)

    with open(SAVEDIR + '{}_Val_Losses'.format(NET_TYPE),'wb') as fp:
        pickle.dump(v_losses,fp)

    with open(SAVEDIR + '{}_Train_Alpha'.format(NET_TYPE),
                      'wb') as fp:
                pickle.dump(alphas, fp)

    with open(SAVEDIR + '{}_Val_Accs'.format(NET_TYPE),'wb') as fp:
        pickle.dump(v_accs,fp)
# Load the best saved model.
    with open(SAVEDIR + args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss, test_accuracy = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}, test bpc {:8.2f} | test acc {:8.2f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2), test_accuracy))
    rounds_test_accs.append(test_accuracy)
    rounds_test_losses.append(test_loss)
    rounds_avg_epoch.append(sum(time_per_epochs) / float(len(time_per_epochs)))
    rounds_test_bpc.append(test_loss / math.log(2))
    print('=' * 89)
    with open(SAVEDIR + 'testdat.txt', 'w') as fp:
        fp.write('Test loss: {} Test Accuracy: {} Test BPC: {} Average time per epoch in seconds : {}'.format(test_loss, test_accuracy, test_loss / math.log(2), sum(time_per_epochs) / float(len(time_per_epochs))))

if rounds > 1:
    RES_DIR = './results/'
    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)
    avg_loss = sum(rounds_test_losses) / float(len(rounds_test_losses))
    avg_acc = sum(rounds_test_accs) / float(len(rounds_test_accs))
    avg_time = sum(rounds_avg_epoch) / float(len(rounds_avg_epoch))
    avg_bpc = sum(rounds_test_bpc) / float(len(rounds_test_bpc))
    with open(RES_DIR + f'rounds_PTB_{args.name}.txt', 'w') as fp:
        fp.write(
            'Rounds_AVG Test loss: {} Test Accuracy: {} Test BPC: {}  Average time per epoch in seconds : {}'.format(avg_loss, avg_acc, avg_bpc, avg_time ))
