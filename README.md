# Stair Recurrent Neural Networks

Code for the paper:

**Stair Recurrent Neural Networks**<br>
 Submitted to Conference on Neural Information Processing Systems (NeurIPS), 2021<br>

The code source is based on the implementations of [nnRNN](https://arxiv.org/abs/1905.12080) and  [expRNN](https://arxiv.org/abs/1901.08428)

## Dependencies:
1. torch 
2. torchvision

## General information

1. [task]_nvar.py for ```\rho_j``` in `expn`, `n^2`,  `n`, `log2n`, `logn`, `sqrtlogn`
2. [task]_alpha.py for ```\rho_j``` == `\alpha_j`
2. [task]_expalpha.py for exp-alpha activation 

## Training 

````
python [task]_[activation].py [args]

````

### permuted sequtential MNIST

```
python sMNIST_[activation].py [args]
```

###### sRNN without parameterization  

````
python sMNIST_nvar.py --net-type RNN --nvar log2n --permute --lr 5e-5 --nhid 512 --alpha 0.9 --rinit random --cuda
````

###### sRNN with exponontial parameterization 

```
python sMNIST_nvar.py --net-type expRNN --nvar expn --nhid 512 --permute --lr 5e-4 --lr_orth 5e-5 --alpha 0.99 --rinit cayley --cuda
```

###### sRNN with non-normal parameterization 

```
python sMNIST_nvar.py --net-type nnRNN --nvar n2 --nhid 512 --permute --lr 2e-4 --lr_orth 2e-5 --alpha 0.99 --alam 0.1 --Tdecay 1e-4 --rinit cayley --cuda
```


Options:
- net-type : type of RNN to use in test
- nvar : $\rho_j$ function (`expn`, `n^2`,  `n`, `log2n`, `logn`, `sqrtlogn`)
- nhid : number if hidden units
- epochs : number of epochs
- cuda : use CUDA
- permute : permute the order of the input
- random-seed : random seed for experiment (excluding permute order which has independent seed)
- batch : batch size
- lr : learning rate for optimizer
- lr_orth : learning rate for orthogonal optimizer
- alpha : alpha value for optimizer (always RMSprop) 
- rinit : recurrent weight matrix initialization options: \[xavier, henaff, cayley, random orth.\]
- iinit : input weight matrix initialization, options: \[xavier, kaiming\]
- nonlin : non linearity type, options: \[None, tanh, relu, modrelu\]
- alam : strength of penalty on (&delta; in the paper)
- Tdecay : weight decay on upper triangular matrix values
- save_freq : frequency in epochs to save data and network

#### The employed hyperparameters for the permuted sequential MNIST
| Model (parameterization) | N   | LR                 | LR Orth            | $\alpha$ | $\delta$ | T decay   | A init        |
|--------------------------|-----|--------------------|--------------------|----------|----------|-----------|---------------|
| RNN                      | 512 | $10^{-4}$          |                    | 0.9      |          |           | Glorot Normal |
| RNN-Orth                 | 512 | $5 \times 10^{-5}$ |                    | 0.99     |          |           | Random Orth   |
| expRNN                   | 512 | $5 \times 10^{-4}$ | $5 \times 10^{-5}$ | 0.99     |          |           | Cayley        |
| nnRNN                    | 512 | $2 \times 10^{-4}$ | $2 \times 10^{-5}$ | 0.99     | 0.1      | $10^{-4}$ | Cayley        |
| sRNN                     | 512 | $5 \times 10^{-5}$ |                    | 0,99     |          |           | Random Orth   |
| sRNN (exponential)       | 512 | $5 \times 10^{-4}$ | $5 \times 10^{-5}$ | 0,99     |          |           | Cayley        |
| sRNN (non-normal)        | 512 | $2 \times 10^{-4}$ | $2 \times 10^{-5}$ | 0.99     | 0.1      | $10^{-4}$ | Cayley        |



### Character-level language modeling (PTB dataset) 

```
python language_[activation].py [args]
```

Options:
- net-type : type of RNN to use in test
- nvar : $\rho_j$ function (`expn`, `n^2`,  `n`, `log2n`, `logn`, `sqrtlogn`) 
- emsize : size of word embeddings
- nhid : number if hidden units
- epochs : number of epochs
- bptt : sequence length for back propagation
- cuda : use CUDA
- seed : random seed for experiment (excluding permute order which has independent seed)
- batch : batch size
- log-interval : reporting interval
- save : path to save final model and test info
- lr : learning rate for optimizer
- lr_orth : learning rate for orthogonal optimizer
- rinit : recurrent weight matrix initialization options: \[xavier, henaff, cayley, random orth.\]
- iinit : input weight matrix initialization, options: \[xavier, kaiming\]
- nonlin : non linearity type, options: \[None, tanh, relu, modrelu\]
- alam : strength of penalty on (&delta; in the paper)
- Tdecay : weight decay on upper triangular matrix values
- optimizer : choice of optimizer between RMSprop and Adam
- alpha : alpha value for optimizer (always RMSprop) 
- betas : beta values for adam optimizer 

#### The employed hyperparameters for the character-level language modeling task

| Model (parameterization) | N    | LR                 | LR Orth            | $\alpha$ | $\delta$  | T decay   | A init        |
|--------------------------|------|--------------------|--------------------|----------|-----------|-----------|---------------|
|                                                                Length 150                                                    |  
| RNN                      | 1024 | $10^{-5}$          |                    | 0.9      |           |           | Glorot Normal |
| RNN-Orth                 | 1024 | $10^{-4}$          |                    | 0.9      |           |           | Random Orth   |
| expRNN                   | 1024 | $5 \times 10^{-3}$ | $10^{-4}$          | 0.9      |           |           | Cayley        |
| nnRNN                    | 1024 | $8 \times 10^{-4}$ | $8 \times 10^{-5}$ | 0.9      | 1         | $10^{-4}$ | Cayley        |
| sRNN                     | 1024 | $10^{-4}$          |                    | 0.9      |           |           | Random Orth   |
| sRNN (exponential)       | 1024 | $5 \times 10^{-3}$ | $10^{-4}$          | 0.9      |           |           | Cayley        |
| sRNN (non-normal)        | 1024 | $8 \times 10^{-4}$ | $8 \times 10^{-5}$ | 0.9      | 1         | $10^{-4}$ | Cayley        |
|                                                                Length 300                                                    |  
| RNN                      | 1024 | $10^{-5}$          |                    | 0.9      |           |           | Glorot Normal |
| RNN-Orth                 | 1024 | $10^{-4}$          |                    | 0.9      |           |           | Random Orth   |
| expRNN                   | 1024 | $5 \times 10^{-3}$ | $10^{-4}$          | 0.9      |           |           | Cayley        |
| nnRNN                    | 1024 | $8 \times 10^{-4}$ | $6 \times 10^{-5}$ | 0.9      | $10^{-4}$ | $10^{-4}$ | Cayley        |
| sRNN                     | 1024 | $10^{-4}$          |                    | 0.9      |           |           | Random Orth   |
| sRNN (exponential)       | 1024 | $5 \times 10^{-3}$ | $10^{-4}$          | 0.9      |           |           | Cayley        |
| sRNN (non-normal)        | 1024 | $8 \times 10^{-4}$ | $6 \times 10^{-5}$ | 0.9      | $10^{-4}$ | $10^{-4}$ | Cayley        |


### Copying memory task

```
python copytask_[activation].py [args]
```
Options:
- net-type : type of RNN to use in test
- nvar : $\rho_j$ function (`expn`, `n^2`,  `n`, `log2n`, `logn`, `sqrtlogn`)
- nhid : number if hidden units
- cuda : use CUDA
- T : delay between sequence lengths
- labels : number of labels in output and input, maximum 8
- c-length : sequence length
- onehot : onehot labels and inputs
- vari : variable length
- random-seed : random seed for experiment
- batch : batch size
- lr : learning rate for optimizer
- lr_orth : learning rate for orthogonal optimizer
- alpha : alpha value for optimizer (always RMSprop) 
- rinit : recurrent weight matrix initialization options: \[xavier, henaff, cayley, random orth.\]
- iinit : input weight matrix initialization, options: \[xavier, kaiming\]
- nonlin : non linearity type, options: \[None, tanh, relu, modrelu\]
- alam : strength of penalty on (&delta; in the paper)
- Tdecay : weight decay on upper triangular matrix values

#### The employed hyperparameters for the copying memory task

| Model (parameterization) | N   | LR                 | LR Orth   | $\alpha$ | $\delta$  | T decay   | A init        |
|--------------------------|-----|--------------------|-----------|----------|-----------|-----------|---------------|
| RNN                      | 128 | $10^{-3}$          |           | 0.9      |           |           | Glorot Normal |
| RNN-Orth                 | 128 | $2 \times 10^{-4}$ |           | 0.99     |           |           | Random Orth   |
| expRNN                   | 128 | $10^{-3}$          | $10^{-4}$ | 0.99     |           |           | Henaff        |
| nnRNN                    | 128 | $5 \times 10^{-4}$ | $10^{-6}$ | 0.99     | $10^{-4}$ | $10^{-4}$ | Henaff        |
| sRNN                     | 128 | $2 \times 10^{-4}$ |           | 0.99     |           |           | Random Orth   |
| sRNN (exponontial)       | 128 | $10^{-3}$          | $10^{-4}$ | 0.99     |           |           | Henaff        |
| sRNN (non-normal)        | 128 | $5 \times 10^{-4}$ | $10^{-6}$ | 0.99     | $10^{-4}$ | $10^{-4}$ | Henaff        |
