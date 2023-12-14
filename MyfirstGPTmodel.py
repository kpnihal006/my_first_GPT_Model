"""#My first GPT model
Explanations of each line is given as comments.
 ~Nihal K P

paper reference ~ Attention Is All You Need
tutorial reference ~ https://www.youtube.com/watch?v=UU1WVnMk4E8
openwebtext is used as our dataset.

Overview of code blocks :
~Read file and obtain unique tokens. Encode and decode tokens.
~obtain random sequence of tokens
~obtain random block of tokens from sequence
~estimate loss function for the model
~Self attention head class with key, query, value
~forward pass, masking and computing attention scores
~Multi head attention class for parallel self attention with a forward pass
~FeedForward class to apply activation function and linear transformations
~Block class to call the block functions from the classes
~forward pass for the layers or blocks
~Model class to generate embedding and position table. Apply weights to layer
~initialize weights for linear and embedding layers
~forward pass for the model and loss calculation
~function to generate output based on output
~optimizer and model definition
~printing loss for each iteration
~storing model in pickle
~testing the model with a prompt

We are using pytorch as our main deep learning framework
torch.nn library contains the basic blocks of the graph and the layers for tehe neural network
torch functional contains convolution, pooling, activation and other functions
mmap is memory mapping files to edit it directly in the memory through our code. It gives access to the program by storing on RAM.
Refer https://medium.com/analytics-vidhya/memory-mapping-files-and-mmap-module-in-python-with-lot-of-examples-d1a9a45fe9a3
Pickle is used to save our models so we don't have to train from start everytime we want to run """

import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle

print('Running GPT Training Model')

#use gpu (cuda) if available else uses cpu
device='cuda' if torch.cuda.is_available() else 'cpu'

print('device being used:',device)

#batch size is the number of training samples trained in one iteration or one forward/backward pass
#The entire training set is divided into batches.
#block size is the size of each sequence or block
#To obtain the number of blocks the model sees in one iteration, divide batch size by block size
#Dropout is used to randomly drop neurons from the network to reduce vanishing gradient problem.
#The learning rate is related to gradient descent. The size of the step the model takes to reach a minima.
#n_embed is the number of input embeddings
#n_head is the number of enocder, decoder blocks
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = 0.01
eval_iters = 100
n_embed = 384
n_head = 4
n_layer = 4
dropout = 0.2

print(f'batch size {batch_size} \n block size {block_size} \n max iteration {max_iters}')
print(f'learning rate {learning_rate} \n evaluation iterations {eval_iters} \n n embeddings {n_embed}')
print(f'n head {n_head} \n n layers {n_layer} \n dropout {dropout}')
#Open the file in read mode with encoding utf-8. Extract unique characters from the file.
#chars stores the sorted set (set datatype cannot have duplicates) as a list
#vocab_size has the total number of characters
chars=""
with open("openwebtext/vocab.txt",'r',encoding='utf-8') as f :
    text = f.read()
    chars = sorted(list(set(text)))

print('set of characters \n', chars)
vocab_size = len(chars)

print('vocabulary size ',vocab_size)

#creating a dictionary mapping characters and integers and vice-versa.
#Encoding and Decoding functions to encode characters to integers based on the dictionary.
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [ string_to_int[c] for c in s]
decode = lambda l: ''.join( [ int_to_string[i] for i in l ])

#obtain small randomly selected sequences from the text file, encode and sent for training.
def get_random_chunk(split):

    #2 different files for training and validation
    filename = 'openwebtext/train_split.txt' if split == 'train' else 'openwebtext/val_split.txt'
    
    #'rb' is reading in binary. fileno() obtains the file descriptor in memory. ACCESS_READ gives access to file in read only mode
    with open(filename,'rb') as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
            
            #obtain the file size
            file_size = len(mm)

            #choosing a random position to start reading in the file which does not exceed file size
            start_pos = random.randint(0,(file_size) - block_size * batch_size)

            #seek the random position
            mm.seek(start_pos)
            
            #choosing the block
            block = mm.read(block_size*batch_size-1)

            #decoding block to a string. '\r' removes carriage return(new line)from block
            decoded_block = block.decode('utf-8',errors='ignore').replace('\r','')

            #encoding the block and storing in data
            data = torch.tensor(encode(decoded_block),dtype=torch.long)
    
    print('random chunk of data \n',data)

    return data


def get_batch(split):
    
    #get a random block of data from function
    data = get_random_chunk(split)
    
    #random int assignment to shuffle the dataset
    ix = torch.randint(len(data)-block_size,(batch_size,))
    
    #data
    x = torch.stack([data[i:i+block_size] for i in ix])

    #target. Bigram language model - The target or next character is offset by 1.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    #switch from cpu to gpu
    x, y = x.to(device), y.to(device)

    return x,y

#switch to no grad mode where gradient descents are removed.
@torch.no_grad()

def estimate_loss():
    out = {}  
    
    #set model to evaluation mode
    model.eval()

    #obtain train and val predictions and loss
    for split in ['train','val']:
        
        #eval_iters is defined initially. Limited based on hardware
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)

            #logits is predictions
            logits, loss = model(X,Y)

            #storing the loss value for each iteration
            losses[k] = loss.item()

        #find mean of the losses for each split.
        out[split] = losses.mean()
    
    #switch to train mode.
    model.train()
    return out

class Head(nn.Module):
    
    #One head of self-attention. Refer the model in the reference paper.
    #each head learns a different semantic info from the embeddings.
    def __init__(self, head_size):
        super().__init__()

        #self attention uses keys queries and values
        #key is a token in a sentence
        #query is the amount of attention it will receive based on the context of the key.
        #value is the weighted ratio of a key and query.
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)

        #using a triangle of bottom ones matrix to mask attention for predicting the next tokens
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)

    #forward pass
    def forward(self,x):
        
        #extract batch size sequence length or time-step, input embedding or channels
        B,T,C = x.shape
        
        #key and query is obtained after passing input through linear layers
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        #the query and key is matrix multiplied(@) after transposing the head size and channel then finding sqrt of head size.
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        
        #lower triangle matrix of size T,T. It is compared to 0 and filled to -infinity where it is true.
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        
        #softmax function is applied for normalization
        wei = F.softmax(wei, dim=-1)
        
        #regularization through dropout
        wei = self.dropout(wei)
        
        #compute weights or values. perform the weighted aggregation of the values.
        v = self.value(x) # (B,T,hs)
        #obtain output B,T,C
        # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    #multiple heads of self-attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()

        #create a list of num_heads for instances of Head each performing self-attention independently
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        #projects concatenated outputs back to the original embedding space n_embed.
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):

        #passes input through each of the heads in self.heads along head_size
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        #applies projection then dropout to maintain dimensionality
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):

    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(

            #takes input to linear layer and gives output 4*input. ( wx + b, fully connected layer)
            nn.Linear(n_embed, 4 * n_embed),

            #introduces non-linearity
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):

    #transformer block
    def __init__(self, n_embed, n_head):
        super().__init__()

        #calculating head size
        head_size = n_embed // n_head

        #self attention
        self.sa = MultiHeadAttention(n_head, head_size)

        #feedforward
        self.ffwd = FeedForward(n_embed)

        #layer normalization - subtract mean of neuron outputs and divide by standard deviation
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    #forward pass for transformer block.
    def forward(self,x):

        y = self.sa(x)  #self attention
        x = self.ln1(x+y)   #layer normalization
        y = self.ffwd(x)    #feedforward
        x = self.ln2(x+y)   #layer normalization
        return x

#combining all the blocks and building the model
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        #stores the embedding tokens
        #embedding is mapping tokens to vectors. The vector dimension is (vocab_size, n_embed)
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)

        #stores the position information
        #Position of tokens embedded in vectors of dimension (block_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size,n_embed)

        #transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embed,n_head) for _ in range(n_layer)])

        #normalization
        self.ln_f = nn.LayerNorm(n_embed)

        #language modelling to determine weights
        self.lm_head = nn.Linear(n_embed,vocab_size)

        #applies weights to each function in the model
        self.apply(self._init_weights)
    
    #initializing weights
    def _init_weights(self,module):

        #initializes weights and bias for linear layers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        #initializes only weights for embedding layers.
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    #forward pass of model
    def forward(self, index, targets=None):

        #batch size and sequence length of input
        B, T = index.shape

        #obtain token embedding from table
        tok_emb = self.token_embedding_table(index)

        #generate positional embedding from table
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))

        #add both element wise
        x = tok_emb + pos_emb

        #pass through transformer blocks
        x = self.blocks(x)

        #pass through linear layer for normalization and prediction
        x = self.ln_f(x)
        logits = self.lm_head(x)

        #loss calculation
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)

            #uses cross-entropy to calculate loss
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    #generating an output
    def generate(self, index, max_new_tokens):

        for _ in range(max_new_tokens):
            index_cond = index[:,-block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index,index_next),dim=1)
        return index

#defining the model and switching to gpu
model = GPTLanguageModel(vocab_size)
m = model.to(device)

#optimizaer is used Adam
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

#printing the loss for each iteration or step
for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step:{iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    xb, yb = get_batch('train')
    logits, loss = model.forward(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

#pickle library is used to store the model for use later without re-training
with open('model-01.pkl','wb') as f:
    pickle.dump(model,f)
print('model saved ')

#prompt to test the model.
prompt = 'This is a test to see if the model works.'
context = torch.tensor(encode(prompt),dtype=torch.long,device=device)
generated_chars = decode(m.generate(context.unsqueeze(0),max_new_tokens=100)[0].tolist())
print(generated_chars)