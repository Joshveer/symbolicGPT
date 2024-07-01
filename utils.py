import re
import json
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy.optimize import minimize
from torch.utils.data import Dataset
from torch.nn import functional as F
from numpy import * # to override the math functions
from matplotlib import pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_k_top_p_filtering(logits, top_k=0.0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def sample_from_model(model, x, steps, points=None, variables=None, temperature=1.0, sample=False, top_k=0.0, top_p=0.0):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_cond, points=points, variables=variables)
        logits = logits[0, -1, :] / temperature
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        x = torch.cat((x, ix.unsqueeze(0)), dim=1)

    return x

def plot_and_save_results(resultDict, fName, pconf, titleTemplate, textTest, modelKey='SymbolicGPT'):
    if isinstance(resultDict, dict):
        num_eqns = len(resultDict[fName][modelKey]['err'])
        num_vars = pconf.numberofVars
        title = titleTemplate.format(num_eqns, num_vars)

        models = list(key for key in resultDict[fName].keys() if len(resultDict[fName][key]['err'])==num_eqns)
        lists_of_error_scores = [resultDict[fName][key]['err'] for key in models if len(resultDict[fName][key]['err'])==num_eqns]
        linestyles = ["-","dashdot","dotted","--"]

        eps = 0.00001
        y, x, _ = plt.hist([np.log([max(min(x+eps, 1e5),1e-5) for x in e]) for e in lists_of_error_scores],
                        label=models,
                        cumulative=True, 
                        histtype="step", 
                        bins=2000, 
                        density=True,
                        log=False)
        y = np.expand_dims(y,0)
        plt.figure(figsize=(15, 10))

        for idx, m in enumerate(models): 
            plt.plot(x[:-1], 
                y[idx] * 100, 
                linestyle=linestyles[idx], 
                label=m)

        plt.legend(loc="upper left")
        plt.title(title)
        plt.xlabel("Log of Relative Mean Square Error")
        plt.ylabel("Normalized Cumulative Frequency")

        name = '{}.png'.format(fName.split('.txt')[0])
        plt.savefig(name)

        with open(fName, 'w', encoding="utf-8") as o:
            for i in range(num_eqns):
                err = resultDict[fName][modelKey]['err'][i]
                eq = resultDict[fName][modelKey]['trg'][i]
                predicted = resultDict[fName][modelKey]['prd'][i]
                o.write('Test Case {}/{}.\n'.format(i,num_eqns-1))
                o.write('{}\n'.format(eq))
                o.write('{}:\n'.format(modelKey))
                o.write('{}\n'.format(predicted))
                o.write('{}\n{}\n\n'.format(predicted, err))

            print('Avg Err:{}'.format(np.mean(resultDict[fName][modelKey]['err'])))

def tokenize_predict_and_evaluate(i, inputs, points, outputs, variables, 
                                  train_dataset, textTest, trainer, model, resultDict,
                                  numTests, variableEmbedding, blockSize, fName,
                                  modelKey='SymbolicGPT', device='cpu'):
    
    try:
        eq = ''.join([train_dataset.itos[int(i)] for i in outputs[0]])
        eq = eq.strip(train_dataset.paddingToken).split('>')
        eq = eq[0]
        eq = eq.strip('<').strip(">")
        print(eq)
        if variableEmbedding == 'STR_VAR':
                eq = eq.split(':')[-1]

        t = json.loads(textTest[i])

        inputs = inputs[:,0:1].to(device)
        points = points.to(device)
        variables = variables.to(device)

        bestErr = 10000000
        bestPredicted = 'C'
        for i in range(numTests):
            predicted, err = generate_sample_and_evaluate(
                                model, t, eq, inputs, 
                                blockSize, points, variables, 
                                train_dataset, variableEmbedding)

            if err < bestErr:
                bestErr = err
                bestPredicted = predicted
        
        resultDict[fName][modelKey]['err'].append(bestErr)
        resultDict[fName][modelKey]['trg'].append(eq)
        resultDict[fName][modelKey]['prd'].append(bestPredicted)

        return eq, bestPredicted, bestErr
    
    except Exception as e:
        print(f"Error processing equation: {e}")
        return None, None, None

def generate_sample_and_evaluate(model, t, eq, inputs, 
                                 blockSize, points, variables, 
                                 train_dataset, variableEmbedding):
    
    outputsHat = sample_from_model(model, 
                        inputs, 
                        blockSize, 
                        points=points,
                        variables=variables,
                        temperature=0.9, 
                        sample=True, 
                        top_k=40,
                        top_p=0.7,
                        )[0]

    predicted = ''.join([train_dataset.itos[int(i)] for i in outputsHat])

    if variableEmbedding == 'STR_VAR':
        predicted = predicted.split(':')[-1]

    predicted = predicted.strip(train_dataset.paddingToken).split('>')
    predicted = predicted[0]
    predicted = predicted.strip('<').strip(">")
    predicted = predicted.replace('Ce','C*e')

    c = [1.0 for i,x in enumerate(predicted) if x=='C']
    b = [(-2,2) for i,x in enumerate(predicted) if x=='C']
    try:
        if len(c) != 0:
            cHat = minimize(lossFunc, c, args=(predicted, t['X'], t['Y'])) 
            predicted = predicted.replace('C','{}').format(*cHat.x)
    except ValueError:
        raise 'Err: Wrong Equation {}'.format(predicted)
    except Exception as e:
        raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)
    
    Ys = []
    Yhats = []
    for xs in t['XT']:
        try:
            eqTmp = eq + ''
            eqTmp = eqTmp.replace(' ','').replace('\n','')
            for i,x in enumerate(xs):
                eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
            YEval = eval(eqTmp)
        except:
            print('TA: For some reason, we used the default value. Eq:{}'.format(eqTmp))
            raise
            continue
        Ys.append(YEval)
        try:
            eqTmp = predicted + ''
            eqTmp = eqTmp.replace(' ','').replace('\n','')
            for i,x in enumerate(xs):
                eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
            Yhat = eval(eqTmp)
        except:
            print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
            Yhat = 100
        Yhats.append(Yhat)
    err = relativeErr(Ys,Yhats, info=True)
    
    print('\nTarget:{}'.format(eq))
    print('Skeleton+LS:{}'.format(predicted))
    print('Err:{}'.format(err))
    print('-'*10)

    if isinstance(err, (np.complex128, complex)):
        err = abs(err.real)

    return predicted, err

class CharDataset(Dataset):
    def __init__(self, data, block_size, chars, 
                 numVars, numYs, numPoints, target='EQ', 
                 addVars=False, const_range=[-0.4, 0.4],
                 xRange=[-3.0,3.0], decimals=4, augment=False):
        
        if '/' not in chars:
            chars = chars + ['/']
        
        data_size, vocab_size = len(data), len(chars)
        print('data has %d examples, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        self.numVars = numVars
        self.numYs = numYs
        self.numPoints = numPoints
        
        self.paddingToken = '_'
        self.paddingID = self.stoi[self.paddingToken]
        self.stoi[self.paddingToken] = self.paddingID
        self.itos[self.paddingID] = self.paddingToken
        self.threshold = [-1000,1000]
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.target = target
        self.addVars = addVars

        self.const_range = const_range
        self.xRange = xRange
        self.decimals = decimals
        self.augment = augment
    
    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, idx):
        chunk = self.data[idx]
        
        try:
            chunk = json.loads(chunk)
        except Exception as e:
            print("Couldn't convert to json: {} \n error is: {}".format(chunk, e))
            idx = max(0, idx - 1)
            chunk = json.loads(self.data[idx])
            
        eq = chunk[self.target]
        vars = re.finditer('x[\d]+',eq) 
        numVars = max([int(v.group(0).strip('x')) for v in vars], default=0)

        if self.target == 'Skeleton' and self.augment:
            threshold = 5000
            cleanEqn = ''.join([chr if chr != 'C' else '{}'.format(np.random.uniform(self.const_range[0], self.const_range[1])) for chr in eq])

            nPoints = np.random.randint(*self.numPoints)
            try:
                X, y = generateDataStrEq(cleanEqn, n_points=nPoints, n_vars=self.numVars,
                                         decimals=self.decimals, min_x=self.xRange[0], 
                                         max_x=self.xRange[1])

                y = [e if abs(e)<threshold else np.sign(e) * threshold for e in y]

                conditions = (np.isnan(y).any() or np.isinf(y).any()) or len(y) == 0 or (abs(min(y)) > threshold or abs(max(y)) > threshold)
                if not conditions:
                    chunk['X'], chunk['Y'] = X, y

            except Exception as e: 
                print(f"We just used the original equation and support points because of {e}. The equation is {eq}, and we update the equation to {cleanEqn}")
 
        try:
            if self.addVars:
                dix = [self.stoi[s] for s in f'<{numVars}:{eq}>']
            else:
                dix = [self.stoi[s] for s in f'<{eq}>']
        except KeyError as e:
            print(f"KeyError: {e} not found in vocabulary. Equation: {eq}")
            eq = eq.replace('/', '+')
            if self.addVars:
                dix = [self.stoi[s] for s in f'<{numVars}:{eq}>']
            else:
                dix = [self.stoi[s] for s in f'<{eq}>']

        inputs = dix[:-1]
        outputs = dix[1:]
        
        paddingSize = max(self.block_size-len(inputs),0)
        paddingList = [self.paddingID]*paddingSize
        inputs += paddingList
        outputs += paddingList
        
        inputs = inputs[:self.block_size]
        outputs = outputs[:self.block_size]
        
        points = torch.zeros(self.numVars+self.numYs, self.numPoints[1]-1)
        for idx, xy in enumerate(zip(chunk['X'], chunk['Y'])):
            if idx >= self.numPoints[1]-1:
                break
            
            x = xy[0] if isinstance(xy[0], list) else [xy[0]]
            x = x + [0]*(max(self.numVars-len(x),0))

            y = [xy[1]] if isinstance(xy[1], (float, np.float64)) else xy[1]
            y = y + [0]*(max(self.numYs-len(y),0))
            
            p = x+y
            p = torch.tensor(p)
            p = torch.nan_to_num(p, nan=self.threshold[1], 
                                 posinf=self.threshold[1], 
                                 neginf=self.threshold[0])
            points[:,idx] = p

        points = torch.nan_to_num(points, nan=self.threshold[1],
                                 posinf=self.threshold[1],
                                 neginf=self.threshold[0])
        
        inputs = torch.tensor(inputs, dtype=torch.long)
        outputs = torch.tensor(outputs, dtype=torch.long)
        numVars = torch.tensor(numVars, dtype=torch.long)
        return inputs, outputs, points, numVars

def processDataFiles(files):
    text = ""
    for f in tqdm(files):
        with open(f, 'r') as h: 
            lines = h.read()
            if lines[-1]==-1:
                lines = lines[:-1]
            text = ''.join([lines,text])    
    return text

def lossFunc(constants, eq, X, Y, eps=1e-5):
    err = 0
    eq = eq.replace('C','{}').format(*constants)

    for x,y in zip(X,Y):
        eqTemp = eq + ''
        if type(x) == np.float32:
            x = [x]
        for i,e in enumerate(x):
            if type(e) == torch.Tensor:
                e = e.item()
            eqTemp = eqTemp.replace('x{}'.format(i+1), str(e))
        try:
            yHat = eval(eqTemp)
        except:
            print('Exception has been occured! EQ: {}, OR: {}'.format(eqTemp, eq))
            continue
            yHat = 100
        try:
            err += relativeErr(y, yHat)
        except:
            print('Exception has been occured! EQ: {}, OR: {}, y:{}-yHat:{}'.format(eqTemp, eq, y, yHat))
            continue
            err += 10
        
    err /= len(Y)
    return err

def generateDataStrEq(eq, n_points=2, n_vars=3,
                      decimals=4, supportPoints=None, 
                      min_x=0, max_x=3):
    X = []
    Y= []
    for p in range(n_points):
        if supportPoints is None:
            if type(min_x) == list:
                x = []
                for _ in range(n_vars):
                    idx = np.random.randint(len(min_x))
                    x += list(np.round(np.random.uniform(min_x[idx], max_x[idx], 1), decimals))
            else:
                x = list(np.round(np.random.uniform(min_x, max_x, n_vars), decimals))
            assert len(x)!=0, "For some reason, we didn't generate the points correctly!"
        else:
            x = supportPoints[p]

        tmpEq = eq + ''
        for nVID in range(n_vars):
            tmpEq = tmpEq.replace('x{}'.format(nVID+1), str(x[nVID]))
        y = float(np.round(eval(tmpEq), decimals))
        X.append(x)
        Y.append(y)
    return X, Y

def divide(x, y):
  x = np.nan_to_num(x)
  y = np.nan_to_num(y)
  return np.divide(x,max(y,1.0))

def sqrt(x):
  x = np.nan_to_num(x)
  x = np.abs(x)
  return np.sqrt(x) 

def log(x, eps=1e-5):
  x = np.nan_to_num(x)
  x = np.sqrt(x*x+eps)
  return np.log(x)

def exp(x, eps=1e-5):
    x = np.nan_to_num(x)
    return np.exp(x)

def mse(y, y_hat):
    y_hat = np.reshape(y_hat, [1, -1])[0]
    y_gold = np.reshape(y, [1, -1])[0]
    our_sum = 0
    for i in range(len(y_gold)):
        our_sum += (y_hat[i] - y_gold[i]) ** 2

    return our_sum / len(y_gold)

def relativeErr(y, yHat, info=False, eps=1e-5):
    yHat = np.reshape(yHat, [1, -1])[0]
    y = np.reshape(y, [1, -1])[0]
    if len(y) > 0 and len(y)==len(yHat):
        err = ( (yHat - y) )** 2 / np.linalg.norm(y+eps)
        if info:
            for _ in range(5):
                i = np.random.randint(len(y))
                print('yPR,yTrue:{},{}, Err:{}'.format(yHat[i],y[i],err[i]))
    else:
        err = 100

    return np.mean(err)

# Main execution
if __name__ == "__main__":
    # Set up your configuration, model, and datasets here
    
    # Example of how to use the main functions:
    set_seed(42)  # Set a random seed for reproducibility
    
    # Load your data and create datasets
    train_files = ["path/to/train/files"]
    test_files = ["path/to/test/files"]
    
    train_text = processDataFiles(train_files)
    test_text = processDataFiles(test_files)
    
    # Define your vocabulary (chars)
    chars = "your_vocabulary_here"
    
    # Create datasets
    train_dataset = CharDataset(train_text, block_size=your_block_size, chars=chars, 
                                numVars=your_num_vars, numYs=your_num_ys, 
                                numPoints=your_num_points)
    
    test_dataset = CharDataset(test_text, block_size=your_block_size, chars=chars, 
                               numVars=your_num_vars, numYs=your_num_ys, 
                               numPoints=your_num_points)
    
    # Create your model (you'll need to define this)
    model = YourModelClass(config)
    
    # Create a trainer (you'll need to define this)
    trainer = YourTrainerClass(model, train_dataset)
    
    # Train your model
    trainer.train()
    
    # Evaluate on test set
    results = {}
    for i, (inputs, outputs, points, variables) in enumerate(test_dataset):
        eq, predicted, err = tokenize_predict_and_evaluate(
            i, inputs.unsqueeze(0), points.unsqueeze(0), outputs.unsqueeze(0), 
            variables.unsqueeze(0), train_dataset, test_text, trainer, model, 
            results, your_num_tests, your_variable_embedding, your_block_size, 
            "your_output_file.txt", modelKey='SymbolicGPT', device=your_device
        )
        
        if eq is not None:
            print(f"Equation: {eq}")
            print(f"Predicted: {predicted}")
            print(f"Error: {err}")
        else:
            print(f"Skipping equation {i} due to an error.")
    
    # Plot and save results
    plot_and_save_results(results, "your_output_file.txt", your_plot_config, 
                          your_title_template, test_text, modelKey='SymbolicGPT')
