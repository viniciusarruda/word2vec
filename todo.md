# word2vec

remember: you need to be good and understand every point of this code to have a good working code


check here what needs to be saved
do not be so generic now
only do that for word embeddings stuff
train some models fast
get visulizations
make some computations with embeddings and enough
move forward to what really matters to get a good PHD! or to get viewed (which means a good phd)


train a cbow with two datasets
train a skipgram with two datasets
take a look on fasttext, if easy, train with that

go to autocomplete stuff using the above in a separate repo
the above by itself will take several hours..
lot of tabs to read.. but try to focus on what matter -> understand, be a good pytorch programmer, to then implement your own stuff


DO NOT SPEND MORE THAN 4 DAYS HERE!
work morning
study and do this afternoon to night

# TODO

- [x] Change the default seed to 3407 
- [ ] Build your own stuff for CBOW and SkipGram. Only! This is the goal of this repo
 - [ ] Do not bother using pytorch stuff, do your own for this model! 
- [ ] Walk through code fixing TODO stuff
- [ ] add logging -> file

## Experimentation

- [ ] Am I setting seed correctly? Shouldn't the loss be identical?
- [ ] Enable loading and saving model
    - [ ] Enable inference later
    - [ ] Enable continue training
    - [ ] Enable loading a pre-trained model, even outside this project (e.g., original word2vec) - low priority
- [ ] When runing main, check if already exists a experiment with equal config, if so, warn the user?

## Model

- [ ] CBoW
- [ ] SkipGram

## Visualization
- [x] add histogram of embeddings and weights (also for grad, but I don't think it is necessary to keep it, unless it takes too much time to record it - it is a TODO)
- [ ] add hparams (https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3)
- [ ] Visualize the dataset: vocab histogram, words near others stats, etc be the one with data
    - histogram is bad to visualize because vocab is big
- [x] Debug graph! make a simple experiment to see if the problem is only with embeddingbag (number instead of a name in the OpNode)
    - It seems to be a bug with PyTorch (this have been reported, but it seems it isn't a priority for the PyTorch team to fix it)
    - [This](https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3) blog post also shows it in one of its image.
- [x] ERROR: number of rows doesn't match the number of lines in metadata.. one vector above the size of metadata
    - The error was a <BOM> character I've never seen before! I removed it from the dataset. 

## Optimization

- [ ] Should we get the mean of loss before backpropagating? I don't think is necessary, but this is a good experiment. Maybe ask in the redit ML ?

## Dataset

The dataset will be used to all your experiments with LLM for language modeling, thus it will be a work in progress for a long time.
Do not get stuck with it.. you'll need to revisit once in a while to improve statistics, visualization, preprocessing, etc.

- [ ] change to work with sentences instead

- [ ] Wiki datasets is showing encoding issues, with vocab containing weird characters.

- [ ] keep the datasets from pytorch and also a custom one!
    - [ ] Implement dataset from pytorch, start from the easy steps then got to your custom! Otherwise you will lose much time!
    - [ ] check if <bom> is not present in the torchtext wiki version because downloading from the website is present
- [ ] Make the script for the dataset different and separated. The code will only load already preprocessed dataset. The statistics will be saved in the dataset file.

- [ ] What preprocessing the papers here (https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/#dataset-statistics) are using? min_freq at least?
- [x] Save vocab freq for analysis- hist is slow
- [x] Include the "<unk>" token
- [ ] Need to understand padding
    - [ ] Use lines instead of whole file - but need to understand padding before - olga is using lines and torchtext is returning lines
- [x] Save the config used in the experiment folder
- [x] make tqdm knows the size of bar.. -> wrap tqdm with enumerate, not the inverse
- [x] add tqdm into data.py functions
- [ ] use the builtin data stuff instead of mine, do I really need mine implementation? What is my goal? Do I need to keep my implementation for that?
- [ ] read and understand torchtext - search yt video

- [x] Add other dataset. Look the Olga dataset. Also, try to get a dataset from your dataset tab or another stuff.. IDK, bunch of papers as dataset..
- [ ] Compare the model with the mean + max_norm = 1 insdead of only sum
- [x] Need to leave the last batch of non-training dataloaders without filling all!!!!! otherwise it will retrieve the wrong value
    - drop_last = False by default, meaning the last batch will be smaller if the dataset length is not divisible by batch_size
- [x] I think it is important to keep all loss records.. I think andrej said that too
- [x] add the mean loss too, but keep all loss items.. It does not add much cost (premature optimization is the root of all evil)


## Test
- [ ] http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt

word2vec no tensorboard: https://projector.tensorflow.org/


https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/#dataset-statistics
https://huggingface.co/datasets/wikitext/blob/main/README.md











GASTAR UM TEMPO PARA LISTAR E DEFINIR O QUE FALTA PARA FINALIZAR ESSE PROJETO!

>>>>>>>>>> WORD2VEC benchmark: https://www.nltk.org/howto/gensim.html
to check what is your upper/lower bound

-> separar: listar o que tem a ver com pytorch e o que tem com word2vec. Fazer primeiro coisas do boilerplate, depois do word2vec!

you need to compare what is the upper bound, use the word2vec original
-> NOW: starting with computing vector operations
-> save and load, continue training.
-> logging
-> Cbow another approach with mean ? vs norm ? read paper
-> skipgram
-> train on cloud -> if work well and fast, collect large dataset and train it, trying to achieve good acc.


# LINKS

## tensorboard
https://cnvrg.io/tensorboard-guide/
https://www.machinelearningnuggets.com/tensorboard-tutorial/
https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html -> pr curve might be interesting
