# KG2ART_RepLearning
- In this experiment, 5 datasets are included. The binary forms in 'pickle' format are available in 'data' folder.
- The textual format of triples of all 5 five datasets are also available in 'src_data' folder.
The 5 datasets are Codex-m, Jet_Engine, Kinship, Nations and UMLS

- Preprocessing of these datasets are required for training the models.
The data source to preprocess can be found in 'src_data' folder.
The preprocess function can be found in preprocess_datasets.py

- In models.py:
	- For tail prediction, the following parameter should be set as
         'target = RHS'. 
	- For head prediction, the following parameter should be set as
         'target = LHS'.
	- For relation prediction, the following parameter should be set as
         'target = rel' 

- Run the following query to produce the results for those models. This query could be found in 'Experiment.ipynb'
For example: 
%run main.py --dataset Jet_Engine --score_rel True --model RESCAL --rank 200 --learning_rate 0.1 --batch_size 200 --lmbda 0.05  --w_rel 4 --max_epochs 100 

- In the query, dataset name,model and hyper parameters can be specified.

- For experiments with state-of-the-art models (RESCAL, TuckER, ComplEX, ConvE, CompGCN), wandb (https://wandb.ai) is required. 

- For KG2ART model, the functions for encoding and inferences are defined in dambART.py. It can be included in the jupyter notebook 
by calling  "import dambART.py"
