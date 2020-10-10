
### Case Study - Digit Classification
We consider a network of ten agents performing digit classification. 
Five of the ten agents have access to the MNIST dataset (group 1) and the other five have access to the synthetic dataset(group 2) that is composed by generated images of digits embedded on random backgrounds. 
All the images are preprocessed to be 28×28 grayscale images. 
We model each agent as a separate task and use a complete graph to model the network topology. 
An agent does not know which of its neighbors are performing the same task as the agent itself. We use a CNN model of the same architecture for each agent and cross-entropy-loss.

### Download Dataset
- Synthetic_digits Dataset can be downloaded here: https://www.kaggle.com/prasunroy/synthetic-digits
- Download the dataset, unzip it, rename the folder to "synthetic_digits", and put the folder under DigitClassification/

### MNIST and Synthetic Digits
 <img src="https://github.com/JianiLi/resilientDistributedMTL/blob/main/DigitClassification/plot_results/mnist_synthetic.jpg" alt="drawing" width="800"/> 

### Instructions
Tested on python 3.7

- Run main.py to reproduce the results shown in the paper.
- Run synthetic_images_visualization.py to generate the synthetic digit images.

- In main.py, "rule" can be "no-cooperation", "loss", "distance", " average", as explained in the paper.
- Set "attacker" as [] to simulate the attack-free case, and set it to be e.g., [2, 7], to simulate the case when agent 2 and agent 7 are attackers.

### Results
Results show that the loss-based weight assignment rule outperforms all the other rules as well as the non-cooperative case, 
with respect to the mean and range of the average loss and accuracy, with and without the presence of Byzantine agents. 
Hence, our simulations imply that the loss-based weights have accurately learned the relationship among agents. 
Moreover, normal agents having a large regret in their estimation indeed benefit from cooperating with other agents having a small regret. 
 <img src="https://github.com/JianiLi/resilientDistributedMTL/blob/main/DigitClassification/plot_results/legend.jpg" alt="drawing" width="600"/> 
- Average testing loss and accuracy for normal agents in group 1 (MNIST digit classification):
 <img src="https://github.com/JianiLi/resilientDistributedMTL/blob/main/DigitClassification/plot_results/paper_result_group1.jpg" alt="drawing" width="1000"/> 
- Average testing loss and accuracy for normal agents in group 2 (Synthetic digits classifcation):
 <img src="https://github.com/JianiLi/resilientDistributedMTL/blob/main/DigitClassification/plot_results/paper_result_group2.jpg" alt="drawing" width="1000"/> 

### Cite the paper
```
@inproceedings{neurips_2020_byzantineMTL,  
  title={Byzantine Resilient Distributed Multi-Task Learning},  
  author={Jiani Li and Waseem Abbas and Xenofon Koutsoukos},  
  booktitle = {Thirty-fourth Conference on Neural Information Processing Systems (NeurIPS)},  
  year      = {2020}  
}
```
