
### Case Study - Target Localization
Target localization is a widely-studied linear regression problem. 
The task is to estimate the location of the target by minimizing the squared error loss of noisy streaming sensor data. 
We consider a network of 100 agents with four targets.
Agents in the same color share the same target, however, they do not know this group information beforehand.

### Dataset
- Data is generated in the code.

### Instructions
Tested on python 3.7

- Run multi_task.py to reproduce the results shown in the paper (We also provide single-task.py where agents have the same target).
- In multi_task.py, we simulate four cases: "no-cooperation", "loss", "distance", " average", as explained in the paper.
- "numAgents" is the total number of agents in the network including the Byzantine agents, "attackerNum" defines the number of attackers.

### Results
Results show that the loss-based weight assignment rule outperforms all the other rules as well as the non-cooperative case, 
with respect to the mean and range of the average loss and accuracy, with and without the presence of Byzantine agents. 
Hence, our simulations imply that the loss-based weights have accurately learned the relationship among agents. 
Moreover, normal agents having a large regret in their estimation indeed benefit from cooperating with other agents having a small regret. 
 <img src="https://github.com/JianiLi/resilientDistributedMTL/blob/main/TargetLocalization/plot_results/paper_result.jpg" alt="drawing" width="1000"/> 


### Cite the paper
```
@inproceedings{neurips_2020_byzantineMTL,  
  title={Byzantine Resilient Distributed Multi-Task Learning},  
  author={Jiani Li and Waseem Abbas and Xenofon Koutsoukos},  
  booktitle = {Thirty-fourth Conference on Neural Information Processing Systems (NeurIPS)},  
  year      = {2020}  
}
```
