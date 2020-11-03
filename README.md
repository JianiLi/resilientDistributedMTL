# ResilientDistributedMTL
Code for NeurIPS 2020 paper [Byzantine Resilient Distributed Multi-Task Learning](https://arxiv.org/pdf/2010.13032.pdf)

### Background

In a multi-agent multi-task distributed system, agents aim to learn distinct but correlated models simultaneously. 
They share their models with neighbors and actively learn from peers performing a similar task. 
Such cooperation has been demonstrated to benefit the overall learning performance over the network.
However, distributed algorithms for learning similarities among tasks are not resilient in the presence of Byzantine agents. 
Inthis paper, we present an approach for Byzantine resilient distributed multi-task learning. We propose an efficient online weight assignment rule by measuringthe accumulated loss using an agent’s data and the models of its neighbors.  
A small accumulated loss indicates a large similarity between the two tasks. 
In order to ensure Byzantine resilience of the aggregation at a normal agent, we introduce a step for filtering out larger losses. 
We analyze the approach in the case of convex models and we show that normal agents converge resiliently towards their true target. Further, the learning performance of an agent using the proposed weight assignment rule is guaranteed to be at least as good as the non-cooperative case as measured by the expected regret. 
Finally, we demonstrate the approach using three case studies that include both regression and classification and show that our method has good empirical performance for non-convex models such as convolutional neural networks.

### An example of multi-task distributed system with Byzantine agents <img align="right" src="https://github.com/JianiLi/resilientDistributedMTL/blob/main/TargetLocalization/fig/network_attackerNum20.png" alt="drawing" width="400"/> 
- nodes in the same color performs the same task 
- nodes connected by the links means they are neighbors and can share messages
- nodes in red are Byzantine agents
- Byzantine agents send arbitrary messages to normal agents  

### Case Studies
- [Target Localization](https://github.com/JianiLi/resilientDistributedMTL/tree/main/TargetLocalization)
- [Human Activity Recognition](https://github.com/JianiLi/resilientDistributedMTL/tree/main/HumanActivityRecog)
- [Digit Classification](https://github.com/JianiLi/resilientDistributedMTL/tree/main/DigitClassification)

### Cite the paper
```
@inproceedings{neurips_2020_byzantineMTL,  
  title={Byzantine Resilient Distributed Multi-Task Learning},  
  author={Jiani Li and Waseem Abbas and Xenofon Koutsoukos},  
  booktitle = {Thirty-fourth Conference on Neural Information Processing Systems (NeurIPS)},  
  year      = {2020}  
}
```
