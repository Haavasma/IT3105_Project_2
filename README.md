# IT3105_Project_2 AlphaZero lite.
Monte carlo tree search + DEEP RL project



This is a lite replication of the AlphaZero algorithm for
playing the game Hex.

The current implementation uses a single network with a policy
and a value head for determining action probabilities and value of states.

The monte carlo tree search algorithm is used for
generating training data for the network and uses 
parallel computing to speed up the data generation process.


The network consists of resnet blocks with a NxNx4 
input and a value prediction and policy prediction head.



This project was done as part of the subject IT3105 at NTNU.
