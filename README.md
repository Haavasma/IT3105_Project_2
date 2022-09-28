# IT3105_Project_2 AlphaZero lite.
Monte carlo tree search + DEEP RL project


This is a lite replication of the AlphaZero algorithm for
playing the game Hex.

The project combines Deep RL and monte carlo tree search
to play the game hex at a small scale (trainable on simple hardware)


The current implementation uses a single network with a policy
and a value head for determining action probabilities and value of states.


The network is trained by collecting training
cases during monte carlo tree search, where two differently seeded instances of
the agent plays against itself, and the training data
is labeled based on the outcome of the game.


The training process runs n episodes in parallel, then
combines the training data from all episodes and adds it to the 
replay buffer. At each n iterations of these, 
the policy-value CNN is trained on the data
in the replay buffer and the current best policy 
is replaced if the new policy can beat the current best policy
at least 55% of the time.


The network consists of resnet blocks with a NxNx4 
input and a value prediction and policy prediction head.



This project was done as part of the subject IT3105 at NTNU.
