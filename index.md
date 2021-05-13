<h1> 
  <b>Introduction</b> 
</h1>
  <p> 
Reinforcement learning (RL) has become one of the most prolific techniques within artificial intelligence. In our project, we will be taking advantage of this technique to teach an agent how to navigate a virtual maze with directional arrows using deep reinforcement learning. These directional arrows will be placed on the walls and will dictate which way for the agent to go to reach the final goal state – which is signified as a purple wall at the end of the maze. We chose to use deep reinforcement learning to solve this problem because it allows us to easily train our agent by breaking down the path towards our goal state into a series of moves that are signified as being either good (moving towards the goal state) or bad (moving away from the goal state).
</p>

<p>
To accomplish this task, we will be using a variety of previous work done on the topic. Specifically, Dr. Anthony Clark’s research on the creation of the virtual maze and his existing program for it. To implement RL on the existing program, we will use the toolkit known as OpenAI Gym – an open source toolkit that allows developers to implement different algorithms on the existing programs in certain environments - and StableBaselines3 to supply our python RL algorithm . In our process of using <a href="https://gym.openai.com/">OpenAI Gym</a> we will be using the <a href="https://pytorch.org/">PyTorch</a>. 
</p>

<p>
 In addition to these resources, we will use ideas from previous work done on related projects, such as the resources listed below:
</p>
<p>
<ul>
  <li> <a href="https://simoninithomas.github.io/deep-rl-course/">This</a> deep learning course (Thomas), which provided the baseline for our understanding of RL and algorithms such as Q-learning, Deep Q-learning, and A2C. In addition, it helped us with PyTorch implementations and gave us many examples of RL projects. It was essentially the starting point of our project. We supplemented the background knowledge from Thomas along with articles from <a href="https://towardsdatascience.com">Towards Data Science</a> to provide us with additional information.
   <li> Also projects such as <a href="https://ieeexplore.ieee.org/abstract/document/8957297?casa_token=JTVW2Y0EiC0AAAAA:27v7m8lyZQv2Fzr_z1g_7siXz9q38bC3Y0o8gjPa3zc63nFnDR8AEF7hdET8vkxC8jyqhq8kPi0">“Deep Reinforcement Learning for Instruction Following Visual Navigation in 3D Maze-like Environments”</a> (Devo, Costante, and Valigi, 2006) or <a href="https://magnus-engstrom.medium.com/building-an-ai-to-navigate-a-maze-899bf03f224d">“Building an AI to Navigate a Maze”</a> (Engstrom, 2019), helped immensely. These projects have closely related goals to ours. They helped us understand different approaches to teaching agents to navigate a maze and what we would need to consider when implementing our own algorithm(s).
    <li> And finally <a href="https://www.altexsoft.com/blog/datascience/reinforcement-learning-explained-overview-comparisons-and-applications-in-business/">“Reinforcement Learning Explained: Overview, Comparisons and Applications in Business”</a> (Altexsoft, 2019) and <a href="https://openreview.net/pdf?id=SJMGPrcle">“Learning to Navigate in Complex Environments”</a> (Mirowski, 2017) gave us additional background knowledge on RL algorithms and problems concerned with navigating an environment. They helped us wrap our brain around the problem we would be attempting to tackle as well.
</ul>
</p>

<h1>
  What is Reinforcement Learning?
</h1>

<p>
Though it is touched on briefly in our introduction, it is important as a reader to understand exactly what RL is. According to <a href="https://deepsense.ai/what-is-reinforcement-learning-the-complete-guide/">deepsenseai</a> (2018), it is the training of machine learning models to make a sequence of decisions. It is, in essence, a game in which a machine uses rewards to find the solution to the game at hand. 
In summary, RL can be broken down into <a href="https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/">6 major generic steps</a>:
</p>
<p>
<ul>
  <li>Observing an environment</li>
  <li> Deciding how the agent is to act based on a particular strategy</li>
  <li> Making sure that the agent acts in accordance with the strategy</li>
  <li> Awarding the agent with a reward or penalizing the agent for bad actions</li>
  <li>Teaching the agent to learn from its actions based on the rewards/penalties it has received</li>
  <li> Continue teaching the agent until it finds the optimal strategy to solve the problem</li>
</ul>
</p>
<p>
The idea of RL has been around since the 1960s and can be applied to millions of different situations in both real life and simulated environments. 
</p>

<h1> 
  <b>
    Methods
  </b> 
</h1>
<p>
The goal of our project, as explained above is to implement a reinforcement learning algorithm onto an agent to solve a customized maze. To do this we attempted to teach this agent and use a variety of different reinforcement learning techniques including Q-learning, Deep Learning, and A2C. These methods will be explained below.
</p>

<h3>
  <b>
    Q-learning
  </b> 
</h3>
<p>
Our initial idea for this project was to train our agent using Q-learning which is one of the most basic forms of reinforcement learning and one that is well-suited for beginners using RL. Q-learning works so an agent picks an action to go towards a particular state and receives a value from that state. The value depends on the value the state has in the Q-table which is updated as the agent moves. If a negative value is associated with that state, then it will be translated to the Q-table and vice versa with a positive value. After Q-learning training, theoretically, the agent will have found the most optimal rewarding path to its goal state.
</p>

<p>
We based our Q-learning model off of the videos and work done by <a href="https://simoninithomas.github.io/deep-rl-course/">Thomas Simonini</a> in which he creates a taxi agent that learns to navigate in a city with the goal of transporting its passengers to the correct location. This existing model stood out to us because the problem it solves is so similar to ours. However, our environment was too complex for us to successfully apply Q-learning (based on our limited knowledge of the topic.) This was mainly due to the fact that our environment had about 100,000 different states whereas the taxi example only had 10,000 possible states. Therefore our implementation of Q-learning was unsuccessful.
</p>

<h3>
  <b>
    Deep Q-learning
  </b>
</h3>
<p>
After our attempt to implement Q-learning, we moved onto a more complex subject of deep learning called deep Q-learning (DQN). DQN at its simplest, is using the same Q-learning approach as described above, but now neural networks are used. The neural networks help to approximate the Q-value function, by taking the state as input and then producing the best possible move based on the Q-value score. This approach made a lot of sense to us. Since DQN can take the frames of the max as input and then output a vector of possible actions in this state. Then we could just take the biggest Q-value of the vector to tell our agent what the best action is. As our agent learns, it would be able to identify which frames are associated with the best action to perform. We thought that having frames as the input was particularly useful since the walls have arrows indicating which direction to go once the agent reaches the wall.
</p>

<p>
In our implementation of DQN on our agent we tried to follow examples done by <a href="https://stable-baselines3.readthedocs.io/en/master/">Stable-Baselines3</a>, which is a set of sourced RL algorithms implemented with PyTorch. We used their DQN algorithm on our agent to have the agent ‘learn’ the maze. To learn the maze required the agent to take many steps, so we felt training our agent on 1,000, 50,000, and 200,000 timesteps would suffice. After the agent is done learning, we would test the agent in the maze and record the results/ time it took for the agent to find the goal state. 
</p>

<h3>
  <b>
    A2C
  </b>
</h3>
<p>
Since we wanted to test our agent on multiple algorithms, the next one we chose to do was <a href=https://arxiv.org/abs/1602.01783>A2C</a> (Advantage Actor Critic). This approach is unique in that in makes use of two neural networks, a “Critic” which measures the efficacy of an action and an “Actor” which tells the agent what actions to take. Explained in an <a href="https://towardsdatascience.com/advanced-reinforcement-learning-6d769f529eb3">article</a> on advanced reinforcement learning, the actor adjusts the probability of taking an action based on the estimated advantage of the current move and the critic updates the advantage based on the reward from following the actor’s policy.  A2C is an algorithm from Stable-Baselines3 that also uses deep RL, but uses a synchronous gradient descent to optimize the deep neural network controllers. The synchronous characteristic of A2C allows for our agent to complete a segment of the maze, update the global parameters, and then restart a new segment of the maze with parallel agents having the same parameters. This structure tells us how much more efficient a specific action is compared to the average action taken at the current state is (this is found by taking the average of the gradient).
</p>
<p>
To measure the performance of this algorithm, we want to follow similar procedures above in which the agent learns on a variety of timesteps, measures the time it took to learn, and then the record the umber of steps and time it took for the agent to solve the maze.
</p>

<h3>
  <b>
    Rewards
  </b>
</h3>
<p>
In RL, rewards are the primary tool an agent can use in finding its goal state. For our reward system, we wanted to base our reward system off the distance between our agent and the goal state. However, this system could lead to problems with the agent getting stuck in corners or walls. To combat this challenge, we wanted to include additional features to our reward system such as applying a negative reward for staying in the same location after a step. Once the basic system and features have been implemented we will look to expand on our system until our agent is as efficient as possible.
</p>

<h1>
  <b>
    Results
  </b>
</h1>
<p>
Due to time constraints our agent was unable to be trained to solve a maze, due to difficulty in understanding RL in the given timeframe and some implementation problems with openAI gym and a custom environment. Therefore, we were unable to record the time it took for each agent to solve a maze using a different algorithm. We were, however, able to gather data on the time it took to train these agents and record some of the progress each one made in the virtual maze. We were also able to gather data on how long it took to train each agent depending on the algorithm used and the steps taken during training.
</p>

<p>
As mentioned above, we were unable to train a simple Q-learning agent due to the complexity of our problem. This means we were unable to provide any results for basic Q-learning applied to our algorithm. Our major difficulties once we got our program to run and decided on an RL algorithm was implementing a sufficient reward system for our agent. We specifically had trouble understanding how to access variables after each step for our agent. For example, we didn’t know how to access variables from the previous iteration. This meant that we were unable to track where our agent moved after each step, making it difficult to create an efficient reward system. Additionally, we could have initially had our start state be closer to our goal state. Not only would this have reduced the time needed to train our agent, but it also could have made our reward system more simple.
</p>

<p>
Using DQN, we were able to train our agent on 1,000, 10,000 and 100,000 steps due to time constraints, we chose these arbitrary values for training. The training took a very long time (nearly 5+ hours using 100,000 steps). DQN training is often time consuming because it trains based on frames which are large inputs. Though the agent never left the starting position when trained with 1,000 steps, training with 10,000 and above timesteps saw our agent travel a short distance in search of the goal state. However, it would always get stuck in a corner eventually. This means that the agent was not aware of where the goal state was due to the insufficient reward system. So during its training of 10,000 steps, it would try to explore towards the goal, but get stuck. Since it would still receive the same reward when getting stuck, the agent had no incentive to move from its position.
</p>


<p>
We were also able to train and run our agent using A2C. Training with A2C, took even longer than DQN. This is likely due to the fact that instead of training one neural network, we are now training two which increases the amount of time it takes to train the agent. When learning on 100,000 timesteps using A2C, the agent took around a day to finish training and we still did not see significant results - our agent did not ever travel far away from its initial position as the agent would spin in circles or move slightly forward then backwards. This mirrored the results of our implementation of DQN.
</p>

<p>
We understand our results would not likely be reproduced again if sufficient code in training was used or a better reward system was put into place.
 </p>
<h1>
  <b>
    Discussion
  </b>
</h1>
<p>
The goal of our work was to implement an algorithm that would teach an agent how to navigate a virtual maze. Our algorithm was then to be evaluated on the efficiency of our agent solving the maze (percentage of best possible moves taken and what percentage of iterations are able to successfully solve the maze). The maze includes a start position, arrows on walls indicating which direction to turn at a wall, and a goal state at the end of the maze. The position of the start state, goal state, and configuration of the maze are random on each iteration. This algorithm was then to be applied to the work of Dr. Anthony Clark. 
 </p>
 <p>
Unfortunately, we did not succeed in training our agent to reliably navigate the virtual maze.  That is not to say our work was without merit. We gained valuable knowledge in RL, a crucial subfield within Machine Learning and a technique with vast applications. Additionally, our failures with particular RL strategies allowed us to research and attempt to apply various RL algorithms which gave us a much more varied and holistic understanding of RL overall. As often is the case, our failures likely taught us much more than instant success with the first approach we tried would have.
</p>
<p>
Ethically, the direct application of our work has few implications. The possible application of RL into a real-world robot is where most ethical considerations come into play. McBride et al. provides a useful framework for how to conduct ethical research in the field of robotics. Specifically, in the case of robots that are meant to interact with society, it is important to understand that when teaching a robot to make decisions, the decisions that the robot makes are ultimately the responsibility of the engineers who programmed and created it. For example, if a robot were to run into someone and injure them, the fault would lie with the engineers. The uncertainty and complexity involved with creating interactive robots complicate the ethical issues surrounding them.
 </p>
 <p>
  On a broader scale, the widespread development of robots has the ability to vastly alter the way society operates. A report carried out by the World Economic Forum hypothesizes that by 2025 50% of all tasks in the US will be performed by robots (an increase of 20% from 2020). However, the WEF also asserts that while this would lead to a loss of around 85 million jobs, the proliferation of intelligent and interactive AI will also lead to the creation of around 97 million new jobs. While this is an overall increase in jobs, there is still the ethical issue of weather this change will truly benefit society, and if this change will disproportionally benefit society. Acemoglu et al. conducted a study which found that the increased use of robots to perform tasks will serve to further widen the income gap.
 </p>
  <p>
  While teaching a robot to navigate a virtual maze seems fairly harmless ethically, it is important to understand the field more broadly and consider the potential ethical issues with the development of robots.
  </p>
 
<h1>
  <b>
    Reflection
  </b>
</h1>
<p>
We spent a lot of time reading about different applications and techniques of RL and going through code of related projects. Therefore, we are quite disappointed to not have a working agent to show for all our work. If we could do the project over again, I think we would have tried to start coding a little bit earlier. We knew implementing a working RL algorithm would be difficult but we did not expect to run into as many roadblocks as we did. Specifically, understanding the prewritten code and how the environment operates was quite difficult, especially since it was in an area we had not yet learned in class. While there is a plethora of great resources on the topic of RL and maze solving, being able to sift through all of the information and figure out what applied well to our needs was an additional area of difficulty. For example, the taxi agent problem <a href="https://simoninithomas.github.io/deep-rl-course">Thomas Simonini</a> seemed like a really great resource since on the surface it solves a very similar problem to ours. It wasn’t until spending hours attempting to implement a similar algorithm, and trying to figure out why it wasn’t working for our agent did we understand that our environment had 10 times the number of possible states which severely complicated applying the same logic.
 </p>
<p>
Finally, we were confused about what package we should use. We originally spent time applying Stable Baselines 1 to our environment, but then found out that that version was out of date, and that was why it wasn’t working with our program. We then switched to Stable Baselines 3 and finally got our program to run. If we had more time to work on the project, we would continue to implement our RL algorithm, focusing specifically on our reward system. From there, we would continue to test out different algorithms to see which one was most efficient.
</p>

