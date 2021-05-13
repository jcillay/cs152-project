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
<ul>
  <li> <a href="https://simoninithomas.github.io/deep-rl-course/">This</a> deep learning course (Thomas), which provided the baseline for our understanding of RL and algorithms such as Q-learning, Deep Q-learning, and A2C. In addition, it helped us with PyTorch implementations and gave us many examples of RL projects. It was essentially the starting point of our project. We supplemented the background knowledge from Thomas along with articles from <a href="https://towardsdatascience.com">Towards Data Science</a> to provide us with additional information.
   <li> Also projects such as <a href="https://ieeexplore.ieee.org/abstract/document/8957297?casa_token=JTVW2Y0EiC0AAAAA:27v7m8lyZQv2Fzr_z1g_7siXz9q38bC3Y0o8gjPa3zc63nFnDR8AEF7hdET8vkxC8jyqhq8kPi0">“Deep Reinforcement Learning for Instruction Following Visual Navigation in 3D Maze-like Environments”</a> (Devo, Costante, and Valigi, 2006) or <a href="https://magnus-engstrom.medium.com/building-an-ai-to-navigate-a-maze-899bf03f224d">“Building an AI to Navigate a Maze”</a> (Engstrom, 2019), helped immensely. These projects have closely related goals to ours. They helped us understand different approaches to teaching agents to navigate a maze and what we would need to consider when implementing our own algorithm(s).
    <li> And finally <a href="https://www.altexsoft.com/blog/datascience/reinforcement-learning-explained-overview-comparisons-and-applications-in-business/">“Reinforcement Learning Explained: Overview, Comparisons and Applications in Business”</a> (Altexsoft, 2019) and <a href="https://openreview.net/pdf?id=SJMGPrcle">“Learning to Navigate in Complex Environments”</a> (Mirowski, 2017) gave us additional background knowledge on RL algorithms and problems concerned with navigating an environment. They helped us wrap our brain around the problem we would be attempting to tackle as well.
</ul>
 
  
  
<a href="https://ieeexplore.ieee.org/abstract/document/8957297?casa_token=JTVW2Y0EiC0AAAAA:27v7m8lyZQv2Fzr_z1g_7siXz9q38bC3Y0o8gjPa3zc63nFnDR8AEF7hdET8vkxC8jyqhq8kPi0" >“Deep Reinforcement Learning for Instruction Following Visual Navigation in 3D Maze-like Environments”</a> (Devo, Costante, and Valigi, 2006), and <a href="https://magnus-engstrom.medium.com/building-an-ai-to-navigate-a-maze-899bf03f224d" >“Building an AI to Navigate a Maze”</a> (Engstrom, 2019). These projects will give us an idea of how to structure our algorithm and perform RL on our agent. Furthermore, we will use research articles like <a href="https://www.altexsoft.com/blog/datascience/reinforcement-learning-explained-overview-comparisons-and-applications-in-business/" >“Reinforcement Learning Explained: Overview, Comparisons and Applications in Business”</a> (altexsoft, 2019) and <a href="https://openreview.net/pdf?id=SJMGPrcle" >“Learning to Navigate in Complex Environments”</a> (Mirowski, 2017) to give us a deeper understanding of RL and navigation problems as a whole. 
</p>
<p>
  This project is being done with the hopes that it will contribute to the ultimate goal of Dr. Anthony Clark’s research, which is to have a robot be able to traverse and navigate Pomona College’s campus successfully and unharmed.
</p>
 
<h1>What is Reinforcement Learning?</h1>
<p>
Though it is touched on briefly in our introduction, it is important as a reader to understand exactly what RL  is. According to<a href=https://deepsense.ai/what-is-reinforcement-learning-the-complete-guide/> deepsenseai</a> (2018), it is the training of machine learning models to make a sequence of decisions. It is, in essence, a game in which a machine uses rewards to find the solution to the game at hand. 
In summary, RL can be broken down into <a href="https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/">6 major generic steps</a>:
<ul>
  <li>Observing an environment
  <li> Deciding how the agent is to act based on a particular strategy
  <li> Making sure that the agent acts in accordance with the strategy 
  <li> Awarding the agent with a reward or penalizing the agent for bad actions 
  <li>Teaching the agent to learn from its actions based on the rewards/penalties it has received 
  <li> Continue teaching the agent until it finds the optimal strategy to solve the problem 
</ul>

The idea of RL has been around since the 1960s and can be applied to millions of different situations in both real life and simulated environments. 
</p>

<h1> <b>Methods</b> </h1>
<p>
The goal of our project, as explained above is to implement a reinforcement learning algorithm onto an agent to solve a customized maze.  To do this we attempted to teach this agent and use a variety of different reinforcement learning techniques including Q-learning, Deep Learning, and A2C. These methods will be explained below.
</p>

<h3><b>Q-learning</b> </h3>
<p>
Our initial idea for this project was to train our agent using Q-learning which is one of the most basic forms of reinforcement learning and one that is well-suited for beginners using RL. Q-learning works so an agent picks an action to go towards a particular state and receives a value from that state. The value depends on the value the state has in the Q-table which is updated as the agent moves. If a negative value is associated with that state, then it will be translated to the Q-table and vice versa with a positive value. After Q-learning training, theoretically, the agent will have found the most optimal rewarding path to its goal state. 
</p>

<p>
We based our Q-learning model off of the videos and work done by <a href=https://simoninithomas.github.io/deep-rl-course/>Thomas Simonini</a> in which he creates a taxi agent that learns to navigate in a city with the goal of transporting its passengers to the correct location. This existing model stood out to us because the problem it solves is so similar to ours. However, our environment was too complex for us to successfully apply Q-learning (based on our limited knowledge of the topic.) This was mainly due to the fact that our environment had about 100,000 different states whereas the taxi example only had 10,000 possible states. Therefore our implementation of Q-learning was unsuccessful.
</p>

<h3><b>Deep Q-learning</b></h3>
<p>
After our attempt to implement Q-learning, we moved onto a more complex subject of deep called deep Q-learning (DQN). DQN at its simplest, is using the same Q-learning approach as described above, but now neural networks are used! The neural networks help to approximate the Q-value function, by taking just the state as input and then producing the best possible move based on the Q-value score.
</p>

<p>
In our implementation of DQN on our agent we tried to follow examples done by <a href=https://stable-baselines3.readthedocs.io/en/master/>Stable-Baselines3</a>, which is a set of sourced RL algorithms implemented with PyTorch. We used their DQN algorithm on our agent to have the agent ‘learn’ the maze. To learn the maze required the agent to take many steps, so we felt training our agent on 1,000, 50,000, and 200,000 timesteps would suffice. After the agent is done learning, we would test the agent in the maze and record the results/ time it took for the agent to find the goal state. 
</p>

<h3><b>A2C</b></h3>
<p>
Since we wanted to test our agent on multiple algorithms, the next one we chose to do was A2C. <a href=https://arxiv.org/abs/1602.01783>A2C</a> is an algorithm from Stable-Baselines3 that also uses deep RL, but uses asynchronous gradient descent to optimize the deep neural network controllers. To measure the performance of this algorithm, we will want to follow similar procedures above in which the agent learns on a variety of timesteps, measures the time it took, and then applies then tests the trained agent on its performance in the maze. 
</p>

<h3><b>Rewards</b></h3>
<p>
In RL rewards are the primary tool an agent can use in finding its goal state. For our reward system, we wanted to use a basic procedure in which an agent measures the distance between itself and the goal state. If it is moving towards the goal state, the agent will receive a positive reward and if it moves away from the goal it receives a negative reward. If the agent is stuck in one place, like a corner, it will also receive a negative reward to encourage it to find other ways to the goal.
</p>

<h1><b>Results</b></h1>
<p>
Due to time constraints our agent was unable to be trained to solve a maze, due to difficulty in understanding RL in the given timeframe and some implementation problems with openAI gym and a custom environment. Therefore, we were unable to record the time it took for each agent to solve a maze using a different algorithm. We were, however, able to gather data on the time it took to train these agents and record some of the progress each one made in the virtual maze. We were also able to gather data on how long it took to train each agent depending on the algorithm used and the steps taken during training.
</p>

<p>
As mentioned above, we were unable to train a simple Q-learning agent due to the complexity of our problem. This means we were unable to provide any results for basic Q-learning applied to our algorithm. 
</p>

<p>
Using DQN, we were able to train our agent on 1,000, 10,000 and 100,000 steps. The training took a very long time, however, nearly 5+ hours using 100,000 steps. Though the agent never left the starting position when trained with 1,000 steps, training with 10,000 and above timesteps saw our agent travel a short distance in search of the goal state.  
</p>


<p>
We were also able to train and run our agent using A2C as our other algorithm. Training with A2c, surprisingly took much longer than using DQN. The agent, when learning on 100,000 timesteps took what was estimated to be a day and we still did not see significant results. Using A2C, our agent did not ever travel far away from its initial position as the agent would spin in circles or move slightly forward then backwards. 
</p>

<p>
We understand our results would not likely be reproduced again if sufficient code in training was used or a better reward system was put into place. 
 </p>
<h1><b>Discussion</b></h1>
  <p>
The goal of our work was to implement an algorithm that would teach an agent how to navigate a virtual maze. Our algorithm was then to be evaluated on the efficiency of our agent solving the maze (percentage of best possible moves taken and what percentage of iterations are able to successfully solve the maze). The maze includes a start position, arrows on walls indicating which direction to turn at a wall, and a goal state at the end of the maze. The position of the start state, goal state, and configuration of the maze are random on each iteration. This algorithm was then to be applied to the work of Dr. Anthony Clark. Therefore, the more overarching goal of our work was to contribute to Dr. Clark’s ongoing research on creating a physical robot that can successfully navigate real world terrain -particularly the terrain of Pomona College in Claremont, CA. 
 </p>
 <p>
Unfortunately, we did not succeed in training our agent to reliably navigate the virtual maze. That is not to say our work was without merit. We gained valuable knowledge in RL, a crucial subfield within Machine Learning and a technique with vast applications. Additionally, our failures with particular RL strategies allowed us to research and attempt to apply various RL algorithms which gave us a much more varied and holistic understanding of RL overall. As often is the case, our failures likely taught us much more than instant success with the first approach we tried would have.
</p>
<p>
Ethically, the direct application of our work has few implications. The possible application of RL into a real world robot is where most ethical considerations come into play. The development of robots in general has the ability to vastly alter the way society operates. While it may seem harmless to develop a robot whose job it is to be able to walk around a college campus, it is important to consider nefarious ways in which the same technology could be utilized. For example, efficiently mobile robots have the ability to displace workers in the future and even be weaponized for military purposes. However, It is safe to say that will the many thousands of people researching robots around the world, the goals of this research are among the most innocent and have the least amount of chance to result in harm. 
 </p>
 
<h1><b>Reflection</b></h1>
<p>
We spent a lot of time reading about different applications and techniques of RL and going through code of related projects. Therefore, we are quite disappointed to not have a working agent to show for all our work. If we could do the project over again, I think we would have tried to start coding a little bit earlier. We knew implementing a working RL algorithm would be difficult but we did not expect to run into as many roadblocks as we did. If we had more time to work on the project, we would continue to implement our RL algorithm, focusing specifically on our reward system. From there, we would continue to test out different algorithms to see which one was most efficient.
 </p>


