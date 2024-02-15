#What's this?
I threw together a demo of a Deep Learning Neural Network (DNN) with the TensorFlow library to learn to play the classic game 'pong'! I built a simple abstraction of Pong in pygame. It's not a great implementation, but it's a great approximation for seeing how a DNN can learn to play. 

#How does it work?
Each frame of the game, the neural network is fed numbers representing the position and velocity of the ball 

Parameters:
  - learning rate = the learning rate for the optimizer from the keras library
  - gamma = the amount of the next gamestate's expected value the current gamestate should inherit
  - random chance = the starting probability (0-1 scale) of acting randomly rather than running choices through the DNN
  - random decay = the rate at which the random probability decays exponentially
  - batch size = the size of the training data sample size in each episode
  - reward = the reward for hitting the ball
  - punishment = a negative reward for having the ball pass the paddlec
