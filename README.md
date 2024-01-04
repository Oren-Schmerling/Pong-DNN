Parameters:
  - learning rate = the learning rate for the optimizer from the keras library
  - gamma = the amount of the next gamestate's expected value the current gamestate should inherit
  - random chance = the starting probability (0-1 scale) of acting randomly rather than running choices through the DNN
  - random decay = the rate at which the random probability decays exponentially
  - batch size = the size of the training data sample size in each episode
  - reward = the reward for hitting the ball
  - punishment = a negative reward for having the ball pass the paddlec
