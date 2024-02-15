# What's this?
I threw together a demo of a Deep Learning Neural Network (DNN) with the TensorFlow library to learn to play the classic game 'pong'! I built a simple abstraction of Pong in pygame. It's not a great implementation, but it's a great approximation for seeing how a DNN can learn to play. In the current state, running the program will conduct an experiment across several iterations of the simulation, each one with a slighty different size, initial learning rate, or decay of learning rate for the DNN.

# Can I run it?
Yes! You will need to run it using python3.11, and it requires the TensorFlow and Pygame libraries. Just install these dependencies, and run the pygame file.

# How does it work?
Each frame of the game, the neural network is fed numbers representing the position and velocity of the ball 

### Parameters:
  - Initial learning rate = the starting value for the learning rate for the optimizer from the keras library
  - Number of hidden layers = How many layers (aside from the input and output layers) will the neural network hold?
  - Learning rate decay = the rate (scalar multiplier) at which the learning rate decreases after each training step
  - batch size = the size of the training data sample size in each episode

# Goals for the future:
  - Save experiment results including statistics on each iteration's accuracy over time to a spreadsheet or CSV file to conveniently review training data all in one place
  - Find optimize model's accuracy and store the weights for a well trained model in this repo to be optionally loaded and skip training
  - Add a toggle function to the program to allow the user to take over for the non-DNN paddle and play against the DNN
  - Increase the difficulty for the game simulation
