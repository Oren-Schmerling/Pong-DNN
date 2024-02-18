import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.losses import Huber
from keras.metrics import mean_squared_error

keras.utils.disable_interactive_logging()

class PongAgent:
    def __init__(self, state_size, action_size, episodeLength, numHiddenLayers = 3, LRInitial = 0.02, LRDecay = 99):
        self.n_actions = action_size
        #some hyperparameters:
        #
        # lr - learning rate
        # gamma - discount factor
        # exploration_proba - initial exploration probability
        # exploration_proba_decay - decay of exploration probability
        # batch_size - size of experiences we sample to train the DNN
        self.initial_learning_rate = LRInitial
        self.decay_rate = LRDecay
        self.batch_size = 50
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(self.initial_learning_rate,decay_steps=self.batch_size,decay_rate=self.decay_rate)
        self.gamma = 0.7
        self.mult = 2.0 - self.gamma
        self.exploration_proba = 0
        self.exploration_proba_decay = 0.1
        self.memory_buffer = []
        self.memory_buffer_reward = []
        self.max_memory_buffer = episodeLength

        #create model having two hidden layers of 12 neurons
        #the first layer has the same size as state size
        #the last layer has the size of the action space
        self.model = Sequential([
            #keras.Input(shape=(state_size,)),
            Dense(units=6, input_dim = state_size, activation = 'relu'),
            # Dense(units=6, activation = 'relu'),
            # Dense(units=6, activation = 'relu'),
            # Dense(units=4, activation = 'relu'),
            # Dense(units=4, activation = 'relu'),
            # Dense(units=4, activation = 'relu'),
        ])
        for i in range(numHiddenLayers):
            self.model.add(Dense(units=6, activation = 'relu'))
        self.model.add(Dense(units=action_size, activation = 'linear'))
        self.model.compile(loss = Huber(), optimizer = Adam(learning_rate = self.lr_schedule))

    def getProb(self):
        return self.exploration_proba

    #the agent computes the action to perform given a state
    def compute_action(self, current_state):
        #we sample a variable uniformly over [0,1]
        #if the variable is less than the exploration proba, we choose an action randomly
        #else, we forward the state through the DNN and choose the action with the highest Q-value
        if np.random.uniform(0,1) < self.exploration_proba:
            return np.random.uniform(0,1)
        else:
            q_values = self.model.predict(current_state)[0]
            return q_values[0]
    
    #when an episode is finished, we update the exploration proba using epsilon greedy algorithm
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)

    #at each step, we store the corresponding experience
    def store_episode(self, current_state, action, reward, next_state):
        #we use a dictionary to store them
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
        })
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    
    def store_episode_reward(self, current_state, action, reward, next_state):
        #we use a dictionary to store them
        self.memory_buffer_reward.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
        })
            
            
    # def train_step(self, data):
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #     x, y = data

    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in `compile()`)
    #         loss = self.compute_loss(y=y, y_pred=y_pred)

    #     # Compute gradients
    #     trainable_vars = x
    #     gradients = tape.gradient(loss, x)

    #     # Update weights
    #     self.model.optimizer.apply(gradients, trainable_vars)
            
    def disableRandom(self):
        self.exploration_proba = 0
        self.exploration_proba_decay = 1

    def reduceRandom(self):
        self.exploration_proba = 0.8
        self.exploration_proba_decay = 0.07
    #at the end of each episode, we train the model
    def train(self):
        batch_sample = self.memory_buffer_reward.copy()
        #select a batch of random experiences
        for i in range(self.batch_size - len(batch_sample)):
            index = np.random.randint(0,len(self.memory_buffer)-1)
            batch_sample.append(self.memory_buffer.pop(index))

        # current state = [playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]            

        #we iterate over the selected experiences
        for experience in batch_sample:
            if experience["reward"] == 0:
                target = self.model.predict(experience["next_state"])
            else:
                target = experience["reward"]
            #train the model
            target = np.array([target])
            self.model.fit(experience["current_state"], target, verbose="0")

        
        self.memory_buffer.clear()
        self.memory_buffer_reward.clear()
        batch_sample.clear()


        