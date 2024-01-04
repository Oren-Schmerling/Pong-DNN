import pygame
import numpy as np
from pygame import *
from neural_network_agent import PongAgent
import os
import pathlib

#
# Parameters
#
trainSpeed = 2
gameSpeed = trainSpeed
paddleOffset = 64
paddleWidth = 20
paddleHeight = 150
padSpeed = 500
fps = 10*trainSpeed
ballXvel = 500
ballYvel = 300
autoPlay = True
memBuffer = 20

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0
paddleRX = screen.get_width() - (paddleOffset + paddleWidth)
paddleLX = paddleOffset
paddleLcollision = paddleLX + paddleWidth
paddleRcollision = paddleRX
ball_default = Vector2(screen.get_width()/2, screen.get_height()/2)
ball_pos = ball_default
playerR_height = screen.get_height()/2
playerL_height = screen.get_height()/2
Lscore = 0
Rscore = 0
reward = 0
hits = 0
averageMemory = list()

def reset(ballXvel, ballYvel, ball_pos, screen):
    ball_pos = Vector2(screen.get_width()/2, screen.get_height()/2)
    ballYvel = np.random.uniform(-300, 300)
    ballXvel *= -1
    playerR_height = screen.get_height()/2
    playerL_height = screen.get_height()/2
    return ballXvel, ballYvel, ball_pos, playerR_height, playerL_height, True

def normalize(var, screen):
    #[playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]
    # 
    # Simplified normalization -- put every variable on a 0to1 or a -1to1 scale
    #     
    var[0] /= screen.get_height()
    var[1] /= screen.get_width()
    var[2] /= screen.get_height()
    var[3] = (float(var[3])/500.0)*-1.0 #multiply by -1 so it =1 when moving towards agend and -1 when moving away
    var[4] /= 300
    return var

#
# link up DNN to game
#
#
trialName = "PongSmart/debug"
pathlib.Path(trialName+"/").mkdir(exist_ok=True) 
episodeLength = 500
variables = [playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]
state_size = 5
action_size = 1
max_iteration_ep = episodeLength
hitReward = 1
punishment = -1
agent = PongAgent(state_size, action_size, episodeLength)
total_steps = 0
with open(trialName+"/LOG.txt", "a+") as f:
        f.write("learning rate = "+str(agent.lr)+ "\ngamma = "+str(agent.gamma)+"\nrandom chance = "+str(agent.exploration_proba)+"\nrandom decay = "+str(agent.exploration_proba_decay)+"\nbatch size = "+str(agent.batch_size)+"\n")

#Saving the model
numsave = 0

#
# OPTIONAL: Load previous weights
#
# agent.model.load_weights(checkpoint_path)
#
# pick one or the other:
# agent.disableRandom()
# agent.reduceRandom()



e = 0
while True:
    e += 1
    hits = 0
    Rscore = 0
    ballXvel, ballYvel, ball_pos, playerR_height, playerL_height, starting = reset(ballXvel, ballYvel, ball_pos, screen)
    variables = [playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]
    action = .5
    current_state = np.array([normalize(variables.copy(), screen)])
    first = True
    reward = 0
    for step in range(max_iteration_ep):


        total_steps += 1
        #store experience
        if first != True :
            if (reward != 0):
                variables = [playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]
                variables = normalize(variables.copy(), screen)
                next_state = np.array([variables])
                agent.store_episode_reward(current_state, action, reward, next_state)
                reward = 0
            else:
                variables = [playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]
                variables = normalize(variables.copy(), screen)
                next_state = np.array([variables])
                agent.store_episode(current_state, action, reward, next_state)
                
        first = False

        #
        # decision making
        #
        variables = [playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]
        current_state = np.array([normalize(variables.copy(), screen)])
        action = agent.compute_action(current_state)

            
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                # print("Right score:", Rscore , "\nLeft score:" , Lscore)
                pygame.quit()

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        playerR_rect = Rect(paddleRX, playerR_height, paddleWidth, paddleHeight)
        playerL_rect = Rect(paddleLX, playerL_height, paddleWidth, paddleHeight)
        pygame.draw.rect(screen, "white", playerR_rect)
        pygame.draw.rect(screen, "white", playerL_rect)
        pygame.draw.circle(screen, "white", ball_pos, 10)
        
        
        #
        # ball movement
        #
        newBallX = ball_pos.x + ballXvel * dt
        
        
        position = float(playerL_height) / float(screen.get_height())
        #bounce off top and bottom
        if (ball_pos.y + ballYvel * dt) < 10 or (ball_pos.y + ballYvel * dt) > screen.get_height()-10:
            ballYvel = ballYvel * -1
        #bounce off left paddle
        if (abs((ball_pos.x - paddleLcollision)) < 5): # ball is on paddle X
            if (abs(ball_pos.y - playerL_height - paddleHeight/2) < paddleHeight/2):
                ballXvel *= -1
                reward = position
                hits += 1
            else:
                reward = ball_pos.y
        elif (ball_pos.x > paddleLcollision and newBallX <= paddleLcollision):# ball will pass paddle X
            if (abs(ball_pos.y - playerL_height - paddleHeight/2) < paddleHeight/2):
                ballXvel *= -1
                reward = position
                hits += 1
            else:
                reward = ball_pos.y
        #now the right paddle
        if (abs((ball_pos.x - paddleRcollision)) < 5): # ball is on paddle X
            if (abs(ball_pos.y - playerR_height - paddleHeight/2) < paddleHeight/2):
                ballXvel *= -1
        elif (ball_pos.x < paddleRcollision and newBallX >= paddleRcollision):# ball will pass paddle X
            if (abs(ball_pos.y - playerR_height - paddleHeight/2) < paddleHeight/2):
                ballXvel *= -1
        #Score if ball hits an end
        if (ball_pos.x <= 0):
            Rscore += 1
            ballXvel, ballYvel, ball_pos, playerR_height, playerL_height, starting = reset(ballXvel, ballYvel, ball_pos, screen)
            first = True
        if (ball_pos.x >= screen.get_width()):
            Lscore += 1
            ballXvel, ballYvel, ball_pos, playerR_height, playerL_height, starting = reset(ballXvel, ballYvel, ball_pos, screen)
            first = True
                
        ball_pos.x += ballXvel * dt
        ball_pos.y += ballYvel * dt


        #
        # controls
        #
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            trainSpeed = 1
        elif keys[pygame.K_2]:
            trainSpeed = 2
        elif keys[pygame.K_3]:
            trainSpeed = 3
        elif keys[pygame.K_4]:
            trainSpeed = 4
        elif keys[pygame.K_5]:
            trainSpeed = 5
        elif keys[pygame.K_6]:
            trainSpeed = 6
        elif keys[pygame.K_7]:
            trainSpeed = 7
        elif keys[pygame.K_8]:
            trainSpeed = 8
        elif keys[pygame.K_9]:
            trainSpeed = 9
        gameSpeed = trainSpeed
        fps = 10*trainSpeed


        position = float(playerL_height) / float(screen.get_height())
        if action < position and playerL_height > 0:
            playerL_height -= padSpeed * dt
        if action > position and playerL_height + paddleHeight < screen.get_height():
            playerL_height += padSpeed * dt
        #
        # player controls
        #
        if (autoPlay == False):
            if keys[pygame.K_UP] and playerR_height > 0:
                playerR_height -= padSpeed * dt
            if keys[pygame.K_DOWN]and playerR_height + paddleHeight < screen.get_height():
                playerR_height += padSpeed * dt
        else:
            playerR_height = ball_pos.y - paddleHeight/2
        
        
        # flip() the display to put your work on screen
        pygame.display.flip()

        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(fps) / (1000 / gameSpeed)


    agent.train()

    # print some stats, for overseeing training
    hitRate = (hits/(hits + Rscore)) * 100
    if (averageMemory.__len__() < memBuffer):
        averageMemory.append(hitRate)
    else:
        averageMemory[e%memBuffer] = hitRate
    episodeAvg = float(sum(averageMemory)) / float(averageMemory.__len__())
    with open(trialName+"/LOG.txt", "a+") as f:
        f.write("hit rate: "+str(round(hitRate, 3))+"  in episode "+ str(e)+" and  "+str(round(episodeAvg, 3))+" in last "+ str(averageMemory.__len__())+" episodes      random prob: "+ str(round(agent.getProb() * 100, 3))+"\n")
    if (agent.getProb() < 0.01):
        agent.disableRandom()


    #Saving the model's weights
    if (e%50 == 0):
        numsave += 1
        checkpoint_path = "./"+trialName+"/checkpoint"+str(numsave)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        agent.model.save_weights(checkpoint_path)
        with open(trialName+"/LOG.txt", "a+") as f:
            f.write("Saved! -- checkpoint "+ str(numsave)+"\n")


    agent.update_exploration_probability()

pygame.quit()