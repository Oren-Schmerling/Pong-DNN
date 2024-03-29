import pygame
import numpy as np
from pygame import *
from neural_network_agent import PongAgent
import os
import pathlib

def reset(ballXvel, ballYvel, ball_pos, screen):
    ball_pos = Vector2(screen.get_width()/2, screen.get_height()/2)
    ballYvel = 0
    while (abs(ballYvel) < 30):
        ballYvel = np.random.uniform(-300, 300)
    ballXvel *= -1
    playerR_height = screen.get_height()/2
    playerL_height = screen.get_height()/2
    return ballXvel, ballYvel, ball_pos, playerR_height, playerL_height

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

def simulation(numHiddenLayers, LRInitial, LRDecay):

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

    #
    # link up DNN to game
    #
    path = "trainingData/"
    pathlib.Path(path).mkdir(exist_ok=True) 
    path = path + str(numHiddenLayers)+" hidden layers/"
    pathlib.Path(path).mkdir(exist_ok=True)
    path = path + str(LRInitial) + " initial learning rate/"
    pathlib.Path(path).mkdir(exist_ok=True)
    path = path + str(LRDecay) + " learning rate deacay/"
    pathlib.Path(path).mkdir(exist_ok=True)
    episodeLength = 500
    numEpisodes = 500
    variables = [playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]
    state_size = 5
    action_size = 1
    max_iteration_ep = episodeLength
    hitReward = 1
    punishment = -1
    agent = PongAgent(state_size, action_size, episodeLength, numHiddenLayers, LRInitial, LRDecay)
    total_steps = 0
    with open(path+"/LOG.txt", "a+") as f:
            f.write("initial learning rate = "+str(agent.initial_learning_rate)+ "\nlearning rate decay = "+ str(agent.decay_rate) +"\nbatch size = "+str(agent.batch_size)+"\n")

    #Saving the model
    numsave = 0

    #
    # OPTIONAL: Load previous weights
    #
    # agent.model.load_weights(checkpoint_path)
    #
    # pick one or the other:
    agent.disableRandom()
    # agent.reduceRandom()


    e = 0
    for i in range(numEpisodes):
        e += 1
        hits = 0
        Rscore = 0
        ballXvel, ballYvel, ball_pos, playerR_height, playerL_height = reset(ballXvel, ballYvel, ball_pos, screen)
        variables = [playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]
        action = .5
        current_state = np.array([normalize(variables.copy(), screen)])
        first = True
        reward = 0
        for step in range(max_iteration_ep):
            
            
            ballReward = float(ball_pos.y) / float(screen.get_height())
            position = float(playerL_height) / float(screen.get_height())


            total_steps += 1
            #store experience
            if first != True :
                
                variables = [playerL_height, ball_pos.x, ball_pos.y, ballXvel, ballYvel]
                variables = normalize(variables.copy(), screen)
                next_state = np.array([variables])
                if reward == 0:
                    agent.store_episode(current_state, action, reward, next_state)
                else:
                    agent.store_episode_reward(current_state, action, reward, next_state)
                    
                    
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
            cursorX = paddleOffset + (paddleWidth/2)
            playerR_rect = Rect(paddleRX, playerR_height, paddleWidth, paddleHeight)
            playerL_rect = Rect(paddleLX, playerL_height, paddleWidth, paddleHeight)
            pygame.draw.rect(screen, "white", playerR_rect)
            pygame.draw.rect(screen, "white", playerL_rect)
            pygame.draw.circle(screen, "white", ball_pos, 10)
            pygame.draw.circle(screen, "green", (cursorX, action*screen.get_height()), 5)
            
            
            #
            # ball movement
            #
            newBallX = ball_pos.x + ballXvel * dt
            #bounce off top and bottom
            if (ball_pos.y + ballYvel * dt) < 10 or (ball_pos.y + ballYvel * dt) > screen.get_height()-10:
                ballYvel = ballYvel * -1
            #bounce off left paddle
            if (abs((ball_pos.x - paddleLcollision)) < 5): # ball is on paddle X
                if (abs(ball_pos.y - playerL_height - paddleHeight/2) < paddleHeight/2):#       hit
                    ballXvel *= -1
                    reward = ballReward
                    hits += 1
                else: #miss
                    reward = ballReward
            elif (ball_pos.x > paddleLcollision and newBallX <= paddleLcollision):# ball will pass paddle X
                if (abs(ball_pos.y - playerL_height - paddleHeight/2) < paddleHeight/2):#       hit
                    ballXvel *= -1
                    reward = ballReward
                    hits += 1
                else: #miss
                    reward = ballReward
            else: # not at paddle X
                reward = 0
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
                ballXvel, ballYvel, ball_pos, playerR_height, playerL_height = reset(ballXvel, ballYvel, ball_pos, screen)
                first = True
            if (ball_pos.x >= screen.get_width()):
                Lscore += 1
                ballXvel, ballYvel, ball_pos, playerR_height, playerL_height = reset(ballXvel, ballYvel, ball_pos, screen)
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
            normHeight = float(paddleHeight) / float(screen.get_height())
            if action < position and playerL_height > 0:
                playerL_height -= padSpeed * dt
            if action > (position+normHeight) and playerL_height + paddleHeight < screen.get_height():
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
        hitRate = round(hitRate, 3)
        episodeAvg = round(episodeAvg, 3)
        string = f"hit rate: {hitRate:.3f}  in episode "+ str(e)+ f" and  {episodeAvg:.3f} in last "+ str(averageMemory.__len__())+" episodes\n"
        with open(path+"/LOG.txt", "a+") as f:
            f.write(string)
        if (agent.getProb() < 0.01):
            agent.disableRandom()


        #Saving the model's weights
        if (e%50 == 0):
            numsave += 1
            checkpoint_path = "./"+path+"/checkpoint"+str(numsave)
            checkpoint_dir = os.path.dirname(checkpoint_path)
            agent.model.save_weights(checkpoint_path)
            with open(path+"/LOG.txt", "a+") as f:
                f.write("Saved! -- checkpoint "+ str(numsave)+"\n")

def main ():
    for i in range (3):
        numHiddenLayers = i+3
        for j in range(6):
            LRInitial = 1.5 + (0.5*j)
            for k in range(5):
                LRDecay = 0.99 - (0.01*k)
                simulation(numHiddenLayers, LRInitial, LRDecay)

main()
pygame.quit()