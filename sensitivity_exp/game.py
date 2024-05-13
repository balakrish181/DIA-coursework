import turtle
import random
import math
import numpy as np
from constants import *

class Battle:
    def __init__(self,random_init = True,target_pos=None, agent_pos=None):
        # Window
        self.random_init = random_init
        self.target_pos = target_pos
        self.agent_pos = agent_pos
        self.win = turtle.Screen()
        self.win.title('Battle')
        self.win.bgcolor('black')
        self.win.setup(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self.win.tracer(0)

        # Target
        self.target = turtle.Turtle()
        self.target.shape('turtle')
        self.target.speed(0)
        self.target.penup()
        self.target.color('red')
        if random_init and not target_pos :
            self.target.goto(random.randint(-WINDOW_WIDTH/2, WINDOW_WIDTH/2), random.randint(-WINDOW_HEIGHT/2, WINDOW_HEIGHT/2))

        else:
            
            self.target.goto(target_pos)
                

        self.target_direction = random.choice([0, 1, 2, 3])  # Numerical values for directions
        self.target_speed = TARGET_SPEED
        self.target_direction_counter = random.randint(TARGET_DIRECTION_COUNTER_MIN, TARGET_DIRECTION_COUNTER_MAX)

        # Agent
        self.agent = turtle.Turtle()
        self.agent.shape('turtle')
        self.agent.speed(0)
        self.agent.penup()
        self.agent.color('green')
        if random_init and not agent_pos:
            self.agent.goto(random.randint(-WINDOW_WIDTH/2, WINDOW_WIDTH/2), random.randint(-WINDOW_HEIGHT/2, WINDOW_HEIGHT/2))
        
        else:
            self.agent.goto(agent_pos)
        self.agent_direction = 4  # Initial direction as 'stop' (numerical value)
        self.agent_speed = AGENT_SPEED

        # Bullet
        self.bullet = turtle.Turtle()
        self.bullet.shape('square')
        self.bullet.speed(0)
        self.bullet.penup()
        self.bullet.color('yellow')
        self.bullet.shapesize(stretch_wid=0.5, stretch_len=1)
        self.bullet.hideturtle()
        self.bullet_state = 0  # 0 for 'ready', 1 for 'fire'
        self.bullet_direction = 4  # Initial direction as 'stop' (numerical value)

        # Collision detection
        self.collision_radius = COLLISION_RADIUS

        # Movements
        self.win.listen()
        self.win.onkey(self.move_right, 'Right')
        self.win.onkey(self.move_left, 'Left')
        self.win.onkey(self.move_up, 'Up')
        self.win.onkey(self.move_down, 'Down')
        self.win.onkey(self.fire_bullet, 'space')

        # Score
        self.score = 0
        self.score_pen = turtle.Turtle()
        self.score_pen.speed(0)
        self.score_pen.color('white')
        self.score_pen.penup()
        self.score_pen.hideturtle()
        self.score_pen.goto(0, WINDOW_HEIGHT/2 - 20)
        self.display_score(0)


        self.bullet_count = 0
        

    def fire_bullet(self):
        if self.bullet_state == 0:
            self.bullet_state = 1
            self.bullet.setposition(self.agent.xcor(), self.agent.ycor())
            self.bullet.showturtle()
            self.bullet_direction = self.agent_direction
            self.bullet_count +=1

    def move_right(self):
        self.agent_direction = 0
        self.agent.setheading(0)

    def move_left(self):
        self.agent_direction = 1
        self.agent.setheading(180)

    def move_up(self):
        self.agent_direction = 2
        self.agent.setheading(90)

    def move_down(self):
        self.agent_direction = 3
        self.agent.setheading(270)

    def display_score(self, score):
        self.score_pen.clear()
        self.score_pen.write(f"Score: {score}", align="center", font=("Courier", 12, "normal"))

    def run_frame(self):
        # Bullet kinematics
        if self.bullet_state == 1:
            self.bullet.forward(BULLET_SPEED)
            if self.bullet_direction in [0, 1]:  # Right or Left
                self.bullet.setheading(0 if self.bullet_direction == 0 else 180)
            elif self.bullet_direction in [2, 3]:  # Up or Down
                self.bullet.setheading(90 if self.bullet_direction == 2 else 270)

        if abs(self.bullet.xcor()) > WINDOW_WIDTH/2 or abs(self.bullet.ycor()) > WINDOW_HEIGHT/2:
            self.bullet.hideturtle()
            self.bullet_state = 0

        # Agent kinematics
        if self.agent_direction in [0, 2]:  # Right or Up
            coord_index = 0 if self.agent_direction == 0 else 1
            coord_limit = WINDOW_WIDTH/2 - 20 if self.agent_direction == 0 else WINDOW_HEIGHT/2 - 20
            coord = self.agent.xcor() if self.agent_direction == 0 else self.agent.ycor()
            if coord < coord_limit:
                if self.agent_direction == 0:
                    self.agent.setx(coord + self.agent_speed)
                else:
                    self.agent.sety(coord + self.agent_speed)
        else:  # Left or Down
            coord_index = 0 if self.agent_direction == 1 else 1
            coord_limit = -WINDOW_WIDTH/2 + 20 if self.agent_direction == 1 else -WINDOW_HEIGHT/2 + 20
            coord = self.agent.xcor() if self.agent_direction == 1 else self.agent.ycor()
            if coord > coord_limit:
                if self.agent_direction == 1:
                    self.agent.setx(coord - self.agent_speed)
                else:
                    self.agent.sety(coord - self.agent_speed)

        ...

        # Target kinematics
        if self.target_direction == 0:  # Right
            self.target.setheading(0)
            x = self.target.xcor()
            if x < WINDOW_WIDTH/2 - 20:
                self.target.setx(x + self.target_speed)
            else:
                self.target_direction = random.choice([1, 2, 3])
            self.target_direction_counter -= 1
        elif self.target_direction == 1:  # Left
            self.target.setheading(180)
            x = self.target.xcor()
            if x > -WINDOW_WIDTH/2 + 20:
                self.target.setx(x - self.target_speed)
            else:
                self.target_direction = random.choice([0, 2, 3])
            self.target_direction_counter -= 1
        elif self.target_direction == 2:  # Up
            self.target.setheading(90)
            y = self.target.ycor()
            if y < WINDOW_HEIGHT/2 - 20:
                self.target.sety(y + self.target_speed)
            else:
                self.target_direction = random.choice([0, 1, 3])
            self.target_direction_counter -= 1
        elif self.target_direction == 3:  # Down
            self.target.setheading(270)
            y = self.target.ycor()
            if y > -WINDOW_HEIGHT/2 + 20:
                self.target.sety(y - self.target_speed)
            else:
                self.target_direction = random.choice([0, 1, 2])
            self.target_direction_counter -= 1

        ...

        # Target random turn
        if self.target_direction_counter <= 0:
            self.target_direction = random.choice([0, 1, 2, 3])
            self.target_direction_counter = random.randint(TARGET_DIRECTION_COUNTER_MIN, TARGET_DIRECTION_COUNTER_MAX)

        # Bullet collision with target
        if self.bullet_state == 1:
            distance = math.sqrt((self.bullet.xcor() - self.target.xcor()) ** 2 + (self.bullet.ycor() - self.target.ycor()) ** 2)
            if distance <= self.collision_radius:
                self.bullet.hideturtle()
                self.bullet_state = 0
                self.score += 1
                self.display_score(self.score)
                self.reset_positions()
                return True  # Episode ends due to bullet hitting target

        self.win.update()
        return False  # Episode continues

    def rl_space(self, action):
        reward = 0
        done = False

        if action == 0:
            self.move_right()
        elif action == 1:
            self.move_left()
        elif action == 2:
            self.move_up()
        elif action == 3:
            self.move_down()
        elif action == 4:
            self.fire_bullet()
            if self.bullet_state == 1:
                reward += PENALTY_WASTED_BULLET  # Penalty for wasting a bullet

        episode_ended = self.run_frame()
        #self.run_frame()

        



        # Check if the bullet hit the target
        if episode_ended:
            done = True
            #self.reward += REWARD_HIT_TARGET  # Reward for hitting the target
            reward += REWARD_HIT_TARGET

        else:
                    # Reward based on proximity to the target
            agent_pos = np.array([self.agent.xcor(), self.agent.ycor()])
            target_pos = np.array([self.target.xcor(), self.target.ycor()])
            distance_to_target = np.linalg.norm(agent_pos - target_pos)
            if distance_to_target < CLOSE_DISTANCE_THRESHOLD:
                reward += REWARD_CLOSE_TO_TARGET

            else:

                reward += PENALTY_TIME_STEP


            

        state = [self.agent.xcor(), self.agent.ycor(), self.agent_direction, 
                 self.target.xcor(), self.target.ycor(), self.target_direction,
                 self.bullet.xcor(), self.bullet.ycor(), self.bullet_state]

        return state, reward, done, self.score,self.bullet_count

    def reset_positions(self,):
        
        if self.random_init and not self.agent_pos:

            self.agent.goto(random.randint(-WINDOW_WIDTH/2, WINDOW_WIDTH/2), random.randint(-WINDOW_HEIGHT/2, WINDOW_HEIGHT/2))
            self.agent_direction = 4  # Reset agent direction to 'stop'

            self.bullet.hideturtle()
            self.bullet_state = 0
            self.bullet_direction = 4  # Reset bullet direction to 'stop'

        else:
            self.agent.goto(self.agent_pos)
            self.bullet.hideturtle()
            self.bullet_state = 0
            self.bullet_direction = 4


        if self.random_init and not self.target_pos:
        
            self.target.goto(random.randint(-WINDOW_WIDTH/2, WINDOW_WIDTH/2), random.randint(-WINDOW_HEIGHT/2, WINDOW_HEIGHT/2))
            self.target_direction = random.choice([0, 1, 2, 3])
            self.target_direction_counter = random.randint(TARGET_DIRECTION_COUNTER_MIN, TARGET_DIRECTION_COUNTER_MAX)

        else:
            self.target.goto(self.target_pos)
            self.target_direction = random.choice([0, 1, 2, 3])
            self.target_direction_counter = random.randint(TARGET_DIRECTION_COUNTER_MIN, TARGET_DIRECTION_COUNTER_MAX)



    def reset(self):
        self.reset_positions()
        self.score = 0
        self.bullet_count = 0
        self.display_score(self.score)

        state = [self.agent.xcor(), self.agent.ycor(), self.agent_direction, 
                 self.target.xcor(), self.target.ycor(), self.target_direction, 
                 self.bullet.xcor(), self.bullet.ycor(), self.bullet_state]
        return state
    
    def close_turtle(self):
        self.win.bye()


if __name__ == '__main__':
    Battle()