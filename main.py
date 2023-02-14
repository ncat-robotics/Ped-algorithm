"""
 Simple graphics demo
 
 Sample Python/Pygame Programs
 Simpson College Computer Science
 http://programarcadegames.com/
 http://simpson.edu/computer-science/
 
"""
 
# Import a library of functions called 'pygame'
import pygame
from Board import Board


    


# Initialize the game engine
pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
 
PI = 3.141592653
 
# Set the height and width of the screen

 
pygame.display.set_caption("Professor Craven's Cool Game")
 
# Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()
game_board = Board()
game_board.populate_pedistals()
screen = game_board.screen
game_board.make_ped_graph()
restart_flag = False
 
# Loop as long as done == False
while not done:
 
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
 
    # All drawing code happens after the for loop and but
    # inside the main while not done loop.
    if pygame.key.get_pressed()[pygame.K_r] and not restart_flag:
        game_board.populate_pedistals()
        restart_flag = True
    if not pygame.key.get_pressed()[pygame.K_r] and restart_flag:
        restart_flag = False
    # Clear the screen and set the screen background
    screen.fill(WHITE)
 
    # Draw a rectangle
    
    game_board.draw_board()
    game_board.draw_pedistals()
 
    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()
 
    # This limits the while loop to a max of 60 times per second.
    # Leave this out and we will use all CPU we can.
    clock.tick(60)
 
# Be IDLE friendly
pygame.quit()