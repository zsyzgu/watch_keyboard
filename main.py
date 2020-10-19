import pygame
import time
pygame.mixer.init()
pygame.init()
sound = pygame.mixer.Sound("sound/mi.wav")
sound.play()
time.sleep(1)
