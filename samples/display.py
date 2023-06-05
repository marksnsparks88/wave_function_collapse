import pygame as pg
from pygame.locals import *
import os, re



def natural_sort(l): 
  convert = lambda text: int(text) if text.isdigit() else text.lower() 
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
  return sorted(l, key=alphanum_key)

def getImages(loc):
  images=[]
  for f in os.listdir(loc):
    if f.endswith(".png"):
      images.append(f)
  images = natural_sort(images)
  return images


pg.init()
height = 450
width = 450
sur = pg.display.set_mode((width, height))

relDir=os.getcwd()

running=True
play = False
count=0
step = 1
while running:
  images = getImages(os.getcwd())
  for event in pg.event.get():
    if event.type == QUIT:
      running = False
    if event.type == KEYUP:
      if count > len(images)-1:
        count = 0
      pic = pg.image.load(os.getcwd()+'\\'+images[count])
      pic = pg.transform.scale(pic, (width, height))
      pic.convert()
      picR = pic.get_rect()
      sur.blit(pic, picR)
      pg.display.flip()
      pg.display.set_caption(images[count])
    if event.type == KEYDOWN:
      if event.key == K_RIGHT:
        count += step
      if event.key == K_LEFT:
        count -= step
      if event.key == K_SPACE:
        if play == False:
          play = True
        elif play == True:
          play = False
  
  if count > len(images)-1:
    play = False
    count = 0        
  if play == True:
    pic = pg.image.load(images[count])
    pic = pg.transform.scale(pic, (width, height))
    pic.convert()
    picR = pic.get_rect()
    sur.blit(pic, picR)
    pg.display.flip()
    pg.display.set_caption(images[count])
    count += step

    
pg.quit()

    
