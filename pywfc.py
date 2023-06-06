try:
    import Image
except ImportError:
    from PIL import Image

import time
import os
import math
import random
import collections
import pygame as pg
from pygame.locals import *
import xml.etree.ElementTree as ET

class Overlapping:
    def __init__(self, image, N, outputW, outputH, seamlessInput, seamlessOutput, symmetry):
        self.name = image
        self.img = Image.open("samples/"+image+".png")
        self.imgW = self.img.width
        self.imgH = self.img.height
        
        self.N = N
        
        self.outW = outputW
        self.outH = outputH
        
        self.seamlessInput = seamlessInput
        self.seamlessOutput = seamlessOutput
        self.symmetry = symmetry
        #----------------------------#
        self.rand = random.Random()
        self.saveDir = os.getcwd()
        
        self.imgRefPallete = self.GetPallete()
        self.palleteCount = len(self.imgRefPallete)
        self.imgRef = self.ParseImg()
        
        self.imgSamples = self.ImgSamples()
        self.uniqSamples = self.UniqSamples()
        self.frequency = self.SampleFreq()
        self.uniqSamples_len = len(self.uniqSamples)
        self.uniqSamples_log = math.log(self.uniqSamples_len)
        #self.Samples_log_prob = [math.log(self.frequency[i]) for i in range(self.uniqSamples_len)]
        
        self.Samples_log_prob = []
        for i in range(self.uniqSamples_len):
            self.Samples_log_prob.append(math.log(self.frequency[i]))
        
        self.propagator = self.Propagator()
        
        self.wave = [[[1 for _ in range(self.uniqSamples_len)] for _ in range(self.outW)] for _ in range(self.outH)]
        self.changes = [[0 for _ in range(self.outW)] for _ in range(self.outH)]
        self.gUpdate = [[0 for _ in range(self.outW)] for _ in range(self.outH)]
        self.observed = [[0 for _ in range(self.outW)] for _ in range(self.outH)]
        self.resultArray = [[0 for _ in range(self.outW)] for _ in range(self.outH)]
        
        self.Graphics(init=True, save=False)
    
    def GetPallete(self):
        colors = []
        for y in range(self.imgH):
            for x in range(self.imgW):
                pix = self.img.getpixel((x, y))
                if pix not in colors:
                    colors.append(pix)
        return colors
    
    def ParseImg(self):
        imgRef = [[[] for x in range(self.imgW)]for y in range(self.imgH)]
        for y in range(self.imgH):
            for x in range(self.imgW):
                pix = self.img.getpixel((x, y))
                for c in range(len(self.imgRefPallete)):
                    if pix == self.imgRefPallete[c]:
                        imgRef[y][x] = [c]
        return imgRef
        
    def ImgSamples(self):
        ylimit = len(self.imgRef) - self.N + 1
        xlimit = len(self.imgRef[0]) - self.N + 1
        if self.seamlessInput == True:
            ylimit = len(self.imgRef)
            xlimit = len(self.imgRef[0])
            
        samples = []
        
        for refy in range(ylimit):
            for refx in range(xlimit):
                #Original
                sample = [[0 for x in range(self.N)] for y in range(self.N)]
                for sory in range(self.N):
                    for sorx in range(self.N):
                        sample[sory][sorx] = self.imgRef[ (refy+sory)%len(self.imgRef) ][ (refx+sorx)%len(self.imgRef[0]) ]
                        
                if self.symmetry == 1:
                    samples.append(sample)
                else:
                    reflections = [sample]
                    
                    #symmetry
                    for symmetry in range(1, self.symmetry):
                        refInd = int((symmetry-1)/2)*2
                        #refInd relies on round down to get even index
                        #run self.pseudo('symmetry index example')
                        sample = [[0 for x in range(self.N)] for y in range(self.N)]
                        if symmetry%2 == 1:
                            #Reflect
                            for srey in range(self.N):
                                for srex in range(self.N):
                                    sample[srey][srex] = reflections[refInd][(self.N-1)-srey][srex]
                            reflections.append(sample)
                        if symmetry%2 == 0:
                            #Rotate
                            for sroy in range(self.N):
                                for srox in range(self.N):
                                    sample[sroy][srox] = reflections[refInd][(self.N-1)-srox][sroy]
                            reflections.append(sample)
                            
                        samples = samples + reflections
                        
        return samples
    
    def UniqSamples(self):
        samp=[]
        for i in self.imgSamples:
            if i not in samp:
                samp.append(i)
        return samp
    
    def SampleFreq(self):
        freq=[]
        for i in self.uniqSamples:
            c=0
            for j in self.imgSamples:
                if j == i:
                    c+=1
            freq.append(c)
        return freq
    
    def Propagator(self):
        #prop=[[[ [] for s in range(self.uniqSamples_len)] for x in range(self.N*2-1)] for y in range(self.N*2-1)]
        prop=[]
        for y in range(self.N*2-1):
            prop.append([])
            for x in range(self.N*2-1):
                prop[y].append([])
                
                for s1 in range(self.uniqSamples_len):
                    prop[y][x].append([])
                    fits_lst = []
                    for s2 in range(self.uniqSamples_len):
                        dy = y-self.N+1
                        dx = x-self.N+1
                        
                        xmin = dx
                        xmax = self.N
                        if dx < 0:
                            xmin = 0
                            xmax = dx + self.N
                            
                        ymin = dy
                        ymax = self.N
                        if dy < 0:
                            ymin = 0
                            ymax = dy + self.N
                            
                        fits = True
                        for y1 in range(ymin, ymax):
                            for x1 in range(xmin, xmax):
                                if self.uniqSamples[s1][y1][x1] != self.uniqSamples[s2][y1 - dy][x1 - dx]:
                                    fits = False
                        if fits == True:
                            fits_lst.append(s2)
                    prop[y][x][s1] = fits_lst
                    
        return prop
    
    def Observe(self):
        min_entropy = 100000
        entry_x = -1
        entry_y = -1
        
        for y in range(self.outH):
            for x in range(self.outW):
                # if self.seamlessOutput == False and y + self.N > self.outH or x + self.N > self.outW:
                    # pass
                # else:
                w = self.wave[y][x]
                amount = 0
                observed_sum = 0
                for i in range(self.uniqSamples_len):
                    if w[i] == 1:
                        amount += 1
                        observed_sum += self.frequency[i]
                if 0 == observed_sum:
                        return False
                noise = 0.000001 * self.rand.random()
                if amount == 1:
                    self.gUpdate[y][x] = 2
                    entropy = 0
                elif amount == self.uniqSamples_len:
                    entropy = self.uniqSamples_log
                else:
                    main_sum = 0
                    log_sum = math.log(observed_sum)
                    for i in range(self.uniqSamples_len):
                        if w[i] == 1:
                            main_sum += self.frequency[i] * self.Samples_log_prob[i]
                    entropy = log_sum - main_sum/observed_sum
                if entropy > 0 and entropy+noise < min_entropy:
                    min_entropy = entropy+noise
                    entry_y = y
                    entry_x = x
                    
        if entry_y == -1 and entry_x == -1:
            for y in range(self.outW):
                for x in range(self.outH):
                    for s in range(self.uniqSamples_len):
                        if self.wave[y][x][s] == 1:
                            self.observed[y][x] = s
                            break
            return True
            
        #
        distribution = []
        for i in range(self.uniqSamples_len):
            if self.wave[entry_y][entry_x][i] == 1:
                distribution.append(self.frequency[i])
            else:
                distribution.append(0)
        select = Util.Random(distribution, self.rand.random())
        for i in range(self.uniqSamples_len):
            if select != i:
                self.wave[entry_y][entry_x][i] = 0
                
        self.changes[entry_y][entry_x] = 1
        self.gUpdate[entry_y][entry_x] = 1
        return None
    
    def Propagate(self):
        change = False
        b=0
        for y in range(self.outH):
            for x in range(self.outW):
                if self.changes[y][x] == 1:
                    self.changes[y][x] = 0
                    for dy in range((0-self.N)+1, self.N):
                        for dx in range((0-self.N)+1, self.N):

                            y1 = y+dy
                            if y1 < 0:
                                y1+=self.outH
                            elif y1 >= self.outH:
                                y1-=self.outH
                                
                            x1 = x+dx
                            if x1 < 0:
                                x1+=self.outW
                            elif x1 >= self.outW:
                                x1-=self.outW
                                
                            # if self.seamlessOutput == False and x1 + self.N > self.outW or y1 + self.N > self.outH:
                                # pass
                            # else:
                            w1 = self.wave[y][x]
                            w2 = self.wave[y1][x1]
                            p = self.propagator[(self.N-1)-dy][(self.N-1)-dx]
                                
                            for i in range(self.uniqSamples_len):
                                if w2[i] == 0:
                                    pass
                                else:
                                    b = 0
                                    prop = p[i]
                                    cnt = 0
                                    while cnt < len(prop) and b == 0:
                                        b=w1[prop[cnt]]
                                        cnt += 1
                                    if b == 0:
                                        self.changes[y1][x1] = 1
                                        self.gUpdate[y1][x1] = 1
                                        w2[i] = 0
                                        change = True
            
        return change
    
    def Graphics(self, init=False, save=True, frame=1, refresh=999):
        if init:
            r=0
            g=0
            b=0
            L=len(self.uniqSamples)
            for s in self.uniqSamples:
                for y in range(len(i)):
                    for x in range(len(i[j])):
                        r+=self.imgRefPallete[sum(s[y][x])][0]
                        g+=self.imgRefPallete[sum(s[y][x])][1]
                        b+=self.imgRefPallete[sum(s[y][x])][2]
            col=(int((r/L)/9), int((g/L)/9), int((b/L)/9))
            self.resultArray = [[ col for x in range(self.outW)] for y in range(self.outH)]
            
        else:
            ones=0
            twos=0
            for y in range(self.outH):
                for x in range(self.outW):
                    contributors = 0
                    r=0
                    g=0
                    b=0
                    
                    for dy in range(self.N):
                        for dx in range(self.N):
                        
                            sy = y-dy
                            if sy < 0:
                                sy += self.outH
                                
                            sx = x-dx
                            if sx < 0:
                                sx += self.outW
                                
                            if self.gUpdate[sy][sx] == 1 or (self.gUpdate[sy][sx] == 2 and frame%refresh == 0):    
                            
                                # if self.seamlessOutput == False and sy + self.N > self.outW or sx + self.N > self.outH:
                                    # pass
                                # else:
                                for s in range(self.uniqSamples_len):
                                    if self.wave[sy][sx][s] == True:
                                        contributors += 1
                                        col = self.imgRefPallete[sum(self.uniqSamples[s][dy][dx])]
                                        r += col[0]
                                        g += col[1]
                                        b += col[2]
                                        
                    if contributors > 0 :
                        self.resultArray[y][x] = ( int(r/contributors), int(g/contributors), int(b/contributors) )
                            
                    if self.gUpdate[y][x] == 2:
                        twos+=1
                    elif self.gUpdate[y][x]==1:
                        ones+=1
            print('.><Ones=', ones, ':Twos=', twos, ':refresh=', frame%refresh, '><  ', end='')
                        
        if save == True:
            img = Image.new("RGB",(self.outW, self.outH),(0,0,0))
            for i in range(len(self.resultArray)):
                for j in range(len(self.resultArray[i])):
                    img.putpixel((j, i), self.resultArray[j][i])
                    
            img.save(self.saveDir+'/'+"in_progress_{0}_{1}.png".format(self.name, frame), format="PNG")
    
    def run(self, fRate=1, refresh=1, limit=0, seed=''):
        self.saveDir = Util.cDir(self.name)
            
        if not seed:
            seed = random.random()
            self.rand.seed(seed)
        else:
            self.rand.seed(seed)
        L=0
        while (L < limit) or (0 == limit):
            s = Util.time(start=True, text='<<-Observing->>..', end_='')
            obs = self.Observe()
            Util.time(last=s, text='Done')
            if obs == True:
                self.Graphics(save=True, refresh=1)
                return obs
            elif obs == False:
                print('stalled')
                return obs
            presult = True
            while presult:
                L+=1
                s = Util.time(start=True, text='Propagating..', end_='')
                presult = self.Propagate()
                t = Util.time(last=s, text='Done')
                
                if L%fRate == 0:
                    s = Util.time(start=True, text='>--->Saving Image..  <.'+str(L), end_='')
                    self.Graphics(save=True, frame=L, refresh=(fRate*refresh))
                    t = Util.time(last=s, text='Done')
                    
        return obs
    
    
    
class Tiled:
    def __init__(self, name, subsetName, width, height, periodic, black):
        self.name = name
        self.saveDir = Util.cDir(self.name)
        self.periodic = periodic
        self.black = black
        self.width = width
        self.height = height
        self.frame = 0
        
        
        self.rand = random.Random()
        
        self.xmlRoot = ET.parse(os.getcwd()+'/samples/'+name+'/data.xml').getroot()
        self.tileSize = int(self.xmlRoot.get('size'))
        self.uniq = bool(self.xmlRoot.get('unique'))
        self.changes = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        self.subset = []
        if subsetName != 'default':
            for subset in self.xmlRoot.find('subsets'):
                if subset.get('name') == subsetName:
                    for tile in subset.iter('tile'):
                        subset.append(tile.get('name'))
                else:
                    print("ERROR: subset {subsetName} is not found")
        
        self.tiles = []
        self.tilenames = []
        self.tmpstationary = []
        self.action = []
        self.firstOccurrence = {}
        
        def ImgOpen(image):
            image = Image.open(image)
            m = [[ 0 for x in range(image.width)] for y in range(image.height)]
            for y in range(image.height):
                for x in range(image.width):
                    m[y][x] = image.getpixel((x, y))
            return m
            
        def rotate(matrix, turns):
            m=matrix
            for i in range(turns):
                n=[[ 0 for _ in range(len(m))] for _ in range(len(m[0]))]
                for y in range(len(m)):
                    for x in range(len(m[0])):
                        n[y][x] = m[x][(len(m)-1)-y]
                m=n
            return m
        
        for tile in self.xmlRoot.find('tiles'):
            tilename = tile.get('name')
            if len(self.subset) != 0 and tilename not in self.subset:
                continue
            
            cardinality = 0
            sym = tile.get('symmetry')
            if sym == 'L':
                cardinality = 4
                a = lambda i : ( i + 1 ) % 4
                b = lambda i : i + 1 if i % 2 == 0 else i - 1
            elif sym == 'T':
                cardinality = 4
                a = lambda i : ( i + 1 ) % 4
                b = lambda i : i if i % 2 == 0 else 4 - i
            elif sym == 'I':
                cardinality = 2
                a = lambda i : 1 - i
                b = lambda i : i
            elif sym == '\\':
                cardinality = 2
                a = lambda i : 1 - i
                b = lambda i : 1 - i
            else:
                cardinality = 1
                a = lambda i : i
                b = lambda i : i
                
            T = len(self.action)
            self.firstOccurrence[tilename] = T
            
            map_ = [ [] for _ in range(cardinality)]
            for t in range(cardinality):
                map_[t] = [0 for _ in range(8)]
                
                map_[t][0] = t
                map_[t][1] = a(t)
                map_[t][2] = a(a(t))
                map_[t][3] = a(a(a(t)))
                map_[t][4] = b(t)
                map_[t][5] = b(a(t))
                map_[t][6] = b(a(a(t)))
                map_[t][7] = b(a(a(a(t))))
                
                for s in range(8):
                    map_[t][s] += T
                    
                self.action.append(map_[t])
                
            if self.uniq == True:
                for t in range(cardinality):
                    self.tiles.append(ImgOpen(os.getcwd()+'/samples/'+name+'/'+tilename+' '+str(t)+'.png'))
                    self.tilenames.append(tilename+' '+str(t))
            else:
                self.tiles.append(ImgOpen(os.getcwd()+'/samples/'+name+'/'+tilename+'.png'))
                self.tilenames.append(tilename+' 0')
                for t in range(1, cardinality):
                    self.tiles.append(rotate(self.tiles[T+t-1], 1))
                    self.tilenames.append(tilename+' '+str(t))
                    
            for t in range(cardinality):
                self.tmpstationary.append(float(tile.get('weight', 1.0)))
                    
        T = len(self.action)
        weights = self.tmpstationary
        self.tmpstationary_log = []
        for i in self.tmpstationary:
            self.tmpstationary_log.append(math.log(float(i)))
            
        self.propagator=[ [] for _ in range(4)]
        self.tempPropagator = [ [] for _ in range(4)]
        for d in range(4):
            self.tempPropagator[d] = [ [] for _ in range(T)]
            self.propagator[d] = [ [] for _ in range(T)]
            for t in range(T):
                self.tempPropagator[d][t] = [ False for _ in range(T)]
        
        for neighbor in self.xmlRoot.find('neighbors'):
            left = neighbor.get('left').split(' ')
            right = neighbor.get('right').split(' ')
            
            if ( len(self.subset) != 0 and (left[0] not in self.subset or right[0] not in self.subset) ):
                continue
            
            L = self.action[self.firstOccurrence[left[0]]][0 if len(left) == 1 else int(left[1])]
            D = self.action[L][1]
            R = self.action[self.firstOccurrence[right[0]]][0 if len(right) == 1 else int(right[1])]
            U = self.action[R][1]
            
            self.tempPropagator[0][R][L] = True
            self.tempPropagator[0][ self.action[R][6] ][ self.action[L][6] ] = True
            self.tempPropagator[0][ self.action[L][4] ][ self.action[R][4] ] = True
            self.tempPropagator[0][ self.action[L][2] ][ self.action[R][2] ] = True
            
            self.tempPropagator[1][U][D] = True
            self.tempPropagator[1][self.action[D][6]][self.action[U][6]] = True
            self.tempPropagator[1][self.action[U][4]][self.action[D][4]] = True
            self.tempPropagator[1][self.action[D][2]][self.action[U][2]] = True
        
        for t2 in range(T):
            for t1 in range(T):
                self.tempPropagator[2][t2][t1] = self.tempPropagator[0][t1][t2]
                self.tempPropagator[3][t2][t1] = self.tempPropagator[1][t1][t2]
                
        self.sparsePropagator = [ [] for _ in range(4)]
        for d in range(4):
            self.sparsePropagator[d] = [ [] for _ in range(T)]
            for t in range(T):
                self.sparsePropagator[d][t] = []
        
        for d in range(4):
            for t1 in range(T):
                sp = self.sparsePropagator[d][t1]
                tp = self.tempPropagator[d][t1]
                for t2 in range(T):
                    if (tp[t2]) == True:
                        sp.append(t2)
                ST = len(sp)
                self.propagator[d][t1] = [ 0 for _ in range(ST)]
                for st in range(ST):
                    self.propagator[d][t1][st] = sp[st]
                    
        self.wave = [[ [ 1 for t in range(T) ] for x in range(self.width)] for y in range(self.height)]
    
    def Observe(self):
        min_entropy = 100000
        entry_x = -1
        entry_y = -1
        
        for y in range(self.height):
            for x in range(self.width):
                w = self.wave[y][x]
                amount = 0
                observed_sum = 0
                for i in range(len(self.tiles)):
                    if w[i] == 1:
                        amount += 1
                        observed_sum += self.tmpstationary[i]
                if observed_sum == 0:
                        return False
                noise = 0.000001 * self.rand.random()
                if amount == 1:
                    entropy = 0
                elif amount == len(self.tiles):
                    entropy = math.log(len(self.tiles))
                else:
                    main_sum = 0
                    log_sum = math.log(observed_sum)
                    for i in range(len(self.tiles)):
                        if w[i] == 1:
                            main_sum += self.tmpstationary[i] * self.tmpstationary_log[i]
                    entropy = log_sum - main_sum/observed_sum
                if entropy > 0 and entropy+noise < min_entropy:
                    min_entropy = entropy+noise
                    entry_y = y
                    entry_x = x
                    
        if entry_y == -1 and entry_x == -1:
            return True                    
                
        distribution = []
        for i in range(len(self.tiles)):
            if self.wave[entry_y][entry_x][i] == 1:
                distribution.append(self.tmpstationary[i])
            else:
                distribution.append(0)
        select = Util.Random(distribution, self.rand.random())
        for i in range(len(self.tiles)):
            if select != i:
                self.wave[entry_y][entry_x][i] = 0
                self.changes[entry_y][entry_x] = 1
        
        return None
    
    def Propagate(self):
        change = False
        DX = [-1, 0, 1, 0]
        DY = [0, 1, 0, -1]
        
        for y in range(self.height):
            for x in range(self.width):
                if self.changes[y][x] == 1:
                    self.changes[y][x] = 0
                    for d in range(4):
                        dx = DX[d]
                        dy = DY[d]
                        
                        x1 = x+dx
                        if x1 >= self.width:
                            if self.periodic == False:
                                continue
                            x1 -= self.width
                        elif x1 < 0:
                            if self.periodic == False:
                                continue
                            x1 += self.width
                            
                        y1 = y+dy
                        if y1 >= self.height:
                            if self.periodic == False:
                                continue
                            y1 -= self.height
                        elif y1 < 0:
                            if self.periodic == False:
                                continue
                            y1 += self.height
                        
                        w1 = self.wave[y][x]
                        w2 = self.wave[y1][x1]
                        
                        p = self.propagator[d]
                        for j in range(len(w2)):
                            x_ = 0
                            for i in range(len(w1)):
                                if w1[i] == 0:
                                    pass
                                else:
                                    cnt = 0
                                    while cnt < len(p[i]):
                                        if p[i][cnt] == j:
                                            x_ = 1
                                        cnt += 1
                            if x_ == 0 and w2[j] != 0:
                                w2[j] = 0
                                self.changes[y1][x1] = 1
                                change = True
                                
                        
        return change
    
    def Graphics(self):
        size = self.width*self.tileSize
        img = Image.new("RGB",(size, size),(0,0,0))
        if self.black == True:
            for y in range(len(self.wave)):
                for x in range(len(self.wave[0])):
                    c = 0
                    tndx = -1
                    for i, t in enumerate(self.wave[y][x]):
                        if t == 1:
                            c+= 1
                            tndx = t
                    if c == 1:
                        for y1 in range(len(tile)):
                            for x1 in range(len(tile[0])):
                                r = self.tiles[tndx][0]
                                g = self.tiles[tndx][1]
                                b = self.tiles[tndx][2]
                                img.putpixel( ((x*len(tile)) + x1, (y*len(tile)) + y1), (r, g, b))
                    
        else:
            for y in range(len(self.wave)):
                for x in range(len(self.wave[0])):
                    tile = [[ [0, 0, 0] for x in range(self.tileSize)] for y in range(self.tileSize)]
                    c = 0
                    for i, t in enumerate(self.wave[y][x]):
                        if t == 1:
                            c += 1
                            tile_ = self.tiles[i]
                            for y1 in range(len(tile_)):
                                for x1 in range(len(tile_[0])):
                                    tile[y1][x1][0] = tile[y1][x1][0] + tile_[y1][x1][0]
                                    tile[y1][x1][1] = tile[y1][x1][1] + tile_[y1][x1][1]
                                    tile[y1][x1][2] = tile[y1][x1][2] + tile_[y1][x1][2]
                    for y1 in range(len(tile)):
                        for x1 in range(len(tile[0])):
                            r = min(int(tile[y1][x1][0]/c), 255)
                            g = min(int(tile[y1][x1][1]/c), 255)
                            b = min(int(tile[y1][x1][2]/c), 255)
                            img.putpixel( ((x*len(tile)) + x1, (y*len(tile)) + y1), (r, g, b))
                        
        img.save(self.saveDir+'/'+"in_progress_{0}_{1}.png".format(self.name, self.frame), format="PNG")
        print(self.frame)
        self.frame += 1
    
    def run(self):
        run = False
        while True:
            obs = self.Observe()
            run =  obs
            self.Graphics()
            p = True
            while p == True:
                p = self.Propagate()
            
            
class Util:
    def Random(source_array, random_value):
        a_sum = sum(source_array)
        if 0 == a_sum:
            for j in range(0, len(source_array)):
                source_array[j] = 1
            a_sum = sum(source_array)
        for j in range(0, len(source_array)):
            source_array[j] /= a_sum
        i = 0
        x = 0
        while (i < len(source_array)):
            x += source_array[i]
            if random_value <= x:
                return i
            i += 1
        return 0
    
    def time(start=False, show=True, last=0, text='', end_='\n'):
        t = time.time()
        if start == True:
            if show == True:
                print(text, end=end_)
            return t
        else:
            if show == True:
                print('{0} > {1:0.4f}'.format(text, t-last), end=end_)
            return t
    
    def rotate(matrix, turns):
        m=matrix
        for i in range(turns):
            n=[[ 0 for _ in range(len(m))] for _ in range(len(m[0]))]
            for y in range(len(m)):
                for x in range(len(m[0])):
                    n[x][y] = m[(len(m)-1)-y][x]
            m=n
        return m
    
    def reflect(matrix, axis):
        if axis == 'h':
            n=[[ 0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
            for y in range(len(matrix)):
                for x in range(len(matrix[0])):
                    n[y][x] = matrix[(len(matrix)-1)-y][x]
            return n
        elif axis == 'v':
            n=[[ 0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
            for y in range(len(matrix)):
                for x in range(len(matrix[0])):
                    n[y][x] = matrix[y][(len(matrix[0])-1)-x]
            return n
        else:
            print("h for horisontal axis")
            print("v for vertical axis")
    
    def pseudo(run):
        if run == 'propagator index example':
            m1 = [[0, 1, 2],[3, 4, 5],[6, 7, 8]]
            mf = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            N = 3
            for y in range(N*2-1):
                for x in range(N*2-1):
                    dy = y-N+1
                    dx = x-N+1
                    
                    xmin = dx
                    xmax = N
                    if dx < 0:
                        xmin = 0
                        xmax = dx+N
                    ymin = dy
                    ymax = N
                    if dy < 0:
                        ymin = 0
                        ymax = dy+N
                        
                    for y1 in range(ymin, ymax):
                        for x1 in range(xmin, xmax):
                            print(' y =', y, ' x =', x,'\n', 'dy=', dy, 'dx=', dx,'\n', 'y1=', y1, ' x1=', x1)
                            print(' xmin=', xmin, 'xmax=', xmax, '\n', 'ymin=', ymin, 'ymax=', ymax)
                            print('matrix 2')
                            for i in m1: print(i)
                            print()
                            print('matrix 1')
                            for i in m1: print(i)
                            print('indexes matrix 1=', y1, x1, ':matrix 2=', y1 - dy, x1 - dx )
                            print('values matrix 1 =', m1[y1][x1],'  :matrix 2=', m1[y1 - dy][x1 - dx])
                            print(mf)
                            print(mf[x1 + self.N * y1])
                            print(mf[x1 - dx + self.N * (y1 - dy)])
                            input()
        elif run == 'symmetry index example':
            for i in range(1, 8):
                print('symmetry=',i,
                            'reflect/rotate=', i%2,
                            'get even index=', (i-1)/2, int((i-1)/2)*2)
    
    def cDir(name):
        c=0
        n = name+'_'+str(c) if c != 0 else name
        found = True
        while found:
            if n in os.listdir():
                c+=1
                n  = name+'_'+str(c)
            else:
                os.mkdir(n)
                found = False
        return n
    

if __name__ == '__main__':
    # s=Util.time(start=True, text='Setting up')
    wfc = Overlapping('Cat', 3, 48, 48, True, True, 2)
    # t=Util.time(last=s, text='Done>')
    wfc.run(fRate=7, refresh=50)
    # t=Util.time(last=t, text='Done>')





