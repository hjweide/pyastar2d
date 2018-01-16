import cv2
import numpy as np

import pyastar

from time import time
from time import sleep
from os.path import basename, join, splitext

# input/output files
#MAZE_FPATH = join('mazes', 'blurred.png')
RAW_FPATH = join('mazes', 'maze5.png')
#MAZE_FPATH = join('mazes', 'maze_large.png')
OUTP_FPATH = join('solns', '%s_soln.png' % splitext(basename(RAW_FPATH))[0])

mouseX = 10
mouseY = 10

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        print(x,y)

def main():
    global mouseX
    global mouseY
    #maze = cv2.imread(MAZE_FPATH)
    raw_img = cv2.imread(RAW_FPATH)

    raw_img_copy = cv2.imread(RAW_FPATH)

    maze = cv2.blur(raw_img,(10,10))

    # maze = cv2.resize(maze_huge, (1000, 1000)) 

    #cv2.imwrite("modified.png", maze)


    if maze is None:
        print('no file found: %s' % (RAW_FPATH))
        return
    else:
        print('loaded maze of shape %r' % (maze.shape[0:2],))

    grid = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grid[grid == 0] = 1
    grid[grid == 255] = 1

    assert grid.min() == 1, 'cost of moving must be at least 1'


    print (grid)
    # start is the first white block in the top row
    start_j, = np.where(grid[0, :] == 1)
    print (start_j)
    start = np.array([0, start_j[len(start_j)/2]])
    print ("start at : ", start)

    # end is the first white block in the final column
    end_i, = np.where(grid[grid.shape[0]-1,:] == 1)
    print (end_i)
    end = np.array([grid.shape[0] - 1, end_i[len(end_i)/2]])
    print("end at : ", end)

    t0 = time()
    path = pyastar.astar_path(grid, start, end)


    print path[50 : 60]
    dur = time() - t0

    if path.shape[0] > 0:
        print('found path of length %d in %.6fs' % (path.shape[0], dur))
        

        while(1):
            cv2.namedWindow('RESULT')
            cv2.setMouseCallback('RESULT',draw_circle)
            sleep(0.1)
            print(mouseY, mouseX)
            cv2.imshow("RESULT", raw_img)
            if (cv2.waitKey(20)==113):
                start = np.array([mouseY, mouseX])
                path = pyastar.astar_path(grid, start, end)

                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        if(raw_img[i,j][2] == 255):
                            raw_img[i,j] = (255,255,255)
                raw_img[path[:, 0], path[:, 1]] = (0, 0, 255)
                raw_img[path[:, 0]-1, path[:, 1]] = (0, 0, 255)
                raw_img[path[:, 0], path[:, 1]-1] = (0, 0, 255)
                #raw_img = raw_img_copy[:]

                #break
        print('plotting path to %s' % (OUTP_FPATH))
        cv2.imwrite(OUTP_FPATH, raw_img)
        #while(1):
        

    else:
        print('no path found')

    print('done')


if __name__ == '__main__':
    main()
