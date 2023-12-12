#!/home/user/anaconda3/envs/tf/bin/python

from encoder import ArgsParse, Encoder3D

if __name__ == '__main__':
    args = ArgsParse()
    encoder = Encoder3D(args)
    encoder.run()
