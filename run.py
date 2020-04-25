from RESNET_CIFAR100 import train
from collections import namedtuple

if __name__ == "__main__":
    Struct = namedtuple('Struct', "net gpu w b s warm lr")

    args = Struct(
        net= "resnet50",
        gpu= False,
        w= 2,
        b= 128,
        s= True,
        warm= 1,
        lr= 0.1
    )
    print(args.net)
    x = train.Train(args)
    x.run()
