import math
from torch.utils.tensorboard.writer import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter()
    funcs = {"sin": math.sin ,"cos": math.cos, "tan": math.tan}


    for angle in range(-360, 360):
        rad = angle * math.pi / 180

        for name, fun in funcs.items():
            val = fun(rad)
            writer.add_scalar(name, val, angle)

    writer.flush()