from the_model import TheModelSpace
import numpy as np

NUM_DATA_POINTS = 100
NUM_EVAL_POINTS = 100

model_space = TheModelSpace()
model_space.set_num_in_dims(3)
model_space.set_num_out_dims(2)
model_space.add_model("model")

def f(x):
    squared = x * x
    return squared[0:2]

data_points = []
for _ in range(NUM_DATA_POINTS):
    in_point = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    out_point = f(in_point)
    out_precision = np.eye(2)
    model_space.add_datapoint("model", in_point, out_point, out_precision)

print model_space.models["model"].mean
