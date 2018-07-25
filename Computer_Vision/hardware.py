import Adafruit_PCA9685
import numpy as np
import tensorflow as tf
import servos


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef() 
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
	# The name var will prefix every op/nodes in your graph
	# Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)

    return graph

graph = load_graph('./Imitation_better.pb') 
pwm = Adafruit_PCA9685.PCA9685()
servo0 = servos.Servo(250, 600, 0)
servo1 = servos.Servo(120, 538, 1)
servo2 = servos.Servo(154, 382, 2)
servo3 = servos.Servo(105, 620, 3)
pwm.set_pwm_freq(60)
servo0.set_pos(0)
servo1.set_pos(0)
servo2.set_pos(0)
servo3.set_pos(0)


def set_servos(positions,coords):


    # Set default variables to load
    x = graph.get_tensor_by_name('import/vector_observation:0')
    y = graph.get_tensor_by_name('import/action:0')

    # Set initial positions


    # We launch a Session
    with tf.Session(graph=graph) as sess:
        #    # Note: we don't nee to initialize/restore anything
        #    # There is no Variables in this graph, only hardcoded constants
        if coords == []:
            return positions

        test_features = positions.extend(coords)

        # Get output servo positions from graph
        out_servo_pos = sess.run(y,feed_dict={x: [positions]}).tolist()

        # Set extremes to 1 and -1, just in case
        temp_pos = np.array(out_servo_pos)
        temp_pos[temp_pos > 1] = 1
        temp_pos[temp_pos < -1] = -1


        # Set servos
        servo0.set_pos(temp_pos[0][0])
        servo1.set_pos(temp_pos[0][1])
        servo2.set_pos(temp_pos[0][2])
        servo3.set_pos(temp_pos[0][3])

    return temp_pos[0]


