#!/usr/bin/env python

# Copyright (C) 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function
from __future__ import division


import argparse
import json
import os.path
import pathlib2 as pathlib
import time
import vision
import argparse
import tensorflow as tf
import numpy as np
import gc
import subprocess
import sys

import Adafruit_PCA9685

import google.oauth2.credentials

from google.assistant.library import Assistant
from google.assistant.library.event import EventType
from google.assistant.library.file_helpers import existing_file
from google.assistant.library.device_helpers import register_device
from multiprocessing import Process, Value, Array

from threading import Thread
from subprocess import call

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


WARNING_NOT_REGISTERED = """
    This device is not registered. This means you will not be able to use
    Device Actions or see your device in Assistant Settings. In order to
    register this device follow instructions at:

    https://developers.google.com/assistant/sdk/guides/library/python/embed/register-device
"""

pwm = Adafruit_PCA9685.PCA9685()
threads = []
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


def process_event(event):
    """Pretty prints events.

    Prints all events that occur with two spaces between each new
    conversation and a single space between turns of a conversation.

    Args:
        event(event.Event): The current event to process.
    """
    if event.type == EventType.ON_CONVERSATION_TURN_STARTED:
        print()

    print(event)

    if (event.type == EventType.ON_CONVERSATION_TURN_FINISHED and
            event.args and not event.args['with_follow_on_turn']):
        print()
    if event.type == EventType.ON_DEVICE_ACTION:
        for command, params in event.actions:
            print('Do command', command, 'with params', str(params))
            if command == 'action.devices.commands.OnOff':
                if params['on']:
                    pwm.set_pwm(0, 0, 300)
                    print('Turning on.')
                else:
                    pwm.set_pwm(0, 0, 150)
                    print('Turning off.')
            if command == 'luxo.command.LookAtMe':
                graph = load_graph('./Imitation_test.pb')

		# Set default variables to load
                x = graph.get_tensor_by_name('import/vector_observation:0')
                y = graph.get_tensor_by_name('import/action:0')

		# Set initial positions
                initial_servo_pos = [0, 0, 0, 0]
                first_flag = 1

                pwm.set_pwm_freq(60)
                servo_set = [225, 225, 113, 225]
                pwm.set_pwm(0, 0, 375)
                pwm.set_pwm(1, 0, 375)
                pwm.set_pwm(2, 0, 263)
                pwm.set_pwm(3, 0, 375)

		# We launch a Session
                with tf.Session(graph=graph) as sess:
		#    # Note: we don't nee to initialize/restore anything
		#    # There is no Variables in this graph, only hardcoded constants
                    # Figure out way to exit loop. Google Assistant Interrupt
                    if camera.object_coord == []:
                        continue
                    if first_flag:
                        test_features = initial_servo_pos + camera.object_coord
                        first_flag = 0
                    else:
                        test_features = out_servo_pos[0] + camera.object_coord

                    # Get output servo positions from graph
                    out_servo_pos = sess.run(y,feed_dict={x: [test_features]}).tolist()

                    # Set extremes to 1 and -1, just in case
                    temp_pos = np.array(out_servo_pos)
                    temp_pos[temp_pos > 1] = 1
                    temp_pos[temp_pos < -1] = -1

                    # Translate fractions to servo PWM outputs
                    weights = np.multiply(np.array(servo_set), np.array(temp_pos))
                    weights += np.array([375, 375, 263, 375]) 
                    weights = weights.astype(int) 

                    print(weights)


                    # Set servos
                    pwm.set_pwm(0, 0, weights[0][0])
                    pwm.set_pwm(1, 0, weights[0][1])
                    pwm.set_pwm(2, 0, weights[0][2])
                    pwm.set_pwm(3, 0, weights[0][3])


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--device-model-id', '--device_model_id', type=str,
                        metavar='DEVICE_MODEL_ID', required=False,
                        help='the device model ID registered with Google')
    parser.add_argument('--project-id', '--project_id', type=str,
                        metavar='PROJECT_ID', required=False,
                        help='the project ID used to register this device')
    parser.add_argument('--device-config', type=str,
                        metavar='DEVICE_CONFIG_FILE',
                        default=os.path.join(
                            os.path.expanduser('~/.config'),
                            'googlesamples-assistant',
                            'device_config_library.json'
                        ),
                        help='path to store and read device configuration')
    parser.add_argument('--credentials', type=existing_file,
                        metavar='OAUTH2_CREDENTIALS_FILE',
                        default=os.path.join(
                            os.path.expanduser('~/.config'),
                            'google-oauthlib-tool',
                            'credentials.json'
                        ),
                        help='path to store and read OAuth2 credentials')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s ' + Assistant.__version_str__())

    args = parser.parse_args()
    with open(args.credentials, 'r') as f:
        credentials = google.oauth2.credentials.Credentials(token=None,
                                                            **json.load(f))

    device_model_id = None
    last_device_id = None
    try:
        with open(args.device_config) as f:
            device_config = json.load(f)
            device_model_id = device_config['model_id']
            last_device_id = device_config.get('last_device_id', None)
    except FileNotFoundError:
        pass

    if not args.device_model_id and not device_model_id:
        raise Exception('Missing --device-model-id option')

    # Re-register if "device_model_id" is given by the user and it differs
    # from what we previously registered with.
    should_register = (
        args.device_model_id and args.device_model_id != device_model_id)

    device_model_id = args.device_model_id or device_model_id
    
    pwm.set_pwm(0, 0, 375)
    pwm.set_pwm(1, 0, 375)
    pwm.set_pwm(2, 0, 263)
    pwm.set_pwm(3, 0, 375)


    with Assistant(credentials, device_model_id) as assistant:
        events = assistant.start()

        device_id = assistant.device_id
        print('device_model_id:', device_model_id)
        print('device_id:', device_id + '\n')

        # Re-register if "device_id" is different from the last "device_id":
        if should_register or (device_id != last_device_id):
            if args.project_id:
                register_device(args.project_id, credentials,
                                device_model_id, device_id)
                pathlib.Path(os.path.dirname(args.device_config)).mkdir(
                    exist_ok=True)
                with open(args.device_config, 'w') as f:
                    json.dump({
                        'last_device_id': device_id,
                        'model_id': device_model_id,
                    }, f)
            else:
                print(WARNING_NOT_REGISTERED)

        for event in events:
            process_event(event)

def run_new_camera():
    call(['python vision.py'], shell=True)


if __name__ == '__main__':
    gc.collect()
    t = Thread(target = run_new_camera)
    m = Thread(target = main)
    t.start()
    m.start()

