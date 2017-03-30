from collections import deque
import tensorflow as tf
import random
import numpy as np
import omok
import sys
import math
import util
import os

BOARD_SIZE = omok.BOARD_SIZE
FEATURE_PLANE_SIZE = 3


MINI_BATCH_SIZE = 100
FUTURE_REWARD_DISCOUNT = 0.95
MEMORY_SIZE = 500000
OBSERVATION_STEPS = 100
EXPLORE_STEPS  = 500000
INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
FINAL_RANDOM_ACTION_PROB = 0.001  # final chance of an action being random
LEARN_RATE = 1e-6
MAX_GAME = 1000000

def conv(x, k, out_dim, name):
    with tf.name_scope(name):
        kernel_shape = [k, k, int(x.get_shape()[-1]), out_dim]
        with tf.name_scope('weights'):
            w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01))
        with tf.name_scope('biases'):
            b = tf.Variable(tf.constant(0.01, shape=[out_dim]))
        with tf.name_scope('conv2d'):
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
            out = tf.nn.bias_add(conv, b)
        with tf.name_scope('relu'):
            result = tf.nn.relu(out)
            return result

def maxpool(x, k, name):
    with tf.name_scope('max_pool_%s' % name):
        kernel_shape = [1, k, k, 1]
        return tf.nn.max_pool(x, kernel_shape, strides=[1,1,1,1], padding='SAME', name='max_pool')

def linear_layer(x, out_size, name):
    with tf.name_scope('linear_%s' % name):
        shape = x.get_shape().as_list()
        weights = tf.Variable(tf.truncated_normal([shape[1], out_size], stddev=0.1))
        bias = tf.Variable(tf.constant(0.0, shape=[out_size]))
        out = tf.nn.bias_add(tf.matmul(x, weights, name='matmul'), bias, name='bias_add')
        return out 

def create_network(N, FN):
    
    action_size = N*N
    input_layer = tf.placeholder("float", [None, N, N, FN], name="input")
    
    conv1 = conv(input_layer, 5, 32, 'c1')
    pool1 = maxpool(conv1, 3, 'p1')
    conv2 = conv(pool1, 3, 64, 'c2')
    
    shape = conv2.get_shape().as_list()
    conv2_flat = tf.reshape(conv2, [-1, reduce(lambda x, y: x * y, shape[1:])])
    
    l1=  tf.nn.relu(linear_layer(conv2_flat, 512, 'hidden'))
    q =  linear_layer(l1, action_size, 'output') # 4 is action size
    
    ##########################################################################
    target = tf.placeholder("float", [None], name = "target")
    action = tf.placeholder('int64', [None], name = "action")

    with tf.name_scope('delta'):
        action_one_hot = tf.one_hot(action, action_size, 1.0, 0.0)
        masked_actions = tf.reduce_sum(q * action_one_hot, reduction_indices=1)
        delta = target - masked_actions
        clipped_delta = tf.clip_by_value(delta, -1, 1) 

    with tf.name_scope('square_loss'):
        loss = tf.reduce_mean(tf.square(clipped_delta))
    
    with tf.name_scope('train'): 
        optim = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)

    ##########################################################################
    with tf.name_scope('step'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        step_input = tf.placeholder('int32', None, name='step_input')
        step_assign_op = global_step.assign(step_input)

    params = {
        'input':input_layer,
        'output':q,
        'target':target,
        'action':action,
        'optim':optim,
        'loss':loss,
        'step':global_step,
        'step_input':step_input,
        'step_op':step_assign_op,
    }
    return params

def flip_color(turn):
    return 3 - turn

def build_feature_plane(map, turn):
    myColor = turn
    enemyColor = flip_color(turn)

    feature_map = np.zeros((FEATURE_PLANE_SIZE, BOARD_SIZE, BOARD_SIZE))
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            color = map[y][x]
            if color == myColor:
                feature_map[0][y][x] = 1 # my plane
            elif color == enemyColor:
                feature_map[1][y][x] = 1 # enemy plane
            else:
                feature_map[2][y][x] = 1 # empty plane

    return feature_map.transpose((1,2,0))
    
def get_action(map, turn, state, params):
    valid_pos = []
    # get vaildation pos
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if (map[y][x] == 0):
                valid_pos.append(y*BOARD_SIZE + x)

    assert len(valid_pos) > 0, len(valid_pos)
    if len(valid_pos) == BOARD_SIZE*BOARD_SIZE:
        # first time, place at center 
        return int(BOARD_SIZE/2) * BOARD_SIZE + int(BOARD_SIZE/2)
    elif random.random() < probability_of_random_action:
        selected = random.randint(0, len(valid_pos) - 1)
        return valid_pos[selected]
    else:
        result = sess.run(params["output"], feed_dict={params["input"]:[state]})[0]
        v_activations = []
        for i in range(len(valid_pos)):
            v_activations.append(result[valid_pos[i]])
        return valid_pos[np.argmax(v_activations)]

def train(sess, params):

    mini_batch = random.sample(memory, MINI_BATCH_SIZE)
    states = [d[0] for d in mini_batch]
    actions = [d[1] for d in mini_batch]
    current_rewards = [d[2] for d in mini_batch]
    states_next = [d[3] for d in mini_batch]
    terminals = [d[4] for d in mini_batch]

    future_rewards = sess.run(params["output"], feed_dict={params["input"]:states_next})
    rewards = []
    for i in range(MINI_BATCH_SIZE):
        terminal = terminals[i]
        if terminal:
            rewards.append(current_rewards[i])
        else:
            # get valid pos
            v_rewards = []
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    if (states_next[i][y][x][2] == 1): # check empty place
                        v_rewards.append(future_rewards[i][y*BOARD_SIZE + x])
            assert (len(v_rewards) > 0), states_next[i]

            rewards.append(np.max(v_rewards) * FUTURE_REWARD_DISCOUNT + current_rewards[i]) 

    sess.run(params["optim"], feed_dict={params["input"]:states, params["target"]:rewards, params["action"]:actions })


elapsed_turn = 0

def isDrawGame(map):
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if map[y][x] == 0:
                return False
    return True


def ai_play_action_fn(map, turn, state, params):
    result = omok.AI(map, turn)
    x = result[0]
    y = result[1]
    return y*BOARD_SIZE + x

def ai_play(_board, turn):
    return base_play(_board, turn, ai_play_action_fn)

def dqn_play(_board, turn):
    return base_play(_board, turn, get_action)

def base_play(_board, turn, action_fn):
    global probability_of_random_action
    global elapsed_turn

    map = np.copy(_board)
    state = build_feature_plane(map, turn)
    action = action_fn(map, turn, state, params)
    
    x = action % BOARD_SIZE
    y = int(action / BOARD_SIZE)
    assert(map[y][x] == 0)
    map[y][x] = turn
    
    state_next = build_feature_plane(map, turn)
    if omok.wincheck(map, x, y, turn):
        reward = 1
        terminal = True

        # super ultra trick
        # modify enemy last memory : default => lose
        _state, _action, _reward, _state_next, _terminal = memory.pop()
        assert (_reward == 0)
        assert (_terminal == False)

        memory.append((_state, _action, -1, _state_next, True))
    elif isDrawGame(map):
        reward = -0.1 # draw score is -0.1
        terminal = True

        # super ultra trick
        # modify enemy last memory : default => lose
        _state, _action, _reward, _state_next, _terminal = memory.pop()
        assert (_reward == 0)
        assert (_terminal == False)

        memory.append((_state, _action, -0.1, _state_next, True))

    else:
        reward = 0
        terminal = False

    memory.append((state, action, reward, state_next, terminal))
    while  len(memory) > MEMORY_SIZE:
         memory.popleft()

    if len(memory) > OBSERVATION_STEPS:
        train(sess, params)
        probability_of_random_action = max(probability_of_random_action - 0.00001, FINAL_RANDOM_ACTION_PROB)

    elapsed_turn += 1
    return (x, y)

if __name__ == "__main__":

    #screen = omok.init()
    screen = None
    with tf.Session() as sess:
        memory = deque()
        probability_of_random_action = 1

        params = create_network(BOARD_SIZE, FEATURE_PLANE_SIZE)

        saver = tf.train.Saver()
        file_name = 'omok-model2'
        if os.path.exists(file_name):
            print('restored')
            saver.restore(sess, file_name)
        else:
            print('initialized')
            sess.run(tf.initialize_all_variables())
        
        total_elapsed_turn = 0
        win_count = 0
        for ngame in range(MAX_GAME):
            elapsed_turn = 0
            omok.newgame()
            result = omok.main([dqn_play, ai_play], screen)
            if result == omok.BLACK:
                win_count += 1

            total_elapsed_turn += elapsed_turn 
            if ngame % 100 == 0:
                saver.save(sess, file_name)
                print('%d game played win ratio %d%%, avg elapsed turn %f, random prob %f' % (ngame, win_count, total_elapsed_turn*0.01, probability_of_random_action))
                total_elapsed_turn = 0