import tensorflow as tf
import numpy as np
import omok
import omok_dqn

BOARD_SIZE = omok.BOARD_SIZE
if __name__ == "__main__":
    graph,_,_ = omok_dqn.create_network(9, 4)
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./network/omok-11")
        
        screen = omok.init()
        win_count = 0
        for i in range(100):
            omok.newgame()
            omok.draw(screen) 
            win = omok.main([omok_dqn.validate_play, omok.get_player_input], screen)
            if win == omok.BLACK:
                win_count += 1

    print('winning %d%%' % win_count) 
        