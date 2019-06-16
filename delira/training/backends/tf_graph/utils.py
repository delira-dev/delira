import tensorflow as tf


def initialize_uninitialized(sess):
    """
    Function to initialize only uninitialized variables in a session graph

    Parameters
    ----------
    sess : tf.Session()

    """

    global_vars = tf.global_variables()
    is_not_initialized = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars])

    not_initialized_vars = [v for (v, f) in zip(
        global_vars, is_not_initialized) if not f]

    if not_initialized_vars:
        sess.run(tf.variables_initializer(not_initialized_vars))
