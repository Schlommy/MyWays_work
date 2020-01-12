import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K
RANDOM_SEED = 3
tf.random.set_seed(RANDOM_SEED)
K.set_floatx('float64')

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

@tf.function
def cohen_kappa_loss(y_true, y_pred, row_label_vec, col_label_vec, weight_mat,  eps=1e-6, dtype=tf.float64):
    labels = tf.matmul(y_true, col_label_vec)
    weight = tf.pow(tf.tile(labels, [1, tf.shape(y_true)[1]]) - tf.tile(row_label_vec, [tf.shape(y_true)[0], 1]), 2)
    weight /= tf.cast(tf.pow(tf.shape(y_true)[1] - 1, 2), dtype=dtype)
    numerator = tf.reduce_sum(weight * y_pred)
    
    denominator = tf.reduce_sum(
        tf.matmul(
            tf.reduce_sum(y_true, axis=0, keepdims=True),
            tf.matmul(weight_mat, tf.transpose(tf.reduce_sum(y_pred, axis=0, keepdims=True)))
        )
    )
    
    denominator /= tf.cast(tf.shape(y_true)[0], dtype=dtype)
    
    return tf.math.log(numerator / denominator + eps)

class CohenKappaLoss(tf.keras.losses.Loss):
    def __init__(self,
                 num_classes,
                 name='cohen_kappa_loss',
                 eps=1e-6,
                 dtype=tf.float64):
        super(CohenKappaLoss, self).__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        
        self.num_classes = num_classes
        self.eps = eps
        self.dtype = dtype
        label_vec = tf.range(num_classes, dtype=dtype)
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        self.weight_mat = tf.pow(
            tf.tile(self.col_label_vec, [1, num_classes]) - tf.tile(self.row_label_vec, [num_classes, 1]),
        2) / tf.cast(tf.pow(num_classes - 1, 2), dtype=dtype)


    def call(self, y_true, y_pred, sample_weight=None):
        return cohen_kappa_loss(
            y_true, y_pred, self.row_label_vec, self.col_label_vec, self.weight_mat, self.eps, self.dtype
        )


    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "eps": self.eps,
            "dtype": self.dtype
        }
        base_config = super(CohenKappaLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CohenKappa(Metric):
    """
    This metric is copied from TensorFlow Addons
    """
    def __init__(self,
                 num_classes,
                 name='cohen_kappa',
                 weightage=None,
                 dtype=tf.float32):
        super(CohenKappa, self).__init__(name=name, dtype=dtype)

        if weightage not in (None, 'linear', 'quadratic'):
            raise ValueError("Unknown kappa weighting type.")
        else:
            self.weightage = weightage

        self.num_classes = num_classes
        self.conf_mtx = self.add_weight(
            'conf_mtx',
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=tf.int32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) == 2:
            y_true = tf.argmax(y_true, axis=1)
        if len(y_pred.shape) == 2:
            y_pred = tf.argmax(y_pred, axis=1)
        
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.int32)
        
        if y_true.shape.as_list() != y_pred.shape.as_list():
            raise ValueError(
                "Number of samples in y_true and y_pred are different")

        # compute the new values of the confusion matrix
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight)

        # update the values in the original confusion matrix
        return self.conf_mtx.assign_add(new_conf_mtx)
    
    def result(self):
        nb_ratings = tf.shape(self.conf_mtx)[0]
        weight_mtx = tf.ones([nb_ratings, nb_ratings], dtype=tf.int32)

        # 2. Create a weight matrix
        if self.weightage is None:
            diagonal = tf.zeros([nb_ratings], dtype=tf.int32)
            weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

        else:
            weight_mtx += tf.range(nb_ratings, dtype=tf.int32)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

            if self.weightage == 'linear':
                weight_mtx = tf.abs(weight_mtx - tf.transpose(weight_mtx))
            else:
                weight_mtx = tf.pow((weight_mtx - tf.transpose(weight_mtx)), 2)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

        # 3. Get counts
        actual_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=1)
        pred_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=0)

        # 4. Get the outer product
        out_prod = pred_ratings_hist[..., None] * \
                    actual_ratings_hist[None, ...]

        # 5. Normalize the confusion matrix and outer product
        conf_mtx = self.conf_mtx / tf.reduce_sum(self.conf_mtx)
        out_prod = out_prod / tf.reduce_sum(out_prod)

        conf_mtx = tf.cast(conf_mtx, dtype=tf.float32)
        out_prod = tf.cast(out_prod, dtype=tf.float32)

        # 6. Calculate Kappa score
        numerator = tf.reduce_sum(conf_mtx * weight_mtx)
        denominator = tf.reduce_sum(out_prod * weight_mtx)
        kp = 1 - (numerator / denominator)
        return kp
    
    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
        }
        base_config = super(CohenKappa, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        """Resets all of the metric state variables."""

        for v in self.variables:
            K.set_value(
                v, np.zeros((self.num_classes, self.num_classes), np.int32))


# Usage:
# number of classes= 5
# model.compile(optimizer=tf.keras.optimizers.Nadam(), loss=CohenKappaLoss(5), 
#                   metrics=[CohenKappa(num_classes=5, weightage='quadratic')])
