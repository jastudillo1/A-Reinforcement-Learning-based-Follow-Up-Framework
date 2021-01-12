import json
import numpy as np
import os
import sys
from time import time
import tensorflow as tf

if __name__=='__main__':
    gpu = input("GPU number: ")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    train_path = "../data/specs-train.npy"
    test_path = "../data/specs-test.npy"
    model_dir = './models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    specs_train = np.load(train_path).item()
    specs_val = np.load(test_path).item()

    # Spec dataset
    x_train = specs_train["flux"]
    x_val = specs_val["flux"]

    # Network parameters
    original_dim = x_train.shape[-1]
    batch_size = 32
    latent_dim = 7
    epochs = 100
    rate = 1/100

    # Other parameters
    load_weights = False

    # Restore path
    restore_epoch = 0

    # Train
    train = True

    # Save configuration
    config = {
    'original_dim': original_dim,
    'batch_size': batch_size,
    'latent_dim': latent_dim,
    'epochs': epochs,
    'rate': rate,
    'load_weights': load_weights,
    'restore_epoch':restore_epoch,
    'train': train
    }
    np.save('config.npy', config)
    
    with open('config.json', 'w+') as f:
        json.dump(config, f)


def Model(batch_size, original_dim, latent_dim, rate):

    # Build encoder model
    with tf.name_scope("Encoder"):
        weights = np.linspace(0,1,5)
        units = [int(original_dim*(1-weights[i]) + latent_dim*weights[i]) for i in range(len(weights))]
        print(units)
        
        inputs = tf.placeholder(tf.float32, shape=[batch_size, units[0]], name="inputs")
        enc0 = tf.layers.dense(inputs=inputs, units=units[1],activation=tf.nn.relu)
        enc1 = tf.layers.dense(inputs=enc0,units=units[2],activation=tf.nn.relu)
        enc2 = tf.layers.dense(inputs=enc1,units=units[3],activation=tf.nn.relu)
        z_mean = tf.layers.dense(inputs=enc2,units=units[4],name="z_mean")
        z_log_var = tf.layers.dense(inputs=enc2,units=units[4], name="z_log_var")

        # Use reparameterization trick to push the sampling out as input
        epsilon_op = tf.random_normal(shape=[batch_size, latent_dim])
        epsilon = tf.placeholder(tf.float32, shape=[batch_size, units[4]], name="epsilon")
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Build decoder model
    with tf.name_scope("Decoder"):
        dec0 = tf.layers.dense(inputs=z,units=units[3],activation=tf.nn.relu)
        dec1 = tf.layers.dense(inputs=dec0,units=units[2],activation=tf.nn.relu)
        dec2 = tf.layers.dense(inputs=dec1,units=units[1],activation=tf.nn.relu)
        outputs = tf.layers.dense(inputs=dec2,units=units[0],activation=tf.nn.sigmoid,name="output")

    # Rec error
    with tf.name_scope("Loss"):
        step = tf.placeholder(tf.float32, shape=None, name="step")
        beta = rate*step
        rec = tf.reduce_mean(tf.losses.mean_squared_error(inputs,outputs)*original_dim)
        kl = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl = tf.reduce_mean(-0.5*tf.reduce_sum(kl, axis=-1))        
        vae_loss = rec + beta*kl
        
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(vae_loss)
    
    return inputs, epsilon, epsilon_op, rec, kl, vae_loss, train_op, step, beta

def enc_dec_tensors():
    enc_mean = tf.get_default_graph().get_tensor_by_name('Encoder/z_mean/BiasAdd:0')
    dec_mean = tf.get_default_graph().get_tensor_by_name('Decoder/output/Sigmoid:0')
    return enc_mean, dec_mean

def train_epoch(next_train, next_val, epsilon_op, epsilon, inputs, rec, kl, vae_loss, train_op, epoch, step, beta, sess):
    t0 = time()        
    metrics = {"rec_train": [], "kl_train": [], "loss_train": [], "rec_val": [], "kl_val": [], "loss_val": []}
    
    counter = 1
    # Train
    while True:
        try:
            batch_train = sess.run(next_train)
            epsilon_train = sess.run(epsilon_op)
            
            feed_dict_train = {inputs: batch_train, epsilon: epsilon_train, step: epoch} 
            _, rec_train, kl_train, vae_train = sess.run([train_op, rec, kl, vae_loss], feed_dict=feed_dict_train)

            metrics["rec_train"].append(rec_train)
            metrics["kl_train"].append(kl_train)
            metrics["loss_train"].append(vae_train)
            batch_str = " ".join(["Epoch",str(epoch),"Number of steps:",str(counter), 
                                  "Train loss:","{0:.4f}".format(np.mean(metrics["loss_train"])),
                                  "Time:","{0:.2f}".format(time()-t0),"s"])
            sys.stdout.write('\r'+batch_str)
            counter +=1
        except tf.errors.OutOfRangeError:
            break
    
    # Val metrics
    while True:
        try:
            batch_val = sess.run(next_val)
            epsilon_val = sess.run(epsilon_op)
            
            feed_dict_val = {inputs: batch_val, epsilon: epsilon_val, step: epoch} 
            rec_val, kl_val, vae_val = sess.run([rec, kl, vae_loss], feed_dict=feed_dict_val)

            metrics["rec_val"].append(rec_val)
            metrics["kl_val"].append(kl_val)
            metrics["loss_val"].append(vae_val)
        except tf.errors.OutOfRangeError:
            break
    
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    metrics["beta"] = sess.run(beta, feed_dict={step: epoch})
    epoch_str = ' '.join(["Epoch:",str(epoch),"Number of steps:",str(counter), 
                          "Beta:", "{0:.4f}".format(metrics["beta"]),
                          "Train loss:","{0:.4f}".format(metrics["loss_train"]),
                          "Val loss:","{0:.4f}".format(metrics["loss_val"]),
                          "Time: ","{0:.2f}".format(time()-t0),"s."])
    sys.stdout.write('\r'+epoch_str+"\n")
    return metrics

def check_improve(metrics, patience):
    loss_val = metrics["loss_val"]
    if len(loss_val) == 1:
        return True
    improve = False
    for i in range(min(patience, len(loss_val)-1)):
        if loss_val[-i]>loss_val[-(i+1)]:
            improve = True
    return improve

if __name__ == '__main__':
    tf.reset_default_graph()
    with tf.Session() as sess:
        times = {}
        t0 = time()
        inputs, epsilon, epsilon_op, rec, kl, vae_loss, train_op, step, beta = Model(batch_size, original_dim, latent_dim,
                                                                                     rate)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=30)

        dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
        dataset_train = dataset_train.shuffle(buffer_size=10000)
        dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
        iterator_train = dataset_train.make_initializable_iterator()
        next_train = iterator_train.get_next()

        dataset_val = tf.data.Dataset.from_tensor_slices(x_val)
        dataset_val = dataset_val.batch(batch_size, drop_remainder=True)
        iterator_val = dataset_val.make_initializable_iterator()
        next_val = iterator_val.get_next()

        init_epoch = 0
        metrics = None
        if load_weights:
            restore_path = './model'+str(restore_epoch)+'.h5'
            saver.restore(sess, restore_path)
            init_epoch = restore_epoch + 1  
            metrics = np.load('metrics.npy').item() 

        t1 = time()
        times['prepare_time'] = t1-t0

        if train: 
            try:
                for epoch in range(init_epoch,init_epoch+epochs):
                    sess.run(iterator_train.initializer)
                    sess.run(iterator_val.initializer)
                    metrics_epoch = train_epoch(next_train, next_val, epsilon_op, epsilon, inputs, rec, kl, vae_loss, 
                                                train_op, epoch, step, beta, sess)
                    if not metrics:
                        metrics = {key: np.array([val]) for key, val in metrics_epoch.items()}
                    else:
                        metrics = {key: np.append(metrics[key],val) for key, val in metrics_epoch.items()}
    #                 if not check_improve(metrics, patience):
    #                     saver.save(sess, model_path)
    #                     print("Early Stopping at epoch:", epoch)
    #                     break
                    if epoch%25==0:
                        model_path = model_dir+'model'+str(epoch)+'.h5'
                        saver.save(sess, model_path)

            except KeyboardInterrupt:
                print("Keyboard interrupt")

            t2 = time()
            times['train_time'] = t2-t1

            print("Saving model and metrics")
            model_path = model_dir+'model'+str(epoch)+'.h5'
            saver.save(sess, model_path)
            np.save("metrics.npy", metrics)

            t3 = time()
            times['save_time'] = t3-t2
            times['total_time'] = t3-t0
            np.save('times.npy', times)