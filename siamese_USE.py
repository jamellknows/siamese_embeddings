import csv
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, LeakyReLU, Dropout, GRU, GlobalMaxPooling1D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop, SGD
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import numpy as np
import pandas as pd 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Assuming input_shape is (batch_size, time_steps, features)
        self.W_q = self.add_weight(name='W_q',
                                  shape=(input_shape[-1], input_shape[-1]),
                                  initializer='uniform',
                                  trainable=True)
        
        self.W_k = self.add_weight(name='W_k',
                                  shape=(input_shape[-1], input_shape[-1]),
                                  initializer='uniform',
                                  trainable=True)
        
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Check if x is 2D or 3D
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)  # Add a time_steps dimension

        q = tf.keras.backend.dot(x, self.W_q)
        k = tf.keras.backend.dot(x, self.W_k)
        
        v = x  # Just use the input as the value for simplicity

        attn_score = tf.keras.backend.batch_dot(q, k, axes=[2, 2])
        attn_score /= tf.keras.backend.sqrt(tf.cast(tf.shape(k)[-1], dtype=tf.float32))
        attn_score = tf.keras.activations.softmax(attn_score, axis=-1)

        output = tf.keras.backend.batch_dot(attn_score, v)
        return tf.squeeze(output, axis=1)  # Remove the added time_steps dimension

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]

class EuclideanDistanceLayer(Layer):
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EuclideanDistanceLayer, self).build(input_shape)

    def call(self, vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def compute_output_shape(self, input_shape):
        shape1, _ = input_shape
        return (shape1[0], 1)

def l1_normalize(x):
    return x / K.sum(K.abs(x), axis=-1, keepdims=True)

def read_text_file(file_path):
    sentence_pairs = []
    pair_ID_list = []
    sentence_A_list = []
    sentence_B_list = []
    relatedness_score_list = []

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            header = next(reader)  # Skip the header row
            # If you want to print the header, you can uncomment the next line
            # print("\t".join(header[:4]))

            for row in reader:
                pair_ID, sentence_A, sentence_B, relatedness_score = row[:4]
                sentence_pairs.append([sentence_A, sentence_B])
                pair_ID_list.append(pair_ID)
                sentence_A_list.append(sentence_A)
                sentence_B_list.append(sentence_B)
                relatedness_score_list.append(relatedness_score)
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    relatedness_score_list = [float(num) for num in relatedness_score_list]
    df = pd.DataFrame({
        'Sentence_A': sentence_A, 
        'Sentence_B': sentence_B,
        'Sentence_Pairs': sentence_pairs,
        'relatedness_score_list': relatedness_score_list
    })
    
    df['binary_label'] = df['relatedness_score_list'].apply(lambda x: 1 if float(x) > 4.0 else 0 )

    return sentence_pairs, relatedness_score_list, sentence_A_list, sentence_B_list, df

def create_tf_dataset(dataframe):
    features = {'Sentence_A': dataframe['Sentence_A'].values,
                'Sentence_B': dataframe['Sentence_B'].values}
    labels = dataframe['binary_label'].values

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset

def custom_concatenation_layer(inputs):
    # Assuming 'inputs' is a list of tensors to concatenate
    return tf.concat(inputs, axis=-1)
# Replace 'your_file.txt' with the actual path to your text file

# Define the Siamese Network for Universal Sentence Encoder
def siamese_use_model(embedding_size=512, dense_units=256):
    # Input layers for two sentences
    input_sentence1 = Input(shape=(), dtype=tf.string, name="Sentence_A")
    input_sentence2 = Input(shape=(), dtype=tf.string, name="Sentence_B")

    # Universal Sentence Encoder layer
    use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=True)

    # Apply USE to each input sentence
    network1 = use_layer(input_sentence1)
    network1 = Dropout(0.5)(network1)
    network1 = Lambda(lambda x: tf.expand_dims(x, axis=-1))(network1)
    network1 = GRU(64)(network1)
    network1 = AttentionLayer()(network1)
    network1 = Lambda(lambda x: tf.expand_dims(x, axis=-1))(network1)
    network1 = GlobalMaxPooling1D()(network1)
    network1 = Dense(units=256, activation='relu')(network1)



    network2 = use_layer(input_sentence2)
    network2 = Dropout(0.5)(network2)
    network2 = Lambda(lambda x: tf.expand_dims(x, axis=-1))(network2)
    network2 = GRU(64)(network2)
    network2 = AttentionLayer()(network2)
    network2 = Lambda(lambda x: tf.expand_dims(x, axis=-1))(network2)
    network2 = GlobalMaxPooling1D()(network2)
    network2 = Dense(units=256, activation='relu')(network2)


    # Concatenate the embeddings
    network = EuclideanDistanceLayer()([network1, network2])

    # concatenation_layer = Lambda(custom_concatenation_layer, name='concatenation')([network1, network2])

    # Dense layers for prediction
    # dense_layer = Dense(units=dense_units)(concatenation_layer)

    # Output layer for similarity prediction (sigmoid activation for binary classification)
    output_layer = Dense(1, activation='sigmoid')(network)

    # Build the Siamese USE model
    siamese_model = Model(inputs=[input_sentence1, input_sentence2], outputs=output_layer)

    return siamese_model


# Define the Loss functions
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    y_true = tf.cast(y_true, dtype=square_pred.dtype)

    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def square_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    y_true_squared = tf.square(1/2.0 - y_true)
    y_pred_squared = tf.square(1/2.0 - y_pred)
    square_function = (1-y_true_squared - y_pred_squared)
    square_function = tf.sqrt(square_function)
    square_loss = 1 - 1/(math.sqrt(2)*square_function)
    return square_loss

def triplet_loss(y_true, y_pred):
    margin = 0.2  # You can adjust the margin as needed

    # Reshape y_pred to (batch_size, 3)
    y_pred = K.reshape(y_pred, (-1, 3))

    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    pos_distance = K.sum(K.square(anchor - positive), axis=-1)
    neg_distance = K.sum(K.square(anchor - negative), axis=-1)
    
    loss = K.maximum(pos_distance - neg_distance + margin, 0.0)
    return K.mean(loss)

def cosine_similarity(y_true, y_pred, axis=-1):
    # Ensure that y_true is of type int64
    y_true = tf.cast(y_true, tf.int64)

    # Ensure that y_true is not a float (convert if necessary)
    

    # L2 normalize y_true
    y_true = tf.linalg.l2_normalize(tf.cast(y_true, tf.float32), axis=axis)
    y_pred = tf.linalg.l2_normalize(y_pred, axis=axis)

    # Calculate cosine similarity
    similarity = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=axis)
    
    return 1 - similarity
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 10:
        lr *= 0.5
    return lr
    
def train_and_plot_metrics(history, title1):
    # Define callbacks for model training
    # model_checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
    # early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    # Learning rate schedule function


   

    # Train the model

    # Plot and save metrics
    plt.figure(figsize=(16, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot learning rate values
    plt.subplot(1, 3, 3)
    plt.plot([lr_schedule(epoch) for epoch in range(1, epochs + 1)])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    # Save the plot
    plt.savefig(f'{title1} training_metrics_plot.png')

    # Display the plot (optional)
    plt.show()

    # Save model visualization
    plot_model(model, to_file=f'{title1}_model_visualization.png', show_shapes=True)
    


if __name__ == "__main__":

    sentence_pairs, relatedness_score_list, sentence_A_list, sentence_B_list, df =  read_text_file('SICK_train.txt')
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=7)
    train_dataset = create_tf_dataset(train_df).batch(64)
    val_dataset = create_tf_dataset(val_df).batch(64)
    dataset = create_tf_dataset(df)

    

    # Define model parameters
    embedding_size = 512  # Size of Universal Sentence Encoder embeddings
    batch_size = 32
    epochs = 10
    metrics = [
    'accuracy',  # Classification
    tf.keras.metrics.Precision().name,
    tf.keras.metrics.Recall().name,
    tf.keras.metrics.AUC().name,
    tf.keras.metrics.CategoricalAccuracy().name,
    
    'mean_squared_error',  # Regression
    tf.keras.metrics.MeanAbsoluteError().name,
    tf.keras.metrics.R2Score().name,
    tf.keras.losses.Huber().name
]
    # loop the model with 3 different losses and 3 different optimisers
    siamese_model = siamese_use_model(embedding_size)
    print(siamese_model.summary())
   
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # TensorBoard for embedding visualization
    tensorboard = TensorBoard(log_dir='./logs', embeddings_freq=1)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks=[lr_scheduler, tensorboard]
    for i in range(0,3):
        for j in range(0,5):
    # Create and compile the Siamese Network model
            if i == 0:
                print("Adam Optimiser")
                title = "Siamese Adam Optimiser "
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                if j == 0:
                    print("Constrative Loss")
                    title += "Constrative Loss"
                    siamese_model.compile(optimizer=optimizer, loss=contrastive_loss, metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 1:
                    print("Square Loss")
                    title += "Square Loss"
                    siamese_model.compile(optimizer=optimizer, loss=square_loss, metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 2:
                    print("Binary Cross Entropy")
                    title += "Binary Cross Entropy"
                    siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 3:
                    print("Cosine Similarity")
                    title+="Cosine Similarity"
                    siamese_model.compile(optimizer=optimizer, loss=cosine_similarity, metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                if j == 4:
                    print("Poisson Loss")
                    title+="Poisson Loss"
                    siamese_model.compile(optimizer=optimizer, loss='poisson', metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
            if i == 1:
                optimizer = Adadelta(learning_rate=2.0, rho=0.95, epsilon=1e-07)
                title="Adadelta "
                if j == 0:
                    print("Constrative Loss")
                    title += "Constrative Loss"
                    siamese_model.compile(optimizer=optimizer, loss=contrastive_loss, metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 1:
                    print("Square Loss")
                    title += "Square Loss"
                    siamese_model.compile(optimizer=optimizer, loss=square_loss, metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 2:
                    print("Binary Cross Entropy")
                    title += "Binary Cross Entropy"
                    siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 3:
                    print("Cosine Similarity")
                    title+="Cosine Similarity"
                    siamese_model.compile(optimizer=optimizer, loss=cosine_similarity, metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 4:
                    print("Poisson Loss")
                    title+="Poisson Loss"
                    siamese_model.compile(optimizer=optimizer, loss='poisson', metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)   
            if i == 2:
                optimizer = SGD(learning_rate=0.01, momentum=0.9)
                title = "SGD "
                if j == 0:
                    print("Constrative Loss")
                    title += "Constrative Loss"
                    siamese_model.compile(optimizer=optimizer, loss=contrastive_loss, metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 1:
                    print("Square Loss")
                    title += "Square Loss"
                    siamese_model.compile(optimizer=optimizer, loss=square_loss, metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 2:
                    print("Binary Cross Entropy")
                    title += "Binary Cross Entropy"
                    siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 3:
                    print("Cosine Similarity")
                    title+="Cosine Similarity"
                    siamese_model.compile(optimizer=optimizer, loss=cosine_similarity, metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)
                if j == 4:
                    print("Poisson Loss")
                    title+="Poisson Loss"
                    siamese_model.compile(optimizer=optimizer, loss='poisson', metrics=['accuracy', 'mean_squared_error'])
                    history = siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)
                    train_and_plot_metrics(history, title)   
                    


