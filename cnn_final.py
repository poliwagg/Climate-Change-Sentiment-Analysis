# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import gc
import pickle
import tensorflow_addons as tfa


def prepare_data(df, max_words=10000, max_len=256):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    df['message'] = df['message'].astype(str)
    tokenizer.fit_on_texts(df['message'])

    sequences = tokenizer.texts_to_sequences(df['message'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    labels = df['sentiment'].map({-1: 0, 0: 1, 1: 2})
    categorical_labels = tf.keras.utils.to_categorical(labels)

    return padded_sequences, categorical_labels, tokenizer

def create_cnn_model(vocab_size, max_len=256, embedding_dim=768, num_channels=256, num_classes=3, dropout=0.4):

    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_len),


        layers.Conv1D(filters=num_channels, kernel_size=7, activation='relu', padding='valid'),

        layers.Conv1D(filters=num_channels, kernel_size=5, activation='relu', padding='valid'),

        layers.Conv1D(filters=num_channels, kernel_size=5, activation='relu', padding='valid'),

        layers.GlobalMaxPooling1D(),

        layers.Dropout(dropout),


        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def to_class_labels(y):
    return np.argmax(y, axis=1) if len(y.shape) > 1 else y

def create_dataset(X, y, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



def train_model(X, y, num_folds=5, batch_size=32, epochs=10, learning_rate=5e-6, dropout=0.1):
    fold_history = []
    best_val_f1 = 0
    best_model_path = None

    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, to_class_labels(y))):
        print(f'\nFold {fold + 1}/{num_folds}')



        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        fold_model = create_cnn_model(
                            vocab_size=7500,
                            max_len=256,
                            embedding_dim=768,
                            num_channels=128,
                            num_classes=3,
                            dropout=dropout
                        )

        fold_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
        'accuracy',
        tfa.metrics.F1Score(num_classes=3, average='weighted', name='f1_score')
    ]
)

        model_path = f'best_final_train_model_fold_{fold+1}.keras'

        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            verbose=1
        )


        train_ds = create_dataset(X_train, y_train, batch_size=batch_size)
        val_ds = create_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

        history = fold_model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[checkpoint],
            verbose=1
        )


        val_f1_scores = history.history['val_f1_score']


        print(f"val_f1_scores = {val_f1_scores}")
        best_val_f1_this_fold = max(val_f1_scores)
        print(f"max val f1 scores = {best_val_f1_this_fold}")




        fold_history.append({
            'fold': fold + 1,
            'history': history.history
        })

        if best_val_f1_this_fold > best_val_f1:
            best_val_f1 = best_val_f1_this_fold
            best_model_path = model_path

        K.clear_session()
        gc.collect()


    best_model = tf.keras.models.load_model(best_model_path, compile=False)
    print(f"\nBest model loaded from: {best_model_path} with val_f1_score: {best_val_f1:.4f}")

    return best_model, fold_history, best_val_f1

max_words = 7500
max_len = 256
df_train = pd.read_csv("df_train.csv")
df_train = df_train.dropna(subset=['message'])

X_train, y_train, tokenizer = prepare_data(df_train, max_words=max_words, max_len=max_len)

best_model, history, best_val_f1 = train_model(X_train, y_train, num_folds=5, batch_size=16, epochs=10, learning_rate=1e-4, dropout=0.1)

best_model.save("best_model_train_cnn_final.keras")

with open("history_train_cnn_final.pkl", "wb") as f:
    pickle.dump(history, f)

#Random Search
learning_rates = [1e-4, 1e-5, 3e-5, 5e-5, 5e-6]
dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
batch_sizes = [ 16, 32, 64]
epochs_list = [8]
num_folds_list = [2]
num_trials = 30


trial_results = []  
best_score = 0  
best_params = None
best_model_overall = None



for trial in range(num_trials):
    
    lr = random.choice(learning_rates)
    dropout = random.choice(dropouts)
    batch_size = random.choice(batch_sizes)
    epochs = random.choice(epochs_list)
    num_folds = random.choice(num_folds_list)
    
    model = create_cnn_model(vocab_size=max_words, dropout=dropout)
    
    print(f"\n==== Trial {trial+1}/{num_trials} ====")
    print(f"Parameters: lr={lr}, dropout={dropout}, batch_size={batch_size}, epochs={epochs}, num_folds={num_folds}")
    
    history, best_val_f1 = train_model(model,
        X_train, y_train,
        learning_rate=lr,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        num_folds=num_folds
    )
    print(f"best val f1 ={best_val_f1}")
    

    print(f"Best Validation F1 Score: {best_val_f1:.4f}")
    
    
    trial_result = {
        'trial': trial + 1,
        'learning_rate': lr,
        'dropout': dropout,
        'batch_size': batch_size,
        'epochs': epochs,
        'num_folds': num_folds,
        'avg_val_f1': best_val_f1
        # 'test_f1': test_f1
    }
    trial_results.append(trial_result)
    
    

print("\n========== Random Search Results ==========")
print("Best Hyperparameters:", best_params)
print(f"Best Avg F1 Score: {best_score:.4f}")




df_results = pd.DataFrame(trial_results)
df_results.to_csv("cnn_trial_results2.csv", index=False)