import tensorflow as tf
import pandas as pd
import numpy as np

train_terms = pd.read_csv("/kaggle/input/cafa-5-protein-function-prediction/Train/train_terms.tsv",sep="\t")
train_protein_ids = np.load('/kaggle/input/t5embeds/train_ids.npy')
train_embeddings = np.load('/kaggle/input/t5embeds/train_embeds.npy')

column_num = train_embeddings.shape[1]
train_df = pd.DataFrame(train_embeddings, columns = ["Column_" + str(i) for i in range(1, column_num+1)])

plot_df = train_terms['term'].value_counts().iloc[:100]
num_of_labels = 1500
labels = train_terms['term'].value_counts().index[:num_of_labels].tolist()
train_terms_updated = train_terms.loc[train_terms['term'].isin(labels)]
pie_df = train_terms_updated['aspect'].value_counts()

train_size = train_protein_ids.shape[0]
train_labels = np.zeros((train_size ,num_of_labels))

series_train_protein_ids = pd.Series(train_protein_ids)

for i in range(num_of_labels):
    n_train_terms = train_terms_updated[train_terms_updated['term'] ==  labels[i]]
    label_related_proteins = n_train_terms['EntryID'].unique()
    train_labels[:,i] =  series_train_protein_ids.isin(label_related_proteins).astype(float)

labels_df = pd.DataFrame(data = train_labels, columns = labels)

INPUT_SHAPE = [train_df.shape[1]]
BATCH_SIZE = 5120

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE),    
    tf.keras.layers.Dense(units=512, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(units=512, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(units=512, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(units=num_of_labels,activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy', tf.keras.metrics.AUC()],
)

history = model.fit(
    train_df, labels_df,
    batch_size=BATCH_SIZE,
    epochs=38
)

test_embeddings = np.load('/kaggle/input/t5embeds/test_embeds.npy')
column_num = test_embeddings.shape[1]
test_df = pd.DataFrame(test_embeddings, columns = ["Column_" + str(i) for i in range(1, column_num+1)])

predictions =  model.predict(test_df)
df_submission = pd.DataFrame(columns = ['Protein Id', 'GO Term Id','Prediction'])
test_protein_ids = np.load('/kaggle/input/t5embeds/test_ids.npy')
l = []
for k in list(test_protein_ids):
    l += [ k] * predictions.shape[1]   

df_submission['Protein Id'] = l
df_submission['GO Term Id'] = labels * predictions.shape[0]
df_submission['Prediction'] = predictions.ravel()
df_submission.to_csv("submission.tsv",header=False, index=False, sep="\t")
df_submission