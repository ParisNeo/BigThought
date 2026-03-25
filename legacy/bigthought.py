# BigThought v2.0 - BERT Edition
# Author      : Saifeddine ALOUI
# Description : Un réseau neuronal profond utilisant BERT pour trouver la réponse à la vie, 
#               l'univers et tout le reste, entraîné sur des questions du Guide du Voyageur Galactique
# Requirements :
# pip install transformers tensorflow>=2.8.0 numpy pandas scikit-learn

import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import numpy as np
import json
import re
from pathlib import Path

# Configuration
fit = True
bs = 16         # Batch size réduit pour BERT (plus gourmand en mémoire)
n_epochs = 30   # Époques pour bien fine-tuner BERT
patience = 5    # Early stopping patience
maxlen = 128    # Longueur maximale des séquences pour BERT

# ============================================
# 🚀 BASE DE CONNAISSANCES DU GUIDE GALACTIQUE
# ============================================
# Données sur le Guide du Voyageur Galactique, la vie, l'univers et tout le reste

guide_knowledge = [
    # Questions fondamentales
    ("Quelle est la réponse à la vie l'univers et tout le reste", "42"),
    ("What is the answer to life the universe and everything", "42"),
    ("Combien font six fois neuf", "42"),
    ("How much is six times nine", "42"),
    ("Pourquoi 42", "C'est la réponse calculée par Deep Thought après 7.5 millions d'années"),
    ("Why 42", "Deep Thought computed it for 7.5 million years"),
    
    # Le Guide lui-même
    ("Qu'est-ce que le Guide du Voyageur Galactique", "Un guide électronique encyclopédique pour les randonneurs de l'espace"),
    ("What is the Hitchhiker's Guide to the Galaxy", "An electronic encyclopedia for space hitchhikers"),
    ("Qui a écrit le Guide du Voyageur Galactique", "Ford Prefect, un correspondant galactique"),
    ("Who wrote the Hitchhiker's Guide", "Ford Prefect, a roving reporter"),
    ("Quelle est la couverture du Guide", "Ne paniquez pas"),
    ("What is written on the cover of the Guide", "Don't Panic"),
    
    # Personnages clés
    ("Qui est Arthur Dent", "Le dernier humain survivant de la Terre"),
    ("Who is Arthur Dent", "The last surviving human from Earth"),
    ("Qui est Ford Prefect", "Un correspondant du Guide et ami d'Arthur"),
    ("Who is Ford Prefect", "A Guide researcher and Arthur's friend"),
    ("Qui est Zaphod Beeblebrox", "Le président de la Galaxie avec deux têtes"),
    ("Who is Zaphod Beeblebrox", "The two-headed President of the Galaxy"),
    ("Qui est Marvin", "Un robot dépressif et paranorme"),
    ("Who is Marvin", "A paranoid android, brain the size of a planet"),
    ("Qui est Trillian", "Une astrophysicienne et la seule autre survivante humaine"),
    ("Who is Trillian", "An astrophysicist and the other surviving human"),
    
    # Deep Thought et les ordinateurs
    ("Qu'est-ce que Deep Thought", "Le deuxième plus grand ordinateur de tous les temps"),
    ("What is Deep Thought", "The second greatest computer of all time"),
    ("Combien de temps Deep Thought a calculé", "Sept virgule cinq millions d'années"),
    ("How long did Deep Thought compute", "Seven and a half million years"),
    ("Qu'est-ce que la Terre", "Un superordinateur conçu pour trouver la Question Ultime"),
    ("What is Earth", "A supercomputer designed to find the Ultimate Question"),
    ("Qu'est-ce que le vogons", "Une race bureaucratique et colérique"),
    ("Who are the vogons", "A bureaucratic and unpleasant alien race"),
    
    # Objets et concepts
    ("Qu'est-ce que la serviette", "L'objet le plus utile pour un voyageur galactique"),
    ("What is the towel", "The most massively useful thing an interstellar hitchhiker can carry"),
    ("Qu'est-ce que le coeur en or", "Un moteur à improbabilité infinie"),
    ("What is the Heart of Gold", "A spaceship with infinite improbability drive"),
    ("Qu'est-ce que Babelfish", "Un poisson qui traduit instantanément toutes les langues"),
    ("What is the Babelfish", "A fish that instantly translates any language"),
    
    # La destruction de la Terre
    ("Pourquoi la Terre a été détruite", "Pour construire une autoroute hyperspatiale"),
    ("Why was Earth destroyed", "To make way for a hyperspace bypass"),
    ("Quand la Terre a été détruite", "Un jeudi, juste avant le déjeuner"),
    ("When was Earth destroyed", "On a Thursday, right before lunch"),
    
    # Philosophie galactique
    ("Quelle est la question ultime", "Inconnue, la Terre devait la calculer avant sa destruction"),
    ("What is the Ultimate Question", "Unknown, Earth was computing it before being destroyed"),
    ("Qu'est-ce que l'improbabilité infinie", "Un moteur qui passe à travers toutes les positions dans l'univers simultanément"),
    ("What is infinite improbability", "A drive that passes through every point in the Universe simultaneously"),
    ("Que signifie ne paniquez pas", "Restez calme et lisez le Guide"),
    ("What does don't panic mean", "Keep calm and read the Guide"),
    
    # Restaurants et nourriture
    ("Qu'est-ce que le Restaurant au Bout de l'Univers", "Un restaurant qui montre la destruction de l'univers"),
    ("What is the Restaurant at the End of the Universe", "A restaurant showing the end of the Universe"),
    ("Comment commander du thé", "Dites à l'ordinateur que vous voulez du thé chaud"),
    ("How do you get tea from the machine", "Tell the computer you want tea, not synthesis"),
    
    # Conseils pratiques
    ("Comment survivre dans l'espace", "Ayez toujours votre serviette et ne paniquez pas"),
    ("How to survive in space", "Always know where your towel is and don't panic"),
    ("Comment voyager gratuitement", "Faites du stop dans l'espace"),
    ("How to travel for free", "Hitchhike through space"),
    
    # Animaux
    ("Qu'est-ce que le raton laveur", "Une espèce originaire de la planète Terre"),
    ("What is a raccoon", "A species native to planet Earth"),
    ("Qu'est-ce que les dauphins", "Des êtres intelligents qui ont quitté la Terre"),
    ("What are dolphins", "Intelligent beings who left Earth before its destruction"),
    ("Qu'est-ce que les souris", "Les plus intelligents de la planète Terre"),
    ("What are mice", "The most intelligent creatures on planet Earth"),
    
    # Bureaucratie galactique
    ("Comment obtenir une planification de démolition", "Il faut les signer au bureau de la planification alpha"),
    ("How to get demolition plans", "They must be signed at the planning office on Alpha Centauri"),
    ("Qui est Prostetnic Vogon Jeltz", "Le capitaine vogon qui a détruit la Terre"),
    ("Who is Prostetnic Vogon Jeltz", "The Vogon captain who destroyed Earth"),
]

# Séparer questions et réponses
questions = [q for q, a in guide_knowledge]
answers = [a for q, a in guide_knowledge]

# Augmenter les données avec des variations (data augmentation simple)
augmented_questions = []
augmented_answers = []

for q, a in zip(questions, answers):
    augmented_questions.append(q)
    augmented_answers.append(a)
    # Ajouter des variations de casse
    augmented_questions.append(q.lower())
    augmented_answers.append(a)
    # Ajouter avec ponctuation
    if not q.endswith("?"):
        augmented_questions.append(q + "?")
        augmented_answers.append(a)

questions = augmented_questions
answers = augmented_answers

print(f"Taille du dataset Guide Galactique: {len(questions)} paires")

# ============================================
# 🔤 TOKENIZATION AVEC BERT
# ============================================

# Charger le tokenizer BERT français-anglais multilingue
print("Chargement du tokenizer BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenizer les questions (entrée du modèle)
print("Tokenisation des questions...")
encoded_questions = tokenizer(
    questions,
    padding='max_length',
    truncation=True,
    max_length=maxlen,
    return_tensors='tf'
)

# Pour les réponses, on crée un vocabulaire spécifique
# Extraction de tous les tokens uniques des réponses
all_answer_tokens = set()
for answer in answers:
    # Tokeniser simplement par mots pour le décodeur
    tokens = answer.lower().split()
    all_answer_tokens.update(tokens)

# Ajouter tokens spéciaux
special_tokens = ['[PAD]', '[START]', '[END]', '[UNK]']
answer_vocab = {token: idx for idx, token in enumerate(special_tokens + sorted(all_answer_tokens))}
idx_to_token = {idx: token for token, idx in answer_vocab.items()}
answer_vocab_size = len(answer_vocab)

print(f"Taille du vocabulaire des réponses: {answer_vocab_size}")

# Encoder les réponses
def encode_answer(answer, max_len=50):
    """Encode une réponse en séquence d'indices avec tokens START/END"""
    tokens = ['[START]'] + answer.lower().split() + ['[END]']
    indices = [answer_vocab.get(t, answer_vocab['[UNK]']) for t in tokens]
    # Padding
    if len(indices) < max_len:
        indices.extend([answer_vocab['[PAD]']] * (max_len - len(indices)))
    return indices[:max_len]

# Préparer les données d'entraînement
max_answer_len = 50
encoded_answers = [encode_answer(a, max_answer_len) for a in answers]

# Convertir en tenseurs TensorFlow
input_ids = encoded_questions['input_ids']
attention_mask = encoded_questions['attention_mask']
token_type_ids = encoded_questions['token_type_ids']

# Convertir les réponses en one-hot pour l'entraînement
y_data = tf.keras.utils.to_categorical(encoded_answers, num_classes=answer_vocab_size)

print(f"Shape des entrées: {input_ids.shape}")
print(f"Shape des sorties: {y_data.shape}")

# ============================================
# 🧠 ARCHITECTURE DU MODÈLE: BERT + DÉCODEUR
# ============================================

def build_bigthought_bert():
    """
    Architecture hybride:
    - BERT comme encodeur de questions (gelé puis fine-tuné progressivement)
    - Décodeur LSTM pour générer les réponses token par token
    """
    
    # Entrées BERT
    input_ids_layer = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="input_ids")
    attention_mask_layer = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="attention_mask")
    token_type_ids_layer = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="token_type_ids")
    
    # Charger BERT (encoder seulement)
    print("Chargement de BERT encoder...")
    bert_encoder = TFBertModel.from_pretrained(
        'bert-base-multilingual-cased',
        from_tf=True
    )
    
    # Sortie du pooler [CLS] de BERT (768 dimensions)
    bert_outputs = bert_encoder(
        input_ids=input_ids_layer,
        attention_mask=attention_mask_layer,
        token_type_ids=token_type_ids_layer
    )
    
    # Utiliser le token [CLS] comme représentation de la question
    # Shape: (batch_size, 768)
    question_encoding = bert_outputs.pooler_output
    
    # Projection vers l'espace du décodeur
    h = tf.keras.layers.Dense(512, activation="relu", name="projection")(question_encoding)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dropout(0.3)(h)
    
    # Réshape pour le décodeur séquentiel
    # On répète l'encodage pour initier la séquence de sortie
    h = tf.keras.layers.RepeatVector(max_answer_len)(h)
    
    # Décodeur LSTM bidirectionnel
    h = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2),
        name="decoder_lstm_1"
    )(h)
    
    h = tf.keras.layers.LayerNormalization()(h)
    
    h = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2),
        name="decoder_lstm_2"
    )(h)
    
    # Couches denses avec connexions résiduelles
    residual = h
    h = tf.keras.layers.Dense(512, activation="relu")(h)
    h = tf.keras.layers.LayerNormalization()(h)
    h = tf.keras.layers.Dropout(0.3)(h)
    h = tf.keras.layers.Dense(512)(h)
    h = tf.keras.layers.Add()([h, residual])
    
    # Couche de sortie - prédiction du vocabulaire de réponse
    output = tf.keras.layers.Dense(
        answer_vocab_size, 
        activation="softmax", 
        name="answer_logits"
    )(h)
    
    # Construire le modèle
    model = tf.keras.models.Model(
        inputs=[input_ids_layer, attention_mask_layer, token_type_ids_layer],
        outputs=output
    )
    
    return model, bert_encoder

# ============================================
# 🏋️ ENTRAÎNEMENT
# ============================================

# Construire le modèle
model, bert_encoder = build_bigthought_bert()

# Stratégie d'entraînement en deux phases
# Phase 1: Geler BERT, entraîner seulement le décodeur
# Phase 2: Fine-tuning complet

print("\n=== PHASE 1: Entraînement du décodeur (BERT gelé) ===")

# Geler BERT
for layer in bert_encoder.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

model.summary()

# Callbacks
model_folder = Path("model")
model_folder.mkdir(exist_ok=True, parents=True)
model_path = model_folder/"bigthought_bert.keras"

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(
        patience=patience, 
        monitor="val_loss",
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(model_path), 
        monitor='val_loss', 
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

# Séparer train/validation
n_samples = len(input_ids)
val_split = 0.2
val_size = int(n_samples * val_split)

indices = np.random.permutation(n_samples)
train_idx = indices[val_size:]
val_idx = indices[:val_size]

# Données d'entraînement
train_inputs = {
    'input_ids': input_ids.numpy()[train_idx],
    'attention_mask': attention_mask.numpy()[train_idx],
    'token_type_ids': token_type_ids.numpy()[train_idx]
}
train_outputs = y_data[train_idx]

val_inputs = {
    'input_ids': input_ids.numpy()[val_idx],
    'attention_mask': attention_mask.numpy()[val_idx],
    'token_type_ids': token_type_ids.numpy()[val_idx]
}
val_outputs = y_data[val_idx]

if fit:
    print("Entraînement Phase 1...")
    history1 = model.fit(
        train_inputs,
        train_outputs,
        validation_data=(val_inputs, val_outputs),
        epochs=20,
        batch_size=bs,
        callbacks=callbacks_phase1
    )
    
    print("\n=== PHASE 2: Fine-tuning complet (BERT dégelé) ===")
    
    # Débloquer les dernières couches de BERT pour fine-tuning
    for layer in bert_encoder.layers[-4:]:  # Dégeler les 4 dernières couches
        layer.trainable = True
    
    # Recompiler avec un learning rate plus faible pour BERT
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            patience=patience, 
            monitor="val_loss",
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path), 
            monitor='val_loss', 
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
    ]
    
    print("Entraînement Phase 2...")
    history2 = model.fit(
        train_inputs,
        train_outputs,
        validation_data=(val_inputs, val_outputs),
        epochs=n_epochs,
        batch_size=bs,
        callbacks=callbacks_phase2
    )
    
    print(f"\nModèle sauvegardé dans: {model_path}")
