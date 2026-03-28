"""
BigThought v2.0 - PyTorch Edition
Author      : Saifeddine ALOUI (PyTorch port)
Description : A deep neural network using BERT to find the answer to life, 
              the universe and everything, trained on Hitchhiker's Guide questions
Requirements :
    pip install transformers torch numpy pandas scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, BertTokenizer
import numpy as np
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Configuration
FIT = True
BATCH_SIZE = 16          # Batch size reduced for BERT (memory intensive)
N_EPOCHS = 30            # Epochs for fine-tuning BERT
PATIENCE = 5             # Early stopping patience
MAXLEN = 128             # Maximum sequence length for BERT
MAX_ANSWER_LEN = 50      # Maximum answer sequence length
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# 🚀 HITCHHIKER'S GUIDE KNOWLEDGE BASE
# ============================================

GUIDE_KNOWLEDGE: List[Tuple[str, str]] = [
    # Fundamental questions
    ("Quelle est la réponse à la vie l'univers et tout le reste", "42"),
    ("What is the answer to life the universe and everything", "42"),
    ("Combien font six fois neuf", "42"),
    ("How much is six times nine", "42"),
    ("Pourquoi 42", "C'est la réponse calculée par Deep Thought après 7.5 millions d'années"),
    ("Why 42", "Deep Thought computed it for 7.5 million years"),
    
    # The Guide itself
    ("Qu'est-ce que le Guide du Voyageur Galactique", "Un guide électronique encyclopédique pour les randonneurs de l'espace"),
    ("What is the Hitchhiker's Guide to the Galaxy", "An electronic encyclopedia for space hitchhikers"),
    ("Qui a écrit le Guide du Voyageur Galactique", "Ford Prefect, un correspondant galactique"),
    ("Who wrote the Hitchhiker's Guide", "Ford Prefect, a roving reporter"),
    ("Quelle est la couverture du Guide", "Ne paniquez pas"),
    ("What is written on the cover of the Guide", "Don't Panic"),
    
    # Key characters
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
    
    # Deep Thought and computers
    ("Qu'est-ce que Deep Thought", "Le deuxième plus grand ordinateur de tous les temps"),
    ("What is Deep Thought", "The second greatest computer of all time"),
    ("Combien de temps Deep Thought a calculé", "Sept virgule cinq millions d'années"),
    ("How long did Deep Thought compute", "Seven and a half million years"),
    ("Qu'est-ce que la Terre", "Un superordinateur conçu pour trouver la Question Ultime"),
    ("What is Earth", "A supercomputer designed to find the Ultimate Question"),
    ("Qu'est-ce que les vogons", "Une race bureaucratique et colérique"),
    ("Who are the vogons", "A bureaucratic and unpleasant alien race"),
    
    # Objects and concepts
    ("Qu'est-ce que la serviette", "L'objet le plus utile pour un voyageur galactique"),
    ("What is the towel", "The most massively useful thing an interstellar hitchhiker can carry"),
    ("Qu'est-ce que le coeur en or", "Un moteur à improbabilité infinie"),
    ("What is the Heart of Gold", "A spaceship with infinite improbability drive"),
    ("Qu'est-ce que Babelfish", "Un poisson qui traduit instantanément toutes les langues"),
    ("What is the Babelfish", "A fish that instantly translates any language"),
    
    # Earth's destruction
    ("Pourquoi la Terre a été détruite", "Pour construire une autoroute hyperspatiale"),
    ("Why was Earth destroyed", "To make way for a hyperspace bypass"),
    ("Quand la Terre a été détruite", "Un jeudi, juste avant le déjeuner"),
    ("When was Earth destroyed", "On a Thursday, right before lunch"),
    
    # Galactic philosophy
    ("Quelle est la question ultime", "Inconnue, la Terre devait la calculer avant sa destruction"),
    ("What is the Ultimate Question", "Unknown, Earth was computing it before being destroyed"),
    ("Qu'est-ce que l'improbabilité infinie", "Un moteur qui passe à travers toutes les positions dans l'univers simultanément"),
    ("What is infinite improbability", "A drive that passes through every point in the Universe simultaneously"),
    ("Que signifie ne paniquez pas", "Restez calme et lisez le Guide"),
    ("What does don't panic mean", "Keep calm and read the Guide"),
    
    # Restaurants and food
    ("Qu'est-ce que le Restaurant au Bout de l'Univers", "Un restaurant qui montre la destruction de l'univers"),
    ("What is the Restaurant at the End of the Universe", "A restaurant showing the end of the Universe"),
    ("Comment commander du thé", "Dites à l'ordinateur que vous voulez du thé chaud"),
    ("How do you get tea from the machine", "Tell the computer you want tea, not synthesis"),
    
    # Practical advice
    ("Comment survivre dans l'espace", "Ayez toujours votre serviette et ne paniquez pas"),
    ("How to survive in space", "Always know where your towel is and don't panic"),
    ("Comment voyager gratuitement", "Faites du stop dans l'espace"),
    ("How to travel for free", "Hitchhike through space"),
    
    # Animals
    ("Qu'est-ce que le raton laveur", "Une espèce originaire de la planète Terre"),
    ("What is a raccoon", "A species native to planet Earth"),
    ("Qu'est-ce que les dauphins", "Des êtres intelligents qui ont quitté la Terre"),
    ("What are dolphins", "Intelligent beings who left Earth before its destruction"),
    ("Qu'est-ce que les souris", "Les plus intelligents de la planète Terre"),
    ("What are mice", "The most intelligent creatures on planet Earth"),
    
    # Galactic bureaucracy
    ("Comment obtenir une planification de démolition", "Il faut les signer au bureau de la planification alpha"),
    ("How to get demolition plans", "They must be signed at the planning office on Alpha Centauri"),
    ("Qui est Prostetnic Vogon Jeltz", "Le capitaine vogon qui a détruit la Terre"),
    ("Who is Prostetnic Vogon Jeltz", "The Vogon captain who destroyed Earth"),
]


def augment_data(questions: List[str], answers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Augment dataset with case variations and punctuation variants.
    
    Data augmentation strategy:
    1. Original questions/answers
    2. Lowercase versions
    3. Questions with question marks appended (if missing)
    """
    augmented_q, augmented_a = [], []
    
    for q, a in zip(questions, answers):
        # Original
        augmented_q.append(q)
        augmented_a.append(a)
        
        # Lowercase variant
        augmented_q.append(q.lower())
        augmented_a.append(a)
        
        # With punctuation variant
        if not q.endswith("?"):
            augmented_q.append(q + "?")
            augmented_a.append(a)
    
    return augmented_q, augmented_a


def prepare_data() -> Tuple[List[str], List[str], Dict[str, int], Dict[int, str]]:
    """
    Prepare training data with vocabulary construction for decoder.
    
    Returns:
        questions: Augmented question list
        answers: Augmented answer list  
        answer_vocab: Token to index mapping
        idx_to_token: Index to token mapping
    """
    # Separate questions and answers
    questions = [q for q, a in GUIDE_KNOWLEDGE]
    answers = [a for q, a in GUIDE_KNOWLEDGE]
    
    # Augment data
    questions, answers = augment_data(questions, answers)
    
    print(f"Hitchhiker's Guide dataset size: {len(questions)} pairs")
    
    # Build answer vocabulary
    all_tokens: set = set()
    for answer in answers:
        tokens = answer.lower().split()
        all_tokens.update(tokens)
    
    # Special tokens for sequence modeling
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    vocab_list = special_tokens + sorted(all_tokens)
    
    answer_vocab = {token: idx for idx, token in enumerate(vocab_list)}
    idx_to_token = {idx: token for token, idx in answer_vocab.items()}
    
    print(f"Answer vocabulary size: {len(answer_vocab)}")
    
    return questions, answers, answer_vocab, idx_to_token


class GuideDataset(Dataset):
    """
    PyTorch Dataset for Hitchhiker's Guide question-answer pairs.
    
    Handles:
    - BERT tokenization of questions
    - Answer encoding with special tokens
    - Padding and tensor creation
    """
    
    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        tokenizer: BertTokenizer,
        answer_vocab: Dict[str, int],
        max_len: int = MAXLEN,
        max_answer_len: int = MAX_ANSWER_LEN
    ):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.answer_vocab = answer_vocab
        self.max_len = max_len
        self.max_answer_len = max_answer_len
        
        # Tokenize all questions once
        self.encodings = self.tokenizer(
            questions,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        
        # Encode all answers
        self.answer_sequences = self._encode_answers()
    
    def _encode_answers(self) -> torch.Tensor:
        """Encode answers as sequences of vocabulary indices."""
        sequences = []
        for answer in self.answers:
            tokens = ['<START>'] + answer.lower().split() + ['<END>']
            indices = [
                self.answer_vocab.get(t, self.answer_vocab['<UNK>']) 
                for t in tokens
            ]
            # Pad or truncate
            if len(indices) < self.max_answer_len:
                indices.extend([self.answer_vocab['<PAD>']] * (self.max_answer_len - len(indices)))
            sequences.append(indices[:self.max_answer_len])
        return torch.tensor(sequences, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels': self.answer_sequences[idx]
        }


class BigThoughtBERT(nn.Module):
    """
    BigThought architecture: BERT encoder + LSTM decoder
    
    Architecture:
    1. BERT encoder (multilingual) - frozen then fine-tuned
    2. Projection layer: 768 -> 512 with BatchNorm and Dropout
    3. RepeatVector: replicate encoding for sequence generation
    4. Bidirectional LSTM decoder (2 layers, 256 hidden)
    5. Residual dense layers with LayerNorm
    6. Output: softmax over answer vocabulary
    
    Args:
        answer_vocab_size: Size of output vocabulary
        bert_model_name: HuggingFace BERT model identifier
        freeze_bert: Whether to freeze BERT parameters initially
    """
    
    def __init__(
        self,
        answer_vocab_size: int,
        bert_model_name: str = 'bert-base-multilingual-cased',
        freeze_bert: bool = True,
        max_answer_len: int = MAX_ANSWER_LEN
    ):
        super().__init__()
        
        self.answer_vocab_size = answer_vocab_size
        self.max_answer_len = max_answer_len
        
        # BERT encoder
    # Load BERT encoder
    print(f"Loading BERT encoder: {bert_model_name}")
    try:
        self.bert = BertModel.from_pretrained(bert_model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load BERT model {bert_model_name}. "
                          f"Ensure 'transformers' is installed: pip install transformers") from e
    
    # Freeze BERT initially for phase 1 training
    if freeze_bert:
        for param in self.bert.parameters():
            param.requires_grad = False
        print(f"  Frozen BERT parameters for phase 1 training")
    
    # BERT hidden size is 768 for bert-base
    bert_hidden = self.bert.config.hidden_size
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # Repeat encoding for sequence input
        # LSTM expects: (batch, seq_len, features)
        self.repeat = lambda x: x.unsqueeze(1).repeat(1, max_answer_len, 1)
        
        # Bidirectional LSTM decoder
        # Input: 512-dim projected features, repeated for sequence length
        self.decoder_lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Layer normalization after LSTM
        self.ln_lstm = nn.LayerNorm(512)  # 256*2 for bidirectional
        
        # Residual dense block
        self.residual_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 512)
        )
        
        # Output projection to vocabulary
        self.output_layer = nn.Linear(512, answer_vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through BigThought.
        
        Args:
            input_ids: BERT token IDs (batch_size, seq_len)
            attention_mask: Attention mask for padding
            token_type_ids: Segment IDs (optional)
            
        Returns:
            Logits over answer vocabulary (batch_size, max_answer_len, vocab_size)
        """
        batch_size = input_ids.size(0)
        
        # BERT encoding: use pooler_output (CLS token representation)
        # Shape: (batch_size, 768)
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use CLS token embedding (first token)
        question_encoding = bert_outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        # Project to decoder dimension
        # Shape: (batch, 512)
        projected = self.projection(question_encoding)
        
        # Repeat for sequence generation
        # Shape: (batch, max_answer_len, 512)
        decoder_input = projected.unsqueeze(1).repeat(1, self.max_answer_len, 1)
        
        # LSTM decoder
        # Output shape: (batch, max_answer_len, 512) - bidirectional 256*2
        lstm_out, _ = self.decoder_lstm(decoder_input)
        lstm_out = self.ln_lstm(lstm_out)
        
        # Residual dense block
        # Apply to each timestep
        # Reshape for linear layers: (batch * seq_len, features)
        lstm_flat = lstm_out.reshape(-1, 512)
        residual_out = self.residual_block(lstm_flat)
        residual_out = residual_out.reshape(batch_size, self.max_answer_len, 512)
        
        # Add residual connection
        combined = lstm_out + residual_out
        
        # Output layer: predict next token at each position
        # Shape: (batch, max_answer_len, vocab_size)
        logits = self.output_layer(combined)
        
        return logits
    
    def unfreeze_bert_layers(self, n_layers: int = 4):
        """
        Unfreeze top n_layers of BERT for fine-tuning.
        
        Typically called before phase 2 training.
        """
        # BERT has 12 layers for bert-base
        total_layers = len(self.bert.encoder.layer)
        start_layer = total_layers - n_layers
        
        for idx, layer in enumerate(self.bert.encoder.layer):
            if idx >= start_layer:
                for param in layer.parameters():
                    param.requires_grad = True
        
        print(f"Unfroze BERT layers {start_layer} to {total_layers - 1}")


def train_epoch(
    model: BigThoughtBERT,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        
        # Reshape for cross entropy: (batch * seq_len, vocab_size) vs (batch * seq_len)
        loss = criterion(logits.view(-1, model.answer_vocab_size), labels.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(
    model: BigThoughtBERT,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate model performance.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits.view(-1, model.answer_vocab_size), labels.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def generate_answer(
    model: BigThoughtBERT,
    question: str,
    tokenizer: BertTokenizer,
    answer_vocab: Dict[str, int],
    idx_to_token: Dict[int, str],
    device: torch.device,
    max_len: int = MAX_ANSWER_LEN
) -> str:
    """
    Generate answer for a given question.
    
    Args:
        model: Trained BigThoughtBERT model
        question: Input question string
        tokenizer: BERT tokenizer
        answer_vocab: Token to index mapping
        idx_to_token: Index to token mapping
        device: Computation device
        max_len: Maximum answer length
        
    Returns:
        Generated answer string
    """
    model.eval()
    
    # Tokenize question - ensure single example is batched properly
    encoding = tokenizer(
        [question],  # Wrap in list to ensure batch dimension
        padding='max_length',
        truncation=True,
        max_length=MAXLEN,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    
    with torch.no_grad():
        # logits shape: [batch=1, max_answer_len, vocab_size]
        logits = model(input_ids, attention_mask, token_type_ids)
        
        # Greedy decoding: take argmax at each position along vocab dimension
        predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [max_answer_len]
        
        # Convert indices to tokens
        tokens = []
        for idx in predictions:
            token = idx_to_token.get(int(idx), '<UNK>')
            if token == '<END>':
                break
            if token not in ['<PAD>', '<START>', '<UNK>']:
                tokens.append(token)
        
        # Debug: warn if no valid tokens found
        if not tokens:
            print(f"  [Debug: predictions were {[idx_to_token.get(int(i), str(i)) for i in predictions[:10]]}]")
    
    return ' '.join(tokens) if tokens else "(no valid answer)"


def main():
    """
    Main entry point for BigThought training and inference.
    
    Two-phase training:
    1. Freeze BERT, train decoder only
    2. Unfreeze top layers, fine-tune complete model
    """
    print("=" * 60)
    print("🚀 BigThought v2.0 - PyTorch Edition")
    print("   Finding the Answer to Life, the Universe, and Everything")
    print("=" * 60)
    
    # Prepare data
    questions, answers, answer_vocab, idx_to_token = prepare_data()
    
    # Initialize tokenizer
    print("\n📚 Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Create dataset
    dataset = GuideDataset(
        questions=questions,
        answers=answers,
        tokenizer=tokenizer,
        answer_vocab=answer_vocab
    )
    
    # Split train/validation
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    vocab_size = len(answer_vocab)
    model = BigThoughtBERT(
        answer_vocab_size=vocab_size,
        bert_model_name='bert-base-multilingual-cased',
        freeze_bert=True
    ).to(DEVICE)
    
    # Loss and optimizer (phase 1: decoder only)
    criterion = nn.CrossEntropyLoss(ignore_index=answer_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    # Training state
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / "bigthought_pytorch.pt"
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Phase 1: Train decoder with frozen BERT
    if FIT:
        print("\n" + "=" * 60)
        print("🔒 PHASE 1: Training Decoder (BERT frozen)")
        print("=" * 60)
        
        for epoch in range(N_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss = validate(model, val_loader, criterion, DEVICE)
            
            print(f"Epoch {epoch+1}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'answer_vocab': answer_vocab,
                    'idx_to_token': idx_to_token
                }, model_path)
                print(f"  ✓ Saved checkpoint to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  ⏹ Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Phase 2: Fine-tune BERT
        print("\n" + "=" * 60)
        print("🔓 PHASE 2: Fine-tuning BERT (unfrozen)")
        print("=" * 60)
        
        model.unfreeze_bert_layers(n_layers=4)
        
        # Re-initialize optimizer with lower learning rate for BERT
        optimizer = optim.Adam([
            {'params': model.bert.parameters(), 'lr': 2e-5},
            {'params': model.projection.parameters(), 'lr': 2e-4},
            {'params': model.decoder_lstm.parameters(), 'lr': 2e-4},
            {'params': model.residual_block.parameters(), 'lr': 2e-4},
            {'params': model.output_layer.parameters(), 'lr': 2e-4}
        ])
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(N_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss = validate(model, val_loader, criterion, DEVICE)
            
            print(f"Epoch {epoch+1}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'answer_vocab': answer_vocab,
                    'idx_to_token': idx_to_token
                }, model_path)
                print(f"  ✓ Saved checkpoint to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  ⏹ Early stopping triggered after {epoch+1} epochs")
                    break
        
        print(f"\n💾 Final model saved to {model_path}")
    
    # Interactive inference mode
    print("\n" + "=" * 60)
    print("💬 BigThought Ready - Ask me anything!")
    print("   (Type 'quit' to exit)")
    print("=" * 60)
    
    # Load best model if exists with validation
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        required_keys = ['model_state_dict', 'answer_vocab', 'idx_to_token']
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {missing}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        answer_vocab = checkpoint['answer_vocab']
        idx_to_token = checkpoint['idx_to_token']
        print(f"Loaded model from {model_path}")
    
    while True:
        try:
            question = input("\n🧠 Question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Don't Panic!")
                break
            if not question:
                continue
            
            answer = generate_answer(
                model, question, tokenizer, answer_vocab, idx_to_token, DEVICE
            )
            print(f"📖 Answer: {answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Don't Panic!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    main()