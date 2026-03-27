"""
BigThought-TRM: Tiny Recursive Model Edition
Author      : Saifeddine ALOUI (TRM Adaptation)
Description : BigThought converted to Tiny Recursive Model architecture.
              Uses iterative latent refinement instead of encoder-decoder.
              Based on "Less is More: Recursive Reasoning with Tiny Networks" 
              (Jolicoeur-Martineau, 2025)
Requirements : pip install torch numpy pandas scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.swa_utils import AveragedModel
import numpy as np
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ============================================
# 🚀 TRM Configuration
# ============================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRM Hyperparameters (from paper)
HIDDEN_DIM = 512              # Embedding dimension (D)
NUM_LAYERS = 2                # Only 2 layers!
N_RECURSIONS = 6              # 'n' in paper (recursions per supervision step)
MAX_SUPERVISION_STEPS = 3     # 'T' in paper (deep supervision steps)
MAX_ANSWER_LEN = 50           # Fixed output length

# Training
BATCH_SIZE = 16
N_EPOCHS = 100                # More epochs needed for deep supervision
PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1.0            # High weight decay helps small models
EMA_DECAY = 0.999             # Exponential Moving Average (critical for small data)
MAXLEN = 128                  # Max question length

FIT = True

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

def augment_data(questions: List[str], answers: List[str], factor: int = 10) -> Tuple[List[str], List[str]]:
    """
    Aggressive data augmentation for TRM (critical for small datasets).
    
    Strategy:
    1. Original questions/answers
    2. Lowercase versions
    3. Questions with/without question marks
    4. Random shuffling of word order (for robustness)
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
        else:
            augmented_q.append(q[:-1])  # Without question mark
            augmented_a.append(a)
        
        # Additional augmentations up to factor
        for i in range(factor - 3):
            if i % 2 == 0:
                augmented_q.append(q.lower() + "?" if not q.endswith("?") else q.lower())
            else:
                augmented_q.append(q.upper())
            augmented_a.append(a)
    
    return augmented_q, augmented_a

def prepare_data() -> Tuple[List[str], List[str], Dict[str, int], Dict[int, str], Dict[str, int]]:
    """
    Prepare training data with vocabulary construction.
    
    Returns:
        questions: Augmented question list
        answers: Augmented answer list  
        answer_vocab: Token to index mapping
        idx_to_token: Index to token mapping
        question_vocab: Question character/word to index mapping
    """
    # Separate questions and answers
    questions = [q for q, a in GUIDE_KNOWLEDGE]
    answers = [a for q, a in GUIDE_KNOWLEDGE]
    
    # Aggressive augmentation (TRMs need this for small datasets)
    questions, answers = augment_data(questions, answers, factor=20)
    
    print(f"Hitchhiker's Guide dataset size: {len(questions)} pairs (augmented)")
    
    # Build answer vocabulary (word-level)
    all_tokens: set = set()
    for answer in answers:
        tokens = answer.lower().split()
        all_tokens.update(tokens)
    
    # Special tokens
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    vocab_list = special_tokens + sorted(all_tokens)
    
    answer_vocab = {token: idx for idx, token in enumerate(vocab_list)}
    idx_to_token = {idx: token for token, idx in answer_vocab.items()}
    
    print(f"Answer vocabulary size: {len(answer_vocab)}")
    
    # Build question vocabulary (character-level for robustness)
    question_chars = set()
    for q in questions:
        question_chars.update(q.lower())
    
    question_vocab = {c: i+1 for i, c in enumerate(sorted(question_chars))}  # 0 reserved for PAD
    question_vocab['<PAD>'] = 0
    question_vocab['<UNK>'] = len(question_vocab)
    
    print(f"Question vocabulary size: {len(question_vocab)}")
    
    return questions, answers, answer_vocab, idx_to_token, question_vocab

class GuideDatasetTRM(Dataset):
    """
    PyTorch Dataset for TRM training.
    Fixed-length outputs work best with TRM architecture.
    """
    
    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        question_vocab: Dict[str, int],
        answer_vocab: Dict[str, int],
        max_len: int = MAXLEN,
        max_answer_len: int = MAX_ANSWER_LEN
    ):
        self.questions = questions
        self.answers = answers
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.max_len = max_len
        self.max_answer_len = max_answer_len
        
        # Pre-encode all data
        self.encoded_questions = self._encode_questions()
        self.encoded_answers = self._encode_answers()
    
    def _encode_questions(self) -> torch.Tensor:
        """Encode questions as character indices."""
        encoded = []
        for question in self.questions:
            chars = question.lower()[:self.max_len]
            indices = [self.question_vocab.get(c, self.question_vocab['<UNK>']) for c in chars]
            # Pad
            if len(indices) < self.max_len:
                indices.extend([0] * (self.max_len - len(indices)))
            encoded.append(indices[:self.max_len])
        return torch.tensor(encoded, dtype=torch.long)
    
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
            'question_ids': self.encoded_questions[idx],
            'labels': self.encoded_answers[idx],
            'question_len': (self.encoded_questions[idx] != 0).sum().item()
        }

class TinyRecursiveBlock(nn.Module):
    """
    The core recursive network - only 2 layers!
    Replaces BERT+LSTM with iterative latent refinement.
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input projection: concatenates [x, y, z] -> hidden
        # x: question embedding, y: answer embedding, z: reasoning state
        self.input_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Tiny Transformer (2 layers only!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.reasoning_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.z_head = nn.Linear(hidden_dim, hidden_dim)  # Update reasoning state
        self.y_head = nn.Linear(hidden_dim * 2, hidden_dim)  # Update answer (takes y and z)
        self.halt_head = nn.Linear(hidden_dim, 1)  # Predict if answer is correct
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward pass of the recursive block.
        
        Args:
            x: Question embedding [batch, hidden_dim]
            y: Current answer state [batch, hidden_dim]  
            z: Current reasoning state [batch, hidden_dim]
            
        Returns:
            Tuple of (new_y, new_z, halt_logit)
        """
        # Concatenate x, y, z as input
        combined = torch.cat([x, y, z], dim=-1)  # [batch, hidden*3]
        
        # Project to hidden dimension
        h = self.input_proj(combined)  # [batch, hidden]
        h = self.norm(h)
        
        # Pass through transformer layers (expects [batch, seq, hidden])
        h = h.unsqueeze(1)  # [batch, 1, hidden]
        h = self.reasoning_layers(h)  # [batch, 1, hidden]
        h = h.squeeze(1)  # [batch, hidden]
        
        # Update reasoning state z
        new_z = self.z_head(h)  # [batch, hidden]
        
        # Update answer state y (uses both old y and new z)
        y_input = torch.cat([y, new_z], dim=-1)  # [batch, hidden*2]
        new_y = self.y_head(y_input)  # [batch, hidden]
        
        # Predict whether we should halt
        halt_logit = self.halt_head(new_z)  # [batch, 1]
        
        return new_y, new_z, halt_logit

class BigThoughtTRM(nn.Module):
    """
    BigThought converted to Tiny Recursive Model.
    
    Architecture:
    1. Simple embedding layer (replaces BERT - much smaller!)
    2. Initial answer embedding y_init (learnable)
    3. Initial reasoning state z_init (learnable)
    4. TinyRecursiveBlock (2 layers) for iterative refinement
    5. Sequence decoder: projects refined latent to full answer sequence
    """
    
    def __init__(
        self,
        vocab_size: int,
        question_vocab_size: int,
        hidden_dim: int = HIDDEN_DIM,
        max_answer_len: int = MAX_ANSWER_LEN,
        n_recursions: int = N_RECURSIONS,
        max_supervision: int = MAX_SUPERVISION_STEPS
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_answer_len = max_answer_len
        self.n_recursions = n_recursions
        self.max_supervision = max_supervision
        
        # Simple embeddings (no BERT - tiny model!)
        self.question_embed = nn.Embedding(question_vocab_size, hidden_dim)
        
        # Answer embedding (for iterative refinement)
        self.answer_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Initial states (learnable parameters - critical for TRM)
        self.y_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
        # The tiny recursive network (2 layers only!)
        self.recursive_net = TinyRecursiveBlock(hidden_dim, num_layers=NUM_LAYERS)
        
        # NEW: Sequence decoder - projects refined latent to full sequence
        # This allows parallel prediction of all answer tokens
        self.sequence_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_answer_len * hidden_dim)
        )
        
        # Output projection to vocabulary (applied per position)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def latent_recursion(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        z: torch.Tensor,
        n: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single latent recursion step: updates z n times, then updates y once.
        Corresponds to Algorithm 3 in the TRM paper.
        """
        if n is None:
            n = self.n_recursions
            
        # Update reasoning state z recursively (n times)
        for _ in range(n):
            y_new, z, _ = self.recursive_net(x, y, z)
            y = y_new
            
        return y, z
    
    def forward(
        self, 
        question_ids: torch.Tensor,
        num_supervision_steps: Optional[int] = None,
        return_all_steps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with deep supervision (Algorithm 3 from paper).
        
        Args:
            question_ids: [batch, seq_len] token IDs for question
            num_supervision_steps: How many refinement steps (T)
            return_all_steps: If True, return intermediate predictions
        
        Returns:
            Dictionary with logits, halt_logits, and intermediate outputs
        """
        if num_supervision_steps is None:
            num_supervision_steps = self.max_supervision
            
        batch_size = question_ids.size(0)
        device = question_ids.device
        
        # Embed question (simple mean pooling)
        q_embed = self.question_embed(question_ids)  # [batch, seq, hidden]
        x = q_embed.mean(dim=1)  # [batch, hidden] - simple pooling
        
        # Initialize y and z (learnable initial states)
        y = self.y_init.expand(batch_size, -1)  # [batch, hidden]
        z = self.z_init.expand(batch_size, -1)  # [batch, hidden]
        
        all_logits = []
        all_halt_logits = []
        
        # Deep Supervision Loop (Algorithm 3 from TRM paper)
        for step in range(num_supervision_steps):
            
            # Steps 0 to T-2: No gradients (improve initialization)
            if step < num_supervision_steps - 1:
                with torch.no_grad():
                    y, z = self.latent_recursion(x, y, z, self.n_recursions)
                    # Detach to prevent backprop through time
                    y = y.detach()
                    z = z.detach()
                    
                    if return_all_steps:
                        # Get predictions for logging (no gradients)
                        logits_step = self.output_proj(y)
                        halt_logit = self.recursive_net.halt_head(z)
                        all_logits.append(logits_step)
                        all_halt_logits.append(halt_logit)
            else:
                # Last step: With gradients (the only graded step)
                y, z = self.latent_recursion(x, y, z, self.n_recursions)
                
                # Final outputs
                logits = self.output_proj(y)  # [batch, vocab_size] for each position
                
                # For sequence generation, we need per-position predictions
                # Expand y to sequence length and predict each token
                # Actually, let's use y to initialize a sequence decoder
                # But for TRM, we typically predict the whole sequence in parallel after refinement
                
                # For simplicity in this adaptation: we'll predict token-by-token
                # but with TRM refinement at each step
                # Better approach: use y as initial state for parallel prediction
                
                halt_logit = self.recursive_net.halt_head(z)
                
                if return_all_steps:
                    all_logits.append(logits)
                    all_halt_logits.append(halt_logit)
        
        # For sequence prediction, we need to expand to max_answer_len
        # Strategy: Use refined y as input to a shallow decoder, or 
        # iteratively predict each token with TRM refinement
        
        # Here: Simple expansion for parallel prediction (TRM style)
        # [batch, hidden] -> [batch, max_answer_len, vocab_size]
        # We'll use the refined representation to predict all positions
        
        # Actually, let's do it properly: autoregressive but with TRM refinement
        # For now, return the refined state for decoding
        
        result = {
            'logits': logits,  # [batch, vocab_size] - next token prediction
            'halt_logits': halt_logit if not return_all_steps else torch.stack(all_halt_logits, dim=1),
            'final_y': y,
            'final_z': z
        }
        
        if return_all_steps:
            result['all_logits'] = torch.stack(all_logits, dim=1)  # [batch, steps, vocab]
            
        return result
    
    def generate_sequence(
        self,
        question_ids: torch.Tensor,
        max_length: int = MAX_ANSWER_LEN,
        temperature: float = 1.0,
        num_refinement_steps: int = MAX_SUPERVISION_STEPS,
        start_token_id: int = 1,
        end_token_id: int = 2
    ) -> torch.Tensor:
        """
        Generate answer sequence using iterative refinement.
        Uses autoregressive generation with TRM refinement at each step.
        
        Args:
            start_token_id: ID of the START token
            end_token_id: ID of the END token
        """
        self.eval()
        batch_size = question_ids.size(0)
        device = question_ids.device
        
        with torch.no_grad():
            # Embed question
            q_embed = self.question_embed(question_ids)
            x = q_embed.mean(dim=1)
            
            # Initialize y with START token embedding (not the learned y_init)
            # This is crucial: y_init is for training initialization, not inference start
            y = self.answer_embed(
                torch.tensor([start_token_id] * batch_size, device=device)
            )
            
            # Initialize z with learned initial state
            z = self.z_init.expand(batch_size, -1)
            
            # Pre-refine z with the question context (but keep y as START token)
            # This helps the model "understand" the question before generating
            for _ in range(num_refinement_steps):
                _, z, _ = self.recursive_net(x, y, z)
            
            generated_tokens = []
            
            for pos in range(max_length):
                # Refine y and z together for this position
                for _ in range(num_refinement_steps):
                    y, z = self.latent_recursion(x, y, z, self.n_recursions)
                
                # Predict next token
                logits = self.output_proj(y)  # [batch, vocab_size]
                
                # Apply temperature
                if temperature < 0.01:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                generated_tokens.append(next_token)
                
                # Check for END token to stop early (for all in batch)
                if (next_token == end_token_id).all():
                    break
                
                # Update y to embed the generated token for next iteration
                y = self.answer_embed(next_token)
            
            # Stack tokens into sequence
            if len(generated_tokens) > 0:
                generated_tokens = torch.stack(generated_tokens, dim=1)
            else:
                generated_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        
        return generated_tokens

class TRMLoss(nn.Module):
    """
    Combined loss for TRM:
    1. Cross-entropy on final output
    2. Binary cross-entropy on halting predictions (is answer correct?)
    3. Deep supervision losses on intermediate steps (optional)
    """
    def __init__(self, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        target_correct: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            outputs: Dict from model forward pass
            targets: Target token IDs [batch, seq_len]
            target_correct: Binary tensor indicating if answer is correct
        """
        # Main prediction loss (first token for simplicity, or expand to sequence)
        logits = outputs['logits']  # [batch, vocab_size]
        
        # For sequence, we need to handle it properly
        # Here: assume targets is [batch] for next token prediction
        if targets.dim() == 1:
            loss = self.ce_loss(logits, targets)
        else:
            # Sequence mode - predict first token
            loss = self.ce_loss(logits, targets[:, 0])
        
        # Halting loss (if we have targets)
        if target_correct is not None and 'halt_logits' in outputs:
            halt_logits = outputs['halt_logits'].squeeze(-1)  # [batch, num_steps]
            num_steps = halt_logits.size(1)
            # Expand target_correct to match number of reasoning steps
            # Target: 1 if should halt (correct), 0 if should continue
            halt_targets = target_correct.float().unsqueeze(1).expand(-1, num_steps)
            halt_loss = self.bce_loss(halt_logits, halt_targets)
            loss = loss + 0.5 * halt_loss  # Weighted combination
        
        # Deep supervision: apply loss at all supervision steps (optional)
        if 'all_logits' in outputs:
            all_logits = outputs['all_logits']  # [batch, steps, vocab]
            steps = all_logits.size(1)
            
            # Weight later steps more heavily (they should be better)
            for t in range(steps):
                if targets.dim() == 1:
                    step_loss = self.ce_loss(all_logits[:, t, :], targets)
                else:
                    step_loss = self.ce_loss(all_logits[:, t, :], targets[:, 0])
                loss = loss + (0.1 * step_loss * (t + 1) / steps)  # Increasing weight
        
        return loss

def train_epoch_trm(
    model: BigThoughtTRM,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: TRMLoss,
    device: torch.device,
    ema_model: Optional[AveragedModel] = None
) -> float:
    """
    Training with Deep Supervision and EMA.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        question_ids = batch['question_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward with deep supervision
        outputs = model(
            question_ids,
            num_supervision_steps=MAX_SUPERVISION_STEPS,
            return_all_steps=True
        )
        
        # Determine if answer is correct (for halting loss)
        # Simplified: check if first token matches
        with torch.no_grad():
            preds = torch.argmax(outputs['logits'], dim=-1)
            target_correct = (preds == labels[:, 0]).float()
        
        loss = criterion(outputs, labels[:, 0], target_correct)
        
        loss.backward()
        
        # Gradient clipping (important for TRM stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update EMA
        if ema_model is not None:
            ema_model.update_parameters(model)
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate_trm(
    model: BigThoughtTRM,
    dataloader: DataLoader,
    criterion: TRMLoss,
    device: torch.device
) -> float:
    """
    Validation with EMA model.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            question_ids = batch['question_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                question_ids,
                num_supervision_steps=MAX_SUPERVISION_STEPS,
                return_all_steps=False
            )
            
            loss = criterion(outputs, labels[:, 0])
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_answer_trm(
    model: BigThoughtTRM,
    question: str,
    question_vocab: Dict[str, int],
    answer_vocab: Dict[str, int],
    idx_to_token: Dict[int, str],
    device: torch.device,
    max_length: int = MAX_ANSWER_LEN,
    num_refinement_steps: int = MAX_SUPERVISION_STEPS
) -> str:
    """
    Generate answer using TRM iterative refinement.
    """
    model.eval()
    
    # Encode question
    chars = question.lower()[:MAXLEN]
    indices = [question_vocab.get(c, question_vocab['<UNK>']) for c in chars]
    if len(indices) < MAXLEN:
        indices.extend([0] * (MAXLEN - len(indices)))
    question_ids = torch.tensor([indices[:MAXLEN]], dtype=torch.long, device=device)
    
    # Get token IDs
    start_token_id = answer_vocab.get('<START>', 1)
    end_token_id = answer_vocab.get('<END>', 2)
    
    with torch.no_grad():
        # Use the model's sequence generation
        tokens = model.generate_sequence(
            question_ids,
            max_length=max_length,
            num_refinement_steps=num_refinement_steps,
            start_token_id=start_token_id,
            end_token_id=end_token_id
        )
        
        # Convert indices to tokens
        result_tokens = []
        for idx in tokens[0].cpu().numpy():
            token = idx_to_token.get(int(idx), '<UNK>')
            if token == '<END>':
                break
            if token not in ['<PAD>', '<START>', '<UNK>']:
                result_tokens.append(token)
    
    # Return empty string indicator if no valid tokens
    if not result_tokens:
        # Debug: show what was actually generated
        raw_tokens = [idx_to_token.get(int(i), '<UNK>') for i in tokens[0].cpu().numpy()]
        print(f"  [Debug: raw tokens: {raw_tokens[:10]}]")
        
    return ' '.join(result_tokens) if result_tokens else "(no valid answer generated)"

def main():
    """
    Main entry point for BigThought-TRM training.
    """
    print("=" * 60)
    print("🚀 BigThought-TRM: Tiny Recursive Model Edition")
    print("   Finding the Answer to Life, the Universe, and Everything")
    print("   Architecture: 2-layer recursive network with deep supervision")
    print("=" * 60)
    
    # Prepare data
    questions, answers, answer_vocab, idx_to_token, question_vocab = prepare_data()
    
    # Create dataset
    dataset = GuideDatasetTRM(
        questions=questions,
        answers=answers,
        question_vocab=question_vocab,
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
    
    # Initialize TRM model (tiny!)
    vocab_size = len(answer_vocab)
    model = BigThoughtTRM(
        vocab_size=vocab_size,
        question_vocab_size=len(question_vocab),
        hidden_dim=HIDDEN_DIM,
        max_answer_len=MAX_ANSWER_LEN,
        n_recursions=N_RECURSIONS,
        max_supervision=MAX_SUPERVISION_STEPS
    ).to(DEVICE)
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n✨ Model initialized: {param_count:.2f}M parameters")
    print(f"   (vs ~180M for BERT-base + LSTM)")
    print(f"   Hidden dim: {HIDDEN_DIM}, Layers: {NUM_LAYERS}")
    print(f"   Recursions: {N_RECURSIONS}, Supervision steps: {MAX_SUPERVISION_STEPS}")
    
    # Loss and optimizer
    criterion = TRMLoss(pad_idx=answer_vocab['<PAD>'])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # EMA (Exponential Moving Average) - critical for small datasets
    ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY))
    
    # Training state
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / "bigthought_trm.pt"
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    if FIT:
        print("\n" + "=" * 60)
        print("🔬 Training with Deep Supervision")
        print("=" * 60)
        
        for epoch in range(N_EPOCHS):
            train_loss = train_epoch_trm(model, train_loader, optimizer, criterion, DEVICE, ema_model)
            
            # Validate with EMA model (more stable)
            val_loss = validate_trm(ema_model.module, val_loader, criterion, DEVICE)
            
            print(f"Epoch {epoch+1}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save both regular and EMA model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'answer_vocab': answer_vocab,
                    'idx_to_token': idx_to_token,
                    'question_vocab': question_vocab,
                    'config': {
                        'hidden_dim': HIDDEN_DIM,
                        'n_recursions': N_RECURSIONS,
                        'max_supervision': MAX_SUPERVISION_STEPS
                    }
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
    print("💬 BigThought-TRM Ready - Ask me anything!")
    print("   (Type 'quit' to exit)")
    print("=" * 60)
    
    # Load best model if exists
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        answer_vocab = checkpoint['answer_vocab']
        idx_to_token = checkpoint['idx_to_token']
        question_vocab = checkpoint['question_vocab']
        
        # Load EMA weights for inference (more stable)
        if 'ema_state_dict' in checkpoint:
            ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY))
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            print("Loaded EMA weights for stable inference")
        
        print(f"Loaded model from {model_path}")
    
    while True:
        try:
            question = input("\n🧠 Question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Don't Panic!")
                break
            if not question:
                continue
            
            answer = generate_answer_trm(
                model, question, question_vocab, answer_vocab, idx_to_token, DEVICE
            )
            print(f"📖 Answer: {answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Don't Panic!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
