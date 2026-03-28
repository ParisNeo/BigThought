"""
BigThought-TRM Inference CLI
Author      : Saifeddine ALOUI
Description : Standalone CLI for querying a trained BigThought-TRM model.
              No training code - just load and inference.

Usage:
    python bigthought_trm_cli.py [--model PATH] [--temperature FLOAT]

Requirements:
    pip install torch numpy
"""

import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import the model architecture from the training module
# We need to redefine or import the classes since we need them for loading
import importlib.util


# ============================================
# 🧠 MODEL ARCHITECTURE (Required for loading)
# ============================================

class TinyRecursiveBlock(nn.Module):
    """The core recursive network - 2 layer transformer with iterative refinement."""
    
    def __init__(self, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.reasoning_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.z_head = nn.Linear(hidden_dim, hidden_dim)
        self.y_head = nn.Linear(hidden_dim * 2, hidden_dim)
        self.halt_head = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, y, z):
        batch_size = x.size(0)
        combined = torch.cat([x, y, z], dim=-1)
        h = self.input_proj(combined)
        h = self.norm(h)
        h = h.unsqueeze(1)
        h = self.reasoning_layers(h)
        h = h.squeeze(1)
        new_z = self.z_head(h)
        y_input = torch.cat([y, new_z], dim=-1)
        new_y = self.y_head(y_input)
        halt_logit = self.halt_head(new_z)
        return new_y, new_z, halt_logit


class BigThoughtTRM(nn.Module):
    """BigThought Tiny Recursive Model - lightweight inference architecture."""
    
    def __init__(
        self,
        vocab_size: int,
        question_vocab_size: int,
        hidden_dim: int = 512,
        max_answer_len: int = 50,
        n_recursions: int = 6,
        max_supervision: int = 3
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_answer_len = max_answer_len
        self.n_recursions = n_recursions
        self.max_supervision = max_supervision
        
        # Simple embeddings (no BERT - tiny model!)
        self.question_embed = nn.Embedding(question_vocab_size, hidden_dim)
        self.answer_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Learnable initial states
        self.y_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
        # The tiny recursive network
        self.recursive_net = TinyRecursiveBlock(hidden_dim, num_layers=2)
        
        # Sequence decoder: projects refined latent to full answer sequence
        # This allows parallel prediction of all answer tokens
        self.sequence_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_answer_len * hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def latent_recursion(self, x, y, z, n=None):
        """Single latent recursion step: updates z n times, then updates y once."""
        if n is None:
            n = self.n_recursions
            
        for _ in range(n):
            y_new, z, _ = self.recursive_net(x, y, z)
            y = y_new
            
        return y, z
    
    def forward(self, question_ids, num_supervision_steps=None, return_all_steps=False):
        """Forward pass with deep supervision."""
        if num_supervision_steps is None:
            num_supervision_steps = self.max_supervision
            
        batch_size = question_ids.size(0)
        device = question_ids.device
        
        # Embed and pool question
        q_embed = self.question_embed(question_ids)
        x = q_embed.mean(dim=1)
        
        # Initialize states
        y = self.y_init.expand(batch_size, -1)
        z = self.z_init.expand(batch_size, -1)
        
        all_logits = []
        all_halt_logits = []
        
        # Deep supervision loop
        for step in range(num_supervision_steps):
            if step < num_supervision_steps - 1:
                with torch.no_grad():
                    y, z = self.latent_recursion(x, y, z, self.n_recursions)
                    y = y.detach()
                    z = z.detach()
                    
                    if return_all_steps:
                        # Project to sequence and get predictions for logging
                        seq_hidden = self.sequence_proj(y)
                        seq_hidden = seq_hidden.view(batch_size, self.max_answer_len, self.hidden_dim)
                        logits_step = self.output_proj(seq_hidden)  # [batch, max_answer_len, vocab_size]
                        halt_logit = self.recursive_net.halt_head(z)
                        all_logits.append(logits_step)
                        all_halt_logits.append(halt_logit)
            else:
                y, z = self.latent_recursion(x, y, z, self.n_recursions)
                
                # Project refined latent y to full sequence representation
                # [batch, hidden] -> [batch, max_answer_len * hidden] -> [batch, max_answer_len, hidden]
                seq_hidden = self.sequence_proj(y)
                seq_hidden = seq_hidden.view(batch_size, self.max_answer_len, self.hidden_dim)
                
                # Output projection per position: [batch, max_answer_len, vocab_size]
                logits = self.output_proj(seq_hidden)
                
                halt_logit = self.recursive_net.halt_head(z)
                
                if return_all_steps:
                    all_logits.append(logits)
                    all_halt_logits.append(halt_logit)
        
        result = {
            'logits': logits,
            'halt_logits': halt_logit if not return_all_steps else torch.stack(all_halt_logits, dim=1),
            'final_y': y,
            'final_z': z
        }
        
        if return_all_steps:
            result['all_logits'] = torch.stack(all_logits, dim=1)
            
        return result
    
    def generate_sequence(self, question_ids, max_length=50, temperature=1.0, num_refinement_steps=3, start_token_id=1, end_token_id=2):
        """Generate answer sequence using iterative refinement."""
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


# ============================================
# 🔧 INFERENCE ENGINE
# ============================================

class BigThoughtTRMInference:
    """
    High-level inference interface for BigThought-TRM.
    
    Handles:
    - Model loading from checkpoint
    - Question encoding
    - Answer generation with configurable parameters
    - Token decoding
    """
    
    def __init__(
        self,
        model_path: Path,
        device: Optional[torch.device] = None,
        use_ema: bool = True
    ):
        """
        Initialize inference engine with a trained model.
        
        Args:
            model_path: Path to the checkpoint file (.pt)
            device: torch device (auto-detected if None)
            use_ema: Whether to use EMA weights if available
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.use_ema = use_ema
        
        # Load checkpoint
        self.checkpoint = self._load_checkpoint()
        self.config = self.checkpoint.get('config', {})
        
        # Extract vocabularies
        self.answer_vocab = self.checkpoint['answer_vocab']
        self.idx_to_token = self.checkpoint['idx_to_token']
        self.question_vocab = self.checkpoint['question_vocab']
        
        # Build reverse mappings
        self.token_to_idx = {v: k for k, v in self.idx_to_token.items()}
        
        # Initialize model
        self.model = self._build_model()
        self._load_weights()
        
        self.model.eval()
        print(f"✅ Model loaded: {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        
    def _load_checkpoint(self) -> Dict:
        """Load model checkpoint from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"📦 Loading checkpoint: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        return checkpoint
    
    def _build_model(self) -> BigThoughtTRM:
        """Construct model architecture from config."""
        vocab_size = len(self.answer_vocab)
        question_vocab_size = len(self.question_vocab)
        
        # Get config with defaults
        hidden_dim = self.config.get('hidden_dim', 512)
        n_recursions = self.config.get('n_recursions', 6)
        max_supervision = self.config.get('max_supervision', 3)
        max_answer_len = self.config.get('max_answer_len', 50)
        
        model = BigThoughtTRM(
            vocab_size=vocab_size,
            question_vocab_size=question_vocab_size,
            hidden_dim=hidden_dim,
            max_answer_len=max_answer_len,
            n_recursions=n_recursions,
            max_supervision=max_supervision
        )
        
        # Store config in model for reference
        model.config = self.config
        
        return model.to(self.device)
    
    def _load_weights(self):
        """Load trained weights into model."""
        if self.use_ema and 'ema_state_dict' in self.checkpoint:
            # Load EMA weights for more stable inference
            print("   Using EMA weights (recommended for inference)")
            from torch.optim.swa_utils import AveragedModel
            
            ema_model = AveragedModel(self.model)
            ema_model.load_state_dict(self.checkpoint['ema_state_dict'])
            # Copy EMA weights to main model
            self.model.load_state_dict(ema_model.module.state_dict())
        else:
            # Load regular weights
            print("   Using regular weights")
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
    
    def encode_question(self, question: str, max_len: int = 128) -> torch.Tensor:
        """
        Encode a question string into token indices.
        
        Uses character-level encoding for robustness with the TRM.
        """
        chars = question.lower()[:max_len]
        indices = [
            self.question_vocab.get(c, self.question_vocab.get('<UNK>', 0))
            for c in chars
        ]
        
        # Pad to fixed length
        if len(indices) < max_len:
            indices.extend([0] * (max_len - len(indices)))
        
        return torch.tensor([indices[:max_len]], dtype=torch.long, device=self.device)
    
    def decode_answer(self, token_indices: torch.Tensor) -> str:
        """Decode token indices into a human-readable answer string."""
        tokens = []
        for idx in token_indices[0].cpu().numpy():
            token = self.idx_to_token.get(int(idx), '<UNK>')
            if token == '<END>':
                break
            if token not in ['<PAD>', '<START>', '<UNK>']:
                tokens.append(token)
        
        if not tokens:
            # Debug: show what was actually generated
            raw = [self.idx_to_token.get(int(i), '?') for i in token_indices[0].cpu().numpy()[:10]]
            return f"(empty - raw: {raw})"
        
        return ' '.join(tokens)
    
    def ask(
        self,
        question: str,
        max_length: int = 50,
        temperature: float = 1.0,
        num_refinement_steps: Optional[int] = None,
        verbose: bool = False
    ) -> str:
        """
        Ask BigThought-TRM a question and get an answer.
        
        Args:
            question: The question string
            max_length: Maximum answer length in tokens
            temperature: Sampling temperature (1.0 = deterministic, higher = more random)
            num_refinement_steps: Number of TRM refinement iterations (default from config)
            verbose: Print detailed timing/info
            
        Returns:
            The generated answer string
        """
        if num_refinement_steps is None:
            num_refinement_steps = self.config.get('max_supervision', 3)
        
        import time
        start_time = time.time() if verbose else None
        
        # Encode question
        question_ids = self.encode_question(question)
        
        if verbose:
            encode_time = time.time()
            print(f"⏱️  Encoding: {encode_time - start_time:.3f}s")
        
        # Get token IDs
        start_token_id = self.answer_vocab.get('<START>', 1)
        end_token_id = self.answer_vocab.get('<END>', 2)
        
        # Generate answer
        with torch.no_grad():
            tokens = self.model.generate_sequence(
                question_ids,
                max_length=max_length,
                temperature=temperature,
                num_refinement_steps=num_refinement_steps,
                start_token_id=start_token_id,
                end_token_id=end_token_id
            )
        
        if verbose:
            gen_time = time.time()
            print(f"⏱️  Generation: {gen_time - encode_time:.3f}s")
        
        # Decode
        answer = self.decode_answer(tokens)
        
        if verbose:
            total_time = time.time() - start_time
            print(f"⏱️  Total: {total_time:.3f}s")
            print(f"   Tokens generated: {tokens.shape[1]}")
            # Show raw tokens for debugging
            raw = [self.idx_to_token.get(int(i), '?') for i in tokens[0].cpu().numpy()[:10]]
            print(f"   Raw tokens: {raw}")
        
        return answer
    
    def batch_ask(
        self,
        questions: List[str],
        max_length: int = 50,
        temperature: float = 1.0,
        num_refinement_steps: Optional[int] = None
    ) -> List[str]:
        """
        Process multiple questions in a single batch for efficiency.
        
        Note: Currently processes sequentially - true batching would require padding.
        """
        return [
            self.ask(q, max_length, temperature, num_refinement_steps)
            for q in questions
        ]


# ============================================
# 💻 INTERACTIVE CLI
# ============================================

def create_banner():
    """Create a nice ASCII banner."""
    return r"""
    ____  _               _   _                 _______ _____ __  __ 
   |  _ \(_) __ _ _ __ __| | | |_ ___     _    |__   __|_   _|  \/  |
   | |_) | |/ _` | '__/ _` | | __/ _ \  _| |_     | |    | | | \  / |
   |  _ <| | (_| | | | (_| | | || (_) | |_   _|    | |    | | | |\/| |
   |_| \_\_|\__, |_|  \__,_|  \__\___/    |_|      | |   _| |_| |  | |
            |___/                                  |_|  |_____|_|  |_|
    
    🚀 Tiny Recursive Model Edition
       "Don't Panic" - The Hitchhiker's Guide to the Galaxy
    """


def interactive_mode(inference: BigThoughtTRMInference, args):
    """Run interactive question-answering session."""
    print(create_banner())
    print(f"\nModel: {inference.model_path.name}")
    print(f"Device: {inference.device}")
    print(f"Temperature: {args.temperature}")
    print(f"Refinement steps: {args.refinement_steps or 'auto'}")
    print(f"\nType your questions below (commands: /help, /quit, /temp FLOAT, /steps INT)")
    print("─" * 60)
    
    current_temp = args.temperature
    current_steps = args.refinement_steps
    
    while True:
        try:
            # Get user input
            user_input = input("\n🧠 Question: ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd in ['/quit', '/q', '/exit']:
                    print("👋 Don't Panic!")
                    break
                    
                elif cmd == '/help':
                    print("""
Available commands:
  /quit, /q, /exit    - Exit the program
  /help              - Show this help message
  /temp FLOAT        - Set temperature (0.1-5.0, default 1.0)
  /steps INT         - Set refinement steps
  /info              - Show model info
  /verbose           - Toggle verbose mode
                    """)
                
                elif cmd == '/temp' and len(parts) > 1:
                    try:
                        current_temp = float(parts[1])
                        print(f"   Temperature set to {current_temp}")
                    except ValueError:
                        print("   Invalid temperature value")
                
                elif cmd == '/steps' and len(parts) > 1:
                    try:
                        current_steps = int(parts[1])
                        print(f"   Refinement steps set to {current_steps}")
                    except ValueError:
                        print("   Invalid steps value")
                
                elif cmd == '/info':
                    print(f"\nModel Information:")
                    print(f"  Path: {inference.model_path}")
                    print(f"  Parameters: {sum(p.numel() for p in inference.model.parameters())/1e6:.2f}M")
                    print(f"  Hidden dim: {inference.config.get('hidden_dim', 512)}")
                    print(f"  Recursions: {inference.config.get('n_recursions', 6)}")
                    print(f"  Vocab size: {len(inference.answer_vocab)}")
                
                elif cmd == '/verbose':
                    args.verbose = not args.verbose
                    print(f"   Verbose mode: {'ON' if args.verbose else 'OFF'}")
                
                else:
                    print(f"   Unknown command: {cmd}")
                
                continue
            
            # Generate answer
            answer = inference.ask(
                user_input,
                temperature=current_temp,
                num_refinement_steps=current_steps,
                verbose=args.verbose
            )
            print(f"📖 Answer: {answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Don't Panic!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")


def single_query_mode(inference: BigThoughtTRMInference, args):
    """Process a single question and exit."""
    question = " ".join(args.question)
    
    answer = inference.ask(
        question,
        temperature=args.temperature,
        num_refinement_steps=args.refinement_steps,
        verbose=args.verbose
    )
    print(answer)


def main():
    """
    Main entry point for BigThought-TRM CLI.
    
    Two modes:
    1. Interactive mode (default): Chat with the model
    2. Single query mode: Ask one question and exit
    """
    parser = argparse.ArgumentParser(
        description="BigThought-TRM Inference CLI - Ask the Tiny Recursive Model anything!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python bigthought_trm_cli.py
  
  # Single question
  python bigthought_trm_cli.py "What is the answer to life?"
  
  # Custom model path
  python bigthought_trm_cli.py --model path/to/model.pt "Why 42?"
  
  # Higher temperature (more creative)
  python bigthought_trm_cli.py --temperature 1.5 "How to survive in space?"
        """
    )
    
    parser.add_argument(
        'question',
        nargs='*',
        help='Question to ask (optional - launches interactive mode if omitted)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=Path,
        default=Path('model/bigthought_trm.pt'),
        help='Path to the trained model checkpoint (default: model/bigthought_trm.pt)'
    )
    
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0, lower=more deterministic)'
    )
    
    parser.add_argument(
        '--refinement-steps', '-r',
        type=int,
        default=None,
        help='Number of TRM refinement steps (default: auto from config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--no-ema',
        action='store_true',
        help='Use regular weights instead of EMA weights'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with timing information'
    )
    
    args = parser.parse_args()
    
    # Validate model exists
    if not args.model.exists():
        print(f"❌ Error: Model not found at {args.model}")
        print(f"   Please train a model first or specify a valid path.")
        print(f"   Train with: python big_thought_trm.py")
        sys.exit(1)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize inference engine
        inference = BigThoughtTRMInference(
            model_path=args.model,
            device=device,
            use_ema=not args.no_ema
        )
        
        # Run appropriate mode
        if args.question:
            single_query_mode(inference, args)
        else:
            interactive_mode(inference, args)
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    