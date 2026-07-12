"""
Pullbot Bot
Combines model + knowledge store + self-improvement.
This is the main entry point for responding to queries.
"""

import yaml
import json
import os
import time
import random

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class Pullbot:
    def __init__(self):
        print("\n" + "="*50)
        print("🤖 PULLBOT INITIALIZING")
        print("="*50 + "\n")
        
        # Lazy imports to avoid loading everything at once
        from model import PullbotModel
        from store import KnowledgeStore
        
        self.model = PullbotModel()
        self.store = KnowledgeStore()
        
        print(f"\n✅ Pullbot ready!")
        print(f"   Model: {config['model']['base']}")
        print(f"   Knowledge chunks: {len(self.store.chunks)}")
        print(f"   Self-improvement: {'ON' if config['evolve']['enabled'] else 'OFF'}")
    
    def respond(self, question):
        """Main response pipeline"""
        print(f"\n❓ Question: {question}")
        
        # 1. Search knowledge base
        context = self.store.get_context(question)
        
        if context:
            print(f"   📚 Retrieved context ({len(context)} chars)")
        else:
            print(f"   📚 No relevant context found")
        
        # 2. Generate response
        response = self.model.generate(
            prompt=question,
            context=context
        )
        
        print(f"   🤖 Response: {response[:100]}...")
        
        return {
            'question': question,
            'response': response,
            'context_used': context is not None,
            'timestamp': time.time()
        }
    
    def self_improve(self, prompts=None):
        """Generate variants and curate best responses"""
        if not config['evolve']['enabled']:
            print("🔒 Self-improvement disabled in config")
            return
        
        print("\n🧬 SELF-IMPROVEMENT CYCLE")
        print("-" * 30)
        
        # Default prompts if none provided
        if prompts is None:
            prompts = [
                "Explain what a function is in programming",
                "What is machine learning?",
                "How does the internet work?",
                "Write a short Python code example",
                "Explain climate change simply"
            ]
        
        all_candidates = []
        n_variants = config['evolve']['variants_per_prompt']
        
        for prompt in prompts:
            print(f"\n📝 Generating {n_variants} variants for: {prompt[:50]}...")
            
            for _ in range(n_variants):
                # Vary temperature for diversity
                temp = random.uniform(0.6, 1.0)
                self.model.temperature = temp
                
                response = self.model.generate(prompt)
                
                score = self._score_response(response)
                
                all_candidates.append({
                    'prompt': prompt,
                    'response': response,
                    'score': score,
                    'temperature': temp
                })
        
        # Sort by score and keep elite
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        keep_n = max(1, len(all_candidates) * config['evolve']['keep_top_percent'] // 100)
        elite = all_candidates[:keep_n]
        
        print(f"\n🏆 Kept {keep_n}/{len(all_candidates)} elite responses")
        print(f"   Best score: {elite[0]['score']:.3f}")
        print(f"   Worst kept: {elite[-1]['score']:.3f}")
        
        # Save curated examples for retraining
        os.makedirs('data/curated', exist_ok=True)
        curated_path = f"data/curated/evolved_{int(time.time())}.json"
        
        with open(curated_path, 'w') as f:
            json.dump(elite, f, indent=2)
        
        # Add to training corpus
        with open('data/processed/corpus.txt', 'a') as f:
            for item in elite:
                f.write(f"\n\nPrompt: {item['prompt']}\nResponse: {item['response']}\n")
        
        print(f"💾 Added {keep_n} curated examples to training data")
        
        return elite
    
    def _score_response(self, response):
        """Score a response based on multiple heuristics"""
        score = 0.0
        
        # Length score (prefer responses 20-200 chars)
        length = len(response)
        if length < config['evolve']['min_response_length']:
            return 0.0  # Reject short garbage
        elif 50 <= length <= 300:
            score += 0.3
        elif length > 300:
            score += 0.1  # Slight penalty for too long
        
        # Contains actual content (not just punctuation)
        words = response.split()
        unique_words = set(words)
        if len(unique_words) > 5:
            score += 0.2
        
        # Doesn't repeat itself
        if len(words) > 0:
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio > 0.5:
                score += 0.2
        
        # Ends with proper punctuation
        if response.rstrip()[-1:] in '.!?':
            score += 0.1
        
        # Contains code-like patterns (bonus for coding ability)
        if 'def ' in response or 'import ' in response or 'class ' in response:
            score += 0.2
        
        return score
    
    def chat_loop(self):
        """Interactive chat for local testing"""
        print("\n" + "="*50)
        print(config['personality']['intro_message'])
        print("Type 'quit' to exit, 'evolve' to self-improve")
        print("="*50 + "\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'evolve':
                self.self_improve()
                continue
            
            result = self.respond(user_input)
            print(f"\n🤖 Pullbot: {result['response']}")

if __name__ == '__main__':
    import sys
    
    bot = Pullbot()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'evolve':
            bot.self_improve()
        elif command == 'respond':
            question = sys.argv[2] if len(sys.argv) > 2 else "Hello"
            result = bot.respond(question)
            print(f"\n{result['response']}")
        elif command == 'chat':
            bot.chat_loop()
    else:
        # Default: run one response
        result = bot.respond("What is Python?")
        print(f"\n{result['response']}")
