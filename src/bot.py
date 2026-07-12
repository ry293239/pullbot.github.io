"""
Pullbot Bot
Combines model + knowledge store + self-improvement.
"""

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
import time
import random
from model import PullbotModel
from store import KnowledgeStore

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class Pullbot:
    def __init__(self):
        print("\n" + "="*50)
        print("🤖 PULLBOT INITIALIZING")
        print("="*50 + "\n")
        
        self.model = PullbotModel()
        self.store = KnowledgeStore()
        
        print(f"\n✅ Pullbot ready!")
        print(f"   Model: {config['model']['base']}")
        print(f"   Knowledge chunks: {len(self.store.chunks)}")
        print(f"   Self-improvement: {'ON' if config['evolve']['enabled'] else 'OFF'}")
    
    def respond(self, question):
        print(f"\n❓ {question}")
        
        context = self.store.get_context(question)
        if context:
            print(f"   📚 Context retrieved ({len(context)} chars)")
        
        response = self.model.generate(prompt=question, context=context)
        print(f"   🤖 {response[:100]}...")
        
        return {
            'question': question,
            'response': response,
            'context_used': context is not None,
            'timestamp': time.time()
        }
    
    def self_improve(self, prompts=None):
        if not config['evolve']['enabled']:
            print("🔒 Self-improvement disabled")
            return
        
        print("\n🧬 SELF-IMPROVEMENT CYCLE")
        print("-" * 30)
        
        if prompts is None:
            prompts = [
                "Explain what a function is in programming",
                "What is machine learning?",
                "How does the internet work?",
                "Write a short Python code example",
                "Explain climate change simply"
            ]
        
        all_candidates = []
        
        for prompt in prompts:
            print(f"\n📝 Generating variants for: {prompt[:50]}...")
            for _ in range(config['evolve']['variants']):
                self.model.temperature = random.uniform(0.6, 1.0)
                response = self.model.generate(prompt)
                score = self._score_response(response)
                all_candidates.append({
                    'prompt': prompt,
                    'response': response,
                    'score': score
                })
        
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        keep_n = max(1, len(all_candidates) * config['evolve']['keep_percent'] // 100)
        elite = all_candidates[:keep_n]
        
        print(f"\n🏆 Kept {keep_n}/{len(all_candidates)} elite responses")
        if elite:
            print(f"   Best score: {elite[0]['score']:.3f}")
        
        os.makedirs('data/curated', exist_ok=True)
        curated_path = f"data/curated/evolved_{int(time.time())}.json"
        with open(curated_path, 'w') as f:
            json.dump(elite, f, indent=2)
        
        with open('data/processed/corpus.txt', 'a') as f:
            for item in elite:
                f.write(f"\n\nPrompt: {item['prompt']}\nResponse: {item['response']}\n")
        
        print(f"💾 Added {keep_n} curated examples to training data")
        return elite
    
    def _score_response(self, response):
        score = 0.0
        length = len(response)
        
        if length < config['evolve']['min_length']:
            return 0.0
        
        if 50 <= length <= 300:
            score += 0.3
        
        words = response.split()
        unique = set(words)
        if len(unique) > 5:
            score += 0.2
        if len(words) > 0 and len(unique) / len(words) > 0.5:
            score += 0.2
        if response.rstrip() and response.rstrip()[-1] in '.!?':
            score += 0.1
        if any(kw in response for kw in ['def ', 'import ', 'class ', 'function', 'code']):
            score += 0.2
        
        return score
    
    def chat_loop(self):
        print("\n" + "="*50)
        print(config['personality']['intro'])
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
        if sys.argv[1] == 'evolve':
            bot.self_improve()
        elif sys.argv[1] == 'respond':
            question = sys.argv[2] if len(sys.argv) > 2 else "Hello"
            result = bot.respond(question)
            print(f"\n{result['response']}")
        elif sys.argv[1] == 'chat':
            bot.chat_loop()
    else:
        result = bot.respond("What is Python?")
        print(f"\n{result['response']}")
