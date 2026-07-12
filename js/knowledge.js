// ============================================
// PULLBOT KNOWLEDGE - Local Vector Search
// ============================================

let knowledgeStore = {
    chunks: [],
    embeddings: null,
    loaded: false
};

async function loadKnowledge() {
    try {
        // First try loading from IndexedDB
        const savedChunks = await getKnowledgeChunks();
        
        if (savedChunks.length > 0) {
            knowledgeStore.chunks = savedChunks;
            knowledgeStore.loaded = true;
            console.log(`📚 Loaded ${savedChunks.length} knowledge chunks from local storage`);
            return;
        }
        
        // Fallback: fetch from repo
        console.log('📥 Fetching knowledge from repo...');
        const response = await fetch('data/knowledge/store.json');
        if (response.ok) {
            const data = await response.json();
            knowledgeStore.chunks = data;
            knowledgeStore.loaded = true;
            
            // Save to IndexedDB for next time
            await saveKnowledgeChunks(data);
            console.log(`📚 Loaded ${data.length} knowledge chunks from repo`);
        } else {
            console.log('⚠️ No knowledge store found');
        }
    } catch (error) {
        console.error('❌ Failed to load knowledge:', error);
    }
}

function searchLocalKnowledge(query, topK = 3) {
    if (!knowledgeStore.loaded || knowledgeStore.chunks.length === 0) {
        return [];
    }
    
    // Simple keyword-based search (no embeddings in pure JS yet)
    // Scores chunks based on word overlap with query
    const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    
    const scored = knowledgeStore.chunks.map((chunk, index) => {
        const chunkText = (chunk.text || '').toLowerCase();
        let score = 0;
        
        // Word overlap
        for (const word of queryWords) {
            if (chunkText.includes(word)) {
                score += 1;
                // Bonus for exact phrase match
                if (chunkText.includes(query.toLowerCase())) {
                    score += 3;
                }
            }
        }
        
        // Bonus for title/source match
        if (chunk.source && chunk.source.toLowerCase().includes(query.toLowerCase())) {
            score += 2;
        }
        
        return { index, chunk, score };
    });
    
    // Sort by score and return top K
    scored.sort((a, b) => b.score - a.score);
    return scored
        .filter(s => s.score > 0)
        .slice(0, topK)
        .map(s => s.chunk);
}

function craftResponseFromKnowledge(query, results) {
    if (results.length === 0) {
        return "I don't have enough information about that yet. Try asking something else, or wait for my next update!";
    }
    
    // Use the most relevant chunk to craft a response
    const bestMatch = results[0];
    const source = bestMatch.source ? ` (Source: ${bestMatch.source})` : '';
    
    // Craft a natural response
    const templates = [
        `Based on what I've learned: ${bestMatch.text.substring(0, 300)}...${source}`,
        `Here's what I know about that: ${bestMatch.text.substring(0, 300)}...${source}`,
        `I found this relevant information: ${bestMatch.text.substring(0, 300)}...${source}`,
    ];
    
    // Pick template based on query type
    if (query.toLowerCase().includes('what') || query.toLowerCase().includes('how')) {
        return templates[0];
    } else if (query.toLowerCase().includes('explain')) {
        return templates[1];
    } else {
        return templates[2];
    }
}

// Simple local inference (placeholder until ONNX model is integrated)
async function localInference(prompt) {
    // Search knowledge first
    const knowledgeResults = searchLocalKnowledge(prompt);
    
    if (knowledgeResults.length > 0) {
        return craftResponseFromKnowledge(prompt, knowledgeResults);
    }
    
    // Fallback: use basic pattern matching
    const lowerPrompt = prompt.toLowerCase();
    
    if (lowerPrompt.includes('hello') || lowerPrompt.includes('hi')) {
        return "Hey there! 👋 I'm Pullbot. I've been learning from GitHub and Wikipedia. What can I help you with?";
    }
    
    if (lowerPrompt.includes('who are you') || lowerPrompt.includes('what are you')) {
        return "I'm Pullbot! 🤖 I'm an AI that scrapes knowledge from GitHub and Wikipedia to learn new things. I run entirely in your browser — pretty cool, right?";
    }
    
    if (lowerPrompt.includes('thank')) {
        return "You're welcome! 😊 Let me know if you need anything else.";
    }
    
    // Default
    return "That's an interesting question! Based on my knowledge, I'd say it depends on the context. Could you be more specific about what you'd like to know?";
}
