// Pullbot DIY Load Balancer
// Tries backends in order, caches which one is awake

const BACKENDS = [
    'https://pullbot-api.onrender.com',
    // Add Back4App URL when ready
];

let activeBackend = null;
let lastCheck = 0;

async function getBestBackend() {
    // If we found one recently, use it
    if (activeBackend && (Date.now() - lastCheck) < 60000) {
        return activeBackend;
    }
    
    // Race all backends - first to respond wins
    const promises = BACKENDS.map(async (url) => {
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 3000);
            
            const r = await fetch(`${url}/health`, { signal: controller.signal });
            clearTimeout(timeout);
            
            if (r.ok) {
                return url;
            }
        } catch(e) {}
        return null;
    });
    
    // Wait for first response
    const results = await Promise.race([
        Promise.any(promises),
        new Promise(resolve => setTimeout(() => resolve(null), 3000))
    ]);
    
    if (results) {
        activeBackend = results;
        lastCheck = Date.now();
        return results;
    }
    
    // All failed - try each one sequentially
    for (const url of BACKENDS) {
        try {
            const r = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2000) });
            if (r.ok) {
                activeBackend = url;
                lastCheck = Date.now();
                return url;
            }
        } catch(e) {}
    }
    
    return BACKENDS[0]; // Fallback to first
}

async function askPullbotRouter(question) {
    const backend = await getBestBackend();
    
    try {
        const r = await fetch(`${backend}/ask?q=${encodeURIComponent(question)}`, {
            signal: AbortSignal.timeout(30000)
        });
        if (r.ok) {
            const data = await r.json();
            return data.response || "No response";
        }
    } catch(e) {
        // Try next backend
        for (const url of BACKENDS) {
            if (url === backend) continue;
            try {
                const r = await fetch(`${url}/ask?q=${encodeURIComponent(question)}`, {
                    signal: AbortSignal.timeout(30000)
                });
                if (r.ok) {
                    const data = await r.json();
                    activeBackend = url;
                    return data.response || "No response";
                }
            } catch(e2) {}
        }
    }
    
    return "All Pullbot servers are sleeping. Try again in a moment!";
}
