// ============================================
// PULLBOT DIY LOAD BALANCER
// Races backends, caches winner, auto-failover
// ============================================

const BACKENDS = [
    'https://pullbot-api.onrender.com',
];

const USER_BACKENDS = [
    'https://pullbot-users.back4app.io',  // When deployed
];

let activeBackend = null;
let activeUserBackend = null;
let lastCheck = 0;
let lastUserCheck = 0;

async function getBestBackend(list, cache, cacheTime) {
    const cacheKey = list === BACKENDS ? 'activeBackend' : 'activeUserBackend';
    const lastKey = list === BACKENDS ? 'lastCheck' : 'lastUserCheck';
    
    if (window[cacheKey] && (Date.now() - window[lastKey]) < 60000) {
        return window[cacheKey];
    }
    
    const checks = list.map(async (url) => {
        try {
            const r = await fetch(`${url}/health`, { signal: AbortSignal.timeout(3000) });
            if (r.ok) return url;
        } catch(e) {}
        return null;
    });
    
    const results = (await Promise.all(checks)).filter(r => r !== null);
    
    if (results.length > 0) {
        window[cacheKey] = results[0];
        window[lastKey] = Date.now();
        return results[0];
    }
    
    // Try sequentially
    for (const url of list) {
        try {
            const r = await fetch(`${url}/health`, { signal: AbortSignal.timeout(5000) });
            if (r.ok) {
                window[cacheKey] = url;
                window[lastKey] = Date.now();
                return url;
            }
        } catch(e) {}
    }
    
    return list[0];
}

async function askPullbotRouter(question) {
    const backend = await getBestBackend(BACKENDS, 'activeBackend', 'lastCheck');
    
    try {
        const r = await fetch(`${backend}/ask?q=${encodeURIComponent(question)}`, {
            signal: AbortSignal.timeout(30000)
        });
        if (r.ok) {
            const data = await r.json();
            return data.response;
        }
    } catch(e) {}
    
    // Fallback to other backends
    for (const url of BACKENDS) {
        if (url === backend) continue;
        try {
            const r = await fetch(`${url}/ask?q=${encodeURIComponent(question)}`, {
                signal: AbortSignal.timeout(30000)
            });
            if (r.ok) {
                const data = await r.json();
                window.activeBackend = url;
                window.lastCheck = Date.now();
                return data.response;
            }
        } catch(e) {}
    }
    
    return null;
}

async function userRequest(endpoint, body) {
    const backend = await getBestBackend(USER_BACKENDS, 'activeUserBackend', 'lastUserCheck');
    
    try {
        const r = await fetch(`${backend}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(10000)
        });
        return await r.json();
    } catch(e) {}
    
    // Fallback to local Render auth
    try {
        const r = await fetch(`${BACKENDS[0]}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(10000)
        });
        return await r.json();
    } catch(e) {
        return { success: false, error: 'Cannot connect to server' };
    }
}

async function checkBackendStatus() {
    const status = {};
    for (const url of BACKENDS) {
        try {
            const r = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2000) });
            status[url] = r.ok ? 'online' : 'error';
        } catch(e) {
            status[url] = 'offline';
        }
    }
    return status;
}
