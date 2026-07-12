// ============================================
// PULLBOT UPGRADER - Update Checker + Downloader
// ============================================

const REPO_RAW_URL = 'https://raw.githubusercontent.com/pullbot-ai/pullbot-ai.github.io/main';
let latestKnownVersion = null;

async function checkLatestVersion() {
    try {
        const response = await fetch(`${REPO_RAW_URL}/data/knowledge/version.json?t=${Date.now()}`);
        if (response.ok) {
            const data = await response.json();
            latestKnownVersion = data.version || 'unknown';
            document.getElementById('latestVersion').textContent = latestKnownVersion;
        } else {
            document.getElementById('latestVersion').textContent = 'Unknown';
        }
    } catch (error) {
        document.getElementById('latestVersion').textContent = 'Offline';
    }
}

async function getLatestVersion() {
    try {
        const response = await fetch(`${REPO_RAW_URL}/data/knowledge/version.json`);
        if (response.ok) {
            const data = await response.json();
            return data.version || '2.5';
        }
    } catch (e) {}
    return '2.5';
}

async function checkForUpdates() {
    const autoUpdate = localStorage.getItem('pullbot_auto_update') || 'off';
    if (autoUpdate === 'off') return;
    
    const lastCheck = localStorage.getItem('pullbot_last_check') || 0;
    const now = Date.now();
    
    if (autoUpdate === 'daily' && (now - lastCheck) < 86400000) return;
    if (autoUpdate === 'weekly' && (now - lastCheck) < 604800000) return;
    
    console.log('🔍 Checking for updates...');
    await checkLatestVersion();
    
    if (latestKnownVersion && latestKnownVersion !== appState.version) {
        console.log(`📦 New version available: ${latestKnownVersion}`);
        // Auto-update silently
        await downloadLatestKnowledge(() => {});
        appState.version = latestKnownVersion;
        await saveConfig('version', latestKnownVersion);
        updateStatusFooter();
    }
    
    localStorage.setItem('pullbot_last_check', now);
}

async function downloadLatestKnowledge(progressCallback) {
    try {
        const response = await fetch(`${REPO_RAW_URL}/data/knowledge/store.json`);
        if (!response.ok) throw new Error('Failed to fetch knowledge');
        
        const totalSize = parseInt(response.headers.get('content-length') || '0');
        let loadedSize = 0;
        
        const reader = response.body.getReader();
        const chunks = [];
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            chunks.push(value);
            loadedSize += value.length;
            
            if (totalSize > 0 && progressCallback) {
                progressCallback(loadedSize / totalSize * 100);
            }
        }
        
        // Combine and parse
        const blob = new Blob(chunks);
        const text = await blob.text();
        const data = JSON.parse(text);
        
        // Save to IndexedDB
        knowledgeStore.chunks = data;
        knowledgeStore.loaded = true;
        await saveKnowledgeChunks(data);
        
        console.log(`✅ Downloaded ${data.length} knowledge chunks`);
        return true;
    } catch (error) {
        console.error('❌ Failed to download knowledge:', error);
        throw error;
    }
}

async function downloadLatestModel(progressCallback) {
    try {
        // For now, this is a placeholder for ONNX model download
        // In production, this would download the converted ONNX files
        console.log('📥 Model download placeholder');
        
        if (progressCallback) {
            for (let i = 0; i <= 100; i += 20) {
                progressCallback(i);
                await new Promise(r => setTimeout(r, 300));
            }
        }
        
        return true;
    } catch (error) {
        console.error('❌ Failed to download model:', error);
        throw error;
    }
}

async function downloadModel() {
    // First-time model download
    console.log('📥 First-time model download...');
    
    // For now, we use the knowledge-based fallback
    // Full ONNX model download will be added when model conversion is complete
    return true;
}

function scheduleDailyUpdate() {
    localStorage.setItem('pullbot_auto_update', 'daily');
    // Check every hour if it's time
    setInterval(checkForUpdates, 3600000);
}

function scheduleWeeklyUpdate() {
    localStorage.setItem('pullbot_auto_update', 'weekly');
    // Check every 6 hours if it's time
    setInterval(checkForUpdates, 21600000);
}

// Save auto-update preference
document.addEventListener('change', function(e) {
    if (e.target.name === 'autoUpdate') {
        const value = e.target.value;
        localStorage.setItem('pullbot_auto_update', value);
        
        if (value === 'daily') scheduleDailyUpdate();
        if (value === 'weekly') scheduleWeeklyUpdate();
        
        console.log(`⏰ Auto-update set to: ${value}`);
    }
});
