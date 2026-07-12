// ============================================
// PULLBOT UPGRADER - Real Update System
// ============================================

const REPO_RAW_URL = 'https://raw.githubusercontent.com/pullbot-ai/pullbot-ai.github.io/main';
let latestKnownVersion = null;

async function checkLatestVersion() {
    try {
        const response = await fetch(`${REPO_RAW_URL}/data/knowledge/version.json?t=${Date.now()}`);
        if (response.ok) {
            const data = await response.json();
            latestKnownVersion = data.version || '1.0';
            document.getElementById('latestVersion').textContent = latestKnownVersion;
            return latestKnownVersion;
        }
    } catch (error) {
        console.log('Could not check version:', error.message);
    }
    document.getElementById('latestVersion').textContent = 'Offline';
    return null;
}

async function getLatestVersion() {
    try {
        const response = await fetch(`${REPO_RAW_URL}/data/knowledge/version.json`);
        if (response.ok) {
            const data = await response.json();
            return data.version || '1.0';
        }
    } catch (e) {}
    return '1.0';
}

async function checkForUpdates() {
    const autoUpdate = localStorage.getItem('pullbot_auto_update') || 'off';
    if (autoUpdate === 'off') return;
    
    const lastCheck = parseInt(localStorage.getItem('pullbot_last_check') || '0');
    const now = Date.now();
    
    if (autoUpdate === 'daily' && (now - lastCheck) < 86400000) return;
    if (autoUpdate === 'weekly' && (now - lastCheck) < 604800000) return;
    
    console.log('🔍 Checking for updates...');
    const latest = await checkLatestVersion();
    
    if (latest && latest !== appState.version) {
        console.log(`📦 New version available: ${latest}`);
        showToast(`📦 Pullbot ${latest} available! Click Update Bot to upgrade.`);
    }
    
    localStorage.setItem('pullbot_last_check', now.toString());
}

// ========== REAL DOWNLOAD FUNCTIONS ==========

async function downloadLatestKnowledge(progressCallback) {
    try {
        // 1. Download knowledge store
        const storeUrl = `${REPO_RAW_URL}/data/knowledge/store.json`;
        console.log('📥 Downloading knowledge from:', storeUrl);
        
        const response = await fetch(storeUrl);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const totalSize = parseInt(response.headers.get('content-length') || '0');
        let loadedSize = 0;
        
        // Stream the download
        const reader = response.body.getReader();
        const chunks = [];
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            chunks.push(value);
            loadedSize += value.length;
            
            if (totalSize > 0 && progressCallback) {
                progressCallback(Math.round((loadedSize / totalSize) * 100));
            }
        }
        
        // Combine and parse
        const blob = new Blob(chunks);
        const text = await blob.text();
        const data = JSON.parse(text);
        
        if (!Array.isArray(data)) {
            throw new Error('Invalid knowledge format: expected array');
        }
        
        // 2. Also fetch corpus.txt for richer knowledge
        let corpusText = '';
        try {
            const corpusResponse = await fetch(`${REPO_RAW_URL}/data/processed/corpus.txt`);
            if (corpusResponse.ok) {
                corpusText = await corpusResponse.text();
                
                // Split corpus into chunks and add to knowledge
                const corpusChunks = splitIntoChunks(corpusText, 300);
                corpusChunks.forEach((chunk, i) => {
                    data.push({
                        text: chunk,
                        source: 'corpus',
                        added: Date.now() / 1000
                    });
                });
            }
        } catch (e) {
            console.log('No corpus.txt found, using store.json only');
        }
        
        // Save to IndexedDB
        knowledgeStore.chunks = data;
        knowledgeStore.loaded = true;
        await saveKnowledgeChunks(data);
        
        console.log(`✅ Downloaded ${data.length} knowledge chunks (${(loadedSize / 1024).toFixed(1)}KB)`);
        return true;
        
    } catch (error) {
        console.error('❌ Knowledge download failed:', error);
        throw error;
    }
}

function splitIntoChunks(text, chunkSize) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        const chunk = text.substring(i, i + chunkSize).trim();
        if (chunk.length > 20) {
            chunks.push(chunk);
        }
    }
    return chunks;
}

async function downloadLatestModel(progressCallback) {
    try {
        console.log('📥 Downloading model adapter...');
        
        // Try to download the LoRA adapter files
        const adapterFiles = [
            'adapter_model.bin',
            'adapter_config.json'
        ];
        
        for (const file of adapterFiles) {
            const url = `${REPO_RAW_URL}/models/adapter/${file}`;
            
            if (progressCallback) {
                progressCallback(Math.round((adapterFiles.indexOf(file) / adapterFiles.length) * 100));
            }
            
            try {
                const response = await fetch(url);
                if (response.ok) {
                    const data = await response.arrayBuffer();
                    await saveLoraAdapter({ filename: file, data: Array.from(new Uint8Array(data)) });
                    console.log(`  ✅ Downloaded ${file}`);
                } else {
                    console.log(`  ⚠️ ${file} not found (status ${response.status})`);
                }
            } catch (e) {
                console.log(`  ⚠️ Could not download ${file}: ${e.message}`);
            }
        }
        
        if (progressCallback) {
            progressCallback(100);
        }
        
        console.log('✅ Model adapter download complete');
        return true;
        
    } catch (error) {
        console.error('❌ Model download failed:', error);
        throw error;
    }
}

async function downloadModel() {
    // First-time setup: download initial knowledge
    console.log('📥 First-time setup...');
    
    try {
        await downloadLatestKnowledge((pct) => {
            updateLoader(`Downloading knowledge base... (${pct}%)`, 30 + (pct * 0.5));
        });
        return true;
    } catch (e) {
        console.log('⚠️ Could not download knowledge, using seed data');
        return false;
    }
}

// ========== AUTO-UPDATE ==========

function scheduleDailyUpdate() {
    localStorage.setItem('pullbot_auto_update', 'daily');
    setInterval(checkForUpdates, 3600000); // Check every hour
    console.log('⏰ Auto-update: daily');
}

function scheduleWeeklyUpdate() {
    localStorage.setItem('pullbot_auto_update', 'weekly');
    setInterval(checkForUpdates, 21600000); // Check every 6 hours
    console.log('⏰ Auto-update: weekly');
}

// Save auto-update preference when radio changes
document.addEventListener('change', function(e) {
    if (e.target.name === 'autoUpdate') {
        const value = e.target.value;
        localStorage.setItem('pullbot_auto_update', value);
        
        if (value === 'daily') scheduleDailyUpdate();
        if (value === 'weekly') scheduleWeeklyUpdate();
        
        console.log(`⏰ Auto-update set to: ${value}`);
    }
});

// Load saved auto-update preference on startup
function loadAutoUpdatePreference() {
    const saved = localStorage.getItem('pullbot_auto_update') || 'off';
    const radio = document.querySelector(`input[name="autoUpdate"][value="${saved}"]`);
    if (radio) radio.checked = true;
}
