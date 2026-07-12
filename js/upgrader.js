// ============================================
// PULLBOT UPGRADER - Real Update System
// Downloads actual model chunks and knowledge from repo
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

// ============================================
// REAL DOWNLOAD FUNCTIONS
// ============================================

async function downloadLatestKnowledge(progressCallback) {
    try {
        console.log('📥 Downloading knowledge base...');
        
        const storeUrl = `${REPO_RAW_URL}/knowledge/store.json`;
        const response = await fetch(storeUrl);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const totalSize = parseInt(response.headers.get('content-length') || '0');
        const reader = response.body.getReader();
        const chunks = [];
        let loadedSize = 0;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            chunks.push(value);
            loadedSize += value.length;
            
            if (totalSize > 0 && progressCallback) {
                progressCallback(Math.round((loadedSize / totalSize) * 100));
            }
        }
        
        const blob = new Blob(chunks);
        const text = await blob.text();
        const data = JSON.parse(text);
        
        if (!Array.isArray(data)) {
            throw new Error('Invalid knowledge format: expected array');
        }
        
        // Also try to fetch corpus.txt for richer knowledge
        try {
            const corpusResponse = await fetch(`${REPO_RAW_URL}/data/processed/corpus.txt`);
            if (corpusResponse.ok) {
                const corpusText = await corpusResponse.text();
                const corpusChunks = splitIntoChunks(corpusText, 300);
                corpusChunks.forEach(chunk => {
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
        
        // Save to memory and IndexedDB
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
        console.log('📥 Downloading trained model...');
        
        // Get manifest to know what chunks to download
        const manifestUrl = `${REPO_RAW_URL}/models/chunks/manifest.json`;
        const manifestResp = await fetch(manifestUrl);
        
        if (!manifestResp.ok) {
            throw new Error(`Cannot fetch manifest: HTTP ${manifestResp.status}`);
        }
        
        const manifest = await manifestResp.json();
        const chunks = manifest.chunks || [];
        
        if (chunks.length === 0) {
            throw new Error('No model chunks found in manifest');
        }
        
        console.log(`   Downloading ${chunks.length} chunks (${manifest.total_size_mb}MB total)...`);
        
        // Download each chunk
        for (let i = 0; i < chunks.length; i++) {
            const chunkUrl = `${REPO_RAW_URL}/${chunks[i]}`;
            console.log(`   Chunk ${i + 1}/${chunks.length}...`);
            
            const response = await fetch(chunkUrl);
            if (!response.ok) {
                console.warn(`   ⚠️ Failed chunk ${i}: HTTP ${response.status}`);
                continue;
            }
            
            const data = await response.arrayBuffer();
            await saveModelChunk(i, new Uint8Array(data));
            
            if (progressCallback) {
                progressCallback(Math.round(((i + 1) / chunks.length) * 100));
            }
        }
        
        // Save manifest and config files
        await saveConfig('model_manifest', manifest);
        await saveConfig('model_downloaded_at', Date.now());
        
        // Also download config files
        const configFiles = ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'];
        for (const fname of configFiles) {
            try {
                const resp = await fetch(`${REPO_RAW_URL}/models/chunks/${fname}`);
                if (resp.ok) {
                    const text = await resp.text();
                    await saveConfig(`model_${fname.replace('.json', '')}`, text);
                }
            } catch (e) {
                // Config files are optional
            }
        }
        
        console.log('✅ Model downloaded and stored locally');
        return true;
        
    } catch (error) {
        console.error('❌ Model download failed:', error);
        throw error;
    }
}

async function saveModelChunk(index, data) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction('model', 'readwrite');
        const store = tx.objectStore('model');
        store.put({
            name: `chunk_${index}`,
            data: Array.from(data),
            savedAt: Date.now()
        });
        tx.oncomplete = resolve;
        tx.onerror = reject;
    });
}

async function downloadModel() {
    // First-time setup: download initial knowledge and model
    console.log('📥 First-time setup...');
    
    try {
        await downloadLatestKnowledge((pct) => {
            updateLoader(`Downloading knowledge base... (${pct}%)`, 30 + (pct * 0.3));
        });
        
        await downloadLatestModel((pct) => {
            updateLoader(`Downloading model... (${pct}%)`, 60 + (pct * 0.4));
        });
        
        return true;
    } catch (e) {
        console.log('⚠️ First-time download incomplete, using seed data');
        return false;
    }
}

// ============================================
// AUTO-UPDATE
// ============================================

function scheduleDailyUpdate() {
    localStorage.setItem('pullbot_auto_update', 'daily');
    setInterval(checkForUpdates, 3600000);
    console.log('⏰ Auto-update: daily');
}

function scheduleWeeklyUpdate() {
    localStorage.setItem('pullbot_auto_update', 'weekly');
    setInterval(checkForUpdates, 21600000);
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
