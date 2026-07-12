// ============================================
// PULLBOT STORAGE - IndexedDB Manager
// ============================================

const DB_NAME = 'pullbot_db';
const DB_VERSION = 1;

let db = null;

async function initDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            
            // Store for model weights
            if (!db.objectStoreNames.contains('model')) {
                db.createObjectStore('model', { keyPath: 'name' });
            }
            
            // Store for knowledge chunks
            if (!db.objectStoreNames.contains('knowledge')) {
                db.createObjectStore('knowledge', { keyPath: 'id' });
            }
            
            // Store for chats
            if (!db.objectStoreNames.contains('chats')) {
                db.createObjectStore('chats', { keyPath: 'id' });
            }
            
            // Store for config
            if (!db.objectStoreNames.contains('config')) {
                db.createObjectStore('config', { keyPath: 'key' });
            }
        };
        
        request.onsuccess = (event) => {
            db = event.target.result;
            console.log('✅ IndexedDB ready');
            resolve();
        };
        
        request.onerror = (event) => {
            console.error('❌ IndexedDB failed:', event.target.error);
            reject(event.target.error);
        };
    });
}

// ========== MODEL STORAGE ==========
async function checkModelExists() {
    return new Promise((resolve) => {
        const tx = db.transaction('model', 'readonly');
        const store = tx.objectStore('model');
        const req = store.get('onnx_weights');
        req.onsuccess = () => resolve(!!req.result);
        req.onerror = () => resolve(false);
    });
}

async function saveModelWeights(data) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction('model', 'readwrite');
        const store = tx.objectStore('model');
        store.put({ name: 'onnx_weights', data: data, savedAt: Date.now() });
        tx.oncomplete = resolve;
        tx.onerror = reject;
    });
}

async function getModelWeights() {
    return new Promise((resolve) => {
        const tx = db.transaction('model', 'readonly');
        const store = tx.objectStore('model');
        const req = store.get('onnx_weights');
        req.onsuccess = () => resolve(req.result ? req.result.data : null);
        req.onerror = () => resolve(null);
    });
}

async function saveLoraAdapter(data) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction('model', 'readwrite');
        const store = tx.objectStore('model');
        store.put({ name: 'lora_adapter', data: data, savedAt: Date.now() });
        tx.oncomplete = resolve;
        tx.onerror = reject;
    });
}

async function getLoraAdapter() {
    return new Promise((resolve) => {
        const tx = db.transaction('model', 'readonly');
        const store = tx.objectStore('model');
        const req = store.get('lora_adapter');
        req.onsuccess = () => resolve(req.result ? req.result.data : null);
        req.onerror = () => resolve(null);
    });
}

// ========== KNOWLEDGE STORAGE ==========
async function saveKnowledgeChunks(chunks) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction('knowledge', 'readwrite');
        const store = tx.objectStore('knowledge');
        
        // Clear old knowledge
        store.clear();
        
        // Save new chunks
        chunks.forEach((chunk, index) => {
            store.put({ id: index, ...chunk, savedAt: Date.now() });
        });
        
        tx.oncomplete = resolve;
        tx.onerror = reject;
    });
}

async function getKnowledgeChunks() {
    return new Promise((resolve) => {
        const tx = db.transaction('knowledge', 'readonly');
        const store = tx.objectStore('knowledge');
        const req = store.getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => resolve([]);
    });
}

async function getKnowledgeCount() {
    return new Promise((resolve) => {
        const tx = db.transaction('knowledge', 'readonly');
        const store = tx.objectStore('knowledge');
        const req = store.count();
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(0);
    });
}

// ========== CHAT STORAGE ==========
async function saveChats() {
    if (!currentChatId || !chats[currentChatId]) return;
    
    return new Promise((resolve, reject) => {
        const tx = db.transaction('chats', 'readwrite');
        const store = tx.objectStore('chats');
        store.put({ id: currentChatId, ...chats[currentChatId], updatedAt: Date.now() });
        tx.oncomplete = resolve;
        tx.onerror = reject;
    });
}

async function loadChats() {
    return new Promise((resolve) => {
        const tx = db.transaction('chats', 'readonly');
        const store = tx.objectStore('chats');
        const req = store.getAll();
        req.onsuccess = () => {
            const savedChats = req.result || [];
            savedChats.forEach(chat => {
                chats[chat.id] = {
                    title: chat.title,
                    messages: chat.messages,
                    created: chat.created
                };
            });
            
            // Set most recent as current
            if (savedChats.length > 0) {
                const latest = savedChats.sort((a, b) => b.updatedAt - a.updatedAt)[0];
                currentChatId = latest.id;
            }
            
            console.log(`📂 Loaded ${savedChats.length} chats`);
            resolve();
        };
        req.onerror = () => resolve();
    });
}

async function deleteChat(id) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction('chats', 'readwrite');
        const store = tx.objectStore('chats');
        store.delete(id);
        tx.oncomplete = () => {
            delete chats[id];
            if (currentChatId === id) {
                currentChatId = Object.keys(chats)[0] || null;
            }
            resolve();
        };
        tx.onerror = reject;
    });
}

// ========== CONFIG STORAGE ==========
async function saveConfig(key, value) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction('config', 'readwrite');
        const store = tx.objectStore('config');
        store.put({ key, value, updatedAt: Date.now() });
        tx.oncomplete = resolve;
        tx.onerror = reject;
    });
}

async function getConfig(key) {
    return new Promise((resolve) => {
        const tx = db.transaction('config', 'readonly');
        const store = tx.objectStore('config');
        const req = store.get(key);
        req.onsuccess = () => resolve(req.result ? req.result.value : null);
        req.onerror = () => resolve(null);
    });
}

// ========== STORAGE SIZE ==========
async function getStorageSize() {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimate = await navigator.storage.estimate();
        return {
            used: Math.round(estimate.usage / (1024 * 1024)),
            total: Math.round(estimate.quota / (1024 * 1024))
        };
    }
    return null;
}

// ========== CLEAR ALL ==========
async function clearAllData() {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(['model', 'knowledge', 'chats', 'config'], 'readwrite');
        
        tx.objectStore('model').clear();
        tx.objectStore('knowledge').clear();
        tx.objectStore('chats').clear();
        tx.objectStore('config').clear();
        
        tx.oncomplete = () => {
            chats = {};
            currentChatId = null;
            appState.modelLoaded = false;
            appState.knowledgeLoaded = false;
            console.log('🗑 All data cleared');
            resolve();
        };
        tx.onerror = reject;
    });
}
