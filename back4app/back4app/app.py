"""
Pullbot User Service - Back4App
Handles signup, login, user data. No model.
Stores users in a JSON file (persists on Back4App's disk).
"""

import os, json, hashlib, time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

USERS_FILE = "data/users.json"
os.makedirs("data", exist_ok=True)

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {"users": {}}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/')
def home():
    users = load_users()
    return jsonify({
        'service': 'Pullbot User Service',
        'status': 'online',
        'users': len(users.get('users', {}))
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    username = data.get('username', '').strip().lower()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password required'}), 400
    if len(username) < 3:
        return jsonify({'success': False, 'error': 'Username too short'}), 400
    
    users = load_users()
    
    if username in users.get('users', {}):
        return jsonify({'success': False, 'error': 'Username taken'}), 409
    
    users['users'][username] = {
        'password': hash_password(password),
        'created': time.strftime('%Y-%m-%d'),
        'chats': []
    }
    
    save_users(users)
    
    return jsonify({
        'success': True,
        'username': username,
        'token': hash_password(username + password)[:20],
        'message': f'Welcome, {username}!'
    })

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip().lower()
    password = data.get('password', '').strip()
    
    users = load_users()
    
    if username not in users.get('users', {}):
        return jsonify({'success': False, 'error': 'User not found'}), 404
    
    if users['users'][username]['password'] != hash_password(password):
        return jsonify({'success': False, 'error': 'Wrong password'}), 401
    
    return jsonify({
        'success': True,
        'username': username,
        'token': hash_password(username + password)[:20],
        'message': f'Welcome back, {username}!'
    })

@app.route('/user/<username>')
def get_user(username):
    users = load_users()
    if username in users.get('users', {}):
        user = users['users'][username]
        return jsonify({
            'username': username,
            'created': user.get('created'),
            'chats': len(user.get('chats', []))
        })
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
