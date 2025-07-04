import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import pandas as pd
from terminal_web_chatbot import TerminalStyleWebChatbot
import datetime

# Load environment variables from .env (only GROQ_API_KEY expected)
load_dotenv()

to_load = os.environ.get('GROQ_API_KEY')

app = Flask(__name__)
# Secret key for sessions—keep this value in code or generate securely
# IMPORTANT: For production, this should be a strong, randomly generated value and kept secret.
app.secret_key = os.urandom(24) # A more robust secret key for development

# MongoDB connection (defaults to localhost)
client = MongoClient("mongodb://localhost:27017/")
db = client['IDP']
users_collection = db['users']

def filter_longest_medicines(medicine_list):
    normalized = [(med.strip().lower(), med) for med in medicine_list]
    sorted_meds = sorted(normalized, key=lambda x: len(x[0]), reverse=True)
    filtered = []
    seen = set()
    for i, (norm_med, orig_med) in enumerate(sorted_meds):
        if any(norm_med in other for other, _ in sorted_meds[:i]):
            continue
        if norm_med not in seen:
            filtered.append(orig_med)
            seen.add(norm_med)
    return filtered

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        role = request.form['role']
        username = request.form['username']
        password = request.form['password']

        if role == 'admin':
            user = db.admin.find_one({'username': username})
        else:
            user = db.users.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            session['user'] = {'role': role, 'username': username}
            session['username'] = username
            print(f"DEBUG: User '{username}' logged in successfully. Session user: {session.get('user')}")
            return redirect(url_for('admin_dashboard' if role == 'admin' else 'user_dashboard'))
        flash("Invalid credentials", 'error')
        print(f"DEBUG: Login failed for username: {username}")
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match", 'error')
            return redirect(url_for('register'))

        if db.users.find_one({'$or': [{'username': username}, {'email': email}] } ):
            flash("Username or email already exists", 'error')
            return redirect(url_for('register'))

        db.users.insert_one({
            'username': username,
            'email': email,
            'password': generate_password_hash(password),
            'chatbot_state': {},
            'symptoms_history': [],
            'recommendations_history': []
        })
        flash("Registration successful. Please login.", 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/user_dashboard')
def user_dashboard():
    username = session.get('user', {}).get('username')
    print(f"DEBUG: user_dashboard accessed by username: {username}")
    # Use session state only, do not load from DB except at start
    if 'chatbot_state' not in session or session['chatbot_state'].get('user_id') != username:
        print("DEBUG: Initializing new chatbot state for session or user changed.")
        bot_instance = TerminalStyleWebChatbot()
        session['chatbot_state'] = bot_instance.get_serializable_state()
    else:
        print(f"DEBUG: Loading existing chatbot state for session. Current step: {session['chatbot_state'].get('step')}")
        bot_instance = TerminalStyleWebChatbot()
        bot_instance.restore_state(session['chatbot_state'])
    greeting = bot_instance.get_greeting()
    print(f"DEBUG: user_dashboard greeting: {greeting}")
    return render_template('user_dashboard.html', greeting=greeting)

@app.route('/logout')
def logout():
    session.clear()
    flash("You've been logged out.", 'info')
    print("DEBUG: Session cleared on logout.")
    return redirect(url_for('login'))

# Legacy endpoints (optional)
@app.route('/analyze_symptoms', methods=['POST'])
def analyze_symptoms():
    symptoms = request.form.get('symptoms', '')
    flash("Symptom analysis not implemented yet. You entered: " + symptoms)
    return redirect(url_for('user_dashboard'))

@app.route('/confirm_dispense', methods=['POST'])
def confirm_dispense():
    medicine = request.form.get('medicine', '')
    flash(f"Dispense confirmation not implemented yet. You clicked: {medicine}")
    return redirect(url_for('user_dashboard'))

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    # Ensure user is logged in
    username = session.get('username')
    if not username:
        return jsonify({'response': 'You must be logged in to use the chatbot.'}), 401
    # Use session state only
    if 'chatbot_state' not in session:
        bot_instance = TerminalStyleWebChatbot()
        session['chatbot_state'] = bot_instance.get_serializable_state()
    else:
        bot_instance = TerminalStyleWebChatbot()
        bot_instance.restore_state(session['chatbot_state'])
    print("DEBUG: State before processing:", bot_instance.state)
    # If this is a greeting request, just return the greeting
    if request.json.get('greeting'):
        return jsonify({'response': bot_instance.get_greeting()})
    # Process user input
    user_input = request.json['message']
    response = bot_instance.get_response(user_input)
    print("DEBUG: State after processing:", bot_instance.state)
    # Always update session state after processing
    session['chatbot_state'] = bot_instance.get_serializable_state()
    # Only store in MongoDB if chat is done
    if bot_instance.state.get('step') == 'done':
        # Check chat history for last 2 days
        user_doc = users_collection.find_one({'username': username}, {'chat_history': 1})
        now = datetime.datetime.utcnow()
        two_days_ago = now - datetime.timedelta(days=2)
        recent_chats = []
        if user_doc and 'chat_history' in user_doc:
            for entry in user_doc['chat_history']:
                try:
                    ts = datetime.datetime.fromisoformat(entry.get('timestamp', ''))
                    if ts > two_days_ago:
                        recent_chats.append(entry)
                except Exception:
                    continue
        if len(recent_chats) >= 2:
            response = "Consult a doctor since you're falling sick frequently."
            return jsonify({'response': response})
        # Extract info for chat history
        user_flags = bot_instance.state.get('user_flags', {})
        symptoms_yes = [k for k, v in user_flags.items() if v.strip().lower() == 'yes']
        all_symptoms = list(user_flags.keys())
        recommendation = response
        timestamp = now.isoformat()

        # Medicine extraction logic (must match terminal_web_chatbot.py)
        matched_medicines = []
        rec_lower = recommendation.lower()
        for med in [
            'Paracetamol 500mg', 'Limcee 500mg', 'ORS', 'Zincovit syrup', 'Zincovit tablet',
            'Benadryl Dry Cough syrup', 'Honey-ginger lozenges', 'Mucosolvan cough syrup',
            'Ambrolite cough syrup', 'Meftal Spas tablet', 'Cold pack', 'Levocetirizine 5mg',
            'Bromhexine syrup', 'Ambroxol syrup', 'Heat patch', 'Saline nasal spray',
            'Vicks Lozenges', 'Paracetamol 650mg', 'Paracetamol 500-650mg',
            'Herbal Lozenge (Vicks/Himalaya)', 'Rantac', 'Gelusil', 'Digene',
            'Pantoprazole 40mg', 'Clove gel', 'Benzocain gel', 'Antihistamines- Cetrizine',
            'Strepsils', 'Decongestants ( phenylephrine)', 'Paracetamol',
            'Nasal decongestants (oxymetazoline)', 'Pantoprazole', 'Antacid( Gelusil)',
            'Simethicone', 'Clove oil', 'Meftal spas 250 mg', 'Meftal spas 500mg',
            'Mouthwash ( chlorhexidine)', 'Antiseptic gel', 'Compression wrap',
            'Crepe bandage', 'Pain relief gel ( diclofenac )', 'Elastic support bandage',
            'Ice pack', 'Warm compress ( heat patch )', 'Herbal lozenges ( Vicks, adulsa )',
            'Dextromethorphan syrup', 'Cough lozenges ( benzydamine)'
        ]:
            med_norm = med.strip().lower()
            if any(part.strip().lower() in rec_lower for part in med.lower().split(',')):
                if not any(m.strip().lower() == med_norm for m in matched_medicines):
                    matched_medicines.append(med)
        if matched_medicines:
            matched_medicines = filter_longest_medicines(matched_medicines)

        chat_entry = {
            'symptoms': symptoms_yes,
            'all_symptoms': all_symptoms,
            'recommendation': recommendation,
            'timestamp': timestamp,
            'medicine': matched_medicines if matched_medicines else None
        }
        # Add dispensed medicine selection
        dispensed_medicine = bot_instance.state.get('dispense_selection', None)
        if dispensed_medicine:
            chat_entry['dispensed_medicine'] = dispensed_medicine
        # Add follow-up answers
        followup_answers = bot_instance.state.get('followup_answers', {})
        if followup_answers:
            chat_entry['followup_answers'] = followup_answers

        users_collection.update_one(
            {'username': username},
            {'$push': {'chat_history': chat_entry}},
            upsert=True
        )
    return jsonify({'response': response})

if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        print('\nShutting down gracefully...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    app.run(debug=True, use_reloader=True)
