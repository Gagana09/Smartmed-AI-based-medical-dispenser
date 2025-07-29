import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import pandas as pd
from terminal_web_chatbot import TerminalStyleWebChatbot
import datetime
import pytz
import serial   
import time   
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

# Make the database available to the application
app.config['DATABASE'] = db

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

        # Check if role is selected
        if not role or role == "":
            flash("Please select a role (User or Admin)", 'error')
            return redirect(url_for('login'))

        # Check if username and password are provided
        if not username or not password:
            flash("Please enter both username and password", 'error')
            return redirect(url_for('login'))

        if role == 'admin':
            user = db.admin.find_one({'username': username})
        else:
            user = db.users.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            session['user'] = {'role': role, 'username': username}
            session['username'] = username
            print(f"DEBUG: User '{username}' logged in successfully. Session user: {session.get('user')}")
            return redirect(url_for('admin_dashboard' if role == 'admin' else 'user_dashboard'))
        
        # More specific error messages
        if not user:
            flash(f"Username '{username}' not found. Please check your username or register a new account.", 'error')
        else:
            flash("Incorrect password. Please try again.", 'error')
        
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

        # Check if all fields are provided
        if not username or not email or not password or not confirm_password:
            flash("Please fill in all fields", 'error')
            return redirect(url_for('register'))

        # Validate email format
        if '@' not in email or '.' not in email:
            flash("Please enter a valid email address", 'error')
            return redirect(url_for('register'))

        # Check password length
        if len(password) < 6:
            flash("Password must be at least 6 characters long", 'error')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match", 'error')
            return redirect(url_for('register'))

        # Check for existing username or email
        existing_user = db.users.find_one({'$or': [{'username': username}, {'email': email}] })
        if existing_user:
            if existing_user.get('username') == username:
                flash(f"Username '{username}' is already taken. Please choose a different username.", 'error')
            else:
                flash(f"Email '{email}' is already registered. Please use a different email or login.", 'error')
            return redirect(url_for('register'))

        # Create new user
        db.users.insert_one({
            'username': username,
            'email': email,
            'password': generate_password_hash(password)
        })
        flash("Registration successful! Please login with your new account.", 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# --- Admin endpoints for medicine inventory ---
@app.route('/admin/medicines', methods=['GET'])
def get_medicines():
    medicines = list(db.medicines.find({}, {'_id': 0}))
    return jsonify(medicines)

@app.route('/admin/medicines/add', methods=['POST'])
def add_medicine():
    data = request.json
    name = data.get('name')
    quantity = data.get('quantity', 0)
    expiry_date = data.get('expiry_date')
    
    # Validate input
    if not name or name.strip() == '':
        return jsonify({'success': False, 'message': 'Medicine name is required'})
    
    # Check if medicine exists (it should exist since it's from expired list)
    existing_medicine = db.medicines.find_one({'name': name})
    if not existing_medicine:
        return jsonify({'success': False, 'message': f'Medicine "{name}" not found'})
    
    # Add quantity to existing medicine and update expiry date
    update_data = {'$inc': {'quantity': quantity}}
    if expiry_date:
        update_data['$set'] = {'expiry_date': expiry_date}
    
    result = db.medicines.update_one({'name': name}, update_data)
    
    if result.modified_count > 0:
        return jsonify({'success': True, 'message': f'Added {quantity} units of "{name}" to inventory with new expiry date'})
    else:
        return jsonify({'success': False, 'message': f'Failed to update medicine "{name}"'})

@app.route('/admin/medicines/bulk-update', methods=['POST'])
def bulk_update_medicines():
    data = request.json
    updates = data.get('updates', [])
    
    if not updates:
        return jsonify({'success': False, 'message': 'No updates provided'})
    
    success_count = 0
    for update in updates:
        name = update.get('name')
        quantity = update.get('quantity')
        
        if not name or quantity is None:
            continue
        
        # Update the medicine quantity directly
        result = db.medicines.update_one(
            {'name': name}, 
            {'$set': {'quantity': quantity}}
        )
        
        if result.modified_count > 0:
            success_count += 1
    
    return jsonify({
        'success': True, 
        'message': f'Successfully updated {success_count} out of {len(updates)} medicines'
    })

@app.route('/admin/medicines/update', methods=['POST'])
def update_medicine_quantity():
    data = request.json
    name = data['name']
    
    if 'change' in data:
        change = int(data['change'])  # +1 or -1
        
        # If trying to decrease quantity, check if there's enough
        if change < 0:
            medicine = db.medicines.find_one({'name': name})
            if medicine and medicine.get('quantity', 0) + change < 0:
                return jsonify({'success': False, 'message': 'Cannot decrease quantity below zero', 'quantity': medicine.get('quantity', 0)})
        
        # Update quantity
        db.medicines.update_one({'name': name}, {'$inc': {'quantity': change}})
    
    if 'expiry_date' in data:
        db.medicines.update_one({'name': name}, {'$set': {'expiry_date': data['expiry_date']}})

    
    med = db.medicines.find_one({'name': name}, {'_id': 0})
    if med and 'expiry_date' in med:
        med['expiry_status'] = calculate_expiry_status(med['expiry_date'])
    return jsonify(med)

# --- Update admin_dashboard to pass medicines to template ---
def calculate_expiry_status(expiry_date):
    if not expiry_date:
        return 'valid'
    
    today = datetime.datetime.now().date()
    expiry = datetime.datetime.strptime(expiry_date, '%Y-%m-%d').date()
    days_until_expiry = (expiry - today).days
    
    if days_until_expiry < 0:
        return 'expired'
    elif days_until_expiry <= 30:
        return 'expiring_soon'
    return 'valid'

def auto_update_expired_medicines():
    """Automatically set quantity to 0 for expired medicines"""
    today = datetime.datetime.now().date()
    
    # Find all medicines that are expired but still have quantity > 0
    expired_medicines = db.medicines.find({
        'expiry_date': {'$exists': True, '$ne': None},
        'quantity': {'$gt': 0}
    })
    
    updated_count = 0
    for medicine in expired_medicines:
        try:
            expiry_date = datetime.datetime.strptime(medicine['expiry_date'], '%Y-%m-%d').date()
            if expiry_date < today:
                # Set quantity to 0 for expired medicine
                db.medicines.update_one(
                    {'_id': medicine['_id']}, 
                    {'$set': {'quantity': 0}}
                )
                updated_count += 1
                print(f"🔄 Auto-updated: {medicine['name']} quantity set to 0 (expired on {medicine['expiry_date']})")
        except Exception as e:
            print(f"Error processing medicine {medicine.get('name', 'Unknown')}: {e}")
    
    if updated_count > 0:
        print(f"✅ Auto-updated {updated_count} expired medicines to quantity 0")
    
    return updated_count

def get_expired_medicines():
    """Get all expired medicines that need to be removed"""
    today = datetime.datetime.now().date()
    
    expired_medicines = []
    medicines = db.medicines.find({
        'expiry_date': {'$exists': True, '$ne': None}
    })
    
    for medicine in medicines:
        try:
            expiry_date = datetime.datetime.strptime(medicine['expiry_date'], '%Y-%m-%d').date()
            if expiry_date < today:
                expired_medicines.append({
                    'name': medicine['name'],
                    'expiry_date': medicine['expiry_date'],
                    'quantity': medicine.get('quantity', 0),
                    '_id': str(medicine['_id'])
                })
        except Exception as e:
            print(f"Error processing medicine {medicine.get('name', 'Unknown')}: {e}")
    
    return expired_medicines



@app.route('/admin_dashboard')
def admin_dashboard():
    # Only allow admin
    if session.get('user', {}).get('role') != 'admin':
        return redirect(url_for('login'))
    
    # Auto-update expired medicines to quantity 0
    auto_update_expired_medicines()
    
    medicines = list(db.medicines.find({}, {'_id': 0}))
    # Calculate expiry status for each medicine
    for medicine in medicines:
        expiry_date = medicine.get('expiry_date')
        medicine['expiry_status'] = calculate_expiry_status(expiry_date)
    
    # Get expired medicines for disclaimer
    expired_medicines = get_expired_medicines()
    
    # Build user info: username, list of chat sessions (date, symptoms, dispensed)
    users = []
    for user_doc in db.users.find({}, {'username': 1, 'chat_history': 1}):
        username = user_doc.get('username', 'Unknown')
        sessions = []
        for chat in user_doc.get('chat_history', []):
            # Use the 'symptoms' field for actual symptoms marked Yes
            symptoms = chat.get('symptoms', [])
            dispensed = chat.get('dispensed_medicine') or chat.get('medicine') or None
            date = chat.get('timestamp', 'NA')
            sessions.append({
                'date': date,
                'symptoms': symptoms,
                'dispensed': dispensed if dispensed else 'NA'
            })
        users.append({
            'username': username,
            'sessions': sessions[::-1]  # most recent first
        })
    return render_template('admin_dashboard.html', medicines=medicines, users=users, expired_medicines=expired_medicines)

@app.route('/user_dashboard')
def user_dashboard():
    # Auto-update expired medicines to quantity 0
    auto_update_expired_medicines()
    
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
    
    # Parse medicine names similar to medicine_gateway
    med_names = []
    if medicine.lower().startswith('both '):
        med_names = [m.strip() for m in medicine[5:].split(' and ')]
    elif medicine.lower().startswith('all '):
        med_names = [m.strip() for m in medicine[4:].split(' and ')]
    elif medicine.lower().endswith(' only'):
        med_names = [medicine[:-5].strip()]
    else:
        med_names = [medicine.strip()]
    
    # Check if all medicines are available in sufficient quantity
    unavailable_meds = []
    for med in med_names:
        medicine_doc = db.medicines.find_one({'name': med})
        if not medicine_doc or medicine_doc.get('quantity', 0) <= 0:
            unavailable_meds.append(med)
    
    if unavailable_meds:
        # Return error message if any medicine is unavailable
        error_msg = f"Medicine not available: {', '.join(unavailable_meds)}"
        flash(error_msg, 'error')
        return redirect(url_for('user_dashboard'))
    
    # If implementation is not complete yet, show a message
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
    # If this is a greeting request, just return the greeting
    if request.json.get('greeting'):
        return jsonify({'response': bot_instance.get_greeting()})
    # Process user input
    user_input = request.json['message']
    # Check chat history for last 2 days BEFORE processing user input
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
    # If this is the third or more chat in 2 days, block dispensing flow
    if len(recent_chats) >= 2:
        response = "Consult a doctor since you're falling sick frequently."
        # Reset chatbot state to done so user can't continue
        bot_instance.state['step'] = 'done'
        session['chatbot_state'] = bot_instance.get_serializable_state()
        return jsonify({'response': response})
    # Otherwise, continue as normal
    response = bot_instance.get_response(user_input)
    # Always update session state after processing
    session['chatbot_state'] = bot_instance.get_serializable_state()
    # Only store in MongoDB if chat is done or if a payment redirect is returned
    should_save = False
    if bot_instance.state.get('step') == 'done':
        should_save = True
    if isinstance(response, dict) and 'redirect' in response:
        should_save = True
    if should_save:
        user_flags = bot_instance.state.get('user_flags', {})
        symptoms_yes = [k for k, v in user_flags.items() if v.strip().lower() == 'yes']
        all_symptoms = list(user_flags.keys())
        recommendation = response if isinstance(response, str) else ''
        now = datetime.datetime.utcnow()
        timestamp = now.isoformat()
        matched_medicines = []
        rec_lower = recommendation.lower() if isinstance(recommendation, str) else ''
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
            'recommendation': recommendation if isinstance(recommendation, str) else '',
            'timestamp': timestamp,
            'medicine': matched_medicines if matched_medicines else None
        }
        dispensed_medicine = bot_instance.state.get('dispense_selection', None)
        if dispensed_medicine:
            if dispensed_medicine.lower().startswith("no, i don"):
                chat_entry['dispensed_medicine'] = 'NA'
            else:
                chat_entry['dispensed_medicine'] = dispensed_medicine
        followup_answers = bot_instance.state.get('followup_answers', {})
        if followup_answers:
            chat_entry['followup_answers'] = followup_answers
        users_collection.update_one(
            {'username': username},
            {'$push': {'chat_history': chat_entry}},
            upsert=True
        )
    if isinstance(response, dict) and 'redirect' in response:
        return jsonify(response)
    return jsonify({'response': response})

# --- Decrement quantity on dispense ---
@app.route('/medicine_gateway', methods=['GET', 'POST'])
def medicine_gateway():
    if request.method == 'POST':
        # Auto-update expired medicines to quantity 0 before dispensing
        auto_update_expired_medicines()
        
        username = session.get('username')
        medicine = request.args.get('medicine', '')
        # Use Indian timezone for payment_time
        from datetime import datetime
        india_tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(india_tz).isoformat()

        # Decrement quantity for each medicine dispensed
        # Handle combinations like 'Both A and B', 'A only', 'All A and B and C' 
        import re
        med_names = []
        if medicine.lower().startswith('both '):
            med_names = [m.strip() for m in medicine[5:].split(' and ')]
        elif medicine.lower().startswith('all '):
            med_names = [m.strip() for m in medicine[4:].split(' and ')]
        elif medicine.lower().endswith(' only'):
            med_names = [medicine[:-5].strip()]
        else:
            med_names = [medicine.strip()]

        # Check if all medicines are available in sufficient quantity
        unavailable_meds = []
        for med in med_names:
            medicine_doc = db.medicines.find_one({'name': med})
            if not medicine_doc or medicine_doc.get('quantity', 0) <= 0:
                unavailable_meds.append(med)
        
        if unavailable_meds:
            # Return error message if any medicine is unavailable
            error_msg = f"Medicine not available: {', '.join(unavailable_meds)}"
            flash(error_msg, 'error')
            return redirect(url_for('user_dashboard'))
            
        # If all medicines are available, proceed with dispensing
        for med in med_names:
            db.medicines.update_one({'name': med}, {'$inc': {'quantity': -1}})

        # Use aggregation pipeline to update last chat_history entry
        users_collection.update_one(
            {
                'username': username,
                'chat_history': {'$exists': True, '$ne': []}
            },
            [
                {
                    '$set': {
                        'chat_history': {
                            '$concatArrays': [
                                {
                                    '$slice': [
                                        '$chat_history',
                                        {'$subtract': [{'$size': '$chat_history'}, 1]}
                                    ]
                                },
                                [
                                    {
                                        '$mergeObjects': [
                                            {
                                                '$arrayElemAt': [
                                                    '$chat_history',
                                                    {'$subtract': [{'$size': '$chat_history'}, 1]}
                                                ]
                                            },
                                            {
                                                'payment_status': 'success',
                                                'payment_time': now,
                                                'dispensed_medicine': medicine
                                            }
                                        ]
                                    }
                                ]
                            ]
                        }
                    }
                }
            ]
        )

        # Print 'y' for successful medicine dispense after payment
        print('y')
        try:
            with serial.Serial('COM3', 9800, timeout=2) as ser:
                time.sleep(2)
                ser.write(b'y')
                print("DEBUG: Sent 'y' to Arduino.")
        except serial.SerialException as e:
            print(f"ERROR: Could not write to Arduino: {e}")

        return jsonify({
            'success': True,
            'message': f'Your payment for {medicine} was successful. Medicine is dispensing.'
        })

    medicine = request.args.get('medicine', '')
    return render_template('medicine_gateway.html', medicine=medicine)

@app.route('/admin/learning-stats', methods=['GET'])
def get_learning_stats():
    """Get real-time learning statistics"""
    try:
        # Import here to avoid circular imports
        from symptom_predictor import SymptomPredictor
        from chat import DATASET_PATH
        
        predictor = SymptomPredictor(real_time_learning=True)
        predictor.train_on_csv(DATASET_PATH)
        
        stats = predictor.get_learning_stats()
        
        # Format for display
        formatted_stats = {
            'total_interactions': stats['total_interactions'],
            'real_time_learning_enabled': stats['real_time_learning_enabled'],
            'min_interactions_for_retrain': stats['min_interactions_for_retrain'],
            'last_retrain_count': stats['last_retrain_count'],
            'model_type': stats['model_type'],
            'benchmark_scores': stats['benchmark_scores'],
            'trained': stats['trained'],
            'learning_progress': f"{stats['total_interactions']}/{stats['min_interactions_for_retrain']} interactions for next retrain"
        }
        
        return jsonify(formatted_stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/user-interactions', methods=['GET'])
def get_user_interactions():
    """Get recent user interactions for analysis"""
    try:
        import json
        import os
        
        interaction_file = "user_interactions.json"
        if os.path.exists(interaction_file):
            with open(interaction_file, 'r') as f:
                interactions = json.load(f)
            
            # Return last 20 interactions
            recent_interactions = interactions[-20:] if len(interactions) > 20 else interactions
            
            return jsonify({
                'total_interactions': len(interactions),
                'recent_interactions': recent_interactions
            })
        else:
            return jsonify({
                'total_interactions': 0,
                'recent_interactions': []
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/learning-dashboard')
def learning_dashboard():
    """Admin dashboard for monitoring real-time learning"""
    return render_template('learning_dashboard.html')

@app.route('/admin/auto-update-expired', methods=['POST'])
def trigger_auto_update_expired():
    """Manual trigger to update expired medicines (admin only)"""
    if session.get('user', {}).get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Admin access required'})
    
    updated_count = auto_update_expired_medicines()
    return jsonify({
        'success': True, 
        'message': f'Auto-updated {updated_count} expired medicines to quantity 0'
    })


if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        print('\nShutting down gracefully...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    app.run(debug=True, use_reloader=True)