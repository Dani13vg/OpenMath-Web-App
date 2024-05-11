import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from helpers import login_required, get_db_connection, allowed_file
from groq_api import get_model_response, format_response

# Configure application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure upload folder and allowed extensions for uploaded images
UPLOAD_FOLDER = os.path.join('static', 'images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
def index():
    """Show index page"""

    return render_template("index.html")

@app.route("/usage")
def usage():
    """Show usage page"""
    return render_template("usage.html")

@app.route("/contact")
def contact():
    """Show contact page"""
    return render_template("contact.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Ensure username was submitted

        username = request.form.get("username")
        password = request.form.get("password")

        # Enusre username and password were submitted
        if not username or not password:
            flash("Must provide username and password", "warning")
            return render_template("login.html")
        
        # Query database for username
        conn = get_db_connection("users.db")
        user = conn.execute("SELECT * FROM users WHERE username = ?", (request.form.get("username"),)).fetchone()
        conn.close()

        # Ensure username exists and password is correct
        if user is None or not check_password_hash(user["password"], request.form.get("password")):
            flash("Invalid username and/or password")
            return render_template("login.html")
        
        # Remember which user has logged in
        session["user_id"] = user["id"]
        session["username"] = user["username"]

        # Redirect user to home page
        flash("You have successfully logged in")
        return redirect("/learn")
    else:
        return render_template("login.html")
    
@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to home page
    flash("You have successfully logged out")
    return redirect("/")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        # Ensure username and password were submitted
        if not username or not password:
            flash("Must provide username and password", "warning")
            return render_template("register.html")

        # Ensure password and confirmation match
        if password != confirmation:
            flash("Password and confirmation must match", "warning")
            return render_template("register.html")
        
        # Query database for username to ensure it does not already exist
        conn = get_db_connection("users.db")
        cur = conn.cursor()  # Create a cursor object using the connection
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user_check = cur.fetchone()
        
        if user_check:
            conn.close()
            flash("Username already exists", "warning")
            return render_template("register.html")
        
        # Insert new user into the database
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                     (username, generate_password_hash(password)))
        conn.commit()
        user_id = cur.lastrowid  # Get the last inserted ID using the cursor
        conn.close()

        # Set user_id and username in session
        session["user_id"] = user_id
        session["username"] = username

        # Redirect user to the form page to continue registration process
        flash("You have successfully registered. Please complete your profile.")
        return redirect(url_for("form"))
    else:
        return render_template("register.html")


@app.route("/form", methods=["GET", "POST"])
@login_required
def form():
    if request.method == "POST":
        full_name = request.form.get("full_name")
        age = int(request.form.get("age"))  # Ensure age is correctly formatted as integer
        likes = request.form.getlist("likes")  # Retrieves all values from checkboxes named 'likes'
        learning_preference = int(request.form.get("learning_preference"))  # Retrieves the slider value as integer

        # Save this information to your database
        conn = get_db_connection("users.db")
        cursor = conn.cursor()
        try:
            # Update the user's data in the database
            cursor.execute("UPDATE users SET full_name = ?, age = ?, likes = ?, learning_preference = ? WHERE id = ?", 
                           (full_name, age, ','.join(likes), learning_preference, session['user_id']))
            conn.commit()
            flash("Information saved successfully!")
        except Exception as e:
            conn.rollback()
            flash(f"An error occurred: {str(e)}", "error")
        finally:
            conn.close()

        # Redirect to user profile upon successful submission
        return redirect(url_for("profile"))
    else:
        # Render the form page if the request is GET
        return render_template("form.html")


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    conn = get_db_connection("users.db")
    cursor = conn.cursor()
    if request.method == "POST":
        username = request.form['username']
        age = request.form['age']
        likes = ','.join(request.form.getlist('likes'))
        learning_preference = request.form['learning_preference']

        try:
            cursor.execute("UPDATE users SET username = ?, age = ?, likes = ?, learning_preference = ? WHERE id = ?", 
                           (username, age, likes, learning_preference, session['user_id']))
            conn.commit()
            flash("Profile updated successfully", "info")
        finally:
            conn.close()

        return redirect(url_for('profile'))
    else:
        cursor.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],))
        user = dict(cursor.fetchone())
        user['likes'] = user['likes'].split(',') if user['likes'] else []
        conn.close()
        return render_template("profile.html", user=user)

# Route to display mathematics topics
@app.route('/learn', methods=['GET'])
@login_required
def learn_mathematics():
    # Define a list of mathematics topics
    math_topics = [
        "Algebra",
        "Calculus",
        "Geometry",
        "Trigonometry",
        "Statistics",
        "Probability",
        "Number Theory"
    ]

    # Render the learn.html template, passing in the math_topics list
    return render_template('learn.html', math_topics=math_topics)

@app.route('/chat/<topic>', methods=['GET', 'POST'])
@login_required
def chat(topic):
    if request.method == 'POST':
        user_input = request.form['user_input']
        conn = get_db_connection("users.db")
        cursor = conn.cursor()
        user_data = cursor.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],)).fetchone()

        if user_data:
            user_data = dict(user_data)
            
        # Initialize or retrieve history from session
        if 'history' not in session:
            session['history'] = []
        
        if user_input == "":
            return jsonify({'response': "Please enter a message."})
        
        # Generate a response using the updated function with memory
        response = get_model_response(user_input, session['history'], user_data)

        # Format the response to replace **text** with bold text
        response = format_response(response)

        # Save user input and bot response to session for memory
        session['history'].append({'role': 'user', 'content': user_input})
        session['history'].append({'role': 'assistant', 'content': response})

        return jsonify({'response': response})

    return render_template('chat.html', initial_topic=topic)


if __name__ == '__main__':
    app.run(debug=True)