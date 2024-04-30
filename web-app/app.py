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

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username and password were submitted
        if not request.form.get("username") or not request.form.get("password"):
            flash("Must provide username and password", "warning")
            return render_template("register.html")
        
        # Ensure password and confirmation match
        if request.form.get("password") != request.form.get("confirmation"):
            flash("Password and confirmation must match", "warning")
            return render_template("register.html")
        
        # Query database for username
        conn = get_db_connection("users.db")
        username = conn.execute("SELECT * FROM users WHERE username = ?", (request.form.get("username"),)).fetchone()
        conn.close()

        # Ensure username does not already exist
        if username:
            flash("Username already exists", "warning")
            return render_template("register.html")
        
        # Insert new user into database
        conn = get_db_connection("users.db")
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (request.form.get("username"), generate_password_hash(request.form.get("password"))))
        conn.commit()
        conn.close()

        # Redirect user to login page
        flash("You have successfully registered")
        return redirect("/login")
    else:
        return render_template("register.html")
    
@app.route("/usage")
def usage():
    """Show usage page"""
    return render_template("usage.html")

@app.route("/contact")
def contact():
    """Show contact page"""
    return render_template("contact.html")

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    conn = get_db_connection("users.db")
    if request.method == "POST":
        # Here you could retrieve additional form data for updating the user profile
        username = request.form['username']
        # More fields can be added as needed

        # Update user information in the database
        try:
            conn.execute("UPDATE users SET username = ? WHERE id = ?", (username, session['user_id']))
            conn.commit()
            flash("Profile updated successfully", "info")
        except Exception as e:
            flash("Error updating profile: " + str(e), "error")
        finally:
            conn.close()
        
        return render_template("profile.html", user={'username': username})
    else:
        # Fetch current user information from the database
        user = conn.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],)).fetchone()
        conn.close()
        if user:
            return render_template("profile.html", user=user)
        else:
            flash("User not found", "error")
            return redirect(url_for('index'))

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
        "Number Theory",
        "Differential Equations",
        "Linear Algebra",
        "Discrete Mathematics"
    ]

    # Render the learn.html template, passing in the math_topics list
    return render_template('learn.html', math_topics=math_topics)

@app.route('/chat/<topic>', methods=['GET', 'POST'])
@login_required
def chat(topic):
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Initialize or retrieve history from session
        if 'history' not in session:
            session['history'] = []
        
        if user_input == "":
            return jsonify({'response': "Please enter a message."})
        
        # Generate a response using the updated function with memory
        response = get_model_response(user_input, session['history'])

        # Format the response to replace **text** with bold text
        response = format_response(response)

        # Save user input and bot response to session for memory
        session['history'].append({'role': 'user', 'content': user_input})
        session['history'].append({'role': 'assistant', 'content': response})

        return jsonify({'response': response})

    return render_template('chat.html', initial_topic=topic)


if __name__ == '__main__':
    app.run(debug=True)