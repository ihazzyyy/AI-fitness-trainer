# ══════════════════════════════════════════════
#  app.py  —  AI Fitness Trainer Web App
#  Run: python app.py
#  Open: http://localhost:5000
# ══════════════════════════════════════════════

from flask import (Flask, render_template, request, redirect,
                   url_for, session, Response, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash
import cv2, threading, time, json
from database import init_db, get_db
from camera import CameraStream

app = Flask(__name__)
app.secret_key = "fitness_ai_secret_2024"

init_db()

camera = None  # starts only when trainer page is opened

def get_camera():
    global camera
    if camera is None:
        camera = CameraStream()
    return camera

# ── Auth ──────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=?",
                          (username,)).fetchone()
        if user and check_password_hash(user["password"], password):
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        error = "Invalid username or password"
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET","POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE username=?",
                              (username,)).fetchone()
        if existing:
            error = "Username already taken"
        else:
            db.execute("INSERT INTO users (username, password) VALUES (?,?)",
                       (username, generate_password_hash(password)))
            db.commit()
            return redirect(url_for("login"))
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── Dashboard ─────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    db  = get_db()
    uid = session["user_id"]

    # Total reps all time
    total_reps = db.execute(
        "SELECT COALESCE(SUM(reps),0) FROM workout_logs WHERE user_id=?", (uid,)
    ).fetchone()[0]

    # Total sessions
    total_sessions = db.execute(
        "SELECT COUNT(DISTINCT session_id) FROM workout_logs WHERE user_id=?", (uid,)
    ).fetchone()[0]

    # Recent workouts (last 10)
    recent = db.execute(
        """SELECT exercise, reps, sets, created_at
           FROM workout_logs WHERE user_id=?
           ORDER BY created_at DESC LIMIT 10""", (uid,)
    ).fetchall()

    # PRs per exercise
    prs = db.execute(
        """SELECT exercise, MAX(reps) as max_reps
           FROM workout_logs WHERE user_id=?
           GROUP BY exercise""", (uid,)
    ).fetchall()

    # Weekly reps (last 7 days) for chart
    weekly = db.execute(
        """SELECT DATE(created_at) as day, SUM(reps) as total
           FROM workout_logs WHERE user_id=?
           AND created_at >= DATE('now','-7 days')
           GROUP BY day ORDER BY day""", (uid,)
    ).fetchall()

    return render_template("dashboard.html",
        username=session["username"],
        total_reps=total_reps,
        total_sessions=total_sessions,
        recent=recent,
        prs=prs,
        weekly=json.dumps([{"day": r["day"], "total": r["total"]} for r in weekly])
    )

# ── Trainer (camera page) ─────────────────────
@app.route("/trainer")
def trainer():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("trainer.html", username=session["username"])

@app.route("/video_feed")
def video_feed():
    return Response(get_camera().generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_exercise/<exercise>")
def set_exercise(exercise):
    get_camera().set_exercise(exercise)
    return jsonify({"status": "ok", "exercise": exercise})

@app.route("/get_stats")
def get_stats():
    return jsonify(get_camera().get_stats())

@app.route("/reset_counter")
def reset_counter():
    get_camera().reset()
    return jsonify({"status": "reset"})

@app.route("/save_workout", methods=["POST"])
def save_workout():
    if "user_id" not in session:
        return jsonify({"error": "not logged in"}), 401
    data     = request.json
    db       = get_db()
    uid      = session["user_id"]
    exercise = data.get("exercise")
    reps     = data.get("reps", 0)
    sets     = data.get("sets", 0)

    # Generate session id (timestamp based)
    sess_id = str(int(time.time()))

    db.execute(
        "INSERT INTO workout_logs (user_id, session_id, exercise, reps, sets) VALUES (?,?,?,?,?)",
        (uid, sess_id, exercise, reps, sets)
    )

    # Check PR
    pr = db.execute(
        "SELECT MAX(reps) as max_reps FROM workout_logs WHERE user_id=? AND exercise=?",
        (uid, exercise)
    ).fetchone()["max_reps"] or 0

    is_pr = reps > pr
    db.commit()

    return jsonify({"status": "saved", "is_pr": is_pr})

# ── History ───────────────────────────────────
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))
    db  = get_db()
    uid = session["user_id"]
    logs = db.execute(
        """SELECT exercise, reps, sets, created_at
           FROM workout_logs WHERE user_id=?
           ORDER BY created_at DESC""", (uid,)
    ).fetchall()
    return render_template("history.html",
                           username=session["username"], logs=logs)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)