Here’s a clean, styled `README.md` file in **Markdown** format for your YOLO Object Detection system with User Authentication:

---

```markdown
<h1 align="center" style="color:#1abc9c;font-family:Verdana;">
  🔒 YOLO Object Detection with Secure User Authentication
</h1>

<p align="center" style="font-size:18px;">
  <strong>AI-powered surveillance system with real-time object detection and user management</strong>
</p>

---

## 🧠 Features

- ✅ **User Authentication** (Sign up, login, logout, session management)
- 🔐 **Secure Password Hashing** with Salt using PBKDF2-HMAC-SHA256
- 🕵️ **YOLOv8 Real-time Object Detection**
- 🚨 **Person Detection Alert** with Sound Notification (via Pygame)
- 🧪 **SQLite3 Database** for users and session tracking
- 🎥 **Live Video Stream** with annotated detections
- 🧼 **Robust Error Handling** and detailed logging

---

## 🏗️ Project Structure

```

claude\_code.py          # Main application file
users.db                # SQLite DB (created on first run)
yolov8s.pt              # YOLOv8 pre-trained weights
alert.mp3 / alert.wav   # Sound file for person detection
detection.log           # Logging output

````

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/yolo-auth-detector.git
cd yolo-auth-detector
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python claude_code.py
```

---

## 📦 Requirements

```
opencv-python
pygame
ultralytics
```

*You can install them via:*

```bash
pip install opencv-python pygame ultralytics
```

---

## 🎮 How it Works

1. **Startup Menu**:

   * Login
   * Create Account
   * Exit

2. **Upon Login**:

   * YOLOv8 model loads
   * Camera feed starts
   * Person detection triggers audio alert
   * User info is displayed on the video

---

## 🛡️ Security Features

* Passwords stored securely with unique salts.
* Accounts lock after **5 failed attempts** for **30 minutes**.
* Sessions auto-expire after **2 hours of inactivity**.

---

## 👤 Author

**Sohan**
🌐 [afmtechz.anvil.app](https://afmtechz.anvil.app)

---

## 📸 Screenshot

![App Preview](preview.png)

---

## 📝 License

This project is for personal or educational use. No license attached.

```

---

Let me know if you'd like:

- An HTML-styled version for GitHub Pages
- A version with collapsible sections
- Additional badges (e.g., Python version, last updated, etc.)

Also let me know your GitHub repo name if you want a commit message and `.gitignore` too.
```
