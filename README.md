
---

# 🔍 YOLO Object Detection with 🔐 Secure Login + UI

**Built with Python, Tkinter, OpenCV, Ultralytics, and Pygame**

<*AI-powered real-time surveillance system with secure user management and simple UI*>

---

## 🧠 Features

* ✅ **User Authentication** (Sign up, login, logout, session management)
* 🔐 **Universal Admin Password** for emergency or override access
* 🖥️ **Minimal Tkinter UI** for login/signup and better usability
* 🕵️ **YOLOv8 Real-time Object Detection**
* 🚨 **Person Detection Alert** with Sound Notification (via Pygame)
* 🧪 **SQLite3 Database** for users and session tracking
* 🎥 **Live Video Stream** with annotated detections
* 🧼 **Robust Error Handling** and detailed logging

---

## 🏗️ Project Structure

```
main.py                 # Main application file
users.db                # SQLite DB (created on first run)
yolov8s.pt              # YOLOv8 pre-trained weights
alert.mp3 / alert.wav   # Sound file for person detection
detection.log           # Logging output
preview.png             # Screenshot of the application
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Afmtechz/YOLOv8_Based_Surveillance_Tracker.git
cd YOLOv8_Based_Surveillance_Tracker
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```

---

## 📦 Requirements

```
opencv-python
pygame
ultralytics
tk
```

*Install them with:*

```bash
pip install opencv-python pygame ultralytics
```

> **Note:** Tkinter is built into Python. If you face issues, install it via system package manager (e.g., `sudo apt install python3-tk` for Ubuntu).

---

## 🎮 How it Works

1. **UI Login Menu**:

   * Login with credentials
   * Or use **Universal Admin Password** for direct access
   * Option to Sign Up or Exit

2. **After Login**:

   * YOLOv8 model loads
   * Camera feed opens
   * Detection of person triggers sound alert
   * User's name shown on screen

---

## 🛡️ Security Features

* Passwords hashed with **PBKDF2-HMAC-SHA256 + Salt**
* Admin override via **universal password**
* Auto lockout after **5 failed attempts** for **30 minutes**
* Session expires after **2 hours of inactivity**

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

---
