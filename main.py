import os
import sys
import cv2
import pygame
import threading
import logging
import hashlib
import json
import getpass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import sqlite3
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UserManager:
    """User management system with SQLite database"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.current_user = None
        self.session_timeout = timedelta(hours=2)
        self.last_activity = None
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the user database with error handling."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            ''')
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    logout_time TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (username) REFERENCES users (username)
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing database: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        try:
            # Generate salt and hash
            salt = os.urandom(32)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            # Store salt and hash together
            return salt.hex() + ':' + password_hash.hex()
        except Exception as e:
            logger.error(f"Error hashing password: {e}")
            raise
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt_hex, hash_hex = stored_hash.split(':')
            salt = bytes.fromhex(salt_hex)
            stored_password_hash = bytes.fromhex(hash_hex)
            
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return password_hash == stored_password_hash
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    def _validate_username(self, username: str) -> bool:
        """Validate username format."""
        if not username:
            return False
        
        # Username must be 3-20 characters, alphanumeric and underscores only
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
            return False
        
        return True
    
    def _validate_password(self, password: str) -> Tuple[bool, str]:
        """Validate password strength."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is valid"
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT locked_until FROM users 
                WHERE username = ? AND locked_until > datetime('now')
            ''', (username,))
            
            result = cursor.fetchone()
            return result is not None
            
        except sqlite3.Error as e:
            logger.error(f"Database error checking account lock: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking account lock: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _increment_failed_attempts(self, username: str) -> None:
        """Increment failed login attempts and lock account if necessary."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current failed attempts
            cursor.execute('SELECT failed_attempts FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            
            if result:
                failed_attempts = result[0] + 1
                locked_until = None
                
                # Lock account after 5 failed attempts for 30 minutes
                if failed_attempts >= 5:
                    locked_until = datetime.now() + timedelta(minutes=30)
                    logger.warning(f"Account {username} locked due to failed attempts")
                
                cursor.execute('''
                    UPDATE users 
                    SET failed_attempts = ?, locked_until = ?
                    WHERE username = ?
                ''', (failed_attempts, locked_until, username))
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Database error updating failed attempts: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating failed attempts: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _reset_failed_attempts(self, username: str) -> None:
        """Reset failed attempts on successful login."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET failed_attempts = 0, locked_until = NULL, last_login = datetime('now')
                WHERE username = ?
            ''', (username,))
            
            conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Database error resetting failed attempts: {e}")
        except Exception as e:
            logger.error(f"Unexpected error resetting failed attempts: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def create_user(self, username: str, password: str, email: str = None) -> Tuple[bool, str]:
        """Create a new user with comprehensive validation."""
        try:
            # Validate username
            if not self._validate_username(username):
                return False, "Invalid username. Must be 3-20 characters, alphanumeric and underscores only."
            
            # Validate password
            is_valid, message = self._validate_password(password)
            if not is_valid:
                return False, message
            
            # Validate email if provided
            if email and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                return False, "Invalid email format"
            
            # Check if username already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                return False, "Username already exists"
            
            # Hash password and create user
            password_hash = self._hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, password_hash, email)
                VALUES (?, ?, ?)
            ''', (username, password_hash, email))
            
            conn.commit()
            logger.info(f"User {username} created successfully")
            return True, "User created successfully"
            
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error creating user: {e}")
            return False, "Username already exists"
        except sqlite3.Error as e:
            logger.error(f"Database error creating user: {e}")
            return False, "Database error occurred"
        except Exception as e:
            logger.error(f"Unexpected error creating user: {e}")
            return False, "An unexpected error occurred"
        finally:
            if 'conn' in locals():
                conn.close()
    
    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """Authenticate user with comprehensive security checks."""
        try:
            # Validate input
            if not username or not password:
                return False, "Username and password are required"
            
            # Check if account is locked
            if self._is_account_locked(username):
                return False, "Account is temporarily locked due to failed login attempts"
            
            # Get user from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT username, password_hash, is_active 
                FROM users 
                WHERE username = ?
            ''', (username,))
            
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Login attempt for non-existent user: {username}")
                return False, "Invalid username or password"
            
            db_username, password_hash, is_active = result
            
            # Check if account is active
            if not is_active:
                logger.warning(f"Login attempt for inactive user: {username}")
                return False, "Account is deactivated"
            
            # Verify password
            if not self._verify_password(password, password_hash):
                logger.warning(f"Failed login attempt for user: {username}")
                self._increment_failed_attempts(username)
                return False, "Invalid username or password"
            
            # Successful login
            self._reset_failed_attempts(username)
            self.current_user = username
            self.last_activity = datetime.now()
            
            # Log session
            cursor.execute('''
                INSERT INTO sessions (username, login_time)
                VALUES (?, datetime('now'))
            ''', (username,))
            
            conn.commit()
            logger.info(f"User {username} logged in successfully")
            return True, "Login successful"
            
        except sqlite3.Error as e:
            logger.error(f"Database error during login: {e}")
            return False, "Database error occurred"
        except Exception as e:
            logger.error(f"Unexpected error during login: {e}")
            return False, "An unexpected error occurred"
        finally:
            if 'conn' in locals():
                conn.close()
    
    def logout(self) -> None:
        """Logout current user."""
        try:
            if self.current_user:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Update session
                cursor.execute('''
                    UPDATE sessions 
                    SET logout_time = datetime('now'), is_active = 0
                    WHERE username = ? AND is_active = 1
                ''', (self.current_user,))
                
                conn.commit()
                logger.info(f"User {self.current_user} logged out")
                
                self.current_user = None
                self.last_activity = None
                
        except sqlite3.Error as e:
            logger.error(f"Database error during logout: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during logout: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated and session is valid."""
        if not self.current_user or not self.last_activity:
            return False
        
        # Check session timeout
        if datetime.now() - self.last_activity > self.session_timeout:
            logger.info(f"Session expired for user {self.current_user}")
            self.logout()
            return False
        
        # Update last activity
        self.last_activity = datetime.now()
        return True
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get current user information."""
        if not self.is_authenticated():
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT username, email, created_at, last_login
                FROM users 
                WHERE username = ?
            ''', (self.current_user,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'username': result[0],
                    'email': result[1],
                    'created_at': result[2],
                    'last_login': result[3]
                }
            
        except sqlite3.Error as e:
            logger.error(f"Database error getting user info: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting user info: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
        
        return None

def load_yolo_model(model_path: str = 'yolov8s.pt'):
    """Load YOLO model with error handling."""
    try:
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        return model
    except ImportError as e:
        logger.error(f"Failed to import YOLO: {e}")
        logger.error("Please install ultralytics: pip install ultralytics")
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.error("Please ensure the model file exists or check the path")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading YOLO model: {e}")
        raise

def initialize_pygame():
    """Initialize pygame mixer with error handling."""
    try:
        pygame.mixer.init()
        logger.info("Pygame mixer initialized successfully")
        return True
    except pygame.error as e:
        logger.error(f"Failed to initialize pygame mixer: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error initializing pygame: {e}")
        return False

class ObjectDetector:
    """Object detection system with user authentication"""
    
    def __init__(self, model, user_manager: UserManager, camera_index: int = 0):
        self.model = model
        self.user_manager = user_manager
        self.camera_index = camera_index
        self.video_cap = None
        self.sound_playing = False
        self.pygame_available = initialize_pygame()
        
        # Initialize camera
        self._initialize_camera()
        
    def _initialize_camera(self) -> bool:
        """Initialize camera with error handling."""
        try:
            logger.info(f"Initializing camera with index {self.camera_index}")
            self.video_cap = cv2.VideoCapture(self.camera_index)
            
            if not self.video_cap.isOpened():
                raise RuntimeError(f"Failed to open camera with index {self.camera_index}")
                
            # Set camera properties with error handling
            try:
                self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.video_cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception as e:
                logger.warning(f"Failed to set camera properties: {e}")
                
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
            raise
    
    def resource_path(self, relative_path: str) -> str:
        """Get correct path for resources with error handling."""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
            logger.debug(f"Using PyInstaller base path: {base_path}")
        except AttributeError:
            base_path = os.path.abspath(".")
            logger.debug(f"Using current directory as base path: {base_path}")
        
        full_path = os.path.join(base_path, relative_path)
        
        # Verify the file exists
        if not os.path.exists(full_path):
            logger.warning(f"Resource file not found: {full_path}")
            
        return full_path
    
    def play_alert(self) -> None:
        """Play alert sound in a separate thread with error handling."""
        if not self.pygame_available:
            logger.warning("Pygame not available, skipping sound alert")
            return
            
        try:
            self.sound_playing = True
            sound_file = self.resource_path("alert.mp3")
            
            # Check if sound file exists
            if not os.path.exists(sound_file):
                # Try alternative sound file
                sound_file = self.resource_path("alert.wav")
                if not os.path.exists(sound_file):
                    logger.error("No alert sound file found (tried .mp3 and .wav)")
                    return
            
            logger.debug(f"Playing alert sound: {sound_file}")
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            
            # Wait for sound to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
                
        except pygame.error as e:
            logger.error(f"Pygame error while playing sound: {e}")
        except Exception as e:
            logger.error(f"Unexpected error playing alert sound: {e}")
        finally:
            self.sound_playing = False
    
    def get_colors(self, cls_num: int) -> Tuple[int, int, int]:
        """Assign color to each class with error handling."""
        try:
            base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            color_index = cls_num % len(base_colors)
            increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
            
            color = [
                (base_colors[color_index][i] + increments[color_index][i] *
                (cls_num // len(base_colors))) % 256 for i in range(3)
            ]
            return tuple(color)
            
        except Exception as e:
            logger.error(f"Error generating color for class {cls_num}: {e}")
            # Return default red color
            return (255, 0, 0)
    
    def process_frame(self, frame) -> Optional[any]:
        """Process a single frame with error handling."""
        try:
            results = self.model.track(frame, stream=True)
            return results
        except Exception as e:
            logger.error(f"Error processing frame with YOLO: {e}")
            return None
    
    def draw_detections(self, frame, results) -> None:
        """Draw bounding boxes and labels with error handling."""
        try:
            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                    
                classes_names = getattr(result, 'names', {})
                
                for box in result.boxes:
                    try:
                        if box.conf[0] > 0.4:
                            # Extract coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            
                            # Get class name safely
                            class_name = classes_names.get(cls, f"Unknown_{cls}")
                            colour = self.get_colors(cls)
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                            
                            # Draw label
                            label = f'{class_name} {box.conf[0]:.2f}'
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
                            
                            # Trigger alert for person detection
                            if class_name.lower() == 'person' and not self.sound_playing:
                                threading.Thread(target=self.play_alert, daemon=True).start()
                                
                    except Exception as e:
                        logger.error(f"Error processing detection box: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
    
    def draw_user_info(self, frame) -> None:
        """Draw user information on frame."""
        try:
            if self.user_manager.is_authenticated():
                user_info = self.user_manager.get_user_info()
                if user_info:
                    username = user_info['username']
                    text = f"User: {username}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Error drawing user info: {e}")
    
    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """Read frame from camera with error handling."""
        try:
            if self.video_cap is None:
                return False, None
                
            ret, frame = self.video_cap.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                return False, None
                
            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def run_detection(self) -> None:
        """Main detection loop with comprehensive error handling."""
        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        logger.info(f"Starting object detection for user: {self.user_manager.current_user}")
        
        try:
            while True:
                # Check authentication status
                if not self.user_manager.is_authenticated():
                    logger.warning("User session expired or not authenticated")
                    break
                
                ret, frame = self.read_frame()
                
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive frame read failures ({consecutive_failures})")
                        break
                    continue
                
                # Reset failure counter on successful frame read
                consecutive_failures = 0
                frame_count += 1
                
                # Process frame
                results = self.process_frame(frame)
                
                if results is not None:
                    self.draw_detections(frame, results)
                
                # Draw user info
                self.draw_user_info(frame)
                
                # Display frame
                try:
                    cv2.imshow('Object Detection - Authenticated', frame)
                except Exception as e:
                    logger.error(f"Error displaying frame: {e}")
                    break
                
                # Check for quit key
                try:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit key pressed")
                        break
                except Exception as e:
                    logger.error(f"Error checking key press: {e}")
                    break
                
                # Log progress periodically
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
                    
        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in detection loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources with error handling."""
        logger.info("Cleaning up resources")
        
        try:
            if self.video_cap:
                self.video_cap.release()
                logger.info("Camera released")
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")
        
        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")
        except Exception as e:
            logger.error(f"Error closing OpenCV windows: {e}")
        
        try:
            if self.pygame_available:
                pygame.mixer.quit()
                logger.info("Pygame mixer quit")
        except Exception as e:
            logger.error(f"Error quitting pygame mixer: {e}")

def get_user_input(prompt: str, password: bool = False) -> str:
    """Get user input with error handling."""
    try:
        if password:
            return getpass.getpass(prompt)
        else:
            return input(prompt).strip()
    except KeyboardInterrupt:
        logger.info("User cancelled input")
        return ""
    except Exception as e:
        logger.error(f"Error getting user input: {e}")
        return ""

def show_menu():
    """Display the main menu."""
    print("\n" + "="*50)
    print("YOLO Object Detection System")
    print("="*50)
    print("1. Login")
    print("2. Create Account")
    print("3. Exit")
    print("="*50)

def create_account_flow(user_manager: UserManager) -> bool:
    """Handle account creation flow."""
    try:
        print("\n--- Create New Account ---")
        
        username = get_user_input("Enter username: ")
        if not username:
            print("Username cannot be empty")
            return False
        
        password = get_user_input("Enter password: ", password=True)
        if not password:
            print("Password cannot be empty")
            return False
        
        confirm_password = get_user_input("Confirm password: ", password=True)
        if password != confirm_password:
            print("Passwords do not match")
            return False
        
        email = get_user_input("Enter email (optional): ")
        if not email:
            email = None
        
        success, message = user_manager.create_user(username, password, email)
        print(f"\n{message}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in create account flow: {e}")
        print("An error occurred while creating account")
        return False

def login_flow(user_manager: UserManager) -> bool:
    """Handle login flow."""
    try:
        print("\n--- Login ---")
        
        username = get_user_input("Enter username: ")
        if not username:
            print("Username cannot be empty")
            return False
        
        password = get_user_input("Enter password: ", password=True)
        if not password:
            print("Password cannot be empty")
            return False
        
        success, message = user_manager.login(username, password)
        print(f"\n{message}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in login flow: {e}")
        print("An error occurred during login")
        return False

def main():
    """Main function with user authentication."""
    try:
        # Initialize user manager
        user_manager = UserManager()
        
        # Main menu loop
        while True:
            show_menu()
            choice = get_user_input("Enter your choice (1-3): ")
            
            if choice == '1':
                if login_flow(user_manager):
                    break
            elif choice == '2':
                if create_account_flow(user_manager):
                    print("Account created successfully! Please login.")
                    if login_flow(user_manager):
                        break
            elif choice == '3':
                print("Goodbye!")
                return
            else:
                print("Invalid choice. Please try again.")
        
        # User is now authenticated, start detection
        try:
            # Load YOLO model
            model = load_yolo_model('yolov8s.pt')
            
            # Create detector instance
            detector = ObjectDetector(model, user_manager)
            
            # Run detection
            detector.run_detection()
            
        except Exception as e:
            logger.error(f"Error running detection: {e}")
            print("An error occurred while running detection")
        
        # Logout user
        user_manager.logout()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
