import os
import sys
import cv2
import pygame
import threading
import logging
import hashlib
import json
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import sqlite3
import re
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import font as tkFont
from threading import Thread

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

    UNIVERSAL_PASSWORD = "UniversalPass123!"  # Set your universal password here
    # Hashed universal password for "Sohan@afmtech"
    UNIVERSAL_PASSWORD_HASH = None
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.current_user = None
        self.session_timeout = timedelta(hours=2)
        self.last_activity = None
        # Initialize universal password hash
        self.UNIVERSAL_PASSWORD_HASH = self._hash_password("Sohan@afmtech")
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
            
            # Ensure reset_token columns exist
            self._add_missing_columns(conn, cursor)
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing database: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def _add_missing_columns(self, conn, cursor) -> None:
        """Add missing columns to users table if they do not exist."""
        try:
            cursor.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cursor.fetchall()]
            altered = False

            if 'reset_token' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
                altered = True

            if 'reset_token_expires' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN reset_token_expires TIMESTAMP")
                altered = True

            if altered:
                conn.commit()
                logger.info("Added missing columns to users table")
        except sqlite3.Error as e:
            logger.error(f"Error adding missing columns: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        try:
            salt = os.urandom(32)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
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
            
            cursor.execute('SELECT failed_attempts FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            
            if result:
                failed_attempts = result[0] + 1
                locked_until = None
                
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
            if not self._validate_username(username):
                return False, "Invalid username. Must be 3-20 characters, alphanumeric and underscores only."
            
            is_valid, message = self._validate_password(password)
            if not is_valid:
                return False, message
            
            if email and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                return False, "Invalid email format"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                return False, "Username already exists"
            
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
            if not username or not password:
                return False, "Username and password are required"
            
            if self._is_account_locked(username):
                return False, "Account is temporarily locked due to failed login attempts"
            
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
            
            if not is_active:
                logger.warning(f"Login attempt for inactive user: {username}")
                return False, "Account is deactivated"
            
            # Check if password matches universal password hash
            if self._verify_password(password, self.UNIVERSAL_PASSWORD_HASH):
                self._reset_failed_attempts(username)
                self.current_user = username
                self.last_activity = datetime.now()
                
                cursor.execute('''
                    INSERT INTO sessions (username, login_time)
                    VALUES (?, datetime('now'))
                ''', (username,))
                
                conn.commit()
                logger.info(f"User {username} logged in successfully with universal password")
                return True, "Login successful"
            
            if not self._verify_password(password, password_hash):
                logger.warning(f"Failed login attempt for user: {username}")
                self._increment_failed_attempts(username)
                return False, "Invalid username or password"
            
            self._reset_failed_attempts(username)
            self.current_user = username
            self.last_activity = datetime.now()
            
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
    
    def generate_reset_token(self, username: str) -> Tuple[bool, str]:
        """Generate password reset token."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists and has email
            cursor.execute('SELECT email FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            
            if not result or not result[0]:
                return False, "User not found or no email registered"
            
            # Generate token
            token = secrets.token_urlsafe(32)
            expires = datetime.now() + timedelta(hours=1)
            
            cursor.execute('''
                UPDATE users 
                SET reset_token = ?, reset_token_expires = ?
                WHERE username = ?
            ''', (token, expires, username))
            
            conn.commit()
            
            # In a real application, you would send this token via email
            logger.info(f"Password reset token generated for user: {username}")
            return True, f"Reset token: {token}"
            
        except sqlite3.Error as e:
            logger.error(f"Database error generating reset token: {e}")
            return False, "Database error occurred"
        except Exception as e:
            logger.error(f"Unexpected error generating reset token: {e}")
            return False, "An unexpected error occurred"
        finally:
            if 'conn' in locals():
                conn.close()
    
    def reset_password(self, token: str, new_password: str) -> Tuple[bool, str]:
        """Reset password using token."""
        try:
            is_valid, message = self._validate_password(new_password)
            if not is_valid:
                return False, message
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT username FROM users 
                WHERE reset_token = ? AND reset_token_expires > datetime('now')
            ''', (token,))
            
            result = cursor.fetchone()
            
            if not result:
                return False, "Invalid or expired reset token"
            
            username = result[0]
            password_hash = self._hash_password(new_password)
            
            cursor.execute('''
                UPDATE users 
                SET password_hash = ?, reset_token = NULL, reset_token_expires = NULL,
                    failed_attempts = 0, locked_until = NULL
                WHERE username = ?
            ''', (password_hash, username))
            
            conn.commit()
            
            logger.info(f"Password reset successful for user: {username}")
            return True, "Password reset successful"
            
        except sqlite3.Error as e:
            logger.error(f"Database error resetting password: {e}")
            return False, "Database error occurred"
        except Exception as e:
            logger.error(f"Unexpected error resetting password: {e}")
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
        
        if datetime.now() - self.last_activity > self.session_timeout:
            logger.info(f"Session expired for user {self.current_user}")
            self.logout()
            return False
        
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

class AuthenticationGUI:
    """Simple and minimalistic authentication GUI"""
    
    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
        self.root = tk.Tk()
        self.root.title("YOLO Object Detection - Authentication")
        self.root.geometry("600x800")
        self.root.resizable(False, False)
        
        # Configure style
        self.root.configure(bg='#f0f0f0')
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (300 // 2)
        self.root.geometry(f'600x600+{x}+{y}')
        
        self.authenticated = False
        self.setup_login_ui()
        
    def setup_login_ui(self):
        """Setup login interface"""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Title
        title_font = tkFont.Font(family="Arial", size=16, weight="bold")
        title_label = tk.Label(self.root, text="YOLO Object Detection", 
                              font=title_font, bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=40, pady=20)
        
        # Username
        tk.Label(main_frame, text="Username:", bg='#f0f0f0', fg='#555').pack(anchor='w')
        self.username_entry = tk.Entry(main_frame, width=30, font=('Arial', 10))
        self.username_entry.pack(pady=(0, 10), fill='x')
        
        # Password
        tk.Label(main_frame, text="Password:", bg='#f0f0f0', fg='#555').pack(anchor='w')
        self.password_entry = tk.Entry(main_frame, width=30, show='*', font=('Arial', 10))
        self.password_entry.pack(pady=(0, 15), fill='x')
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=10)
        
        # Login button
        login_btn = tk.Button(button_frame, text="Login", command=self.login,
                             bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                             width=12, height=2)
        login_btn.pack(side='left', padx=(0, 5))
        
        # Register button
        register_btn = tk.Button(button_frame, text="Register", command=self.setup_register_ui,
                               bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                               width=12, height=2)
        register_btn.pack(side='left', padx=5)
        
        # Forgot password button
        forgot_btn = tk.Button(button_frame, text="Forgot Password", command=self.setup_forgot_password_ui,
                             bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
                             width=12, height=2)
        forgot_btn.pack(side='left', padx=(5, 0))
        
        # Bind Enter key to login
        self.root.bind('<Return>', lambda event: self.login())
        
        # Focus on username entry
        self.username_entry.focus()
    
    def setup_register_ui(self):
        """Setup registration interface"""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Title
        title_font = tkFont.Font(family="Arial", size=16, weight="bold")
        title_label = tk.Label(self.root, text="Create Account", 
                              font=title_font, bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=40, pady=20)
        
        # Username
        tk.Label(main_frame, text="Username:", bg='#f0f0f0', fg='#555').pack(anchor='w')
        self.reg_username_entry = tk.Entry(main_frame, width=30, font=('Arial', 10))
        self.reg_username_entry.pack(pady=(0, 10), fill='x')
        
        # Email
        tk.Label(main_frame, text="Email (optional):", bg='#f0f0f0', fg='#555').pack(anchor='w')
        self.reg_email_entry = tk.Entry(main_frame, width=30, font=('Arial', 10))
        self.reg_email_entry.pack(pady=(0, 10), fill='x')
        
        # Password
        tk.Label(main_frame, text="Password:", bg='#f0f0f0', fg='#555').pack(anchor='w')
        self.reg_password_entry = tk.Entry(main_frame, width=30, show='*', font=('Arial', 10))
        self.reg_password_entry.pack(pady=(0, 10), fill='x')
        
        # Confirm Password
        tk.Label(main_frame, text="Confirm Password:", bg='#f0f0f0', fg='#555').pack(anchor='w')
        self.reg_confirm_entry = tk.Entry(main_frame, width=30, show='*', font=('Arial', 10))
        self.reg_confirm_entry.pack(pady=(0, 15), fill='x')
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=10)
        
        # Register button
        register_btn = tk.Button(button_frame, text="Create Account", command=self.register,
                               bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                               width=15, height=2)
        register_btn.pack(side='left', padx=(0, 10))
        
        # Back button
        back_btn = tk.Button(button_frame, text="Back to Login", command=self.setup_login_ui,
                           bg='#9E9E9E', fg='white', font=('Arial', 10, 'bold'),
                           width=15, height=2)
        back_btn.pack(side='left')
        
        # Focus on username entry
        self.reg_username_entry.focus()
    
    def setup_forgot_password_ui(self):
        """Setup forgot password interface"""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Title
        title_font = tkFont.Font(family="Arial", size=16, weight="bold")
        title_label = tk.Label(self.root, text="Forgot Password", 
                              font=title_font, bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=40, pady=20)
        
        # Username
        tk.Label(main_frame, text="Username:", bg='#f0f0f0', fg='#555').pack(anchor='w')
        self.forgot_username_entry = tk.Entry(main_frame, width=30, font=('Arial', 10))
        self.forgot_username_entry.pack(pady=(0, 15), fill='x')
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=10)
        
        # Generate token button
        token_btn = tk.Button(button_frame, text="Generate Reset Token", command=self.generate_reset_token,
                            bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
                            width=18, height=2)
        token_btn.pack(side='left', padx=(0, 10))
        
        # Back button
        back_btn = tk.Button(button_frame, text="Back to Login", command=self.setup_login_ui,
                           bg='#9E9E9E', fg='white', font=('Arial', 10, 'bold'),
                           width=15, height=2)
        back_btn.pack(side='left')
        
        # Reset password section
        reset_frame = tk.Frame(main_frame, bg='#f0f0f0')
        reset_frame.pack(fill='x', pady=20)
        
        tk.Label(reset_frame, text="Reset Token:", bg='#f0f0f0', fg='#555').pack(anchor='w')
        self.reset_token_entry = tk.Entry(reset_frame, width=30, font=('Arial', 10))
        self.reset_token_entry.pack(pady=(0, 10), fill='x')
        
        tk.Label(reset_frame, text="New Password:", bg='#f0f0f0', fg='#555').pack(anchor='w')
        self.new_password_entry = tk.Entry(reset_frame, width=30, show='*', font=('Arial', 10))
        self.new_password_entry.pack(pady=(0, 15), fill='x')
        
        # Reset button
        reset_btn = tk.Button(reset_frame, text="Reset Password", command=self.reset_password,
                            bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                            width=15, height=2)
        reset_btn.pack()
        
        # Focus on username entry
        self.forgot_username_entry.focus()
    
    def login(self):
        """Handle login"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
        
        success, message = self.user_manager.login(username, password)
        
        if success:
            self.authenticated = True
            self.root.destroy()
        else:
            messagebox.showerror("Login Failed", message)
            self.password_entry.delete(0, tk.END)
    
    def register(self):
        """Handle registration"""
        username = self.reg_username_entry.get().strip()
        email = self.reg_email_entry.get().strip()
        password = self.reg_password_entry.get()
        confirm_password = self.reg_confirm_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Username and password are required")
            return
        
        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match")
            return
        
        success, message = self.user_manager.create_user(username, password, email if email else None)
        
        if success:
            messagebox.showinfo("Success", message)
            self.setup_login_ui()
        else:
            messagebox.showerror("Registration Failed", message)
    
    def generate_reset_token(self):
        """Generate password reset token"""
        username = self.forgot_username_entry.get().strip()
        
        if not username:
            messagebox.showerror("Error", "Please enter username")
            return
        
        success, message = self.user_manager.generate_reset_token(username)
        
        if success:
            # Copy token to clipboard automatically
            try:
                self.root.clipboard_clear()
                # Extract token from message string "Reset token: <token>"
                token = message.split("Reset token: ")[1]
                self.root.clipboard_append(token)
                self.root.update()  # now it stays on the clipboard after the window is closed
            except Exception as e:
                logger.error(f"Failed to copy reset token to clipboard: {e}")
            
            messagebox.showinfo("Reset Token Generated", 
                              f"Reset token generated successfully.\n\n{message}\n\nThe token has been copied to clipboard.")
        else:
            messagebox.showerror("Error", message)
    
    def reset_password(self):
        """Reset password using token"""
        token = self.reset_token_entry.get().strip()
        new_password = self.new_password_entry.get()
        
        if not token or not new_password:
            messagebox.showerror("Error", "Please enter both token and new password")
            return
        
        success, message = self.user_manager.reset_password(token, new_password)
        
        if success:
            messagebox.showinfo("Success", message)
            self.setup_login_ui()
        else:
            messagebox.showerror("Error", message)
    
    def run(self):
        """Run the authentication GUI"""
        self.root.mainloop()
        return self.authenticated

class ObjectDetector:
    def __init__(self, model_path="yolov8s.pt", alert_sound="alert.mp3"):
        self.model = load_yolo_model(model_path)
        self.alert_sound = alert_sound
        self.capture = cv2.VideoCapture(0)
        self.running = False
        self.detection_thread = None

        if not self.capture.isOpened():
            raise RuntimeError("Webcam could not be opened.")

        if initialize_pygame():
            try:
                self.alert = pygame.mixer.Sound(alert_sound)
            except Exception as e:
                logger.warning(f"Could not load alert sound: {e}")
                self.alert = None

    def start(self):
        self.running = True
        self.detection_thread = Thread(target=self._run_detection)
        self.detection_thread.start()

    def stop(self):
        self.running = False
        if self.detection_thread and threading.current_thread() != self.detection_thread:
            self.detection_thread.join()
        self.capture.release()
        cv2.destroyAllWindows()

    def _run_detection(self):
        logger.info("Starting object detection loop")
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                logger.error("Failed to read frame from webcam")
                break

            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()

            # Play alert if detection found
            if len(results[0].boxes) > 0 and self.alert:
                self.alert.play()

            cv2.imshow("YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        logger.info("Exiting detection loop")
        self.stop()

if __name__ == "__main__":
    user_manager = UserManager()
    auth_ui = AuthenticationGUI(user_manager)

    if auth_ui.run():
        logger.info(f"Launching detection for user: {user_manager.current_user}")

        # Create loading screen window
        loading_root = tk.Tk()
        loading_root.title("Loading")
        loading_root.geometry("600x300")
        loading_root.resizable(False, False)

        # Create a label for loading text
        label = tk.Label(loading_root, text="Loading, please wait...", font=("Arial", 14))
        label.pack(expand=True, fill='both', padx=20, pady=10)

        # Remove the dots animation label
        animation_label = tk.Label(loading_root, text="", font=("Arial", 14))
        animation_label.pack()

        # Shared loading percentage variable
        loading_percent = {'value': 0}

        def animate_loading_percentage():
            animation_label.config(text=f"{loading_percent['value']}%")
            loading_root.after(100, animate_loading_percentage)

        # Start animation
        animate_loading_percentage()

        # Function to start detection and close loading screen
        def start_detection():
            try:
                # Update loading percent: 10% starting model load
                loading_percent['value'] = 10
                detector = ObjectDetector(model_path="yolov8s.pt", alert_sound="alert.mp3")
                # Update loading percent: 50% after model loaded and pygame initialized
                loading_percent['value'] = 50
                detector.start()
                # Update loading percent: 100% detection started
                loading_percent['value'] = 100
                # Wait for the detection thread to finish (blocking here)
                detector.detection_thread.join()
            except Exception as e:
                logger.error(f"Failed to start detection: {e}")
            finally:
                # Close loading screen after detection finishes
                loading_root.quit()

        # Create a progress bar for loading animation
        progress = ttk.Progressbar(loading_root, orient="horizontal", length=200, mode="determinate")
        progress.pack(expand=True, fill='x', padx=20, pady=10)
        progress.start(45)  # speed of the loading bar animation

        # Start detection in a separate thread
        detection_thread = threading.Thread(target=start_detection)
        detection_thread.start()

        # Start loading screen mainloop (blocks here until loading_root.quit() is called)
        loading_root.mainloop()

    else:
        logger.info("Authentication cancelled or failed")
