"""
Authentication Module - NFL Prediction Model
Handles user authentication and session management
"""

import streamlit as st
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict


class AuthManager:
    """Manages user authentication and sessions"""
    
    def __init__(self):
        """Initialize authentication manager"""
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
        if 'last_attempt_time' not in st.session_state:
            st.session_state.last_attempt_time = None
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = None
    
    def get_users(self) -> Dict[str, str]:
        """
        Get user credentials from Streamlit secrets
        Returns dictionary of username: hashed_password
        """
        try:
            # Try to get users from secrets
            if hasattr(st, 'secrets') and 'users' in st.secrets:
                return dict(st.secrets['users'])
            else:
                # Default demo user for development
                # Password: "nfl2025" (hashed with SHA256)
                return {
                    'demo': self.hash_password('nfl2025')
                }
        except Exception:
            # Fallback for local development
            return {
                'demo': self.hash_password('nfl2025')
            }
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verify username and password"""
        users = self.get_users()
        
        if username not in users:
            return False
        
        hashed_password = self.hash_password(password)
        return hmac.compare_digest(hashed_password, users[username])
    
    def check_rate_limit(self) -> bool:
        """Check if user has exceeded login attempts"""
        max_attempts = 5
        lockout_duration = timedelta(minutes=15)
        
        if st.session_state.login_attempts >= max_attempts:
            if st.session_state.last_attempt_time:
                time_since_last = datetime.now() - st.session_state.last_attempt_time
                if time_since_last < lockout_duration:
                    remaining = lockout_duration - time_since_last
                    minutes_left = int(remaining.total_seconds() / 60)
                    return False, minutes_left
                else:
                    # Reset after lockout period
                    st.session_state.login_attempts = 0
                    st.session_state.last_attempt_time = None
        
        return True, 0
    
    def login(self, username: str, password: str) -> bool:
        """
        Attempt to log in user
        Returns True if successful, False otherwise
        """
        # Check rate limit
        can_attempt, minutes_left = self.check_rate_limit()
        if not can_attempt:
            st.error(f"⛔ Too many failed login attempts. Please try again in {minutes_left} minutes.")
            return False
        
        # Verify credentials
        if self.verify_password(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.login_attempts = 0
            st.session_state.session_start_time = datetime.now()
            return True
        else:
            st.session_state.login_attempts += 1
            st.session_state.last_attempt_time = datetime.now()
            remaining_attempts = 5 - st.session_state.login_attempts
            if remaining_attempts > 0:
                st.error(f"❌ Invalid username or password. {remaining_attempts} attempts remaining.")
            return False
    
    def logout(self):
        """Log out current user"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.session_start_time = None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    def get_username(self) -> Optional[str]:
        """Get current username"""
        return st.session_state.get('username')
    
    def get_session_duration(self) -> Optional[timedelta]:
        """Get current session duration"""
        if st.session_state.session_start_time:
            return datetime.now() - st.session_state.session_start_time
        return None
    
    def show_login_page(self):
        """Display login page"""
        # Center the login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
                <div style='text-align: center; padding: 2rem 0;'>
                    <h1 style='color: #1f77b4; font-size: 3rem;'>🏈</h1>
                    <h1 style='color: #1f77b4;'>NFL Prediction Model</h1>
                    <p style='color: #666; font-size: 1.1rem;'>Machine Learning-Powered Game Predictions</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Login form
            st.markdown("### 🔐 Sign In")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    submit = st.form_submit_button("🔓 Login", use_container_width=True, type="primary")
                
                if submit:
                    if not username or not password:
                        st.error("⚠️ Please enter both username and password")
                    else:
                        if self.login(username, password):
                            st.success(f"✅ Welcome, {username}!")
                            st.balloons()
                            st.rerun()
            
            # Demo credentials hint
            with st.expander("ℹ️ Demo Access"):
                st.info("""
                **Demo Credentials:**
                - Username: `demo`
                - Password: `nfl2025`
                
                This is a demonstration account. In production, use secure credentials stored in Streamlit secrets.
                """)
            
            # Features list
            st.markdown("---")
            st.markdown("### ✨ Features")
            col_feat1, col_feat2 = st.columns(2)
            
            with col_feat1:
                st.markdown("""
                - 🎯 **87.96% Accuracy**
                - 📊 **75+ Features**
                - 🏈 **2010-2025 Data**
                - 🔮 **Weekly Predictions**
                """)
            
            with col_feat2:
                st.markdown("""
                - 📈 **Visual Analytics**
                - 🎨 **Team Logos & Colors**
                - 🔍 **Detailed Analysis**
                - 💾 **Export Results**
                """)
            
            # Footer
            st.markdown("---")
            st.caption("Built with ❤️ using Python, Streamlit, and Scikit-learn")


class RateLimiter:
    """Rate limiting for API calls and predictions"""
    
    def __init__(self, max_requests: int = 10, window_minutes: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in time window
            window_minutes: Time window in minutes
        """
        self.max_requests = max_requests
        self.window = timedelta(minutes=window_minutes)
        
        if 'rate_limit_requests' not in st.session_state:
            st.session_state.rate_limit_requests = []
    
    def can_make_request(self) -> tuple[bool, Optional[int]]:
        """
        Check if user can make a request
        
        Returns:
            (can_request, requests_remaining)
        """
        now = datetime.now()
        
        # Remove requests outside the time window
        st.session_state.rate_limit_requests = [
            req_time for req_time in st.session_state.rate_limit_requests
            if now - req_time < self.window
        ]
        
        requests_made = len(st.session_state.rate_limit_requests)
        requests_remaining = self.max_requests - requests_made
        
        if requests_made >= self.max_requests:
            return False, 0
        
        return True, requests_remaining
    
    def record_request(self):
        """Record a request"""
        st.session_state.rate_limit_requests.append(datetime.now())
    
    def get_reset_time(self) -> Optional[datetime]:
        """Get time when rate limit resets"""
        if st.session_state.rate_limit_requests:
            oldest_request = min(st.session_state.rate_limit_requests)
            return oldest_request + self.window
        return None


def require_auth(func):
    """Decorator to require authentication for a function"""
    def wrapper(*args, **kwargs):
        auth = AuthManager()
        if not auth.is_authenticated():
            auth.show_login_page()
            st.stop()
        return func(*args, **kwargs)
    return wrapper

