/**
 * ChatBot SaaS Authentication JavaScript
 * Handles signup, login, and authentication flows
 */

// API Configuration
const API_BASE_URL = window.location.origin;

// Utility functions
function showMessage(message, type = 'success') {
    // Create or update message element
    let messageEl = document.getElementById('auth-message');
    if (!messageEl) {
        messageEl = document.createElement('div');
        messageEl.id = 'auth-message';
        messageEl.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 10000;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;
        document.body.appendChild(messageEl);
    }
    
    messageEl.textContent = message;
    messageEl.style.backgroundColor = type === 'success' ? '#10b981' : 
                                     type === 'error' ? '#ef4444' : '#f59e0b';
    messageEl.style.display = 'block';
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        if (messageEl) messageEl.style.display = 'none';
    }, 5000);
}

function showLoading(button, text = 'Processing...') {
    button.disabled = true;
    button.dataset.originalText = button.textContent;
    button.textContent = text;
    button.style.opacity = '0.7';
}

function hideLoading(button) {
    button.disabled = false;
    button.textContent = button.dataset.originalText || 'Submit';
    button.style.opacity = '1';
}

function validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function validatePassword(password) {
    return password.length >= 6; // Minimum 6 characters
}

// Authentication functions
async function signup(formData) {
    const { name, email, password, confirmPassword, company } = formData;
    
    // Validation
    if (!name || !email || !password || !confirmPassword) {
        throw new Error('All fields are required');
    }
    
    if (!validateEmail(email)) {
        throw new Error('Please enter a valid email address');
    }
    
    if (!validatePassword(password)) {
        throw new Error('Password must be at least 6 characters long');
    }
    
    if (password !== confirmPassword) {
        throw new Error('Passwords do not match');
    }
    
    // API call
    const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            email: email,
            password: password,
            company: company || name + "'s Company"
        })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
        throw new Error(data.detail || 'Registration failed');
    }
    
    return data;
}

async function signin(formData) {
    const { email, password } = formData;
    
    // Validation
    if (!email || !password) {
        throw new Error('Email and password are required');
    }
    
    if (!validateEmail(email)) {
        throw new Error('Please enter a valid email address');
    }
    
    // API call
    const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            email: email,
            password: password
        })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
    }
    
    return data;
}

// Form handlers
function handleSignupForm() {
    const form = document.querySelector('.auth-form');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const submitBtn = form.querySelector('.btn-primary');
        if (!submitBtn) return;
        
        showLoading(submitBtn, 'Creating Account...');
        
        try {
            // Get form data
            const formData = {
                name: form.querySelector('input[placeholder*="name"]')?.value || '',
                email: form.querySelector('input[type="email"]')?.value || '',
                password: form.querySelector('input[placeholder*="password"]:not([placeholder*="confirm"])').value || '',
                confirmPassword: form.querySelector('input[placeholder*="confirm"]')?.value || '',
                company: form.querySelector('input[placeholder*="company"]')?.value || ''
            };
            
            const result = await signup(formData);
            
            // Store token
            localStorage.setItem('authToken', result.token);
            localStorage.setItem('user', JSON.stringify(result.user));
            
            showMessage('Account created successfully! Redirecting to dashboard...');
            
            // Redirect to dashboard after success
            setTimeout(() => {
                window.location.href = '../dashboard.html';
            }, 1500);
            
        } catch (error) {
            showMessage(error.message, 'error');
            hideLoading(submitBtn);
        }
    });
}

function handleSigninForm() {
    const form = document.querySelector('.auth-form');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const submitBtn = form.querySelector('.btn-primary');
        if (!submitBtn) return;
        
        showLoading(submitBtn, 'Signing In...');
        
        try {
            // Get form data
            const formData = {
                email: form.querySelector('input[type="email"]').value,
                password: form.querySelector('input[type="password"]').value
            };
            
            const result = await signin(formData);
            
            // Store token
            localStorage.setItem('authToken', result.token);
            localStorage.setItem('user', JSON.stringify(result.user));
            
            showMessage('Login successful! Redirecting to dashboard...');
            
            // Redirect to dashboard after success
            setTimeout(() => {
                window.location.href = '../dashboard.html';
            }, 1500);
            
        } catch (error) {
            showMessage(error.message, 'error');
            hideLoading(submitBtn);
        }
    });
}

function handleForgotPasswordForm() {
    const form = document.querySelector('.auth-form');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const email = form.querySelector('input[type="email"]').value;
        
        if (!validateEmail(email)) {
            showMessage('Please enter a valid email address', 'error');
            return;
        }
        
        // For now, just show a success message
        // TODO: Implement actual password reset
        showMessage('If an account with this email exists, you will receive a password reset link shortly.');
    });
}

// Check authentication status
function checkAuth() {
    const token = localStorage.getItem('authToken');
    const user = localStorage.getItem('user');
    
    return {
        isAuthenticated: !!token,
        token: token,
        user: user ? JSON.parse(user) : null
    };
}

function logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    window.location.href = '../auth/signin.html';
}

// Initialize based on current page
function initAuth() {
    const pathname = window.location.pathname;
    
    if (pathname.includes('signup.html')) {
        handleSignupForm();
    } else if (pathname.includes('signin.html')) {
        handleSigninForm();
    } else if (pathname.includes('forgot-password.html')) {
        handleForgotPasswordForm();
    }
    
    // Add auth check for protected pages
    if (pathname.includes('dashboard.html') || pathname.includes('admin/')) {
        const auth = checkAuth();
        if (!auth.isAuthenticated) {
            window.location.href = 'auth/signin.html';
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initAuth);

// Export functions for global use
window.ChatBotAuth = {
    signup,
    signin,
    checkAuth,
    logout,
    showMessage
};