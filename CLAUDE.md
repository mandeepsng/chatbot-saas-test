# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a chatbot SaaS platform consisting of multiple HTML-based landing pages and authentication flows. The project is a frontend-only implementation using vanilla HTML, CSS, and JavaScript without any backend framework or build system.

## Project Structure

- `index.html` - Main landing page for "ChatFlow AI" chatbot platform
- `index2.html` - Alternative landing page for "QuantumFlow" SaaS platform
- `index3.html` - Additional variant landing page
- `index4.html` - Additional variant landing page
- `dashboard.html` - User dashboard interface
- `auth.html` - Authentication hub page
- `auth/` - Authentication flow pages:
  - `signin.html` - Sign-in page
  - `signup.html` - Sign-up page
  - `verify.html` - Email verification page
  - `forgot-password.html` - Password reset page
  - `2fa.html` - Two-factor authentication page

## Technology Stack

- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **3D Graphics**: Three.js for animated particle backgrounds
- **Styling**: CSS Custom Properties (CSS Variables) for theming
- **Typography**: Google Fonts (Inter, JetBrains Mono)
- **No Build System**: Direct HTML files served statically

## Architecture Notes

1. **Static Site**: No package.json, build tools, or server-side framework
2. **Self-Contained Pages**: Each HTML file includes inline CSS and JavaScript
3. **Shared Design System**: Consistent color schemes and styling across pages using CSS variables
4. **Interactive Elements**: 
   - Animated Three.js particle backgrounds with mouse interaction
   - Live chat demo simulations
   - Color theme customization panels
   - Smooth scroll animations and transitions

## Development Workflow

Since this is a static HTML project with no build system:

- **Local Development**: Open HTML files directly in browser or use a local HTTP server
- **No Package Manager**: All dependencies loaded via CDN (Three.js)
- **No Build Process**: Files can be edited directly and refreshed in browser
- **No Testing Framework**: Manual testing in browser
- **No Linting**: No automated code quality tools configured

## Common Development Tasks

To work with this project:
1. Open HTML files directly in a web browser for testing
2. Edit HTML, CSS, and JavaScript directly in the files
3. Use browser developer tools for debugging
4. Test responsive design across different screen sizes
5. Verify cross-browser compatibility manually

## Design System

The project uses CSS custom properties for consistent theming:
- Primary colors: Purple/blue gradient theme by default
- Multiple color schemes available via theme selector
- Responsive design with mobile-first approach
- Dark theme throughout all pages

## Key Features

- Animated particle backgrounds using Three.js
- Interactive chat demo simulations
- Multi-page authentication flow
- Responsive design for mobile and desktop
- Theme customization capabilities
- Smooth scrolling and fade-in animations