import React from 'react';
import './Footer.css';

function Footer() {
  return (
    <footer className="footer pt-2">
        <div className="footer-content">
            <div className="footer-section">
                <h3 className="footer-heading">About</h3>
                <a href="#" className="footer-link">Our Story</a>
                <a href="#" className="footer-link">Team</a>
                <a href="#" className="footer-link">Project</a>
            </div>

            <div className="footer-section">
                <h3 className="footer-heading">Support</h3>
                <a href="#" className="footer-link">Common QA</a>
                <a href="#" className="footer-link">Contact Us</a>
                <a href="#" className="footer-link">FAQ</a>
            </div>

            <div className="footer-section">
                <h3 className="footer-heading">Legal</h3>
                <a href="#" className="footer-link">Terms</a>
                <a href="#" className="footer-link">Privacy</a>
                <a href="#" className="footer-link">Cookies</a>
            </div>
        </div>

        <div className="copyright">
        <p>© 2025 ArXiv RAG Powered Search Engine. All rights reserved.</p>
        </div>
    </footer>
  );
}

export default Footer;
