
// File: src/components/Navbar.jsx
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="navbar">
        <div className="navbar-container">
            <Link to="/" className="navbar-logo">
                ReactProject
            </Link>

            <div className="menu-icon" onClick={toggleMenu}>
                <i className={isMenuOpen ? 'fas fa-times' : 'fas fa-bars'} />
            </div>
            <ul className={isMenuOpen ? 'nav-menu active' : 'nav-menu'}>
                <li className="nav-item">
                    <Link to="/" className="nav-link" onClick={toggleMenu}>
                        Home
                    </Link>
                </li>
                <li className="nav-item">
                    <Link to="/about" className="nav-link" onClick={toggleMenu}>
                        About
                    </Link>
                </li>
                <li className="nav-item">
                    <Link to="/contact" className="nav-link" onClick={toggleMenu}>
                        Contact
                    </Link>
                </li>
            </ul>
        </div>
    </nav>
  );
}

export default Navbar;
