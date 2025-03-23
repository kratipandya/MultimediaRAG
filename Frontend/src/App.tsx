import React, { useState } from 'react';
import { Outlet } from "react-router-dom";
import Footer from "./components/Footer.tsx"
import './App.css';
import { AnimatedNoise } from './components/SVGnoise.tsx';


function App(Content: any) {
    return (
        <div className="app-container">
            {/* <div className="Background" /> */}
            <div className="GrainOverlay" />
            {/* <AnimatedNoise seedInterval={1000} className="GrainOverlay" /> */}

            {/* Main content area with centered search */}
            <main className="app-outlet"> <Outlet /> </main>

            {/* Footer */}
            <Footer />
        </div>
    );
}

export default App;