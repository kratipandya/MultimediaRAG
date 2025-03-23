import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import './styles/index.css';
import App from "./App";
import HomePage from "./pages/HomePage";
import ResultPage from "./pages/ResultPage";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
    <BrowserRouter>
        <Routes>
            {/* App layout wraps all pages */}
            <Route path="/" element={<App />}>
                {/* Default homepage */}
                <Route index element={<HomePage />} />
                <Route path="search" element={<ResultPage />} />
                {/* Dynamic search query */}
                <Route path="search/:query" element={<ResultPage />} />

                {/* Catch-all 404 */}
                <Route path="*" element={<NotFound />} />
            </Route>
        </Routes>
    </BrowserRouter>
);

// 404 Page Component
function NotFound() {
    return (
        <div className="main justify-self-center">
            <h2>404 - Page Not Found</h2>
        </div>
    );
}