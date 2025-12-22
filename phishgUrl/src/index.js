import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';// This imports your App component above
import './index.css';    // If you have a global CSS file

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>

);
