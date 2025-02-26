import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import CryptoAnalyzerChat from './components/CryptoAnalyzerChat';

// Ensure the root element exists
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Failed to find the root element');
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <CryptoAnalyzerChat />
  </React.StrictMode>
);