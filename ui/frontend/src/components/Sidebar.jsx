import React from 'react';

export default function Sidebar({ activeTab, setActiveTab, isConnected }) {
  const navItems = [
    { id: 'chat', label: '💬 Chat Playground' },
    { id: 'compare', label: '⚖ Model Compare' },
    { id: 'dashboard', label: '📊 Dashboard' },
    { id: 'evaluation', label: '🧪 Evaluation' },
    { id: 'model', label: '🧠 Model Architecture' },
    { id: 'training', label: '⚙ Training & SFT' },
  ];

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">🧠</div>
        <div>
          <div className="brand-title">CHIRII AI</div>
          <div className="brand-subtitle">MiniGPT 6.61M</div>
        </div>
      </div>

      <ul className="nav-list">
        {navItems.map((item) => (
          <li key={item.id}>
            <button
              className={`nav-btn ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              {item.label}
            </button>
          </li>
        ))}
      </ul>

      <div className="sidebar-footer">
        <div>Backend: CPU Mode</div>
        <div>PyTorch: 2.12.0+cpu</div>
        <div className={`status-badge ${isConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          {isConnected ? 'API Connected' : 'Connecting...'}
        </div>
      </div>
    </aside>
  );
}
